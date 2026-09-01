from typing import Callable

from env_config import make_env
import numpy as np
from utils.gif_recorder import save_gif
from utils.metrics import save_csv, summarize, run_episodes

SLOWER, IDLE, FASTER = 0, 1, 2

# Above this speed an agent inside the junction box is treated as crossing at
# pace (committed) rather than stalled; sits between the 0 and 4.5 m/s targets.
_COMMIT_SPEED = 4.0
# An agent may hold at the box edge only this far past the entrance; deeper in,
# waiting is more dangerous than clearing, so it commits.
_HOLD_DEPTH = 3.0
# Time-headway thresholds for a vehicle ahead on the ego's own track: brake
# under the first, stop accelerating under the second. Classic car-following
# guard — CPA alone reacts too late in a platoon, where closing rate stays
# near zero until the leader suddenly brakes.
_HEADWAY_BRAKE_S = 0.9
_HEADWAY_COAST_S = 1.6
_VEHICLE_LENGTH = 5.0
_LANE_HALFWIDTH = 2.0


def _present_rows(obs_i: np.ndarray) -> np.ndarray:
    """Rows 1+ of an agent's (15, 7) observation that hold a real vehicle."""
    others = obs_i[1:]
    return others[others[:, 0] > 0]


def _drop_row_at(rows: np.ndarray, position: np.ndarray, eps: float) -> np.ndarray:
    """Remove the row (if any) whose x/y both match `position` within eps."""
    if len(rows) == 0:
        return rows
    matches = (np.abs(rows[:, 1] - position[0]) < eps) & (np.abs(rows[:, 2] - position[1]) < eps)
    return rows[~matches]


def _min_time_to_conflict(position, velocity, threat_rows, conflict_radius, min_closing):
    """Smallest time-to-conflict over the threats on an actual collision course.

    A threat's closing rate is the speed at which the gap shrinks; vehicles
    with closing rate at or below `min_closing` (receding, parallel, or
    near-static) are ignored entirely, so they can never cause a brake.
    Of the rest, only vehicles whose closest point of approach passes within
    `conflict_radius` count — a car sweeping by a lane over is not a conflict."""
    min_ttc = np.inf
    for row in threat_rows:
        rel_p = row[1:3] - position
        rel_v = row[3:5] - velocity
        gap = max(float(np.hypot(rel_p[0], rel_p[1])), 1e-6)
        approach = -float(np.dot(rel_p, rel_v))  # > 0 iff the gap is shrinking
        if approach / gap <= min_closing:
            continue
        t_cpa = approach / float(np.dot(rel_v, rel_v))
        miss = rel_p + rel_v * t_cpa
        if float(np.hypot(miss[0], miss[1])) < conflict_radius:
            min_ttc = min(min_ttc, t_cpa)
    return min_ttc


def _lead_gap(position, velocity, rows):
    """Bumper-to-bumper gap to the nearest vehicle ahead on the ego's own
    track (within half a lane laterally), or inf if the track is clear."""
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < 0.1:
        return np.inf  # no direction of travel while stopped
    heading = velocity / speed
    best = np.inf
    for row in rows:
        rel_p = row[1:3] - position
        along = float(np.dot(rel_p, heading))
        lateral = abs(float(rel_p[0] * heading[1] - rel_p[1] * heading[0]))
        if along > 0.0 and lateral < _LANE_HALFWIDTH:
            best = min(best, along - _VEHICLE_LENGTH)
    return best


def _reaches_box_within(position, velocity, occupied_radius, horizon, min_closing):
    """True if a vehicle is inside the junction box, or moving toward it fast
    enough to reach it within `horizon` seconds (radial approximation, which
    also covers turning vehicles that straight-line extrapolation misses)."""
    d = float(np.hypot(position[0], position[1]))
    if d < occupied_radius:
        return True
    approach_rate = -float(np.dot(position, velocity)) / max(d, 1e-6)
    return approach_rate > min_closing and (d - occupied_radius) / approach_rate < horizon


def _right_of_way(entry_step, dist, tie_eps):
    """Deterministic winner: earlier arrival at the junction zone; before either
    has entered, the closer agent; every tie falls back to agent 0."""
    e0, e1 = entry_step
    if e0 is not None and e1 is not None:
        return 0 if e0 <= e1 else 1
    if e0 is not None:
        return 0
    if e1 is not None:
        return 1
    return 1 if dist[1] < dist[0] - tie_eps else 0


def make_rule_policy(
    junction_radius: float = 20.0,
    conflict_ttc: float = 2.0,
    release_margin: float = 0.5,
    occupied_radius: float = 18.0,
    conflict_radius: float = 4.0,
    min_closing: float = 0.5,
    tie_eps: float = 0.5,
    match_eps: float = 1.0,
) -> Callable:
    """Build a deterministic rule policy for the two controlled agents.

    junction_radius: start negotiating right-of-way this close to center (m)
    conflict_ttc:    brake when min time-to-conflict drops below this (s)
    release_margin:  keep braking until TTC recovers above conflict_ttc + margin (s)
    occupied_radius: junction-box radius for the occupancy test and the yield
                     stop-line (m); the physical box half-width is ~11 m, and
                     a line outside it keeps waiting agents clear of the path
                     that left-turning traffic sweeps near the corners
    conflict_radius: a vehicle is a conflict only if its closest point of
                     approach passes within this distance of the ego (m)
    min_closing:     closing rates at or below this never count as threats,
                     and slower-moving vehicles count as stationary (m/s)
    tie_eps:         distance slack for pre-entry right-of-way ties (m)
    match_eps:       position tolerance to spot the other agent in a row (m)

    Returns policy_fn(obs) -> (action0, action1). The policy is stateful
    (hysteresis latch + arrival order), so it exposes policy_fn.reset(),
    which run_episodes calls at every episode start, and policy_fn.params.
    """
    state = {}

    def reset():
        state["step"] = 0
        state["entry_step"] = [None, None]
        state["braking"] = [False, False]

    reset()

    def policy_fn(obs):
        pos = [obs[i][0, 1:3] for i in range(2)]
        vel = [obs[i][0, 3:5] for i in range(2)]
        dist = [float(np.hypot(pos[i][0], pos[i][1])) for i in range(2)]
        moving = [float(np.hypot(vel[i][0], vel[i][1])) > min_closing for i in range(2)]

        # Arrival order at the junction zone decides right-of-way.
        for i in range(2):
            if state["entry_step"][i] is None and dist[i] < junction_radius:
                state["entry_step"][i] = state["step"]
        state["step"] += 1
        winner = _right_of_way(state["entry_step"], dist, tie_eps)

        # An agent contests the junction while it is moving and inside the box
        # or still heading toward it; a crossed-and-departing winner (or one
        # stalled/crashed on the way) stops contesting, freeing the other agent.
        contesting = [
            moving[i]
            and dist[i] < junction_radius
            and (dist[i] < occupied_radius or float(np.dot(pos[i], vel[i])) < 0.0)
            for i in range(2)
        ]

        actions = []
        for i in range(2):
            j = 1 - i
            others = _present_rows(obs[i])
            speed = float(np.hypot(vel[i][0], vel[i][1]))

            # The right-of-way holder must not brake for the yielding agent,
            # or both stall on mutual detection; the yielder keeps the winner in
            # its threat set as a safety backstop.
            threats = _drop_row_at(others, pos[j], match_eps) if winner == i else others
            min_ttc = _min_time_to_conflict(pos[i], vel[i], threats, conflict_radius, min_closing)

            # Hysteresis: engage below conflict_ttc, release only above
            # conflict_ttc + release_margin, so the brake does not chatter.
            if state["braking"][i]:
                state["braking"][i] = min_ttc < conflict_ttc + release_margin
            else:
                state["braking"][i] = min_ttc < conflict_ttc

            # Gap acceptance: the box is blocked while any moving vehicle is in
            # it or will reach it within conflict_ttc seconds — "empty right
            # now" is stale within a second at these speeds. Only movers count:
            # the conflict brake guards against stationary ones, and a wreck or
            # a parked yielder must not block traffic forever. obs rows can
            # miss the other agent (no rear view), so it is checked explicitly.
            occupied = (
                moving[j]
                and _reaches_box_within(pos[j], vel[j], occupied_radius, conflict_ttc, min_closing)
            ) or any(
                float(np.hypot(row[3], row[4])) > min_closing
                and _reaches_box_within(row[1:3], row[3:5], occupied_radius, conflict_ttc, min_closing)
                for row in others
            )

            # Right-of-way, or an uncontested junction, grants leave to enter —
            # crucial for un-parking a yielder after a full stop, since IDLE
            # holds target speed 0 forever.
            priority = winner == i or not contesting[j]

            # Car-following guard against a vehicle ahead on the ego's own
            # track — the tailgate brake outranks every flow rule below.
            gap_ahead = _lead_gap(pos[i], vel[i], others)
            guard_speed = max(speed, 2.0)
            tailgating = gap_ahead < guard_speed * _HEADWAY_BRAKE_S
            close_follow = gap_ahead < guard_speed * _HEADWAY_COAST_S

            committed = speed > _COMMIT_SPEED or dist[i] < occupied_radius - _HOLD_DEPTH
            if tailgating:
                actions.append(SLOWER)
            elif dist[i] < occupied_radius and committed:
                # Inside the box at pace (or already too deep to wait): clear
                # it. Stopping inside the junction to wait for a gap is the
                # worst option against cross traffic, so this overrides the
                # conflict brake.
                actions.append(FASTER)
            elif dist[i] < occupied_radius:
                # Crept over the line and stalled: hold until the box empties.
                actions.append(SLOWER if occupied else FASTER)
            elif state["braking"][i]:
                actions.append(SLOWER)
            elif dist[i] >= junction_radius:
                # Open road: keep speed up, but do not close in on a leader.
                actions.append(IDLE if close_follow else FASTER)
            elif priority and not occupied:
                # Enter: clear box and right-of-way.
                actions.append(IDLE if close_follow else FASTER)
            else:
                actions.append(SLOWER)  # yield: stop at the box entrance

        return tuple(actions)

    policy_fn.reset = reset
    policy_fn.params = {
        "junction_radius": junction_radius,
        "conflict_ttc": conflict_ttc,
        "release_margin": release_margin,
        "occupied_radius": occupied_radius,
        "conflict_radius": conflict_radius,
        "min_closing": min_closing,
        "tie_eps": tie_eps,
        "match_eps": match_eps,
    }
    return policy_fn


if __name__ == "__main__":
    policy = make_rule_policy()

    # --- one visual demo episode (seeded, so the GIF is reproducible) ---
    demo_env = make_env(render_mode="rgb_array", normalize_obs=False)
    obs, info = demo_env.reset(seed=0)
    policy.reset()
    frames = [demo_env.render()]
    for step in range(50):
        obs, reward, terminated, truncated, info = demo_env.step(policy(obs))
        frames.append(demo_env.render())
        if all(terminated) or truncated:
            break
    save_gif(frames, "output/phase1_episode.gif")

    # --- 300-episode bulk evaluation for the README metrics ---
    eval_env = make_env(render_mode=None, normalize_obs=False)
    results = run_episodes(eval_env, policy, seed_offset=0)
    summary = summarize(results, policy_frequency=eval_env.unwrapped.config["policy_frequency"])
    save_csv(results, "output/phase1_results.csv")

    print("Phase 1 summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
