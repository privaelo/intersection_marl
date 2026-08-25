from env_config import make_env
import numpy as np
from utils.gif_recorder import save_gif
from utils.metrics import save_csv, summarize, run_episodes

SLOWER, IDLE, FASTER = 0, 1, 2
JUNCTION_RADIUS = 20     # (in meters) start paying attention this close to center
CONFLICT_DISTANCE = 10   # (in meters) treat another vehicle this close as a conflict

def rule_based_action(obs_i: np.ndarray) -> int:
    """obs_i: this agent's own (15, 7) Kinematics observation.
    Row 0 is the ego vehicle itself; rows 1+ are nearby vehicles, absolute coords."""
    ego_x, ego_y = obs_i[0, 1], obs_i[0, 2]
    dist_to_junction = np.hypot(ego_x, ego_y)  # road network is centered at origin

    others = obs_i[1:]
    present = others[others[:, 0] > 0]  # presence flag
    if len(present) == 0:
        return IDLE

    nearest_dist = np.hypot(present[:, 1] - ego_x, present[:, 2] - ego_y).min()

    if dist_to_junction < JUNCTION_RADIUS and nearest_dist < CONFLICT_DISTANCE:
        return SLOWER
    return IDLE

def policy_fn(obs):
    return tuple(rule_based_action(obs_i) for obs_i in obs)

if __name__ == "__main__":
    # --- one visual demo episode---
    demo_env = make_env(render_mode="rgb_array", normalize_obs=False)
    obs, info = demo_env.reset()
    frames = [demo_env.render()]
    for step in range(50):
        obs, reward, terminated, truncated, info = demo_env.step(policy_fn(obs))
        frames.append(demo_env.render())
        if all(terminated) or truncated:
            break
    save_gif(frames, "output/phase1_episode.gif")

    # --- 30-episode bulk evaluation for the README metrics ---
    eval_env = make_env(render_mode=None, normalize_obs=False)
    results = run_episodes(eval_env, policy_fn, n_episodes=30, max_steps=50, seed_offset=0)
    summary = summarize(results, policy_frequency=eval_env.unwrapped.config["policy_frequency"])
    save_csv(results, "output/phase1_results.csv")

    print("Phase 1 summary:", summary)