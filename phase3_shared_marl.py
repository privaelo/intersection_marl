"""Phase 3: Cooperative MARL with a shared-parameter network.

Phase 2 gave each vehicle its own Q-network, so from either agent's point of
view the other was a piece of the environment that kept changing shape. Here
both vehicles act from *one* network and fill *one* replay buffer: there is a
single policy, so there is no separate opponent to go stale. Every transition
either vehicle generates trains the policy that drives both of them.

That is the only change. The observation, reward, discount, optimizer, replay
buffer, evaluator and training seeds are all Phase 2's, so the difference
between this file and phase2_independent_rl.py *is* the experiment.

    python phase3_shared_marl.py           # full run (~7 min)
    python phase3_shared_marl.py --smoke   # 30-episode pipeline check
"""
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env_config import make_env
from utils.gif_recorder import save_gif
from utils.metrics import run_episodes, save_csv, summarize
from utils.q_network import QNetwork, select_action, train_step, update_target_network
from utils.replay_buffer import ReplayBuffer

SLOWER, IDLE, FASTER = 0, 1, 2
N_AGENTS = 2
N_ACTIONS = 3

# Same knob as Phase 2, same default, so the input the network sees is identical.
INCLUDE_OTHER_AGENT = False
OBS_DIM = 105 + (7 if INCLUDE_OTHER_AGENT else 0)

SEED = 0
N_EPISODES = 300       # Phase 2's greedy eval read 22.0% at ep 250 and 22.0% at
                       # ep 3000, so the extra 2,750 episodes bought nothing
MAX_STEPS = 50         # dead margin; the env truncates at 20 (duration / policy_frequency)
GAMMA = 0.95           # 1/(1-gamma) = 20 steps, matching the episode length
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
BUFFER_CAPACITY = 10_000
LEARNING_STARTS = 500   # ~3.6k env steps in a 300-episode run against Phase 2's ~36k;
                        # Phase 2's 1000 would spend a quarter of training on pure noise
TARGET_SYNC = 200       # ~18 hard syncs, in the same ballpark as Phase 2's ~72 over 10x
                        # the episodes
UPDATES_PER_STEP = 2    # Phase 2 took one gradient step per network per env step, i.e.
                        # one step per stored transition. Both agents feed one buffer
                        # here, so it gains 2 transitions per env step; two updates keep
                        # the gradient-steps-per-transition ratio identical.
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_FRAC = 0.5    # linear decay over the first half of training

# At 2 transitions x ~3.6k env steps the buffer holds ~7.2k of 10k and never evicts.
# Phase 2 kept it small on purpose, to forget transitions describing an opponent that
# no longer existed. With one shared policy there is no separate opponent, so the
# capacity is left where it was rather than retuned.

TRAIN_SEED_OFFSET = 20_000   # identical to Phase 2: episodes 0-299 are seeds 20000-20299
CURVE_SEED_OFFSET = 5_000    # mid-training probes, disjoint from the reported evals
EVAL_EVERY = 25              # 12 curve points, the same resolution as the Phase 2 plot
EVAL_EPISODES = 50

# Earlier phases, seeds 0-299 (see README)
PHASE1_COLLISION_PCT, PHASE1_ARRIVAL_PCT = 21.0, 87.8
PHASE1_TIMEOUT_PCT, PHASE1_CROSS_S = 0.7, 9.50
PHASE2_COLLISION_PCT, PHASE2_ARRIVAL_PCT = 19.7, 89.2
PHASE2_TIMEOUT_PCT, PHASE2_CROSS_S = 0.5, 9.07

# dataviz palette, light mode
_SURFACE, _INK, _INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
_MUTED, _GRIDLINE, _AXIS = "#898781", "#e1e0d9", "#c3c2b7"
_SERIES = ("#2a78d6", "#eb6834")


class SharedAgent:
    """One online net, one target net, one optimizer, one buffer — for both vehicles."""

    def __init__(self):
        self.q_net = QNetwork(OBS_DIM, N_ACTIONS)
        self.target_net = QNetwork(OBS_DIM, N_ACTIONS)
        update_target_network(self.q_net, self.target_net)
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)


def agent_state(obs, i):
    """Agent i's network input: its own (15, 7) observation flattened, plus the
    other agent's ego row when INCLUDE_OTHER_AGENT. Always a fresh copy, so it
    is safe to hand straight to the replay buffer."""
    own = obs[i].flatten()
    if not INCLUDE_OTHER_AGENT:
        return own
    return np.concatenate([own, obs[1 - i][0]])


def epsilon_at(episode, n_episodes):
    """Linear decay over the first EPS_DECAY_FRAC of training, then flat."""
    decay_episodes = max(1, int(EPS_DECAY_FRAC * n_episodes))
    progress = min(1.0, episode / decay_episodes)
    return EPS_START + (EPS_END - EPS_START) * progress


def make_greedy_policy(agent):
    """policy_fn(obs) -> (action0, action1), epsilon=0, both from the same net."""
    def policy_fn(obs):
        return tuple(
            select_action(agent.q_net, agent_state(obs, i), 0.0, N_ACTIONS)
            for i in range(N_AGENTS)
        )
    return policy_fn


def evaluate(env, policy_fn, n_episodes, seed_offset):
    results = run_episodes(env, policy_fn, n_episodes=n_episodes,
                           max_steps=MAX_STEPS, seed_offset=seed_offset)
    summary = summarize(results, policy_frequency=env.unwrapped.config["policy_frequency"])
    return results, summary


def train(env, eval_env, agent, n_episodes, learning_starts, target_sync,
          eval_every, eval_episodes):
    """Run the shared-parameter DQN training loop. Returns (episode log, eval curve)."""
    episode_log, eval_curve = [], []
    global_step = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=TRAIN_SEED_OFFSET + episode)
        states = [agent_state(obs, i) for i in range(N_AGENTS)]
        done = [False, False]
        returns = [0.0, 0.0]
        losses = []
        epsilon = epsilon_at(episode, n_episodes)
        steps = 0

        for steps in range(1, MAX_STEPS + 1):
            actions = tuple(
                IDLE if done[i]
                else select_action(agent.q_net, states[i], epsilon, N_ACTIONS)
                for i in range(N_AGENTS)
            )
            obs, reward, terminated, truncated, info = env.step(actions)
            next_states = [agent_state(obs, i) for i in range(N_AGENTS)]

            for i in range(N_AGENTS):
                if done[i]:
                    continue
                # Both agents push into the same buffer. Store the *true* terminal
                # flag only: truncation at step 20 is a time limit, not the end of
                # the world, and bootstrapping must continue through it or the
                # policy learns a false horizon.
                agent.buffer.push(states[i], actions[i], float(reward[i]),
                                  next_states[i], bool(terminated[i]))
                returns[i] += float(reward[i])

            # marked after the pushes, so the step into a terminal state is kept
            for i in range(N_AGENTS):
                done[i] = done[i] or bool(terminated[i])

            states = next_states
            global_step += 1

            if global_step >= learning_starts:
                for _ in range(UPDATES_PER_STEP):
                    loss = train_step(agent.q_net, agent.target_net, agent.optimizer,
                                      agent.buffer, BATCH_SIZE, GAMMA)
                    if loss is not None:
                        losses.append(loss)

            if global_step % target_sync == 0:
                update_target_network(agent.q_net, agent.target_net)

            if all(done) or truncated:
                break

        episode_log.append({
            "episode": episode,
            "steps": steps,
            "epsilon": round(epsilon, 4),
            "return_0": round(returns[0], 3),
            "return_1": round(returns[1], 3),
            "mean_loss": round(float(np.mean(losses)), 5) if losses else None,
            "buffer_size": len(agent.buffer),
        })

        if (episode + 1) % eval_every == 0:
            _, summary = evaluate(eval_env, make_greedy_policy(agent),
                                  eval_episodes, CURVE_SEED_OFFSET)
            eval_curve.append({"episode": episode + 1, **summary})
            print(f"  ep {episode + 1}/{n_episodes} eps={epsilon:.3f} "
                  f"collision {summary['collision_rate_pct']}% "
                  f"arrival {summary['arrival_rate_pct']}% "
                  f"cross {summary['avg_time_to_cross_s']}s", flush=True)

    return episode_log, eval_curve


def plot_learning_curve(episode_log, eval_curve, path):
    """Training return on top, greedy-eval rates below. Shared episode axis,
    never a second y-axis on one plot. The dashed references are Phase 2's,
    since Phase 2 is the run this phase is trying to beat."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7), dpi=150, sharex=True, facecolor=_SURFACE)

    episodes = np.array([r["episode"] for r in episode_log])
    window = max(1, min(100, len(episodes) // 10))
    kernel = np.ones(window) / window
    for i in range(N_AGENTS):
        raw = np.array([r[f"return_{i}"] for r in episode_log], dtype=float)
        smoothed = np.convolve(raw, kernel, mode="valid")
        ax_top.plot(episodes[window - 1:], smoothed, color=_SERIES[i],
                    linewidth=2, label=f"agent {i}")
    ax_top.set_ylabel(f"Training return (mean of {window})", color=_INK_2)
    ax_top.set_title("Phase 3 — shared-parameter DQN training", color=_INK)
    ax_top.legend(frameon=False, labelcolor=_INK_2)

    if eval_curve:
        ep = [r["episode"] for r in eval_curve]
        ax_bot.plot(ep, [r["collision_rate_pct"] for r in eval_curve], "-o",
                    color=_SERIES[0], linewidth=2, markersize=5, label="collision rate")
        ax_bot.plot(ep, [r["arrival_rate_pct"] for r in eval_curve], "-o",
                    color=_SERIES[1], linewidth=2, markersize=5, label="arrival rate")
    ax_bot.axhline(PHASE2_COLLISION_PCT, color=_SERIES[0], linestyle="--",
                   linewidth=1.2, alpha=0.5, label="Phase 2 collision")
    ax_bot.axhline(PHASE2_ARRIVAL_PCT, color=_SERIES[1], linestyle="--",
                   linewidth=1.2, alpha=0.5, label="Phase 2 arrival")
    ax_bot.set_xlabel("Training episode", color=_INK_2)
    ax_bot.set_ylabel("Greedy eval (%)", color=_INK_2)
    ax_bot.legend(frameon=False, labelcolor=_INK_2, ncol=2)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor(_SURFACE)
        ax.grid(color=_GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_AXIS)

    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def print_summary(label, summary):
    print(f"{label}:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def print_comparison(summary):
    """All three phases on the same 300 seeds, scored by the same evaluator."""
    header = f"{'Phase':<24}{'Collision':>22}{'Arrival':>10}{'Timeout':>10}{'Cross':>8}"
    print("\n" + header)
    print("-" * len(header))
    print(f"{'1. Rule-based':<24}{PHASE1_COLLISION_PCT:>21.1f}%"
          f"{PHASE1_ARRIVAL_PCT:>9.1f}%{PHASE1_TIMEOUT_PCT:>9.1f}%{PHASE1_CROSS_S:>7.2f}s")
    print(f"{'2. Independent DQN':<24}{PHASE2_COLLISION_PCT:>21.1f}%"
          f"{PHASE2_ARRIVAL_PCT:>9.1f}%{PHASE2_TIMEOUT_PCT:>9.1f}%{PHASE2_CROSS_S:>7.2f}s")
    interval = f"[{summary['collision_rate_lo_pct']}, {summary['collision_rate_hi_pct']}]"
    print(f"{'3. Shared parameters':<24}"
          f"{str(summary['collision_rate_pct']) + '% ' + interval:>22}"
          f"{summary['arrival_rate_pct']:>9.1f}%{summary['timeout_rate_pct']:>9.1f}%"
          f"{summary['avg_time_to_cross_s']:>7.2f}s")


def main():
    smoke = "--smoke" in sys.argv
    n_episodes = 30 if smoke else N_EPISODES
    learning_starts = 100 if smoke else LEARNING_STARTS
    target_sync = 50 if smoke else TARGET_SYNC
    eval_every = 15 if smoke else EVAL_EVERY
    eval_episodes = 5 if smoke else EVAL_EPISODES
    final_episodes = 20 if smoke else 300
    # Smoke artifacts get their own names; a pipeline check must never clobber
    # the results of a real run.
    tag = "_smoke" if smoke else ""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Phase 1 evaluates on raw meters; the nets are trained on normalized input,
    # so every env here must agree. A mismatch fails silently, not loudly.
    env = make_env(render_mode=None, normalize_obs=True)
    eval_env = make_env(render_mode=None, normalize_obs=True)
    assert env.unwrapped.config["observation"]["observation_config"].get("normalize", True)

    agent = SharedAgent()

    print(f"Training {n_episodes} episodes "
          f"(input {OBS_DIM}, other agent {'visible' if INCLUDE_OTHER_AGENT else 'hidden'})...",
          flush=True)
    t0 = time.perf_counter()
    episode_log, eval_curve = train(env, eval_env, agent, n_episodes,
                                    learning_starts, target_sync,
                                    eval_every, eval_episodes)
    print(f"Training done in {(time.perf_counter() - t0) / 60:.1f} min\n")

    save_csv(episode_log, f"output/phase3_training{tag}.csv")
    save_csv(eval_curve, f"output/phase3_eval_curve{tag}.csv")
    plot_learning_curve(episode_log, eval_curve, f"output/phase3_learning_curve{tag}.png")
    torch.save({
        "q_net": agent.q_net.state_dict(),
        "episode": n_episodes,
        "hyperparams": {
            "gamma": GAMMA, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "buffer_capacity": BUFFER_CAPACITY, "target_sync": target_sync,
            "eps_start": EPS_START, "eps_end": EPS_END, "seed": SEED,
            "updates_per_step": UPDATES_PER_STEP,
        },
    }, f"output/phase3_shared{tag}.pt")

    greedy = make_greedy_policy(agent)
    results, summary = evaluate(eval_env, greedy, final_episodes, 0)
    save_csv(results, f"output/phase3_results{tag}.csv")
    print_summary(f"Phase 3 greedy eval ({final_episodes} episodes, seeds 0-{final_episodes - 1})",
                  summary)

    _, holdout = evaluate(eval_env, greedy, final_episodes, 1000)
    print_summary(f"Held-out (seeds 1000-{1000 + final_episodes - 1})", holdout)

    demo_env = make_env(render_mode="rgb_array", normalize_obs=True)
    obs, info = demo_env.reset(seed=0)
    frames = [demo_env.render()]
    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = demo_env.step(greedy(obs))
        frames.append(demo_env.render())
        if all(terminated) or truncated:
            break
    save_gif(frames, f"output/phase3_episode{tag}.gif")

    print_comparison(summary)


if __name__ == "__main__":
    main()
