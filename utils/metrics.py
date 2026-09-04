"""Shared episode-loop evaluator"""
import csv
import numpy as np


def run_episodes(env, policy_fn, n_episodes=300, max_steps=50, seed_offset=0):
    """policy_fn(obs) -> action tuple for both agents.
    Returns a list of per-episode result dicts."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        # Stateful policies (e.g. Phase 1's hysteresis rule) expose an optional
        # reset() to clear per-episode state; plain functions are unaffected.
        reset_hook = getattr(policy_fn, "reset", None)
        if callable(reset_hook):
            reset_hook()
        agent_done_step = [None, None]
        agent_outcome = [None, None]

        for step in range(max_steps):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            for i in range(2):
                if agent_done_step[i] is None and terminated[i]:
                    # step + 1, not step: env.step has already advanced the clock,
                    # so this counts policy steps elapsed rather than the loop
                    # index. An agent finishing on the first iteration has lived
                    # one step, not zero.
                    agent_done_step[i] = step + 1
                    # terminated[i] = crashed or has_arrived, so the crash flag decides.
                    # (Reward sign is unreliable: an off-road crash scores 0.0.)
                    crashed = env.unwrapped.controlled_vehicles[i].crashed
                    agent_outcome[i] = "crashed" if crashed else "arrived"

            if all(d is not None for d in agent_done_step) or truncated:
                break

        for i in range(2):
            if agent_done_step[i] is None:
                agent_outcome[i] = "timeout"

        results.append({
            "episode": ep,
            "agent0_outcome": agent_outcome[0],
            "agent0_step": agent_done_step[0],
            "agent1_outcome": agent_outcome[1],
            "agent1_step": agent_done_step[1],
        })
    return results


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n.
    Returns (lo, hi) as fractions in [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(results, policy_frequency=1):
    """Aggregate collision rate and average time-to-cross across episodes.
    Every rate carries its 95% Wilson interval as flat *_lo_pct/*_hi_pct keys."""
    n = len(results)
    outcomes = [r[f"agent{i}_outcome"] for r in results for i in (0, 1)]
    n_agent_slots = len(outcomes)

    episode_had_collision = sum(
        1 for r in results
        if r["agent0_outcome"] == "crashed" or r["agent1_outcome"] == "crashed"
    )
    arrivals = [
        r[f"agent{i}_step"] for r in results for i in (0, 1)
        if r[f"agent{i}_outcome"] == "arrived"
    ]

    def rate_with_interval(name, k, denom):
        lo, hi = wilson_interval(k, denom)
        return {
            f"{name}_pct": round(100 * k / denom, 1),
            f"{name}_lo_pct": round(100 * lo, 1),
            f"{name}_hi_pct": round(100 * hi, 1),
        }

    return {
        "episodes": n,
        **rate_with_interval("collision_rate", episode_had_collision, n),
        **rate_with_interval("per_agent_collision_rate", outcomes.count("crashed"), n_agent_slots),
        **rate_with_interval("arrival_rate", outcomes.count("arrived"), n_agent_slots),
        **rate_with_interval("timeout_rate", outcomes.count("timeout"), n_agent_slots),
        "avg_time_to_cross_s": round(float(np.mean(arrivals)) / policy_frequency, 2) if arrivals else None,
    }


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)