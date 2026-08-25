"""Shared episode-loop evaluator"""
import csv
import numpy as np


def run_episodes(env, policy_fn, n_episodes=30, max_steps=50, seed_offset=0):
    """policy_fn(obs) -> action tuple for both agents.
    Returns a list of per-episode result dicts."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        agent_done_step = [None, None]
        agent_outcome = [None, None]

        for step in range(max_steps):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            for i in range(2):
                if agent_done_step[i] is None and terminated[i]:
                    agent_done_step[i] = step
                    agent_outcome[i] = "crashed" if reward[i] < 0 else "arrived"

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


def summarize(results, policy_frequency=1):
    """Aggregate collision rate and average time-to-cross across episodes."""
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

    return {
        "episodes": n,
        "collision_rate_pct": round(100 * episode_had_collision / n, 1),
        "per_agent_collision_rate_pct": round(100 * outcomes.count("crashed") / n_agent_slots, 1),
        "arrival_rate_pct": round(100 * outcomes.count("arrived") / n_agent_slots, 1),
        "timeout_rate_pct": round(100 * outcomes.count("timeout") / n_agent_slots, 1),
        "avg_time_to_cross_s": round(float(np.mean(arrivals)) / policy_frequency, 2) if arrivals else None,
    }


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)