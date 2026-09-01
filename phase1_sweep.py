"""Grid sweep over the Phase 1 rule's thresholds.

Every cell replays the same episodes (fixed seed offset), so cells differ only
by policy. Writes all cells to output/phase1_sweep.csv, the Pareto frontier
over (collision rate, avg time-to-cross) to output/phase1_frontier.csv, and a
frontier plot to output/phase1_frontier.png.

Cells run in parallel worker processes; the first cell is timed alone, and the
sweep aborts before writing anything if the projected total exceeds the
30-minute budget (override with --force).
"""
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_config import make_env
from phase1_rule_based import make_rule_policy
from utils.metrics import run_episodes, summarize, save_csv

N_EPISODES = 300
SEED_OFFSET = 0
TIME_BUDGET_S = 1800

GRID = {
    "conflict_ttc": [2.0, 3.0, 4.0, 5.0],    # caution horizon: brake trigger + accepted gap
    "release_margin": [0.5, 1.5, 3.0],       # hysteresis width
    "occupied_radius": [12.0, 15.0, 18.0],   # stop-line / occupancy radius around the box
}

# Palette (dataviz reference, light mode) — the PNG is embedded in the README.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_ACCENT = "#2a78d6"


def evaluate_cell(cell):
    """Run one grid cell in its own process: fresh policy, fresh env."""
    policy = make_rule_policy(**cell)
    env = make_env(render_mode=None, normalize_obs=False)
    results = run_episodes(env, policy, n_episodes=N_EPISODES, max_steps=50,
                           seed_offset=SEED_OFFSET)
    summary = summarize(results, policy_frequency=env.unwrapped.config["policy_frequency"])
    env.close()
    return {**policy.params, "seed_offset": SEED_OFFSET, **summary}


def pareto_frontier(rows):
    """Non-dominated rows over (collision_rate_pct, avg_time_to_cross_s), both
    minimized. Cells with no arrivals have no crossing time and are excluded.
    Returns (frontier sorted by collision rate, number of excluded cells)."""
    eligible = [r for r in rows if r["avg_time_to_cross_s"] is not None]
    frontier = []
    for a in eligible:
        dominated = any(
            b["collision_rate_pct"] <= a["collision_rate_pct"]
            and b["avg_time_to_cross_s"] <= a["avg_time_to_cross_s"]
            and (b["collision_rate_pct"] < a["collision_rate_pct"]
                 or b["avg_time_to_cross_s"] < a["avg_time_to_cross_s"])
            for b in eligible
        )
        if not dominated:
            frontier.append(a)
    frontier.sort(key=lambda r: (r["collision_rate_pct"], r["avg_time_to_cross_s"]))
    return frontier, len(rows) - len(eligible)


def plot_frontier(rows, frontier, path):
    eligible = [r for r in rows if r["avg_time_to_cross_s"] is not None]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    ax.scatter(
        [r["collision_rate_pct"] for r in eligible],
        [r["avg_time_to_cross_s"] for r in eligible],
        s=30, color=_MUTED, alpha=0.55, linewidths=0, label="all cells", zorder=2,
    )

    fx = [r["collision_rate_pct"] for r in frontier]
    fy = [r["avg_time_to_cross_s"] for r in frontier]
    ax.errorbar(
        fx, fy,
        xerr=(
            [r["collision_rate_pct"] - r["collision_rate_lo_pct"] for r in frontier],
            [r["collision_rate_hi_pct"] - r["collision_rate_pct"] for r in frontier],
        ),
        fmt="none", ecolor=_ACCENT, elinewidth=1.2, capsize=2, alpha=0.45, zorder=3,
    )
    ax.plot(fx, fy, "-o", color=_ACCENT, linewidth=2, markersize=6,
            label="Pareto frontier", zorder=4)

    for k, r in enumerate(frontier):
        ax.annotate(
            f"ttc={r['conflict_ttc']:g} rel={r['release_margin']:g} occ={r['occupied_radius']:g}",
            (r["collision_rate_pct"], r["avg_time_to_cross_s"]),
            textcoords="offset points", xytext=(6, 7 if k % 2 == 0 else -13),
            fontsize=7, color=_INK_2,
        )

    ax.set_xlabel("Collision rate (%, 95% Wilson CI)", color=_INK_2)
    ax.set_ylabel("Avg time-to-cross (s)", color=_INK_2)
    ax.set_title(
        f"Phase 1 rule sweep — {len(rows)} cells × {N_EPISODES} episodes "
        f"(seeds {SEED_OFFSET}–{SEED_OFFSET + N_EPISODES - 1})",
        color=_INK,
    )
    ax.grid(color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.legend(frameon=False, labelcolor=_INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def main():
    cells = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    workers = max(1, min((os.cpu_count() or 2) - 1, 8))

    # Time one cell before committing to the rest.
    t0 = time.perf_counter()
    first_row = evaluate_cell(cells[0])
    per_cell = time.perf_counter() - t0
    batches = 1 + -(-(len(cells) - 1) // workers)  # first cell + ceil(rest / workers)
    projected = per_cell * batches
    print(f"cell 1/{len(cells)}: {per_cell:.0f} s/cell -> projected total "
          f"~{projected / 60:.0f} min on {workers} workers")
    if projected > TIME_BUDGET_S and "--force" not in sys.argv:
        print(f"Projected sweep exceeds {TIME_BUDGET_S / 60:.0f} min budget; "
              "nothing written. Rerun with --force to proceed anyway.")
        sys.exit(1)

    rows = [first_row]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, row in enumerate(pool.map(evaluate_cell, cells[1:]), start=2):
            rows.append(row)
            print(f"cell {k}/{len(cells)}: ttc={row['conflict_ttc']:g} "
                  f"rel={row['release_margin']:g} occ={row['occupied_radius']:g} -> "
                  f"collision {row['collision_rate_pct']}% "
                  f"[{row['collision_rate_lo_pct']}, {row['collision_rate_hi_pct']}], "
                  f"cross {row['avg_time_to_cross_s']} s")

    save_csv(rows, "output/phase1_sweep.csv")
    frontier, n_excluded = pareto_frontier(rows)
    save_csv(frontier, "output/phase1_frontier.csv")
    plot_frontier(rows, frontier, "output/phase1_frontier.png")

    print(f"\n{len(rows)} cells written to output/phase1_sweep.csv "
          f"({n_excluded} excluded from the frontier for zero arrivals)")
    print(f"{len(frontier)} frontier cells written to output/phase1_frontier.csv:")
    for r in frontier:
        print(f"  ttc={r['conflict_ttc']:g} rel={r['release_margin']:g} "
              f"occ={r['occupied_radius']:g}: collision {r['collision_rate_pct']}% "
              f"[{r['collision_rate_lo_pct']}, {r['collision_rate_hi_pct']}], "
              f"cross {r['avg_time_to_cross_s']} s, "
              f"arrival {r['arrival_rate_pct']}%, timeout {r['timeout_rate_pct']}%")


if __name__ == "__main__":
    main()
