# Decision-Making Under Uncertainty: Highway Intersection Use Case

This project is a comparative study of decision-making algorithms for multi-agent coordination, using a highway intersection crossing as a use case. It starts with a rule-based control (baseline), then Deep Q-Learning (both agents learn simultaneously, each treating the other as part of a changing environment), and Cooperative Multi-Agent RL (shared rewards and penalties across agents).

### Study materials

- [*Algorithms for Decision Making*](https://algorithmsbook.com/decisionmaking/) by Mykel J. Kochenderfer, Tim A. Wheeler, Kyle H. Wray 
- Stanford's AA228/CS238 DecisionMaking Under Uncertainty, [Course Materials](https://drive.google.com/drive/u/0/folders/1Fs7ad1-YTjVtFocmihgkoja25rSOup0Z) 

## Environment

- **Simulator:** [highway-env](https://highway-env.farama.org/)  (`intersection-multi-agent-v2`)
- **Agents:** 2 controlled vehicles, cooperative/general-sum intersection crossing <!-- **Action space:** `Discrete(3)` per agent — `0=SLOWER, 1=IDLE, 2=FASTER` || **Observation:** Kinematics, 15 nearby vehicles × 7 features, per-agent egocentric-->
- **Shared config:** [`env_config.py`](./env_config.py)


## Phase 1: Rule-based Control

Rule-based logic with reactive agents, built from four interacting rules:

- **Time-to-conflict braking.** Each vehicle's closing rate and closest point of approach are computed from relative position *and* velocity, so only vehicles actually on a collision course trigger a brake. Anything receding or passing wide is ignored. The brake engages below `conflict_ttc` (2s) and releases only above `conflict_ttc + release_margin`, so it latches instead of chattering.
- **Right-of-way by arrival order.** The agent that reaches the junction zone first goes; ties fall back to agent index. Without this, both agents run identical code, brake on mutual detection, and neither clears.
- **Gap acceptance.** An agent enters only when the junction box is free of vehicles that are in it or will reach it within `conflict_ttc`, "empty right now" goes stale within a second at these speeds.
- **Box discipline.** Once committed inside the junction, the agent accelerates out rather than stopping to wait. Stopping inside the box is the worst place to be against cross traffic, and since `IDLE` holds the current target speed, an agent braked to a stop stays parked until it emits `FASTER`.

Thresholds are parameters of `make_rule_policy(...)`, not module constants, so they can be swept; the defaults are the setting the sweep below selected.

**Results (300 episodes, 95% Wilson intervals):**

| Metric | Value |
|---|---|
| Collision rate | 21.0% |
| Arrival rate (per agent) | 87.8% |
| Timeout rate (per agent) | 0.7% |
| Avg. time-to-cross | 8.50s |

<!-- 
| Metric | Value |
|---|---|
| Collision rate | 21.0% [16.8, 26.0] |
| Arrival rate (per agent) | 87.8% [85.0, 90.2] |
| Timeout rate (per agent) | 0.7% [0.3, 1.7] |
| Avg. time-to-cross | 8.50s |
300 episodes keeps every interval under 12 points wide, which is what makes a phase-to-phase comparison meaningful — at 30 episodes a rate near 33% carries an interval of roughly ±17 points, wide enough to swallow the differences the three phases are meant to show. Outcomes are read from each vehicle's crash flag rather than the sign of its reward: the intersection reward mixes collision, arrival, and speed terms, and an off-road crash scores exactly 0.0, so reward sign is not a reliable proxy for crash state.

Because these thresholds were chosen on these 300 episodes, the rule was re-checked on 300 episodes it had never seen (seeds 1000–1299): collision 21.3% [17.1, 26.3], arrival 88.5%, timeout 0.2%, 8.48s to cross. The selected setting is not an artifact of the sweep's episodes. -->

#### Sample episode
![Phase 1 episode](./output/phase1_episode.gif)

#### Threshold sweep

`phase1_sweep.py` grids the three thresholds that shape the safety/latency trade-off — `conflict_ttc` (2–5s), `release_margin` (0.5–3s), and `occupied_radius` (12–18m) — at 36 cells × 300 episodes, every cell replaying the same episodes so the cells differ only by policy.

![Phase 1 sweep Pareto frontier](./output/phase1_frontier.png)

<!--**The frontier is a single point,** because the two objectives turn out not to be in tension: the safest setting is also the fastest. Collision rate and crossing time rise together with `conflict_ttc`, from 21.0% / 8.50s at 2s to ~30% / 11s at 5s. More caution buys no safety here — braking early in an intersection leaves the agent stopped in the conflict zone for longer, which creates more exposure than the earlier brake removes. The winning cell (`conflict_ttc=2`, `release_margin=0.5`, `occupied_radius=18`) dominates all 35 others and is the module default.

Two caveats. The optimum sits on the boundary of the grid (lowest `conflict_ttc`, highest `occupied_radius` tested), so a better setting may lie outside the swept range. And the collision intervals are ~±5 points wide and overlap heavily among the low-`conflict_ttc` cells, so the ordering within that group is not resolved at 300 episodes — the separation between the `conflict_ttc=2` and `conflict_ttc≥4` groups is the finding that holds. -->

Full grid in [`output/phase1_sweep.csv`](./output/phase1_sweep.csv), frontier in [`output/phase1_frontier.csv`](./output/phase1_frontier.csv).

## Phase 2: Independent Deep Q-Learning

*In Progress...*

## Phase 3: Cooperative Learning (Shared Parameter Network)

*In Progress...*

## Cross-Phase Comparison

Rates carry 95% Wilson intervals where the sample size supports them.

| Phase | Collision Rate | Arrival Rate | Timeout Rate | Avg. Time-to-Cross |
|---|---|---|---|---|
| 1. Rule-based | 21.0% | 87.8% | 0.7% | 8.50s |
| 2. Independent Deep Q-Learning | TBD | TBD | TBD | TBD |
| 3. Cooperative MARL | TBD | TBD | TBD | TBD |

<!-- TODO: 2-3 sentence takeaway once all three rows are filled — what
     changed and why, tied back to the book concepts above -->

## Repo Structure

```
intersection-marl/
├── README.md
├── env_config.py
├── phase1_rule_based.py
├── phase1_sweep.py
├── phase2_independent_rl.py
├── phase3_marl_joint.py
├── output/
└── utils/
    ├── replay_buffer.py
    ├── gif_recorder.py
    └── metrics.py
```

## Running It

```bash
python phase1_rule_based.py     # evaluation + sample-episode GIF
python phase1_sweep.py          # threshold sweep, frontier CSVs + plot
python phase2_independent_rl.py
python phase3_marl_joint.py
```