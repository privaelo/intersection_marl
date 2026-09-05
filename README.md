# Decision-Making Under Uncertainty: Highway Intersection Use Case

This project is a comparative study of decision-making algorithms for multi-agent coordination, using a highway intersection crossing as a use case. It starts with a rule-based control (baseline), then Deep Q-Learning (both agents learn simultaneously, each treating the other as part of a changing environment), and Cooperative Multi-Agent RL (shared parameters across agents).

### Study materials

- [*Algorithms for Decision Making*](https://algorithmsbook.com/decisionmaking/) by Mykel J. Kochenderfer, Tim A. Wheeler, Kyle H. Wray 
- Stanford's AA228/CS238 DecisionMaking Under Uncertainty, [Course Materials](https://drive.google.com/drive/u/0/folders/1Fs7ad1-YTjVtFocmihgkoja25rSOup0Z) 


![Phase 1 episode](./output/phase1_episode.gif)

## Environment

- **Simulator:** [highway-env](https://highway-env.farama.org/)  (`intersection-multi-agent-v2`)
- **Agents:** 2 controlled vehicles, cooperative/general-sum intersection crossing <!-- **Action space:** `Discrete(3)` per agent — `0=SLOWER, 1=IDLE, 2=FASTER` || **Observation:** Kinematics, 15 nearby vehicles × 7 features in absolute world coordinates (junction at the origin), 105 values per agent when flattened. `see_behind` is off, so an agent's rows can omit a vehicle behind it — including the other controlled agent.-->
- **Shared config:** [`env_config.py`](./env_config.py)

## AI Usage
Used coding agent Claude Code (Opus 5) for:
- Helper functions and comments
- Debugging learning models
- Formatting eval results


## Phase 1: Rule-based Control

Rule-based logic with reactive agents, built from four interacting rules:

- **Time-to-conflict braking.** Each vehicle's closing rate and closest point of approach are computed from relative position *and* velocity, so only vehicles actually on a collision course trigger a brake. The brake engages below `conflict_ttc` (2s) and releases only above `conflict_ttc + release_margin`.
- **Right-of-way by arrival order.** The agent that reaches the junction zone first goes; ties fall back to agent index.
- **Gap acceptance.** An agent enters only when the junction box is free of vehicles that are in it or will reach it within `conflict_ttc`
- **Box discipline.** Once committed inside the junction, the agent accelerates out rather than stopping to wait.


**Results (300 episodes):**

| Metric | Value |
|---|---|
| Collision rate | 21.0% |
| Arrival rate (per agent) | 87.8% |
| Timeout rate (per agent) | 0.7% |
| Avg. time-to-cross | 9.50s |


### Threshold sweep

![Phase 1 sweep Pareto frontier](./output/phase1_frontier.png)

**The frontier is a single point,** because the two objectives turn out not to be in tension: the safest setting is also the fastest

## Phase 2: Independent Deep Q-Learning

Each agent trains its own Q-network on its own observation, with no shared weights and no communication. From either agent's point of view the other is simply part of the environment. 

Both networks are the same small MLP: 105 inputs (15 vehicles × 7 features, flattened). Observations are normalized to `[-1, 1]`.

| Hyperparameter | Value | |
|---|---|---|
| Episodes | 3000 | ~36k environment steps |
| Discount γ | 0.95 | 1/(1−γ) = 20 steps, same as the episode length |
| Learning rate | 5e-4 | Adam |
| Batch / buffer | 64 / 10,000 | Buffer deliberately small  |
| Target sync | every 500 steps | hard copy |
| Exploration ε | 1.0 → 0.05 | linear over the first half of training |

The replay buffer is kept small on purpose. Under independent learning the *other* agent's policy keeps changing, so old transitions describe an opponent that no longer exists; a buffer holding roughly a third of the run forgets them at about the right rate.

**Results (300 episodes, 95% Wilson intervals):**

| Metric | Value |
|---|---|
| Collision rate | 19.7% [15.6, 24.5] |
| Arrival rate (per agent) | 89.2% [86.4, 91.4] |
| Timeout rate (per agent) | 0.5% [0.2, 1.5] |
| Avg. time-to-cross | 9.07s |

Held out on 300 unseen episodes (seeds 1000–1299): collision 20.0% [15.9, 24.9], arrival 89.7%, 8.95s to cross. The policy generalizes.

![Phase 2 learning curve](./output/phase2_learning_curve.png)

Training return climbs steadily: agent 0 from 1.9 to about 7.8, agent 1 from 3.6 to about 8.0. But that curve is confounded by the exploration schedule. Return is collected *with* ε-greedy noise still on. The greedy evaluation underneath, measured at ε=0 throughout, reads 22.0% at episode 250 and 22.0% at episode 3000: the policy reached its final quality within the first few hundred episodes and then did not improve for the remaining 2,750.

## Phase 3: Cooperative Learning (Shared Parameter Network)

Both vehicles now act from **one** Q-network and fill **one** replay buffer. There is a single policy, so the other agent can no longer drift out from under you — the non-stationarity Phase 2 was built to expose is gone by construction. Everything else is held at Phase 2's values, including the training seeds, so the difference between the two files is the experiment.

<!-- 
| Changed from Phase 2 | Value | |
|---|---|---|
| Episodes | 300 | Phase 2's greedy eval was flat from ep 250 |
| Learning starts | 500 steps | ~3.6k env steps in this run, not ~36k |
| Target sync | every 200 steps | ~18 hard syncs |
| Updates per env step | 2 | one buffer gains 2 transitions per step, so this holds the gradient-steps-per-transition ratio at Phase 2's |

**Results (300 episodes, 95% Wilson intervals):**

| Metric | Value |
|---|---|
| Collision rate | 24.0% [19.5, 29.1] |
| Arrival rate (per agent) | 87.3% [84.4, 89.8] |
| Timeout rate (per agent) | 0.2% [0.0, 0.9] |
| Avg. time-to-cross | 9.02s |
-->
Held out on 300 unseen episodes (seeds 1000–1299): collision 22.3% [18.0, 27.4], arrival 88.5%, 8.96s to cross. 

![Phase 3 learning curve](./output/phase3_learning_curve.png)

The greedy eval hits 22% by episode 25 and never durably improves on it, drifting between 20% and 28% for the remaining 275. The shared policy reaches its ceiling almost immediately, as Phase 2's did. 

## Cross-Phase Comparison

All rows are 300 episodes on the same seeds (0–299), scored by the same evaluator, with 95% Wilson intervals on the collision rate.

| Phase | Collision Rate | Arrival Rate | Timeout Rate | Avg. Time-to-Cross |
|---|---|---|---|---|
| 1. Rule-based | 21.0% [16.8, 26.0] | 87.8% | 0.7% | 9.50s |
| 2. Independent DQN | 19.7% [15.6, 24.5] | 89.2% | 0.5% | 9.07s |
| 3. Shared-parameter MARL | 24.0% [19.5, 29.1] | 87.3% | 0.2% | 9.02s |

Every collision interval overlaps every other one: across four rules, two independent learners and one shared policy, all three sit near 20%. The methods differ in what they assume about the *other agent*, so a result this flat points at the observation rather than the learning rule.

## Repo Structure

```
intersection-marl/
├── README.md
├── env_config.py
├── phase1_rule_based.py
├── phase1_sweep.py
├── phase2_independent_rl.py
├── phase3_shared_marl.py
├── output/
└── utils/
    ├── q_network.py
    ├── replay_buffer.py
    ├── gif_recorder.py
    └── metrics.py
```
<!-- 
## Running It

```bash
python phase1_rule_based.py       # evaluation + sample-episode GIF
python phase1_sweep.py            # threshold sweep, frontier CSVs + plot (~17 min)
python phase2_independent_rl.py   # train two independent DQNs (~1 hour)
python phase3_shared_marl.py      # train one shared DQN (~7 min)
```
-->