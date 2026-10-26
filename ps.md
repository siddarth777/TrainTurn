# TrainTurn: Smarter Platform Allocation for Indian Railways

## Problem Statement

When trains run late, platforms clog up and delays ripple across the station. Station controllers must decide, in real time, which platform each arriving train should use, or whether it should wait outside. Poor choices cause platform conflicts, trains stuck at the outer signal, and cascading delays that hit high-priority services hardest.

Our reinforcement learning agent observes which platforms are busy, how late each train is, and each train's priority. It then decides where to send every arriving train, or whether to hold it, with the goal of cutting total delay and conflicts.

> This is the original project plan, updated where the implementation differs. README.md and Docs/TECHNICAL_NOTES.md describe what was actually built and run.

## RL Formulation (Summary)

| Component     | Design                                                                                                                                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State**     | Per platform: occupied flag, minutes until free, priority of occupying train. Per upcoming train (next K): scheduled arrival, estimated delay, priority class, dwell time, compatible platforms. Global: time of day, trains waiting at outer signal. |
| **Actions**   | Discrete: assign to platform 1..N, or hold. Invalid actions are masked.                                                                                                                                                                               |
| **Reward**    | Negative priority-weighted waiting per minute, plus costs for outer-signal and upstream queueing, platform changes, and holding while a safe platform is free. Conflicts are prevented by the action mask, not penalized.                              |
| **Algorithm** | Double DQN (custom implementation) with action masking.                                                                                                                                                                                               |
| **Baselines** | FCFS first-free platform, timetable-planned platform with fallback, priority rule, greedy one-step lookahead.                                                                                                                                         |

## Plan

### 1. Simulator (SUMO behind a custom Gymnasium environment)

- Mid-size Indian junction with N platforms, with lengths, route conflict matrix, clearance times, and limited outer-signal capacity.
- Realistic timetable of 80-150 trains per day, tagged by type and priority weight.
- Layered delay model: heavy-tailed primary delays, shared corridor shocks, rake-linkage cascades, and recovery from slack.
- Partial observability: the agent sees noisy arrival estimates that sharpen as a train approaches.
- Train movement is simulated physically in Eclipse SUMO 1.28.0: one-second physical steps inside one-minute dispatch ticks, with a decision requested whenever a train needs a platform. Fully seeded.

### 2. Agent and Training

- Double DQN with replay buffer, target network, and normalized rewards.
- Curriculum support: `configs/curriculum.yaml` runs 5, 10, then 15 trains on 3 platforms. The submitted 8-platform, 128-train model was trained directly, without a curriculum.
- Disruption settings (delay severity, platform closures, peaks, corridor shocks, fog and monsoon multipliers) are configurable. Training used the normal setting only; the others are for stress evaluation.

### 3. Validation

- Sanity test: zero delays plus planned platforms gives zero conflicts.
- Unit tests for platform occupancy and delay accounting.
- Compare simulated delay distributions with real data where available, and state assumptions clearly where not.
- Sensitivity analysis across delay parameters.

### 4. Evaluation

- All policies run on identical held-out seeds (common random numbers), 100+ episodes, mean ± confidence interval.
- Metrics: priority-weighted delay, per-class delay, conflicts, platform changes, maximum delay, platform utilization.
- Out-of-distribution stress tests (heavier delays, closed platforms, peaks, corridor disruption) are available through `evaluate.py --stress` but have not been run yet.

### 5. Deliverables

- GitHub repo with runnable code, trained agent, and README (install, run, reproduce).
- 2-page two-column PDF report with the formal RL formulation and plots:
  - Learning curves
  - Agent vs. baselines
  - Delay vs. disruption severity (needs the `--stress` evaluation, not yet run)
  - Per-class delay
  - Platform-occupancy Gantt chart (FCFS vs. agent)

## Repository Structure

```
TrainTurn/
├── railway_rl/     # station, timetable, delay model, SUMO adapter, Gymnasium env
│   ├── agents/     # Double DQN, replay buffer, Q-network
│   ├── baselines/  # FCFS, planned platform, priority-aware, greedy lookahead
│   └── evaluation/ # metrics, statistics, evaluator, plots
├── configs/        # YAML experiment settings
├── scripts/        # train, evaluate, plots, demo, validation entry points
├── tests/
├── checkpoints/    # trained model and training logs
├── results/        # evaluation outputs and figures
├── multistation/   # seven-station network extension
├── realdata/       # DDU real-data extension
└── README.md
```
