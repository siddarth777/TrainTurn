# TrainTurn: Smarter Platform Allocation for Indian Railways

## Problem Statement

When trains run late, platforms clog up and delays ripple across the station. Station controllers must decide, in real time, which platform each arriving train should use, or whether it should wait outside. Poor choices cause platform conflicts, trains stuck at the outer signal, and cascading delays that hit high-priority services hardest.

Our reinforcement learning agent observes which platforms are busy, how late each train is, and each train's priority. It then decides where to send every arriving train, or whether to hold it, with the goal of cutting total delay and conflicts.

## RL Formulation (Summary)

| Component     | Design                                                                                                                                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State**     | Per platform: occupied flag, minutes until free, priority of occupying train. Per upcoming train (next K): scheduled arrival, estimated delay, priority class, dwell time, compatible platforms. Global: time of day, trains waiting at outer signal. |
| **Actions**   | Discrete: assign to platform 1..N, or hold. Invalid actions are masked.                                                                                                                                                                               |
| **Reward**    | Negative priority-weighted delay per step, penalties for conflicts and late platform changes, small cost per minute of holding.                                                                                                                       |
| **Algorithm** | Double DQN (custom implementation), with PPO + action masking as a fallback.                                                                                                                                                                          |
| **Baselines** | FCFS first-free platform, timetable-planned platform with fallback, priority rule, greedy one-step lookahead.                                                                                                                                         |

## Plan

### 1. Simulator (custom Gymnasium environment)

- Mid-size Indian junction with N platforms, with lengths, route conflict matrix, clearance times, and limited outer-signal capacity.
- Realistic timetable of 80-150 trains per day, tagged by type and priority weight.
- Layered delay model: heavy-tailed primary delays, shared corridor shocks, rake-linkage cascades, and recovery from slack.
- Partial observability: the agent sees noisy arrival estimates that sharpen as a train approaches.
- Fixed 1-minute time step with a decision requested whenever a train needs a platform. Fully seeded.

### 2. Agent and Training

- Double DQN with replay buffer, target network, and normalized rewards.
- Curriculum: start with 3 platforms and ~15 trains, then scale to 6-8 platforms and 100+ trains.
- Domain randomization: delay severity, traffic density, platform closures, disruption days (fog, breakdowns).

### 3. Validation

- Sanity test: zero delays plus planned platforms gives zero conflicts.
- Unit tests for platform occupancy and delay accounting.
- Compare simulated delay distributions with real data where available, and state assumptions clearly where not.
- Sensitivity analysis across delay parameters.

### 4. Evaluation

- All policies run on identical held-out seeds (common random numbers), 100+ episodes, mean ± confidence interval.
- Metrics: priority-weighted delay, per-class delay, conflicts, platform changes, maximum delay, platform utilization.
- Out-of-distribution stress tests: heavier delays and closed platforms.

### 5. Deliverables

- GitHub repo with runnable code, trained agent, and README (install, run, reproduce).
- 2-page two-column PDF report with the formal RL formulation and plots:
  - Learning curves
  - Agent vs. baselines
  - Delay vs. disruption severity
  - Per-class delay
  - Platform-occupancy Gantt chart (FCFS vs. agent)

## Suggested Repo Structure

```
trainturn/
├── env/            # station, timetable, delay model, Gymnasium env
├── agents/         # DQN implementation
├── baselines/      # FCFS, rule-based, priority, lookahead
├── train.py
├── evaluate.py
├── tests/
├── results/        # plots and logs
└── README.md
```
