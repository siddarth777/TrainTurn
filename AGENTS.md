# Repository Guidelines


## Current User Decisions (October 12, 2026)

These decisions supersede earlier conflicting implementation guidance:
- Use **SUMO 1.28.0 through libsumo**, with an original Gymnasium adapter; do not continue expanding the handwritten reference simulator. `simulator.py` remains a deterministic test reference and supplies shared state records; production runs use `SumoSimulator` physical events.
- Use **CUDA only** for PyTorch training and inference. Never fall back to CPU. SUMO, NumPy bookkeeping, and non-neural tests run on CPU. Stop neural runs clearly if CUDA is unavailable.
- Default to **3 platforms, 15 trains, short training, and 3 evaluation scenarios**. Do not execute wide simulations, 100-scenario evaluations, or large curricula without a new user request. Keep larger experiments configurable for later.
- The latest station-size request changes `configs/default.yaml` to **8 platforms and 128 trains**, with a 1,440-minute horizon. `configs/compact.yaml` preserves the 3-platform/15-train learning experiment; do not mix the two when comparing learning results. Evaluation remains three scenarios by default. A full training run on the larger configuration has not been implied by changing its counts.
- The user subsequently authorized increasing training with the compact model. The default is **300 training episodes** on those small scenarios, one optimizer update per decision, episode-zero validation, and heuristic validation references. Keep validation/test evidence distinct. Do not claim improvement merely because a trained checkpoint was saved.
- The requested **5.5 GiB VRAM budget** is a PyTorch allocator ceiling; record real memory usage. It excludes driver/display/other-process allocations and is not a target to fill with unused tensors.
- Keep team names and the GitHub URL blank for now. Report current outputs as preliminary, and mark the PDF as a development draft until trained RL results exist.
- SUMO uses one-second physical steps within one-minute dispatch ticks; this replaces the previous requirement for one-minute physical integration.

## Project Purpose and Assignment Compliance

Develop **Intelligent Railway Platform Allocation and Delay Minimization Using Double Deep Q-Learning**, an original university mini-project simulating a configurable mid-sized Indian railway junction. The primary solution must train and evaluate an actual reinforcement learning agent. Heuristics are comparison baselines only.

Follow `Docs/AI_Assignment2_2026_RL.pdf`. Deliver a runnable GitHub repository with source, environment setup, a trained checkpoint, and reproducibility instructions. Maintain genuine, meaningful development commits; never manufacture training or development history. The proposal must be at most 50 words and explicitly map the real-world problem to RL. Its stated deadline is October 12, 2026, 11:59 PM; verify submission details with the course instructions.

The final report must use two columns and contain at most two pages excluding references, with team members and the GitHub link at the top. Define state representation and dimensionality, actions, the mathematical reward, simulation dynamics, algorithm choice, baseline comparisons, and relevant measured graphs. Obtain actual team details rather than inventing them. If the repository is private, arrange instructor access as required by the assignment.

**Originality is mandatory:** never clone, fork, copy, or slightly modify a public RL project, repository, or another student's work. Use SUMO as a cited, unmodified simulator dependency. Implement the station configuration, adapter, replay buffer, Q-network, Double DQN, training, and evaluation originally using the approved stack. Cite conceptual sources and document modeling assumptions; citations do not authorize copied implementation. Never fabricate results, improvements, data provenance, or training history. Never tamper with another team's submission. Clearly label synthetic timetables and delays; do not present them as real Indian Railways operational data.

## Repository Status and Intended Structure

At initialization, only the assignment PDF exists under `Docs/`; there is no usable Git history, implementation, dependency manifest, or test suite. The following is the intended layout, not a claim that these paths already exist. Create files only when their implementation phase needs them.

- `configs/`: YAML environment and experiment settings.
- `data/`: timetable inputs, generation assumptions, units, and provenance.
- `railway_rl/`: station models, simulator, and `TrainTurnEnv`.
- `railway_rl/agents/`: original Double DQN, replay, and networks.
- `railway_rl/baselines/`: comparison policies.
- `railway_rl/evaluation/`: metrics and statistical comparisons.
- `scripts/`: training, validation, evaluation, and plotting entry points.
- `tests/`: simulator, environment, RL, masking, and reproducibility tests.
- `checkpoints/`: actual trained artifacts and associated configuration metadata.
- `results/`: measured experiment outputs and figures.
- `reports/`: LaTeX report source; preserve the assignment in `Docs/`.
- `README.md`: installation, execution, and exact reproduction instructions.
- `requirements.txt`: verified dependencies.

## Stack, Style, and Local Development

Use Python 3.11+, Gymnasium, PyTorch, NumPy, Pandas, SciPy, Matplotlib, pytest, and YAML configurations. Use PyYAML when a YAML parser is needed. Avoid unnecessary dependencies and verify APIs before using them.

Use four-space indentation, `snake_case` functions/modules/variables, `PascalCase` classes, and `UPPER_CASE` constants. Write original, modular, readable code with type hints on public interfaces and dataclasses where appropriate. Keep functions focused and testable. Prefer explicit simulation logic over hidden side effects. Keep configuration separate from implementation; avoid hard-coded paths and magic constants. Explain non-obvious mathematical and railway assumptions. Never silently swallow exceptions or sacrifice correctness for speed. Update dependent modules and tests when changing interfaces. No formatter or linter is established yet; do not claim otherwise or add tooling without need.

Target an Intel Core i5-12450HX, 16 GB DDR5 RAM, and NVIDIA RTX 3050 Laptop GPU. Use CUDA for neural execution with an explicit failure when unavailable; SUMO remains on CPU. Measure throughput before assuming CUDA helps small DQN workloads. Keep replay capacity, network size, batch size, and experiment volume configurable and memory-conscious. Seed every random generator explicitly, including environment, exploration, replay sampling, NumPy, and PyTorch. Document reproducibility limits across devices and software versions.

Once dependencies and tests exist, use:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python -m pytest tests/test_simulator.py
```

The focused test path is an intended naming example. Inspect actual files before running it. Training and evaluation entry points now exist under `scripts/`; use README for verified commands. Document real entry points, YAML paths, seeds, device options, and checkpoint/evaluation commands in README when implemented; never advertise nonexistent commands.

## Simulator Contract and Physical Safety

Model 3–8 platforms and approximately 15–150 trains per scenario, with configurable platform length, electrification, terminal/through compatibility, occupancy, clearance intervals, approach/departure route conflicts, outer-signal queues, upstream spillback, train classes and priority weights, scheduled/actual arrivals and departures, dwell times, closures, and disruptions. Add correlated heavy-tailed incoming delays, imperfect arrival forecasts, and optional rake-linkage propagation in the appropriate later phases.

Advance internally in one-minute ticks. Expose decisions only when allocation or holding is required. Holding must advance time. Enforce safety in shared simulator logic independently of masks and policies: no overlapping platform occupancy, incompatible assignments, simultaneous conflicting route movements, premature departures, or clearance violations. Define invalid-action handling explicitly without permitting unsafe movement. Ensure infeasible scenarios cannot silently deadlock or lose trains; distinguish completion from truncation and account for unfinished trains.

## Environment, Observations, and Reward

Implement `TrainTurnEnv(gymnasium.Env)` using verified Gymnasium reset/step semantics. Use a fixed-dimensional, normalized observation vector with padding and documented feature ordering, units, bounds, and presence indicators. Include per-platform availability/release estimates, current train features, next K incoming forecasts, compatibility, waiting queues, time of day, and congestion. Never expose hidden actual future arrivals or unrevealed disturbances; future estimates must use only information currently available.

Use `Discrete(max_platforms + 1)`: action `0` holds; action `p` assigns platform `p`. Mask padded, occupied, incompatible, closed, and otherwise unsafe platforms. Define hold feasibility and publish the mask consistently to agents and baselines. Masks must match simulator feasibility.

Document the mathematical reward before implementation. A starting structure is
`r_t = -(alpha * W_t + beta * C_t + eta * H_t + zeta * P_t)`,
where `W_t` integrates controllable priority-weighted waiting across elapsed ticks, `C_t` measures congestion over those ticks, `H_t` penalizes unnecessary holding, and `P_t` counts operational changes. Define each term, units, controllability, and event-counting rules precisely; keep coefficients configurable. Separate unavoidable incoming lateness from station-induced delay and avoid double-counting it. Document intentional overlap between delay and congestion penalties.

## Double DQN and Training

Implement online and target Q-networks, experience replay, masked epsilon-greedy exploration, masked Double DQN Bellman targets, Huber loss, gradient clipping, configurable target updates, checkpoints, and reproducible seeds.

Select the next action with masked online Q-values and evaluate it with the target network. Store next-action masks and elapsed simulated minutes in replay. For per-minute discount `gamma`, use `gamma ** elapsed_minutes` for bootstrapping; explicitly define consistent discounting of interval rewards. True terminal transitions must not bootstrap; handle time-limit truncation explicitly. Guard terminal/all-masked cases against invalid argmax or nonfinite targets. Checkpoints must record enough configuration, normalization, seeds, and training state for their documented inference or resume use.

## Baselines, Evaluation, and Evidence

Implement FCFS, planned timetable platform with fallback, and priority-aware allocation; greedy lookahead is optional. Define queue ordering, tie-breaking, fallback, and hold behavior. All policies must use identical physical constraints and equivalent observable information; none may read privileged futures.

Evaluate on at least 100 held-out scenarios, separate from training and tuning, using common random numbers. Pre-generate exogenous scenario randomness or isolate random streams so policy-dependent decisions do not change disturbances. Preserve scenario IDs, seeds, configurations, checkpoint identity, and raw metrics. Report appropriate confidence intervals, preferably paired scenario comparisons.

Track priority-weighted station-induced delay, final departure lateness, per-class average delay, waiting/congestion events, platform changes, maximum delay, platform utilization, and safety violations. Define denominators, unfinished-train handling, and metric units. Safety violations must be zero.

Generate actual learning curves, RL/baseline comparisons, disruption-severity robustness plots, per-class delay comparisons, and platform occupancy Gantt charts. Distinguish training from evaluation results. Report negative or inconclusive findings honestly; never create placeholder performance results.

## Testing and Required Invariants

Use pytest with `tests/test_*.py` files and `test_*` functions. Every major implementation needs appropriate tests; no numeric coverage threshold is established. Prioritize correctness checks over arbitrary coverage percentages.

Test no double occupancy, invalid assignment, conflicting simultaneous routes, departure before arrival, or clearance violation. Test time advancement on hold, feasible masks, identical-seed/action reproduction, correct delay/reward accounting, terminal and masked Double DQN targets, absence of privileged information, and no silently dropped trains at termination. Add small hand-calculated scenarios and edge cases, including closures, full queues, all platforms unavailable, and truncation. Run relevant tests after modifications and broader tests when shared contracts change. Never say tests passed unless executed; report exact commands, outcomes, and limitations.

## Implementation Order

Complete and verify a minimal working system before advanced realism. Do not develop multiple complex subsystems simultaneously without tests.

1. Station data models and deterministic simulator.
2. Gymnasium environment, observations, rewards, and action masking.
3. Baseline policies and evaluation metrics.
4. Original Double DQN and training pipeline.
5. Realistic delay generation and disruption scenarios.
6. Curriculum training and measured optimization.
7. Held-out evaluation and statistical comparisons.
8. Plots, documentation, and the final two-page report.

## Session, Commit, and Review Rules

Read this file and relevant existing code before edits; inspect the repository before assuming modules exist. Preserve established architecture and user work unless a justified change is needed. Briefly explain important decisions. Deliver complete connected functionality, update tests and callers, run relevant checks when possible, and report actual outcomes. Prioritize correctness, reproducibility, realistic simulation, and assignment compliance. Avoid overengineering and unnecessary files.

Never perform Git commits, pushes, destructive operations, or major dependency changes without explicit authorization. No historical commit convention exists yet; when authorized, use focused, truthful imperative messages such as `Add deterministic platform clearance checks`. PR descriptions should explain the problem, resulting behavior, linked issues when applicable, executed tests, and limitations. Include figures only for actual visual changes or measured experiments. Keep development history genuine.

Initialization is complete. Follow README and the current user decisions above. Never claim a trained model or GPU-tested algorithm until an actual CUDA run succeeds.
