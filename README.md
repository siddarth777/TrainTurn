# TrainTurn: Railway Platform Allocation with Double DQN

TrainTurn is a deep reinforcement learning controller for a busy railway junction. When a train arrives at the outer signal, the agent decides which platform it should use, or whether it should wait. The goal is to cut priority-weighted delay, congestion and platform changes.

The train movements are simulated physically in **Eclipse SUMO 1.28.0** through `libsumo`. A custom Gymnasium environment wraps the simulation, and an original masked **Double DQN**, trained on CUDA, makes the decisions. The agent is compared with four heuristic dispatch rules on identical held-out scenarios. Two extensions test the same trained model, unchanged, in new settings: a seven-station double-track network, and 28 real days of September 2024 traffic at Pt. Deen Dayal Upadhyaya Junction (DDU).

## Results at a glance

**Main experiment.** 8 platforms, 128 trains a day, 100 held-out synthetic days, test seeds from 200000 (`results/station8_128_eval100/`). Every policy completed every train with zero safety violations. Lower is better.

| Policy | Weighted station delay (95% CI) | DQN minus policy (95% CI) |
|---|---|---|
| **Double DQN** | **322.1** [307.5, 338.0] | — |
| Priority-aware | 346.1 [330.6, 363.3] | −24.1 [−30.3, −17.8] |
| Planned platform (timetable) | 374.0 [358.0, 390.6] | −51.9 [−59.2, −44.8] |
| FCFS | 398.6 [382.0, 416.7] | −76.5 [−84.4, −68.7] |
| Greedy lookahead | 515.8 [482.8, 550.7] | −193.7 [−218.7, −168.3] |

The model was trained for 300 episodes on an RTX 3050 laptop GPU: 61,543 optimizer updates in about 18.5 minutes. Validation delay fell from 378.7 before training to 285.2 at the best checkpoint. These results come from **one training seed**.

**Extensions** (zero-shot unless noted). Details and caveats are in each folder's README.

- **Seven-station network** ([multistation/README.md](multistation/README.md)): the agent had the lowest network lateness on all 3 test days and clearly beat FCFS and greedy. Against priority-aware and planned-platform allocation, junction waiting was inconclusive.
- **Real DDU days** ([realdata/README.md](realdata/README.md)), covering 28 days and 420 episodes:
  - On the real timetable with no delays, the agent beat every heuristic, and every confidence interval excludes zero.
  - With real delays the main result is inconclusive.
  - Fine-tuning on real days gave no significant gain over the zero-shot model, but it cut discretionary holds from 9.7 to 2.5 a day.

## Repository layout

| Path | Contents |
|---|---|
| `railway_rl/` | Core package: station models, reference simulator, SUMO network and adapter, Gymnasium environment, observations, masks, rewards |
| `railway_rl/agents/` | Q-network, replay buffer, masked Double DQN with checkpointing |
| `railway_rl/baselines/` | FCFS, planned platform, priority-aware and greedy-lookahead policies |
| `railway_rl/evaluation/` | Per-episode metrics, paired bootstrap statistics, evaluator, plots |
| `railway_rl/training.py` | Training loop, curriculum, validation and checkpoint selection |
| `configs/` | `default.yaml` (8 platforms, 128 trains), `compact.yaml` (3 platforms, 15 trains), `curriculum.yaml`, `learning_diagnostic.yaml`, `stress_test.yaml` |
| `scripts/` | Command-line entry points (see below) |
| `tests/` | 34 core tests, including SUMO and CUDA tests |
| `checkpoints/station8_128/` | Trained `best_model.pt`, untrained `initial_model.pt`, training and validation logs, manifest |
| `results/` | 100-scenario evaluation with figures, plus two interactive HTML pages |
| `multistation/` | Seven-station network extension: code, config, scripts, tests, results, replay page |
| `realdata/` | DDU real-data suite: importer, fine-tuning, processed September 2024 days, tests, results, replay page |
| `data/README.md` | Data provenance and the timetable CSV format |
| `Docs/` | Assignment brief and [TECHNICAL_NOTES.md](Docs/TECHNICAL_NOTES.md) (full RL formulation, railway model and safety design) |
| `reports/proposal.txt` | Project proposal |

## Setup

You need:
- Python 3.11 or later.
- An NVIDIA GPU with a working driver. Neural training and inference are **CUDA-only**, with no CPU fallback. SUMO and the non-neural code run on the CPU.
- Pip installs SUMO 1.28.0, `libsumo` and `netconvert`, so you don't need a separate SUMO install.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/check_runtime.py
```

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_runtime.py
```

`check_runtime.py` prints the torch, SUMO and `netconvert` versions and runs a small CUDA calculation.

**Activate the virtual environment before running anything.** The SUMO adapter finds `netconvert` on `PATH`; on Windows it isn't found unless the environment is active.

Run every command from the repository root.

## Running the project

**Tests**

```bash
python -m pytest tests multistation/tests realdata/tests
```

All 51 tests pass when the raw DDU data is present. Without it, 44 pass and the 7 DDU tests are skipped, as designed. See [Real DDU data](#real-ddu-data) for how to download it.

**Baselines, plots and demo** (fast, uses the compact 3-platform configuration)

```bash
python scripts/validate_simulator.py --config configs/compact.yaml   # zero-delay timetable is safe and delay-free in SUMO
python scripts/compare_baselines.py --config configs/compact.yaml --episodes 3 --output results/my_baselines
python scripts/generate_plots.py --results results/my_baselines
python scripts/demo.py --config configs/compact.yaml --policy timetable --seed 123
python scripts/generate_scenarios.py --config configs/compact.yaml --count 3 --output data/my_scenarios
```

`validate_simulator.py` defaults to `configs/default.yaml`, so without `--config` it checks the full 8-platform, 128-train station (about as fast).

**Evaluate the trained agent**

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/station8_128/best_model.pt --episodes 3 --output results/my_eval
python scripts/generate_plots.py --results results/my_eval --training checkpoints/station8_128
python scripts/demo.py --config configs/default.yaml --policy dqn --checkpoint checkpoints/station8_128/best_model.pt --seed 123
```

- Add `--stress` to `evaluate.py` to run five more settings: severity ×1.5 and ×2, a platform closure, a peak, and a corridor disruption.
- `--episodes 100` reproduces the committed evaluation. That takes much longer, at 500 SUMO episodes.

**Train**

```bash
python scripts/train.py --config configs/compact.yaml --episodes 20 --seed 42 --output checkpoints/my_run
python scripts/train.py --config configs/default.yaml --seed 42 --output checkpoints/my_station8_128   # full 300 episodes
```

A training run writes these files:
- `initial_model.pt`, `best_model.pt` and `final_model.pt`;
- `training.csv` and `validation.csv`;
- `validation_references.json` (heuristic scores on the same validation days);
- a manifest that records whether training improved on the untrained network.

`--warm-start FILE` reuses network weights but restarts the optimizer, replay buffer and exploration.

**Output folders.** Every evaluation and training script refuses to overwrite a folder that already contains a manifest, so pick a new `--output` each time.

### Seven-station network

```bash
python multistation/scripts/evaluate_network.py --checkpoint checkpoints/station8_128/best_model.pt --output multistation/results/my_eval
python multistation/scripts/record_network_replay.py --checkpoint checkpoints/station8_128/best_model.pt --seeds 200000 --output multistation/results/my_replay/replay.json
python multistation/visualization/build_replay.py multistation/results/my_replay/replay.json multistation/results/my_replay.html
```

### Real DDU data

The raw source data (RSTGCN and DataMeet) isn't redistributed. Download and verify it first. The script uses `bash`, `curl` and `unzip`; on Windows, run it from Git Bash.

```bash
realdata/scripts/fetch_data.sh
python realdata/scripts/build_ddu.py --output realdata/data/my_build          # rebuilds the committed day files byte-for-byte
python realdata/scripts/replay_ddu.py --scenarios realdata/data/ddu_sep2024 --checkpoint checkpoints/station8_128/best_model.pt --modes planned actual observed --output realdata/results/my_replay
python realdata/scripts/record_ddu_replay.py --scenarios realdata/data/ddu_sep2024 --checkpoint checkpoints/station8_128/best_model.pt --days "2024-09-27:Median day" --output realdata/results/my_replay.html
python realdata/scripts/finetune_ddu.py --scenarios realdata/data/ddu_sep2024 --output realdata/results/my_finetune
```

`replay_ddu.py` only needs the committed day files in `realdata/data/ddu_sep2024/`. The tests, `build_ddu.py` and fine-tuning need the raw data.

## Viewing results

Open these pages in a browser. They are self-contained, with no server needed.

| Page | Shows |
|---|---|
| `results/project_overview.html` | Project overview |
| `results/station_replay.html` | Animated replay of the single junction |
| `multistation/results/network_replay.html` | Animated replay of the seven-station network: DQN against priority-aware and FCFS |
| `realdata/results/ddu_replay.html` | Animated real DDU days (median, best and worst), switchable between controllers |

Figures for the main experiment are in `results/station8_128_eval100/figures/` as PNG and PDF:
- learning curve;
- policy comparison;
- delay per train class;
- platform-occupancy Gantt chart.

Raw numbers are in each results folder: `summary.json`, `episodes.csv` (one row per policy and scenario), `trains.csv` (one row per train) and `manifest.json` (configuration, seeds, checkpoint and scenario hashes).

## How it works

- **State.** A 229-value vector:
  - per platform: occupancy, estimated release time, closures, route availability and compatibility;
  - the current train: priority, waiting time, length, dwell time and planned platform;
  - the next 6 forecast arrivals, with noisy ETAs;
  - global congestion and time of day.

  The agent never sees true future delays.
- **Actions.** `Discrete(9)`: action 0 holds the train for a minute; actions 1–8 assign a platform. A mask removes every unsafe or incompatible choice, and the same mask is given to all baselines.
- **Reward.** Each minute the reward is `−(1·W + 0.05·O + 0.2·U + 2·P + 0.2·H) / 10`, where:
  - `W` is priority-weighted waiting;
  - `O` and `U` are trains queued at the outer signal and further upstream;
  - `P` is a platform change;
  - `H` is a hold while a safe platform was free.

  Incoming lateness is not charged to the controller.
- **Learning.** Masked Double DQN with a 128×128 MLP, replay, a target network, Huber loss and gradient clipping. Bellman targets discount by `γ^elapsed_minutes` for decisions that span several minutes.
- **Safety.** Reservations are checked before each action, again in the adapter, and SUMO signals add a third layer. Teleports are disabled. Any collision or teleport aborts the episode, and history checks run at the end of every episode.

The full formulation, normalisation table, railway model and assumptions are in [Docs/TECHNICAL_NOTES.md](Docs/TECHNICAL_NOTES.md).

## Limitations

- **Synthetic data.** The synthetic timetables and delays are uncalibrated. The DDU layout and interlocking are simplified, so its absolute delay levels are higher than reality.
- **Evidence.** All results come from one training seed. The extensions use small samples: 3 network days, 28 real days and 10 fine-tuning test days.
- **Control.** Only the junction is learned. The six minor stations in the network extension follow a fixed rule.
- **Process.** `libsumo` allows one simulation per process, so evaluation runs sequentially.

## References

- [Double DQN](https://arxiv.org/abs/1509.06461)
- [SUMO railway simulation](https://sumo.dlr.de/docs/Simulation/Railways.html)
- [libsumo](https://sumo.dlr.de/docs/Libsumo.html)
- [Gymnasium](https://gymnasium.farama.org/api/env/)
- RSTGCN dataset: Chowdhury et al., IEEE T-ITS 2026
- DataMeet railways (CC0)
