# Seven-station network extension

This folder places the original RL-controlled junction inside a synthetic
seven-station, double-track line. Traffic runs in both directions, and the line
is simulated in SUMO 1.28.0. **Only the junction is controlled by the agent.**
The six other stations follow one fixed local rule.

The trained checkpoint is used unchanged. It was trained on the isolated
8-platform, 128-train junction with eastbound traffic only, and it has never
seen the network or westbound trains. Results here are therefore *zero-shot
transfer*, not a new training run.

The original single-station code (`railway_rl/`, `configs/`, `scripts/`) is not
modified. This package imports it.

## Layout

```
        westbound ◂─────────────────────┐            ┌──────────────────── westbound
 A line  AW2 ══ AW1 ══ feeder ══ gate ══╡            ╞══ gate ══ AE1 ══ AE2
        eastbound ─────────────────────▸│  junction  │▸─────────────────── eastbound
                                        │   8 PF,    │
        eastbound ─────────────────────▸│ bidirect.  │▸─────────────────── eastbound
 B line  BW1 ══════════ feeder ══ gate ══╡  RL-ctrl.  ╞══ gate ══ BE1
        westbound ◂─────────────────────┘            └──────────────────── westbound
```

| Station | Tracks per direction | Role |
|---|---|---|
| Junction (J) | 8 shared platforms | Controlled by the Double DQN or a junction baseline |
| AW2, AW1, BW1 | 2, 3, 3 | Minor stations west of J |
| AE1, AE2, BE1 | 3, 2, 3 | Minor stations east of J |

**Track and signalling**

* Every line is double track: one track per direction. Each is split into
  signalled blocks of about 1 km. Each minor station has a main track and loops
  for each direction.
* Junction platforms, and the turnouts leading to them, are *bidirectional*
  tracks (SUMO `bidi` edges). Eastbound trains enter at the west throat, and
  westbound trains enter at the east throat. SUMO's rail signals protect the
  shared tracks against opposing movements. The adapter checks this
  independently every second.
* Teleports are disabled, and any collision or teleport aborts the episode.

**Junction corridors**

| Corridor | As an arrival | As a departure |
|---|---|---|
| A / B | Eastbound, from west line A / B | Eastbound, to east line A / B |
| C / D | Westbound, from east line A / B | Westbound, to west line A / B |

**What the trained agent sees**

* Eastbound junction edges keep the trained network's names, lengths and node
  positions.
* Every junction connection is pinned to the trained network's internal-lane
  speed. Westbound connections take the speed of their eastbound mirror. Adding
  the westbound edges therefore does not change junction running times. A test
  checks all 96 connections.
* The observation (229 features), the 9 actions and the reward are unchanged.
* Westbound trains appear to the agent as corridors C/D. It never saw these
  during training, so they are a deliberate distribution shift (see results).

## How the junction boundary works

| | Original (single station) | This network |
|---|---|---|
| Arrival | Timetable time plus sampled delay. The train appears at the boundary. | The train physically stops at the junction gate, its outer signal, and joins the queue at the next dispatch minute. |
| Admission | Insert the train at rest at `approachX`. | The held train is re-routed into the platform and starts from rest at the gate, as in training. |
| Outer queue | A list with capacity 2. | One hold point per approach corridor, so at most 4 trains. |
| Upstream queue | An unlimited reservoir. | Trains held back by the junction queue *anywhere on their approach line*: feeder blocks, earlier block signals, or inside a minor station after their dwell. |
| Waiting charged to J | Every queued minute. | Every second a queued train is stationary. Membership updates at dispatch minutes, as in the original. |
| Interlocking | All entries share one throat, and all exits share the other. | Reservations are per throat. An eastbound entry and a westbound exit both use the west throat, so they are serialized. With eastbound-only traffic this is exactly the original rule. |
| Departure | The front leaves `downstreamX` (network exit). | The same moment. The train then continues along its departure line. |
| Forecasts | Pre-sampled noisy ETA. | Running trains: ETA from position and remaining scheduled calls. Trains not yet started: the original noisy forecast of their late start. |

No hidden future disturbance is read before it happens: late starts and dwell
extensions stay hidden until they occur. A test checks this for trains that
have not yet started.

**Minor-station rule (not learned).** For each direction:

* Non-stopping trains prefer the main track (1).
* Stopping trains prefer their planned loop.
* The first free track in that order is taken. If none is free, the planned
  track is kept and SUMO's signals hold the train.
* Stopping trains never leave before their scheduled departure (SUMO `until`).

The rule is identical for every junction policy.

**Timetable.** Built in four steps:

1. The original `generate_scenario` with the same seed produces the services,
   their late starts, the junction dwell extensions and the forecast noise.
2. A separate seeded random stream sends half the trains westbound
   (`westbound_share`).
3. The junction timetable is re-planned with throat-aware reservations.
4. Network times are derived around the junction: backwards from the scheduled
   stop at the gate, and forwards from the junction departure. Running time over
   a distance `L` is
   `(L/v + v/2a [from rest] + v/2d [to a stop]) × (1 + run_allowance)`.

**Stopping patterns**

* Premium, superfast and freight trains stop only at J.
* Express trains also call at AW1, BW1, AE1 and BE1.
* Passenger trains call at every station before J. They are terminal services
  at J, so afterwards they run empty to the end of the line.

All data are synthetic and uncalibrated. They are not Indian Railways data.

## Commands

Run from the repository root with the project virtual environment active.

```bash
python -m pytest multistation/tests          # 10 network tests (CPU; SUMO)

# Junction policies inside the network; scenarios are shared by all policies.
python multistation/scripts/evaluate_network.py \
    --checkpoint checkpoints/station8_128/best_model.pt \
    --output multistation/results/<new_directory>
# options: --episodes N (default 3), --seed S (default 200000), --config FILE

# Record replays and build the animated HTML page
python multistation/scripts/record_network_replay.py \
    --checkpoint checkpoints/station8_128/best_model.pt \
    --seeds 200000 200001 200002 --output multistation/results/replay/replay.json
python multistation/visualization/build_replay.py \
    multistation/results/replay/replay.json multistation/results/network_replay.html
```

* Loading the checkpoint requires CUDA. As in the main project, there is no CPU
  fallback.
* Omit `--checkpoint` to evaluate the baselines only.
* An output directory that already contains a manifest is never overwritten.
* Network settings are in `multistation/configs/network.yaml`. It reuses
  `configs/default.yaml` for the junction and only shifts `start_minute` to 30,
  so that the first journeys start after minute 0.

## Preliminary results (3 held-out days, seeds 200000–200002)

Results are stored in `results/network7_bidir_eval3/`. Values are means with
95% scenario-bootstrap intervals over three scenarios. All 15 episodes
completed: every train reached the end of its line, and there were zero safety
violations.

| Junction policy | J weighted waiting | Network weighted lateness | Mean public lateness (min) |
|---|---|---|---|
| Trained DQN (zero-shot) | 221.6 | 2405.2 | 7.98 |
| Priority-aware | 233.3 | 2448.5 | 8.09 |
| Planned platform | 235.9 | 2444.8 | 8.09 |
| FCFS | 264.8 | 2493.2 | 8.20 |
| Greedy lookahead | 413.5 | 2547.1 | 8.38 |

Paired differences, DQN minus baseline, on the same scenarios:

| Baseline | J weighted waiting | Network weighted lateness |
|---|---|---|
| Priority-aware | −11.7 [−28.1, +2.9] | −43.3 [−71.1, −26.0] |
| Planned platform | −14.4 [−39.4, +9.0] | −39.7 [−64.9, −17.3] |
| FCFS | −43.3 [−54.0, −34.2] | −88.0 [−106.2, −78.3] |

Junction waiting by direction, as means over the three days:

| | DQN | Priority-aware | Planned platform | FCFS |
|---|---|---|---|---|
| Eastbound (trained direction) | 95.7 | 114.6 | 97.4 | 119.6 |
| Westbound (unseen corridors C/D) | 125.8 | 118.7 | 138.5 | 145.2 |

What these numbers support:

* The checkpoint transfers safely to a two-direction network it never saw. It
  clearly beats FCFS and greedy lookahead at the junction.
* **Network lateness:** the DQN is lowest on all 3 days.
* **Junction waiting:** against priority-aware and planned-platform allocation
  it is inconclusive. Both intervals include zero, and each heuristic won one
  of the three days.
* **By direction:** the DQN is best for eastbound trains. For westbound trains
  it is slightly worse than priority-aware. This matches the distribution
  shift: it never saw corridors C/D in training.
* **Scale:** late starts dominate network lateness, so the gaps between
  controllers are small in relative terms (about 2% between the DQN and
  priority-aware).
* **Sample size:** three scenarios and one training seed are far too few for a
  general claim. A 100-scenario network evaluation has not been run.

`results/network7_eval3/` comes from the earlier one-way version of this
network. The current code no longer reproduces it, and it is kept only as a
record.

## Limitations

* The minor-station rule is fixed; only the junction learns.
* The agent was not trained on westbound traffic or on the network. Fine-tuning
  in `NetworkTrainTurnEnv`, which is Gymnasium-compatible, is the natural next
  step, but it is not implemented.
* Up to four trains can be held at the junction's outer signals at once, while
  training saw at most two. The observation clips this count at three.
* The 30-minute timetable shift slightly changes the time-of-day features
  relative to training.
* Opposing traffic on the open line is not modelled: there are no single-track
  sections or crossing loops. The only shared track is at the junction.
