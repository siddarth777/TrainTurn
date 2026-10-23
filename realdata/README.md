# Real-data replay at DDU (Pt. Deen Dayal Upadhyaya Junction)

This folder tests the trained junction controller on **real Indian Railways days**.
It replays the September 2024 timetable and delays recorded at Pt. Deen Dayal
Upadhyaya Junction (DDU, formerly Mughalsarai), one of the busiest junctions in
the country.

The checkpoint (`checkpoints/station8_128/best_model.pt`) was trained only on
synthetic days and is used unchanged, so these results are zero-shot. The
original single-station code is reused without modification.

## What is real and what is assumed

| Item | Source |
|---|---|
| Which trains call at DDU, in which order, and their scheduled arrival and departure | **Real**: RSTGCN `train_routes_Sep2024.csv` |
| Each train run's delay, per station and day | **Real**: RSTGCN `train_routes_delays_Sep2024.csv` (September 2024) |
| Direction: west (Varanasi, Prayagraj, Chunar) is corridor A; east (Buxar–Patna, Sasaram–Gaya) is corridor B | **Real**: DataMeet station coordinates; renamed stations PRYJ and PCOI set by hand |
| Train class (premium, superfast, express, passenger) | **Rules** on the real train name, DataMeet type and the numbering convention (`configs/ddu.yaml`) |
| 8 platforms, of which only 4 take trains of 24 or more coaches | **Real** count: Wikipedia and the 2018 CAG audit. Which four, and the lengths of 26 and 22 coaches, are **assumptions** |
| Train lengths (16–24 coaches by class) | **Assumption** |
| Yard moves for trains starting or ending at DDU (rake placed 20 minutes before departure, cleared 15 minutes after arrival) | **Assumption** |
| Dwell overruns | **Zero**. In the dataset, departure delay always equals arrival delay, so real dwell is not observable. |
| Interlocking (one entry and one exit movement at a time, 3-minute movements) | **The trained model's**. The real DDU yard can run some routes in parallel, so the replay is more congested than reality. |
| Freight trains | **Absent** from the data, which covers about 3,900 passenger services. |

**Sources**

* **RSTGCN** (IIT Kharagpur, published in IEEE T-ITS 2026): released for
  research, with no explicit licence. Cite: K. Chowdhury, P. Koley,
  A. Chakraborty, S. Ghosh, *RSTGCN: Railway-centric spatio-temporal graph
  convolutional network for train delay prediction*, IEEE T-ITS, 2026.
* **DataMeet railways**: CC0.

The raw files are not committed. `scripts/fetch_data.sh` downloads them and
checks the SHA-256 of the version used here. The derived day files in
`data/ddu_sep2024/` contain timetables and delays taken from RSTGCN; check the
dataset terms before publishing them.

## How a real day becomes a scenario

* **Day of each run.** The dataset's `date` is the date the journey *started*.
  The day a run reaches DDU is that date plus its day offset along the route.
  The importer checks this: every train-date group starts at the first station
  of the route.
* **Clock and window.** Minute 0 is 00:00 at DDU. A run belongs to the day on
  which it reaches DDU's outer signal, before 23:00. Later arrivals would not
  finish within the trained 1,440-minute horizon; they are counted per day in
  the build manifest (about 7 a day).
* **Delay modes.** The same 28 fully covered days, 3–30 September, are replayed
  in three modes:
  1. **`planned` (step 1).** The real timetable with no delays. It is written
     as a CSV and loaded with the project's original loader
     (`railway_rl.timetable.load_timetable`).
  2. **`actual` (step 2, main).** Each train reaches DDU as late as it was at
     the *last station recorded before DDU*. That delay is exogenous to DDU, so
     the junction's own effect is not double-counted.
  3. **`observed` (sensitivity check).** The arrival delay recorded *at* DDU.
     This already contains whatever waiting the real junction caused, so it
     biases the other way. `actual` overstates lateness at the outer signal,
     because trains made up a median of 6 minutes on the last section.
* **Late trains from the previous evening.** About 7 trains a day were scheduled
  the previous evening but reached DDU after midnight. Their schedule is moved
  to minute 0, and the lateness carried over midnight becomes their incoming
  delay.
* **Forecasts.** The controller's forecast for a train is its delay at the
  station recorded *before* the last one, which is known earlier than the
  incoming delay. No later information is used.

## Commands

Run from the repository root with the project virtual environment active.

```bash
realdata/scripts/fetch_data.sh                                   # download + verify raw data
python realdata/scripts/build_ddu.py --output realdata/data/<new_dir>
python realdata/scripts/replay_ddu.py --scenarios realdata/data/ddu_sep2024 \
    --checkpoint checkpoints/station8_128/best_model.pt --output realdata/results/<new_dir>
#   options: --modes planned actual observed   --days N
python realdata/scripts/record_ddu_replay.py --scenarios realdata/data/ddu_sep2024 \
    --checkpoint checkpoints/station8_128/best_model.pt \
    --days "2024-09-27:Median day" "2024-09-03:Best day" --output realdata/results/ddu_replay.html
python realdata/scripts/finetune_ddu.py --scenarios realdata/data/ddu_sep2024 \
    --output realdata/results/<new_dir>                         # CUDA; split and settings in ddu.yaml
python realdata/scripts/replay_ddu.py --scenarios realdata/data/ddu_sep2024 \
    --checkpoint checkpoints/station8_128/best_model.pt \
    --extra-checkpoint finetuned=realdata/results/ddu_finetune_actual/best_model.pt \
    --first-day 2024-09-21 --last-day 2024-09-30 --modes planned actual observed \
    --output realdata/results/<new_dir>                         # held-out test
python -m pytest realdata/tests        # 7 tests; skipped if the raw data is missing
```

The checkpoint needs CUDA; leave out `--checkpoint` to run the baselines only.
An output directory that already contains a manifest is never overwritten.
Rebuilding the day files is deterministic; a second build was byte-identical.

## Results: 28 real days, 3–30 September 2024 (about 120 trains a day)

Results are in `results/ddu_sep2024_replay/` (planned and actual) and
`results/ddu_sep2024_observed/`. The metric is **weighted station delay**:
priority × minutes of waiting caused at DDU, summed over the day. Values are
daily means with 95% bootstrap intervals over days.

**All 420 episodes** (28 days × 3 modes × 5 controllers) **completed every
train with zero safety violations.**

| Controller | Planned (no delays) | Actual (main) | Observed (sensitivity) |
|---|---|---|---|
| Trained DQN | **422.0** [399, 447] | 729.3 [651, 813] | **491.1** [446, 539] |
| Priority-aware | 454.5 [429, 479] | 723.8 [656, 799] | 539.0 [496, 585] |
| Planned platform | 443.9 [421, 467] | 729.3 [655, 810] | 541.2 [502, 581] |
| FCFS | 440.5 [418, 463] | 768.1 [694, 848] | 558.7 [517, 602] |
| Greedy lookahead | 443.6 [421, 467] | **706.1** [642, 774] | 547.6 [504, 592] |

Paired difference, DQN minus baseline on the same days, with the number of days
the DQN was lower:

| Baseline | Planned | Actual (main) | Observed |
|---|---|---|---|
| Priority-aware | −32.5 [−50.7, −14.2], 20/28 | +5.6 [−49.6, +70.7], 17/28 | −47.8 [−77.8, −15.5], 21/28 |
| Planned platform | −21.9 [−37.8, −6.9], 17/28 | +0.1 [−58.6, +64.8], 17/28 | −50.0 [−79.8, −19.1], 20/28 |
| FCFS | −18.5 [−33.7, −4.2], 17/28 | −38.8 [−95.9, +25.1], 18/28 | −67.6 [−99.5, −34.9], 22/28 |
| Greedy lookahead | −21.5 [−37.3, −7.0], 17/28 | +23.3 [−32.0, +82.5], 14/28 | −56.5 [−88.2, −25.1], 22/28 |

**What this supports**

* **Real timetable (step 1).** On DDU's real timetable without delays, the
  synthetic-trained DQN causes less weighted waiting than every heuristic, and
  every interval excludes zero.
* **Real delays (step 2).** The answer depends on how incoming lateness is
  measured.
  * With the main definition (delay at the previous station), the DQN is level
    with priority-aware and planned-platform allocation, and slightly behind
    greedy lookahead. No difference is significant.
  * With the delay recorded at DDU itself, the DQN is best against all four,
    and every interval excludes zero.
  * We report the main result as **inconclusive**, not as a win.
* **Signs of distribution shift.** On real delays the DQN holds trains by
  choice about 15 times a day; on synthetic days it was about 2. Its worst day,
  9 September, is 555 worse than priority-aware; on that day even the untrained
  network did better (854 against 1,261). Real lateness is larger and burstier
  than in training, with a median of 18 minutes and a mean of 54 at the
  previous station.
* **Absolute values.** The modelled interlocking allows one movement at a time
  through each throat, while real DDU can run some in parallel. Absolute delay
  levels are therefore higher than reality; compare the controllers with each
  other, not with real DDU.

## Fine-tuning on real days (preliminary, one training seed)

The synthetic checkpoint was fine-tuned on real `actual`-mode days with a
chronological split, so every test day is later than every day used to train
or select the model:

| Split | Days | Use |
|---|---|---|
| Train | 3–17 Sep (15 days) | 10 passes in seeded shuffled order = 150 episodes |
| Validation | 18–20 Sep (3 days) | Checkpoint selection, including the unchanged checkpoint |
| Test | 21–30 Sep (10 days) | Touched once, after training ended |

* **Settings.** Warm start from `checkpoints/station8_128/best_model.pt`
  (weights only; replay and optimizer restart). Learning rate 1e-4, epsilon
  0.2 → 0.02 over 6,000 decisions, seed 7, other agent settings unchanged
  (`finetune:` in `configs/ddu.yaml`). The observation, actions and reward are
  those of the original model.
* **Run.** 31,435 optimizer updates on CUDA in 18 minutes
  (`results/ddu_finetune_actual/`).
* **Validation.** Delay fell from 852.5 (unchanged checkpoint) to 729.8 at
  episode 25. After that it drifted between 760 and 1,000, and discretionary
  holds crept back up, so longer training hurt. `best_model.pt` is episode 25.
  `final_model.pt` (episode 150) is tested too, to show the effect of
  over-training.

**Held-out test, 21–30 September** (`results/ddu_finetune_test/`, 210
episodes; all complete, zero safety violations). Mean weighted station delay
per day with 95% bootstrap intervals; discretionary holds per day in
brackets for the learned controllers:

| Controller | Planned | Actual (main) | Observed |
|---|---|---|---|
| Fine-tuned (best, ep. 25) | **429.6** [392, 476] (0.2) | **669.8** [542, 810] (2.5) | **488.3** [417, 552] (0.6) |
| Zero-shot DQN | 437.5 [393, 488] (1.4) | 675.6 [566, 797] (9.7) | 509.1 [439, 584] (6.7) |
| Fine-tuned (final, ep. 150) | 461.0 [432, 491] (5.8) | 708.4 [565, 849] (10.3) | 520.3 [454, 580] (4.9) |
| Greedy lookahead | 452.4 | 708.0 | 584.4 |
| Priority-aware | 463.5 | 725.6 | 566.5 |
| Planned platform | 451.4 | 728.2 | 569.8 |
| FCFS | 451.5 | 778.2 | 593.1 |

Paired differences on the 10 test days (days lower out of 10):

| Comparison | Planned | Actual (main) | Observed |
|---|---|---|---|
| Fine-tuned − zero-shot | −7.8 [−41.4, +18.3], 3 | −5.8 [−55.1, +50.7], 6 | −20.8 [−68.8, +16.6], 4 |
| Fine-tuned − priority-aware | −33.9 [−65.2, −2.1], 8 | −55.8 [−95.0, −9.8], 8 | −78.1 [−101.9, −54.5], 10 |
| Fine-tuned − greedy | −22.8 [−52.5, +5.5], 6 | −38.2 [−87.2, +23.1], 7 | −96.1 [−132.8, −59.5], 10 |
| Zero-shot − priority-aware | −26.1 [−60.7, +7.5], 6 | −50.0 [−87.7, −12.9], 8 | −57.4 [−107.9, 0.0], 8 |

**What this supports**

* **Fine-tuning versus zero-shot: no significant gain.** The fine-tuned model
  has the lowest mean in every mode, but every interval against the zero-shot
  model includes zero. The clearest change is behavioural: discretionary holds
  on real delays fall from 9.7 to 2.5 a day, and its results are steadier
  (observed mode: lower than all four heuristics on all 10 days).
* **Over-training hurts.** The final episode-150 model is worse than the
  zero-shot model in every mode. Checkpoint selection on validation days was
  needed.
* **Test days differ from the full month.** On 21–30 September even the
  zero-shot DQN beats priority-aware with real delays (−50.0, interval
  excludes zero), while over all 28 days it was level. The worst days for the
  zero-shot model (for example 9 September) fall in the training period, so
  this split cannot show whether fine-tuning fixes them.
* **Limits.** One fine-tuning seed, 3 validation days and 10 test days. The
  layout and interlocking assumptions above still apply.

**Natural next steps.** Repeat the fine-tuning with several seeds, use
rolling-origin splits so that bad days such as 9 September are tested, then
model DDU's real route-conflict table instead of a single throat.

## Replay page

`results/ddu_replay.html` animates three real days: the DQN's median, best and
worst day against priority-aware. On each day you can switch between the
trained DQN, the untrained network, priority-aware and FCFS. The recorded
totals equal `results/ddu_sep2024_replay`.
