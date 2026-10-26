"""Real Indian Railways timetable and delay replay at DDU (Pt. Deen Dayal Upadhyaya Jn).

Sources (see ../README.md for provenance and licences):

* RSTGCN September 2024 data: scheduled routes (``train_routes_Sep2024.csv``) and
  per train, start date and station, scheduled and actual times with delays
  (``train_routes_delays_Sep2024.csv``). The ``date`` column is the date the
  journey started; a run reaches DDU on that date plus its day offset along
  the route.
* DataMeet railways (CC0): train type codes and station coordinates.

Day scenarios for the trained single-junction model:

* ``planned``: every run's scheduled times at DDU, with no delays (step 1).
* ``actual``: each run reaches DDU's outer signal late by the delay it actually
  had at the last station recorded before DDU (step 2). That delay is
  exogenous to DDU, so the junction's own effect is not double-counted. Dwell
  extensions are zero, because the dataset's departure delay always equals its
  arrival delay (no dwell observations).
* ``observed`` (sensitivity check): the arrival delay recorded at DDU itself.
  It already contains any real waiting at DDU, so it biases the other way:
  ``actual`` overstates lateness at the outer signal by the time trains make up
  on the last section, ``observed`` includes the real junction's own delays.
* Forecast of a run's delay: the delay at the station recorded before that
  one, i.e. information available earlier than the incoming delay itself.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd
import yaml
from railway_rl.config import load_config
from railway_rl.station import Disturbance, Platform, Scenario, Station, Train

WORD = re.compile(r"[^A-Z]+")
REPO = Path(__file__).resolve().parents[2]


def portable(path: str | Path) -> str:
    """Repository-relative form of a path, so written manifests carry no machine paths."""
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def load_real_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    raw["junction"] = load_config((path.parent / raw["junction_config"]).resolve())
    for key in ("rstgcn", "datameet"):
        raw["raw"][key] = (path.parent / raw["raw"][key]).resolve()
    lengths = raw["station"]["platform_lengths_coaches"]
    if not 3 <= len(lengths) <= raw["junction"]["environment"]["max_platforms"]:
        raise ValueError("Platform count must fit the trained observation (3-8)")
    if "finetune" in raw:
        f = raw["finetune"]
        f["warm_start"] = (path.parent / f["warm_start"]).resolve()
        for key in ("train_days", "validation_days", "test_days"):
            f[key] = [str(d) for d in f[key]]  # YAML reads bare dates as datetime.date
        if not f["train_days"][1] < f["validation_days"][0] <= f["validation_days"][1] < f["test_days"][0]:
            raise ValueError("Fine-tuning split must be chronological: train < validation < test")
    raw["source"] = portable(path)
    return raw


def to_minutes(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, format="%I:%M %p", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute


def classify(name: str, dm_type: str | None, number: str, rules: dict) -> str:
    words = set(WORD.split(str(name).upper()))
    if words & set(rules["premium_words"]) or dm_type in rules["premium_types"]:
        return "premium"
    if words & set(rules["passenger_words"]) or dm_type in rules["passenger_types"]:
        return "passenger"
    if (words & set(rules["superfast_words"]) or dm_type in rules["superfast_types"]
            or number[:2] in rules["superfast_number_prefixes"]):
        return "superfast"
    return "express"


def coaches(category: str, name: str, lengths: dict) -> int:
    words = set(WORD.split(str(name).upper()))
    if category == "premium" and "VANDE" in words:
        return lengths["vande_bharat"]
    if category == "premium" and "SHATABDI" in words:
        return lengths["shatabdi"]
    return lengths[category]


@dataclass(frozen=True)
class Run:
    """One train run through DDU on one start date."""
    number: str
    name: str
    start_date: str
    calendar_date: str          # date the run reaches DDU
    sch_arr: int                # minutes after 00:00 of calendar_date (may exceed 1440 for departure)
    sch_dep: int
    category: str
    length: int
    origin: bool                # journey starts at DDU
    terminus: bool              # journey ends at DDU
    arrival_corridor: str
    departure_corridor: str
    incoming_delay: float | None    # delay at the last recorded station before DDU, minutes
    forecast_delay: float | None    # delay at the recorded station before that one
    observed_ddu_delay: float | None  # arrival delay recorded at DDU itself (reference only)


def build_runs(cfg: dict) -> tuple[list[Run], dict]:
    code = cfg["station"]["code"]
    rst, dm = Path(cfg["raw"]["rstgcn"]), Path(cfg["raw"]["datameet"])
    routes = pd.read_csv(rst / "train_routes_Sep2024.csv", dtype={"trainNumber": str})
    delays = pd.read_csv(rst / "train_routes_delays_Sep2024.csv", dtype={"train": str})
    trains = json.loads((dm / "trains.json").read_text())["features"]
    stations = json.loads((dm / "stations.json").read_text())["features"]
    dm_type = {f["properties"]["number"]: f["properties"].get("type") for f in trains}
    coords = {f["properties"]["code"]: (f["geometry"] or {}).get("coordinates") for f in stations}
    here = coords[cfg["coordinates_code"]]

    def side(station: str | None) -> str | None:
        if station is None:
            return None
        if station in cfg["side_overrides"]:
            return cfg["side_overrides"][station]
        c = coords.get(station)
        return None if not c else ("west" if c[0] < here[0] else "east")

    routes = routes.sort_values(["trainNumber", "stnSerialNumber"])
    routes["t_arr"] = to_minutes(routes["arrivalTime"])
    routes["t_dep"] = to_minutes(routes["departureTime"])
    delays = delays.set_index(["train", "date", "station"]).sort_index()
    runs, skipped = [], {"no_side": 0, "no_times": 0}
    for number, g in routes.groupby("trainNumber", sort=False):
        codes = g.station_code.tolist()
        if code not in codes:
            continue
        # Day offset of each stop: the clock running backwards means a new day.
        day, last, offsets = 0, None, []
        for a, d in zip(g.t_arr, g.t_dep):
            t = a if not math.isnan(a) else d
            if last is not None and not math.isnan(t) and t < last - 1:
                day += 1
            offsets.append(day)
            if not math.isnan(a) and not math.isnan(d) and d < a - 1:
                day += 1  # departs after midnight; later stops are on the next day
            last = d if not math.isnan(d) else t
        i = codes.index(code)
        row = g.iloc[i]
        origin, terminus = i == 0, i == len(codes) - 1
        arr, dep = row.t_arr, row.t_dep
        if origin and math.isnan(arr):
            arr = dep - cfg["day"]["originating_rake_minutes"]
        if terminus and math.isnan(dep):
            dep = arr + cfg["day"]["terminating_clear_minutes"]
        if math.isnan(arr) or math.isnan(dep):
            skipped["no_times"] += 1
            continue
        if origin and arr == dep:
            arr = dep - cfg["day"]["originating_rake_minutes"]
        if terminus and arr == dep:
            dep = arr + cfg["day"]["terminating_clear_minutes"]
        if dep <= arr:
            dep += 1440 if dep < arr else 1  # departure after midnight, or zero-length halt
        prev_side = side(codes[i - 1]) if i > 0 else None
        next_side = side(codes[i + 1]) if not terminus else None
        if prev_side is None and next_side is None:
            skipped["no_side"] += 1
            continue
        # Originating trains come off the yard opposite their departure side; terminating
        # trains clear towards the side they were heading.
        prev_side = prev_side or ("west" if next_side == "east" else "east")
        next_side = next_side or ("east" if prev_side == "west" else "west")
        arrival_corridor = "A" if prev_side == "west" else "B"
        departure_corridor = "A" if next_side == "west" else "B"
        category = classify(row.trainName, dm_type.get(number), number, cfg["classes"])
        length = coaches(category, row.trainName, cfg["lengths"])
        earlier = codes[:i][::-1]
        for date in delays.loc[number].index.get_level_values(0).unique() if number in delays.index.get_level_values(0) else []:
            observed = delays.loc[(number, date)]
            seen = [s for s in earlier if s in observed.index]
            incoming = float(observed.loc[seen[0], "dep_delay"]) if seen else (0.0 if origin else None)
            forecast = float(observed.loc[seen[1], "dep_delay"]) if len(seen) > 1 else None
            at_ddu = float(observed.loc[code, "arr_delay"]) if code in observed.index else None
            if incoming is None and at_ddu is None:
                continue  # run has no observation before or at DDU
            calendar = (pd.Timestamp(date) + pd.Timedelta(days=offsets[i])).strftime("%Y-%m-%d")
            runs.append(Run(number, row.trainName, date, calendar, int(arr), int(dep), category, length,
                            origin, terminus, arrival_corridor, departure_corridor,
                            incoming if incoming is not None else at_ddu, forecast, at_ddu))
    return runs, skipped


def station_from(cfg: dict) -> Station:
    j = cfg["junction"]["station"]
    platforms = tuple(Platform(i + 1, length=n, electrified=True, kind="through")
                      for i, n in enumerate(cfg["station"]["platform_lengths_coaches"]))
    return Station(platforms, j["outer_capacity"], j["entry_minutes"], j["departure_minutes"], j["route_clearance"])


def day_scenario(cfg: dict, runs: list[Run], date: str, mode: str) -> tuple[Scenario, dict]:
    """Scenario for one calendar day at DDU; ``mode`` is "planned", "actual" or "observed"."""
    if mode not in ("planned", "actual", "observed"):
        raise ValueError("mode must be 'planned', 'actual' or 'observed'")
    station = station_from(cfg)
    last = cfg["day"]["last_arrival_minute"]
    day0 = pd.Timestamp(date)
    chosen, info = [], {"late_window": 0, "carried_from_previous_day": 0, "missing_delay": 0}
    for run in runs:
        shift = (pd.Timestamp(run.calendar_date) - day0).days * 1440
        arr, dep = run.sch_arr + shift, run.sch_dep + shift
        delay = {"planned": 0.0, "actual": run.incoming_delay,
                 "observed": run.observed_ddu_delay if run.observed_ddu_delay is not None else run.incoming_delay}[mode]
        if delay is None:
            info["missing_delay"] += 1
            continue
        actual = arr + max(0.0, delay)
        if not 0 <= actual < last:
            if mode != "planned" and last <= actual < 1440 and arr >= 0:
                info["late_window"] += 1
            continue
        chosen.append((actual, arr, dep, delay, run))
    chosen.sort(key=lambda c: (c[0], c[4].number))
    trains, noise, seen_ids = [], [], {}
    releases = {p.id: -10**9 for p in station.platforms}
    for actual, arr, dep, delay, run in chosen:
        if arr < 0:
            # Scheduled the previous day but reached DDU today: the schedule is moved to
            # minute 0 and the lateness carried over midnight becomes its incoming delay.
            info["carried_from_previous_day"] += 1
            arr, dep, delay = 0, dep - arr, actual
        dwell = max(1, int(round(dep - arr)))
        n = seen_ids.get(run.number, 0)
        seen_ids[run.number] = n + 1
        tid = run.number if n == 0 else f"{run.number}-{n + 1}"
        kind = "terminal" if run.origin or run.terminus else "through"
        candidate = Train(tid, run.category, int(round(arr)), int(round(arr)) + dwell, 1, run.length, True, kind,
                          run.arrival_corridor, run.departure_corridor, dwell,
                          cfg["junction"]["scenario"]["priorities"][run.category])
        # Planned platform: compatible platform free longest before the scheduled arrival.
        options = [p for p in station.platforms if p.compatible(candidate)]
        free = [p for p in options if releases[p.id] <= candidate.scheduled_arrival]
        best = min(free, key=lambda p: p.id) if free else min(options, key=lambda p: (releases[p.id], p.id))
        releases[best.id] = max(releases[best.id], candidate.scheduled_arrival) + dwell + station.departure_minutes + best.clearance
        trains.append(Train(**{**candidate.__dict__, "planned_platform": best.id}))
        forecast = run.forecast_delay if (mode != "planned" and run.forecast_delay is not None) else 0.0
        noise.append(Disturbance(int(round(max(0.0, delay))), 0, float(max(0.0, forecast)), 0.0))
    seed = int(date.replace("-", ""))
    provenance = (f"RSTGCN Sep 2024 running data + DataMeet (CC0) at DDU, {date}, {mode}; "
                  "platform lengths, train lengths and yard moves are assumptions")
    info.update(trains=len(trains), mode=mode, date=date)
    return Scenario(station, tuple(trains), tuple(noise), (), seed, provenance), info


def complete_days(runs: list[Run]) -> list[str]:
    """Calendar days fully covered: every run reaching DDU that day started inside September."""
    offsets = max((pd.Timestamp(r.calendar_date) - pd.Timestamp(r.start_date)).days for r in runs)
    days = sorted({r.calendar_date for r in runs})
    first = (pd.Timestamp("2024-09-01") + pd.Timedelta(days=offsets)).strftime("%Y-%m-%d")
    return [d for d in days if first <= d <= "2024-09-30"]


def describe(runs: list[Run]) -> dict:
    df = pd.DataFrame([r.__dict__ for r in runs])
    gained = (df.observed_ddu_delay - df.incoming_delay).dropna()
    return {"runs": len(df), "trains": int(df.number.nunique()),
            "categories": df.drop_duplicates("number").category.value_counts().to_dict(),
            "corridors": {f"{a}->{b}": int(n) for (a, b), n in df.groupby(["arrival_corridor", "departure_corridor"]).size().items()},
            "incoming_delay_median": float(np.nanmedian(df.incoming_delay)),
            "incoming_delay_mean": float(np.nanmean(df.incoming_delay)),
            "real_delay_change_into_ddu_median": float(gained.median()) if len(gained) else None}
