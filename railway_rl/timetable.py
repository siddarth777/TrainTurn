"""Synthetic nominal scheduling, CSV ingestion, and immutable scenario storage."""
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from .station import Platform, Station, Train, Scenario, Disturbance, Closure


def make_station(config: dict) -> Station:
    c = config["station"]
    if c["platforms"]:
        platforms = tuple(Platform(**{**p, "approaches": tuple(p.get("approaches", ["A", "B"])),
                                         "departures": tuple(p.get("departures", ["A", "B"]))})
                          for p in c["platforms"])
    else:
        n = c["platform_count"]
        platforms = tuple(Platform(i, length=(24, 18, 12)[(i - 1) % 3],
                                   # Generated terminal services are electric; keep their platform usable.
                                   electrified=True,
                                   kind="terminal" if i == n and n > 3 else "through")
                          for i in range(1, n + 1))
    return Station(platforms, c["outer_capacity"], c["entry_minutes"],
                   c["departure_minutes"], c["route_clearance"],
                   tuple(tuple(pair) for pair in c["extra_conflicts"]))


def nominal_timetable(config: dict, seed: int) -> tuple[Station, tuple[Train, ...]]:
    """Reserve nominal movements before accepting a scheduled arrival/departure."""
    rng = random.Random(seed)
    station = make_station(config)
    c = config["scenario"]
    reservations: list[tuple[str, int, int]] = []
    releases = {p.id: 0 for p in station.platforms}
    trains = []
    desired = c["start_minute"]

    def free(route: str, start: int, duration: int) -> bool:
        end = start + duration + station.route_clearance
        return all(not (a < end and start < b and (station.conflicts(route, r) or (config["environment"]["backend"] == "sumo" and route.split(":")[0] == r.split(":")[0])))
                   for r, a, b in reservations)

    for i in range(c["trains"]):
        # Dense bursts separated by longer gaps, rather than uniform arrivals.
        peak = (i % 20) / 20 < c["peak_fraction"] or c["mode"] == "peak"
        desired += rng.randint(1, 3) if peak else rng.randint(8, 18)
        category = rng.choices(list(c["priorities"]), [1, 2, 4, 3, 1])[0]
        length = {"premium": 16, "superfast": 22, "express": 18, "passenger": 10, "freight": 24}[category]
        corridor = rng.choice(("A", "B"))
        dwell = rng.randint(3, 8) if category != "freight" else rng.randint(8, 16)
        train = Train(f"SYN{i + 1:04}", category, desired, desired + dwell + 2, 1,
                      length, category != "freight", "terminal" if category == "passenger" else "through",
                      corridor, "B" if corridor == "A" else "A", dwell, c["priorities"][category])
        candidates = []
        for platform in station.platforms:
            if not platform.compatible(train):
                continue
            arrival = max(desired, releases[platform.id])
            entry_route = station.route(train, platform.id, "in")
            while not free(entry_route, arrival, station.entry_minutes):
                arrival += 1
            departure_start = arrival + station.entry_minutes + dwell
            exit_route = station.route(train, platform.id, "out")
            departure_start = max(departure_start, arrival + station.entry_minutes + station.route_clearance)
            while not free(exit_route, departure_start, station.departure_minutes):
                departure_start += 1
            candidates.append((arrival, departure_start, platform.id))
        if not candidates:
            raise ValueError(f"No platform supports generated train {train.id}")
        arrival, departure_start, platform = min(candidates)
        departure = departure_start + station.departure_minutes
        train = replace(train, scheduled_arrival=arrival, scheduled_departure=departure_start if config["environment"]["backend"] == "sumo" else departure,
                        planned_platform=platform)
        if c["rake_links"] and category == "passenger":
            previous = next((t for t in reversed(trains) if t.category == category
                             and t.scheduled_departure + train.turnaround <= arrival
                             and not any(x.rake_from == t.id for x in trains)), None)
            if previous is not None:
                train = replace(train, rake_from=previous.id)
        trains.append(train)
        reservations.extend([(station.route(train, platform, "in"), arrival,
                              arrival + station.entry_minutes + station.route_clearance),
                             (station.route(train, platform, "out"), departure_start,
                              departure + station.route_clearance)])
        releases[platform] = departure + station.platforms[platform - 1].clearance
        desired = arrival  # One dispatcher decision per tick; preserve arrival order.
    return station, tuple(trains)


def load_timetable(path: str | Path) -> tuple[Train, ...]:
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    integers = {"scheduled_arrival", "scheduled_departure", "planned_platform", "length", "dwell", "turnaround"}
    result = []
    for row in rows:
        values = {k: v for k, v in row.items() if v != ""}
        for key in integers & values.keys():
            values[key] = int(values[key])
        if "electric" in values:
            if values["electric"].lower() not in ("true", "false", "1", "0"):
                raise ValueError("CSV electric values must be true/false or 1/0")
            values["electric"] = values["electric"].lower() in ("true", "1")
        if "priority" in values:
            values["priority"] = float(values["priority"])
        result.append(Train(**values))
    return tuple(result)


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(scenario), indent=2, allow_nan=False) + "\n")


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text())
    station = data.pop("station")
    station["platforms"] = tuple(Platform(**{**p, "approaches": tuple(p["approaches"]),
                                            "departures": tuple(p["departures"])}) for p in station["platforms"])
    station["extra_conflicts"] = tuple(tuple(p) for p in station["extra_conflicts"])
    return Scenario(Station(**station), tuple(Train(**t) for t in data.pop("trains")),
                    tuple(Disturbance(**d) for d in data.pop("disturbances")),
                    tuple(Closure(**c) for c in data.pop("closures")), **data)
