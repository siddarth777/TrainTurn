"""Two-direction network timetables built around the junction timetable.

Construction, for one seed:

1. The original ``generate_scenario`` produces the services (classes, priorities,
   lengths, dwells, junction dwell extensions, late starts, forecast noise).
2. A separate seeded stream sends each train eastbound or westbound
   (``westbound_share``). Westbound trains use junction corridors ``C``/``D``.
3. The junction timetable (scheduled arrival, departure and planned platform) is
   re-planned with throat-aware reservations: with two directions, eastbound
   entries share the west throat with westbound exits, and vice versa. For
   eastbound-only traffic this rule equals the original planner's.
4. Network times are derived around the junction: backwards from the scheduled
   stop at the junction gate on the approach line, forwards from the junction
   departure on the departure line. Running time over ``L`` is
   ``(L/v + v/2a [from rest] + v/2d [to a stop]) * (1 + allowance)``.
5. The original incoming delay becomes the train's late start at its origin;
   junction lateness emerges from it plus physical interaction on the lines.

Times are seconds from simulation start. All data are synthetic.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import numpy as np
from railway_rl.delays import generate_scenario
from railway_rl.station import Scenario, Station, Train
from railway_rl.timetable import save_scenario, load_scenario
from .layout import CORRIDOR_LINE, EASTBOUND, J_DOWNSTREAM, J_TURNOUT, layout_from_config

STOP_OFFSET = 10.0  # m between a stopped train's front and the end of its track
HOLD_OFFSET = 5.0   # m between a held train's front and the junction gate
TO_WESTBOUND = {"A": "C", "B": "D"}


@dataclass(frozen=True)
class Call:
    station: str
    stops: bool
    track: int              # planned track for the train's direction; 1 is that direction's main line
    arrival: float          # scheduled front arrival (stoppers) or pass time, seconds
    departure: float        # scheduled departure (stoppers) or pass time, seconds
    dwell: int              # minutes; 0 when passing
    extension: int = 0      # exogenous extra dwell, minutes (revealed only when it happens)


@dataclass(frozen=True)
class TrainPlan:
    train: str
    direction: str          # "e" eastbound or "w" westbound
    origin_time: float      # scheduled insertion, seconds
    start_delay: int        # exogenous late start, minutes
    approach_calls: tuple[Call, ...]   # travel order, before the junction
    gate_arrival: float     # scheduled stop at the junction gate, seconds
    junction_departure: float          # scheduled junction departure start, seconds
    gate_out: float         # scheduled front passage past the junction boundary, seconds
    departure_calls: tuple[Call, ...]  # travel order, after the junction
    sink_arrival: float     # scheduled arrival at the end of the line, seconds
    terminal: bool          # public journey ends at the junction (empty stock afterwards)


@dataclass(frozen=True)
class NetworkScenario:
    junction: Scenario
    plans: tuple[TrainPlan, ...]
    seed: int
    layout_digest: str
    provenance: str = "synthetic two-direction network timetable around a synthetic junction timetable; uncalibrated"

    def plan(self, train_id: str) -> TrainPlan:
        return self._index[train_id]

    def __post_init__(self) -> None:
        if [p.train for p in self.plans] != [t.id for t in self.junction.trains]:
            raise ValueError("One network plan per junction train, in junction order")
        object.__setattr__(self, "_index", {p.train: p for p in self.plans})


def run_seconds(distance: float, from_rest: bool, to_rest: bool, net: dict) -> float:
    v, a, d = net["line_speed"], net["acceleration"], net["deceleration"]
    base = distance / v + (v / (2 * a) if from_rest else 0) + (v / (2 * d) if to_rest else 0)
    return base * (1 + net["run_allowance"])


def stops_at(train: Train, station: str, net: dict) -> bool:
    pattern = net["stops"][train.category]
    return pattern == "all" or station in pattern


def throat(route: str) -> str:
    """'W' or 'E' junction throat used by a route string ``direction:corridor:platform``."""
    direction, corridor, _ = route.split(":")
    eastbound = corridor in EASTBOUND
    return "W" if (direction == "in") == eastbound else "E"


def junction_conflict(station: Station, first: str, second: str) -> bool:
    return throat(first) == throat(second) or station.conflicts(first, second)


def two_direction_junction(config: dict, seed: int) -> Scenario:
    junction_cfg, net = config["junction"], config["network"]
    base = generate_scenario(junction_cfg, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
    westbound = [bool(rng.random() < net["westbound_share"]) for _ in base.trains]
    corridors = EASTBOUND + tuple(TO_WESTBOUND.values())
    station = replace(base.station, platforms=tuple(replace(p, approaches=corridors, departures=corridors)
                                                    for p in base.station.platforms))
    reservations: list[tuple[str, int, int]] = []
    releases = {p.id: 0 for p in station.platforms}

    def free(route: str, start: int, duration: int) -> bool:
        end = start + duration + station.route_clearance
        return all(not (a < end and start < b and junction_conflict(station, route, r)) for r, a, b in reservations)

    trains, previous = [], 0
    for train, west in zip(base.trains, westbound):
        if west:
            train = replace(train, arrival_corridor=TO_WESTBOUND[train.arrival_corridor],
                            departure_corridor=TO_WESTBOUND[train.departure_corridor])
        desired = max(train.scheduled_arrival, previous)
        candidates = []
        for platform in station.platforms:
            if not platform.compatible(train):
                continue
            arrival = max(desired, releases[platform.id])
            entry = station.route(train, platform.id, "in")
            while not free(entry, arrival, station.entry_minutes):
                arrival += 1
            start = max(arrival + station.entry_minutes + train.dwell,
                        arrival + station.entry_minutes + station.route_clearance)
            leave = station.route(train, platform.id, "out")
            while not free(leave, start, station.departure_minutes):
                start += 1
            candidates.append((arrival, start, platform.id))
        arrival, start, platform = min(candidates)
        departure = start + station.departure_minutes
        reservations += [(station.route(train, platform, "in"), arrival,
                          arrival + station.entry_minutes + station.route_clearance),
                         (station.route(train, platform, "out"), start, departure + station.route_clearance)]
        releases[platform] = departure + station.platforms[platform - 1].clearance
        trains.append(replace(train, scheduled_arrival=arrival, scheduled_departure=start, planned_platform=platform))
        previous = arrival
    return Scenario(station, tuple(trains), base.disturbances, base.closures, seed)


def generate_network_scenario(config: dict, seed: int) -> NetworkScenario:
    net = config["network"]
    layout = layout_from_config(net)
    junction = two_direction_junction(config, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    track_len, entry_len, exit_len = (net["station_track_length"], net["station_entry_length"],
                                      net["station_exit_length"])
    stop_point = track_len - STOP_OFFSET
    station_ids = [s.id for s in layout.stations]
    plans = []
    for index, (train, noise) in enumerate(zip(junction.trains, junction.disturbances)):
        # One draw per station for every train (policy-independent, direction-independent stream).
        extensions = {}
        for sid in station_ids:
            hit = rng.random() < net["dwell_extension_probability"]
            size = int(rng.integers(1, net["dwell_extension_max_minutes"] + 1))
            extensions[sid] = size if hit else 0

        def specs(stations, after_junction):
            out = []
            for sid in stations:
                stopping = stops_at(train, sid, net) and not (after_junction and train.kind == "terminal")
                tracks = layout.station(sid).tracks
                out.append((sid, stopping, 2 + index % (tracks - 1) if stopping else 1))
            return out

        # ----- approach line: backwards from the stop at the junction gate -----------------
        approach = specs(layout.stations_on(train.arrival_corridor, True), False)
        links = layout.links_on(train.arrival_corridor, True)
        gate = train.scheduled_arrival * 60.0
        t_next, next_rest = gate, True
        distance = STOP_OFFSET + exit_len + links[-1] - HOLD_OFFSET
        times = []
        for i in range(len(approach) - 1, -1, -1):
            sid, stopping, _ = approach[i]
            departure = t_next - run_seconds(distance, stopping, next_rest, net)
            dwell = net["minor_dwell_minutes"][train.category] if stopping else 0
            times.insert(0, (departure - 60 * dwell, departure, dwell))
            t_next, next_rest = departure - 60 * dwell, stopping
            distance = links[i] + entry_len + exit_len + track_len
        # Inserted with its rear at the line end, already at line speed.
        first = links[0] - train.length * 25 + entry_len + stop_point
        origin = t_next - run_seconds(first, False, next_rest, net)
        if origin < 0:
            raise ValueError(f"{train.id} would start before minute 0; raise junction start_minute")

        # ----- departure line: forwards from the junction departure start ------------------
        leaving = specs(layout.stations_on(train.departure_corridor, False), True)
        links = layout.links_on(train.departure_corridor, False)
        start = float(train.scheduled_departure * 60)
        clear = STOP_OFFSET + J_TURNOUT + J_DOWNSTREAM
        gate_out = start + run_seconds(clear, True, False, net)
        t_prev, prev_rest = start, True
        distance = clear + links[0] + entry_len + stop_point
        after = []
        for i, (sid, stopping, _) in enumerate(leaving):
            arrival = t_prev + run_seconds(distance, prev_rest, stopping, net)
            dwell = net["minor_dwell_minutes"][train.category] if stopping else 0
            after.append((arrival, arrival + 60 * dwell, dwell))
            t_prev, prev_rest = arrival + 60 * dwell, stopping
            distance = STOP_OFFSET + exit_len + links[i + 1] + entry_len + stop_point
        last = STOP_OFFSET + exit_len + links[-1] if leaving else clear + links[-1]
        sink = t_prev + run_seconds(last, prev_rest, False, net)

        def calls(spec, schedule):
            return tuple(Call(sid, stopping, track, round(a, 1), round(d, 1), dwell,
                              extensions[sid] if stopping else 0)
                         for (sid, stopping, track), (a, d, dwell) in zip(spec, schedule))

        plans.append(TrainPlan(train.id, CORRIDOR_LINE[train.arrival_corridor][1], round(origin, 1),
                               int(noise.incoming_delay), calls(approach, times), gate, start,
                               round(gate_out, 1), calls(leaving, after), round(sink, 1),
                               train.kind == "terminal"))
    return NetworkScenario(junction, tuple(plans), seed, layout.digest(junction.station))


def save_network_scenario(case: NetworkScenario, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    junction_file = directory / f"junction_{case.seed}.json"
    network_file = directory / f"network_{case.seed}.json"
    save_scenario(case.junction, junction_file)
    payload = {"seed": case.seed, "layout_digest": case.layout_digest, "provenance": case.provenance,
               "plans": [asdict(p) for p in case.plans]}
    network_file.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return junction_file, network_file


def load_network_scenario(directory: str | Path, seed: int) -> NetworkScenario:
    directory = Path(directory)
    junction = load_scenario(directory / f"junction_{seed}.json")
    data = json.loads((directory / f"network_{seed}.json").read_text())
    plans = tuple(TrainPlan(**{**p, "approach_calls": tuple(Call(**c) for c in p["approach_calls"]),
                               "departure_calls": tuple(Call(**c) for c in p["departure_calls"])})
                  for p in data["plans"])
    return NetworkScenario(junction, plans, data["seed"], data["layout_digest"], data["provenance"])
