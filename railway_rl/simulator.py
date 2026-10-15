"""Minute clock with half-open route reservations and independent platform locks."""
from dataclasses import dataclass
from .station import Scenario, Train


@dataclass
class TrainState:
    status: str = "future"
    signal_arrival: int | None = None
    entry: int | None = None
    platform: int | None = None
    ready: int | None = None
    departure_start: int | None = None
    departure: int | None = None
    station_departure: float | None = None
    waiting: int = 0
    departure_waiting: int = 0
    upstream_waiting: int = 0


@dataclass(frozen=True)
class Movement:
    train: str
    route: str
    start: int
    end: int  # Includes route clearance, not just physical movement.


class Simulator:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.station = scenario.station
        self.time = 0
        self.states = {t.id: TrainState() for t in scenario.trains}
        self.trains = {t.id: t for t in scenario.trains}
        self.noise = dict(zip(self.trains, scenario.disturbances))
        self.occupants: dict[int, str] = {}
        self.release = {p.id: 0 for p in self.station.platforms}
        self.outer: list[str] = []
        self.upstream: list[str] = []
        self.movements: list[Movement] = []
        self.active_movements: list[Movement] = []
        self.occupancy: list[dict] = []
        self.platform_changes = 0
        self.holds = 0
        self.forced_waits = 0
        self.congestion_events = 0
        self._congested = False
        self._events()

    @property
    def current(self) -> Train | None:
        return self.trains[self.outer[0]] if self.outer else None

    @property
    def complete(self) -> bool:
        return all(s.status == "departed" for s in self.states.values())

    def closed(self, platform: int, time: int | None = None) -> bool:
        t = self.time if time is None else time
        return any(c.platform == platform and c.start <= t < c.end for c in self.scenario.closures)

    def route_free(self, route: str, start: int, duration: int) -> bool:
        end = start + duration + self.station.route_clearance
        return all(not (m.start < end and start < m.end and self.station.conflicts(route, m.route))
                   for m in self.active_movements)

    def feasible(self, train: Train, platform: int) -> bool:
        if not 1 <= platform <= len(self.station.platforms):
            return False
        p = self.station.platforms[platform - 1]
        return (p.compatible(train) and platform not in self.occupants
                and self.release[platform] <= self.time and not self.closed(platform)
                and self.route_free(self.station.route(train, platform, "in"), self.time,
                                    self.station.entry_minutes))

    def assign(self, platform: int) -> None:
        train = self.current
        if train is None or not self.feasible(train, platform):
            raise ValueError("Unsafe or absent train-platform assignment")
        state = self.states[train.id]
        state.status = "platform"
        state.entry, state.platform = self.time, platform
        # Scheduled departure includes the physical exit movement.
        state.ready = max(train.scheduled_departure - self.station.departure_minutes,
                          self.time + self.station.entry_minutes + train.dwell
                          + self.noise[train.id].dwell_extension)
        self.occupants[platform] = train.id
        self.outer.pop(0)
        self._reserve(train, platform, "in", self.station.entry_minutes)
        self.occupancy.append({"train": train.id, "category": train.category,
                               "platform": platform, "start": self.time, "end": None})
        self.platform_changes += int(platform != train.planned_platform)
        self._fill_outer()

    def _reserve(self, train: Train, platform: int, direction: str, duration: int) -> None:
        route = self.station.route(train, platform, direction)
        if not self.route_free(route, self.time, duration):
            raise RuntimeError("Conflicting route reservation")
        movement = Movement(train.id, route, self.time,
                            self.time + duration + self.station.route_clearance)
        self.movements.append(movement)
        self.active_movements.append(movement)

    def _fill_outer(self) -> None:
        while self.upstream and len(self.outer) < self.station.outer_capacity:
            train_id = self.upstream.pop(0)
            self.states[train_id].status = "outer"
            self.outer.append(train_id)

    def _events(self) -> None:
        self.active_movements = [m for m in self.active_movements if m.end > self.time]
        for platform, train_id in list(self.occupants.items()):
            state = self.states[train_id]
            if (state.departure_start is not None
                    and self.time >= state.departure_start + self.station.departure_minutes):
                state.status, state.departure = "departed", self.time
                self.release[platform] = self.time + self.station.platforms[platform - 1].clearance
                del self.occupants[platform]
                for event in reversed(self.occupancy):
                    if event["train"] == train_id:
                        event["end"] = self.time
                        break
        arrivals = []
        for train in self.scenario.trains:
            state = self.states[train.id]
            if state.status != "future":
                continue
            arrival = train.scheduled_arrival + self.noise[train.id].incoming_delay
            if train.rake_from is not None:
                previous = self.states[train.rake_from]
                if previous.departure is None:
                    continue
                arrival = max(arrival, previous.departure + train.turnaround)
            if arrival <= self.time:
                arrivals.append((arrival, train.id))
        for arrival, train_id in sorted(arrivals):
            self.states[train_id].signal_arrival = arrival
            self.states[train_id].status = "upstream"
            self.upstream.append(train_id)
        self._fill_outer()
        # Departures take precedence over new entries at a tick boundary.
        ready = sorted((self.trains[t] for t in self.occupants.values()
                        if self.states[t].ready <= self.time
                        and self.states[t].departure_start is None),
                       key=lambda t: (self.states[t.id].ready, -t.priority, t.id))
        for train in ready:
            state = self.states[train.id]
            route = self.station.route(train, state.platform, "out")
            if self.route_free(route, self.time, self.station.departure_minutes):
                self._reserve(train, state.platform, "out", self.station.departure_minutes)
                state.departure_start = self.time
        congested = bool(self.upstream)
        self.congestion_events += int(congested and not self._congested)
        self._congested = congested

    def tick(self) -> dict[str, float]:
        """Account costs over [time,time+1), then process boundary events."""
        weighted = 0.0
        for train_id in self.outer + self.upstream:
            state = self.states[train_id]
            state.waiting += 1
            state.upstream_waiting += int(state.status == "upstream")
            weighted += self.trains[train_id].priority
        for train_id in self.occupants.values():
            state = self.states[train_id]
            if state.departure_start is None and state.ready <= self.time:
                state.departure_waiting += 1
                weighted += self.trains[train_id].priority
        costs = {"weighted_wait": weighted, "outer": float(len(self.outer)),
                 "upstream": float(len(self.upstream))}
        self.time += 1
        self._events()
        return costs

    def assert_safety(self) -> None:
        """Independent history checks used by tests and evaluation."""
        for index, event in enumerate(self.occupancy):
            train = self.trains[event["train"]]
            state = self.states[train.id]
            assert self.station.platforms[event["platform"] - 1].compatible(train)
            assert event["start"] >= state.signal_arrival
            end = event["end"] if event["end"] is not None else self.time + 1
            assert end > event["start"]
            if state.departure is not None:
                assert state.departure >= event["start"] + self.station.entry_minutes + train.dwell
                assert state.departure >= train.scheduled_departure
            for later in self.occupancy[index + 1:]:
                if later["platform"] == event["platform"]:
                    assert later["start"] >= end + self.station.platforms[event["platform"] - 1].clearance
        for i, first in enumerate(self.movements):
            for second in self.movements[i + 1:]:
                assert not (first.start < second.end and second.start < first.end
                            and self.station.conflicts(first.route, second.route))
        statuses = [s.status for s in self.states.values()]
        assert len(statuses) == len(self.scenario.trains)
        assert len(self.outer) <= self.station.outer_capacity
        assert len(set(self.outer + self.upstream + list(self.occupants.values()))) == (
            len(self.outer) + len(self.upstream) + len(self.occupants))
