"""Immutable infrastructure and timetable contracts; times are integer minutes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    id: int
    length: int = 24
    electrified: bool = True
    kind: str = "through"
    approaches: tuple[str, ...] = ("A", "B")
    departures: tuple[str, ...] = ("A", "B")
    clearance: int = 2

    def __post_init__(self) -> None:
        if self.id < 1 or self.length < 1 or self.clearance < 0:
            raise ValueError("Invalid platform identifier, length, or clearance")
        if self.kind not in ("through", "terminal") or not self.approaches or not self.departures:
            raise ValueError("Invalid platform type or corridors")

    def compatible(self, train: "Train") -> bool:
        return (train.length <= self.length
                and (not train.electric or self.electrified)
                and (train.kind != "through" or self.kind == "through")
                and train.arrival_corridor in self.approaches
                and train.departure_corridor in self.departures)


@dataclass(frozen=True)
class Train:
    id: str
    category: str
    scheduled_arrival: int
    scheduled_departure: int
    planned_platform: int
    length: int = 16
    electric: bool = True
    kind: str = "through"
    arrival_corridor: str = "A"
    departure_corridor: str = "B"
    dwell: int = 5
    priority: float = 3.0
    rake_from: str | None = None
    turnaround: int = 10

    def __post_init__(self) -> None:
        if not self.id or not self.category or self.length < 1 or self.dwell < 1:
            raise ValueError("Invalid train identity, length, or dwell")
        if self.scheduled_arrival < 0 or self.scheduled_departure <= self.scheduled_arrival:
            raise ValueError("Departure must follow arrival")
        if self.planned_platform < 1 or self.priority <= 0 or self.turnaround < 0:
            raise ValueError("Invalid train plan, priority, or turnaround")
        if self.kind not in ("through", "terminal"):
            raise ValueError("Invalid train type")


@dataclass(frozen=True)
class Station:
    platforms: tuple[Platform, ...]
    outer_capacity: int = 2
    entry_minutes: int = 1
    departure_minutes: int = 1
    route_clearance: int = 1
    # Each pair lists routes that share additional interlocking infrastructure.
    extra_conflicts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not 3 <= len(self.platforms) <= 8:
            raise ValueError("Use 3–8 platforms")
        if [p.id for p in self.platforms] != list(range(1, len(self.platforms) + 1)):
            raise ValueError("Platform identifiers must be contiguous and one-based")
        if not 1 <= self.outer_capacity <= 3:
            raise ValueError("Outer queue capacity must be 1–3")
        if min(self.entry_minutes, self.departure_minutes) < 1 or self.route_clearance < 0:
            raise ValueError("Invalid movement duration")

    def route(self, train: Train, platform: int, direction: str) -> str:
        corridor = train.arrival_corridor if direction == "in" else train.departure_corridor
        return f"{direction}:{corridor}:{platform}"

    def conflicts(self, first: str, second: str) -> bool:
        _, corridor_a, platform_a = first.split(":")
        _, corridor_b, platform_b = second.split(":")
        return (corridor_a == corridor_b or platform_a == platform_b
                or (first, second) in self.extra_conflicts
                or (second, first) in self.extra_conflicts)

    def conflict_matrix(self) -> dict[str, dict[str, bool]]:
        routes = [f"{d}:{c}:{p.id}" for p in self.platforms
                  for d, corridors in (("in", p.approaches), ("out", p.departures))
                  for c in corridors]
        return {a: {b: self.conflicts(a, b) for b in routes} for a in routes}


@dataclass(frozen=True)
class Disturbance:
    incoming_delay: int = 0
    dwell_extension: int = 0
    estimated_delay: float = 0.0
    forecast_error: float = 0.0

    def __post_init__(self) -> None:
        if min(self.incoming_delay, self.dwell_extension) < 0:
            raise ValueError("Physical disturbances cannot be negative")


@dataclass(frozen=True)
class Closure:
    platform: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.platform < 1 or self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid closure interval")


@dataclass(frozen=True)
class Scenario:
    station: Station
    trains: tuple[Train, ...]
    disturbances: tuple[Disturbance, ...]
    closures: tuple[Closure, ...] = ()
    seed: int = 0
    provenance: str = "synthetic; not calibrated to Indian Railways operations"

    def __post_init__(self) -> None:
        if not self.trains or len(self.trains) != len(self.disturbances):
            raise ValueError("One disturbance record is required per train")
        ids = [t.id for t in self.trains]
        if len(set(ids)) != len(ids):
            raise ValueError("Train identifiers must be unique")
        by_id = {t.id: t for t in self.trains}
        for t in self.trains:
            if t.planned_platform > len(self.station.platforms):
                raise ValueError("Planned platform does not exist")
            if not any(p.compatible(t) for p in self.station.platforms):
                raise ValueError(f"No structurally compatible platform for {t.id}")
            if t.rake_from is not None:
                if t.rake_from not in by_id or by_id[t.rake_from].scheduled_departure >= t.scheduled_arrival:
                    raise ValueError("Rake links must refer to an earlier service (no cycles)")
        if any(c.platform > len(self.station.platforms) for c in self.closures):
            raise ValueError("Closure refers to absent platform")
