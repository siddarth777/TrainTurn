"""SUMO adapter: explicit boundary queues and conservative interlocking reservations.

Platform occupancy/release and completed train movements come from SUMO, not the
reference simulator. One minute of dispatch time contains sixty physical seconds.
"""
from pathlib import Path
import tempfile
import weakref
import libsumo as api
from .station import Scenario, Train
from .simulator import Simulator, Movement
from .sumo_network import network_file


class SumoSimulator(Simulator):
    _owner = None  # libsumo permits one live instance in a process.

    def __init__(self, scenario: Scenario, cache: str | Path | None = None) -> None:
        previous = self._owner() if self._owner else None
        if previous is not None and not previous.closed_connection:
            raise RuntimeError("Close the previous SUMO environment before creating another in this process")
        if any(c not in ("A", "B") for p in scenario.station.platforms for c in p.approaches + p.departures):
            raise ValueError("The supplied SUMO junction supports corridors A and B")
        self.closed_connection = True
        self.safety_violations = 0
        self.pending: dict[str, Movement] = {}
        self.stopped: set[str] = set()
        self.platform_occupancy: list[dict] = []
        self._platform_clear_distance: dict[str, float] = {}
        self._temporary = tempfile.TemporaryDirectory(prefix="railway-sumo-")
        cache_path = Path(cache) if cache else Path(tempfile.gettempdir()) / "railway-sumo-networks-v1"
        net = network_file(scenario.station, cache_path)
        types = Path(self._temporary.name) / "types.rou.xml"
        types.write_text('<routes><vType id="train" vClass="rail" carFollowModel="Rail" '
                         'length="200" maxSpeed="20" accel="0.8" decel="1" emergencyDecel="2" '
                         'sigma="0" minGap="10"/></routes>')
        api.start(["sumo", "--net-file", str(net), "--route-files", str(types), "--step-length", "1",
                   "--seed", str(scenario.seed % (2**31)), "--no-step-log", "true", "--duration-log.disable", "true",
                   "--time-to-teleport", "-1", "--time-to-teleport.railsignal-deadlock", "-1",
                   "--collision.action", "warn", "--collision.check-junctions", "true", "--no-warnings", "true"])
        self.closed_connection = False
        type(self)._owner = weakref.ref(self)
        try:
            super().__init__(scenario)
        except Exception:
            self.close()
            raise

    def _events(self) -> None:
        # Used at dispatch boundaries only; hidden exogenous arrivals become observable here.
        self.active_movements = [m for m in self.active_movements if m.end > self.time]
        arrivals = []
        for train in self.scenario.trains:
            if self.states[train.id].status != "future":
                continue
            arrival = train.scheduled_arrival + self.noise[train.id].incoming_delay
            if train.rake_from:
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
        congested = bool(self.upstream)
        self.congestion_events += int(congested and not self._congested)
        self._congested = congested
        self._release_departures()

    def route_free(self, route: str, start: int, duration: int) -> bool:
        end = start + duration + self.station.route_clearance
        # All entries share one throat; all exits share another. Additional corridor
        # and platform exclusions retain the documented conservative dispatch rules.
        direction = route.split(":")[0]
        return all(not (m.start < end and start < m.end
                        and (direction == m.route.split(":")[0] or self.station.conflicts(route, m.route)))
                   for m in self.active_movements)

    def _begin_movement(self, train: Train, platform: int, direction: str) -> None:
        route = self.station.route(train, platform, direction)
        if not self.route_free(route, self.time, 1):
            raise RuntimeError("Interlocking conflict")
        movement = Movement(train.id, route, self.time, float("inf"))
        self.pending[train.id] = movement
        self.active_movements.append(movement)

    def _end_movement(self, train_id: str) -> None:
        movement = self.pending.pop(train_id)
        self.active_movements.remove(movement)
        closed = Movement(train_id, movement.route, movement.start, self.time + self.station.route_clearance)
        self.movements.append(closed)
        self.active_movements.append(closed)

    def assign(self, platform: int) -> None:
        train = self.current
        if train is None or not self.feasible(train, platform):
            raise ValueError("Unsafe or absent train-platform assignment")
        route = [f"approach{train.arrival_corridor}", f"entry{platform}", f"platform{platform}",
                 f"exit{platform}", f"downstream{train.departure_corridor}"]
        api.route.add(train.id, route)
        api.vehicle.add(train.id, train.id, typeID="train", depart="now", departPos="free", departSpeed="0")
        api.vehicle.setLength(train.id, train.length * 25.0)
        api.vehicle.setVehicleClass(train.id, "rail_electric" if train.electric else "rail")
        stop_pos = self.station.platforms[platform - 1].length * 25 + 30
        # The train remains stopped until the adapter grants a safe departure route.
        api.vehicle.setStop(train.id, f"platform{platform}", pos=float(stop_pos), duration=1e9)
        state = self.states[train.id]
        state.status, state.entry, state.platform = "platform", self.time, platform
        state.ready = float("inf")
        self.occupants[platform] = train.id
        self.outer.pop(0)
        self._begin_movement(train, platform, "in")
        self.occupancy.append({"train": train.id, "category": train.category, "platform": platform,
                               "start": self.time, "end": None})
        self.platform_changes += int(platform != train.planned_platform)
        self._fill_outer()

    def _release_departures(self) -> None:
        ready = sorted((self.trains[t] for t in self.stopped
                        if self.states[t].ready <= self.time and self.states[t].departure_start is None),
                       key=lambda t: (self.states[t.id].ready, -t.priority, t.id))
        for train in ready:
            state = self.states[train.id]
            route = self.station.route(train, state.platform, "out")
            if self.route_free(route, self.time, self.station.departure_minutes):
                api.vehicle.resume(train.id)
                self._begin_movement(train, state.platform, "out")
                state.departure_start = self.time
                state.station_departure = self.time

    def _physical_events(self) -> None:
        if api.simulation.getCollidingVehiclesNumber() or api.simulation.getStartingTeleportNumber():
            self.safety_violations += 1
            raise RuntimeError("SUMO reported a collision or teleport; episode is invalid")
        for train_id in api.simulation.getArrivedIDList():
            state = self.states[train_id]
            if state.departure_start is None:
                raise RuntimeError("Train left the network without a recorded station departure")
            state.departure, state.status = self.time, "departed"
            self.release[state.platform] = self.time + self.station.platforms[state.platform - 1].clearance
            del self.occupants[state.platform]
            self.stopped.discard(train_id)
            self._end_movement(train_id)
            for event in reversed(self.occupancy):
                if event["train"] == train_id:
                    event["end"] = self.time
                    break
        for train_id in api.vehicle.getIDList():
            road = api.vehicle.getRoadID(train_id)
            state = self.states[train_id]
            if road == f"platform{state.platform}" and train_id not in self._platform_clear_distance:
                distance = api.vehicle.getDistance(train_id) - api.vehicle.getLanePosition(train_id)
                self._platform_clear_distance[train_id] = distance + self.station.platforms[state.platform - 1].length * 25 + 40 + self.trains[train_id].length * 25
                self.platform_occupancy.append({"train": train_id, "category": self.trains[train_id].category,
                                               "platform": state.platform, "start": self.time, "end": None})
            if train_id in self._platform_clear_distance and api.vehicle.getDistance(train_id) >= self._platform_clear_distance[train_id]:
                for interval in reversed(self.platform_occupancy):
                    if interval["train"] == train_id and interval["end"] is None:
                        interval["end"] = self.time
                        break
            if train_id not in self.stopped and api.vehicle.isStopped(train_id):
                state, train = self.states[train_id], self.trains[train_id]
                self.stopped.add(train_id)
                self._end_movement(train_id)
                state.ready = max(train.scheduled_departure,
                                  self.time + train.dwell + self.noise[train_id].dwell_extension)

    def tick(self) -> dict[str, float]:
        target = round(self.time) + 1
        costs = {"weighted_wait": 0.0, "outer": 0.0, "upstream": 0.0}
        quantum = 1 / 60 if self.occupants else 1.0
        steps = 60 if self.occupants else 1
        for _ in range(steps):
            self.active_movements = [m for m in self.active_movements if m.end > self.time]
            self._release_departures()
            for train_id in self.outer + self.upstream:
                state = self.states[train_id]
                state.waiting += quantum
                state.upstream_waiting += quantum * (state.status == "upstream")
                costs["weighted_wait"] += quantum * self.trains[train_id].priority
            for train_id in self.occupants.values():
                state = self.states[train_id]
                if state.ready <= self.time and state.departure_start is None:
                    state.departure_waiting += quantum
                    costs["weighted_wait"] += quantum * self.trains[train_id].priority
            costs["outer"] += quantum * len(self.outer)
            costs["upstream"] += quantum * len(self.upstream)
            api.simulationStep(round((self.time + quantum) * 60))
            self.time = api.simulation.getTime() / 60
            self._physical_events()
        self.time = target
        self._events()
        return costs

    def assert_safety(self) -> None:
        # SUMO travel times are physical seconds, not the reference simulator's fixed durations.
        assert self.safety_violations == 0
        assert len(self.outer) <= self.station.outer_capacity
        seen = self.outer + self.upstream + list(self.occupants.values())
        assert len(seen) == len(set(seen))
        assert len(seen) + sum(s.status in ("future", "departed") for s in self.states.values()) == len(self.states)
        for i, event in enumerate(self.occupancy):
            train, state = self.trains[event["train"]], self.states[event["train"]]
            assert self.station.platforms[event["platform"] - 1].compatible(train)
            assert event["start"] >= state.signal_arrival
            assert not any(c.platform == event["platform"] and c.start <= event["start"] < c.end for c in self.scenario.closures)
            if state.departure is not None:
                assert state.departure >= state.departure_start >= state.entry
            for later in self.occupancy[i + 1:]:
                if later["platform"] == event["platform"]:
                    assert event["end"] is not None
                    assert later["start"] >= event["end"] + self.station.platforms[event["platform"] - 1].clearance
        movements = self.movements + list(self.pending.values())
        for i, first in enumerate(movements):
            for second in movements[i + 1:]:
                conflicting = first.route.split(":")[0] == second.route.split(":")[0] or self.station.conflicts(first.route, second.route)
                assert not (first.start < second.end and second.start < first.end and conflicting)
        live = set(api.vehicle.getIDList()) | set(api.simulation.getPendingVehicles()) if not self.closed_connection else set()
        if not self.closed_connection:
            assert live == set(self.occupants.values())

    def close(self) -> None:
        if not self.closed_connection:
            api.close()
            self.closed_connection = True
        self._temporary.cleanup()
