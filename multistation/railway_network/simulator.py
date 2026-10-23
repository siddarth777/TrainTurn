"""Physical seven-station, two-direction network with one controlled junction.

``NetworkJunctionSimulator`` reuses the original junction adapter
(``railway_rl.sumo_simulator.SumoSimulator``) for everything inside the junction:
platform locks, clearance, route reservations, departure release and occupancy
measurement. It changes the junction's boundary and adds a second direction:

* **Two directions.** Eastbound trains (corridors A/B) enter at the west throat,
  westbound trains (C/D) at the east throat. Interlocking reserves a *throat*:
  an eastbound entry and a westbound exit both use the west throat and are
  serialized. For eastbound-only traffic this is the original rule (all entries
  share one throat, all exits the other). Junction platforms are SUMO
  bidirectional tracks, so SUMO's rail signals independently protect against
  opposing movements as well.
* **Arrival as in training.** A train arrives at the junction when it is held
  at the gate (the outer signal) at a dispatch minute. Only then can it be
  admitted, and it starts from rest at the gate exactly as the trained adapter
  inserted trains at rest at the start of ``approachX``. As in the original,
  queue membership changes only at dispatch-minute boundaries, so a train that
  stops a few seconds before the minute is not charged that partial minute.
* **Whole-line queue.** Trains held back by the junction queue anywhere on
  their approach line are in the ``upstream`` queue: on the feeder blocks, at
  an earlier block signal, or held inside a minor station after their dwell
  because the line ahead is blocked. A train counts if it is stationary, is not
  making a scheduled stop, and every train between it and the gate is queued
  too. Their stationary time is charged to the junction. Membership is
  recomputed at every dispatch minute; waiting accrues per second while stationary.
* **Departure.** Junction departure is the moment the front leaves
  ``downstreamX`` (the original model's network exit); the train then continues
  along its departure line.
* **Forecasts.** Trains already running get an ETA from their position and
  remaining scheduled calls. Trains not yet started keep the original
  pre-sampled noisy forecast of their late start. Late starts and dwell
  extensions are never read before they happen.

Minor stations are not controlled by the policy. Each uses one fixed rule per
direction: non-stopping trains prefer that direction's main track (1), stopping
trains prefer their planned loop, and the first free track in that order is taken.
If none is free the planned track is kept and SUMO's rail signals hold the train.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
import weakref
import libsumo as api
from railway_rl.simulator import Simulator
from railway_rl.sumo_simulator import SumoSimulator
from .layout import EASTBOUND, WESTBOUND, layout_from_config
from .scenario import NetworkScenario, STOP_OFFSET, HOLD_OFFSET, junction_conflict

TRACK = re.compile(r"^([A-Z]{2}\d)_([ew])(in|pf|out)(\d)$")
STILL = 0.1  # m/s; below this a train is stationary


@dataclass
class Estimate:
    """Mutable stand-in for ``Disturbance`` read by the junction's observation code."""
    incoming_delay: float
    dwell_extension: int
    estimated_delay: float
    forecast_error: float


class NetworkJunctionSimulator(SumoSimulator):
    def __init__(self, case: NetworkScenario, config: dict, cache: str | Path | None = None) -> None:
        previous = SumoSimulator._owner() if SumoSimulator._owner else None
        if previous is not None and not previous.closed_connection:
            raise RuntimeError("Close the previous SUMO environment before creating another in this process")
        self.case, self.net = case, config["network"]
        self.layout = layout_from_config(self.net)
        if case.layout_digest != self.layout.digest(case.junction.station):
            raise ValueError("Scenario was generated for a different network layout")
        self.closed_connection = True
        self.safety_violations = 0
        self.pending, self.stopped = {}, set()
        self.platform_occupancy: list[dict] = []
        self._platform_clear_distance: dict[str, float] = {}
        self._temporary = tempfile.TemporaryDirectory(prefix="railway-network-")
        network = self.layout.build(case.junction.station, Path(cache) if cache else None)
        types = Path(self._temporary.name) / "types.rou.xml"
        # Identical vehicle dynamics to the trained junction.
        types.write_text('<routes><vType id="train" vClass="rail" carFollowModel="Rail" '
                         'length="200" maxSpeed="20" accel="0.8" decel="1" emergencyDecel="2" '
                         'sigma="0" minGap="10"/></routes>')
        api.start(["sumo", "--net-file", str(network), "--route-files", str(types), "--step-length", "1",
                   "--seed", str(case.seed % (2**31)), "--no-step-log", "true", "--duration-log.disable", "true",
                   "--time-to-teleport", "-1", "--time-to-teleport.railsignal-deadlock", "-1",
                   "--collision.action", "warn", "--collision.check-junctions", "true", "--no-warnings", "true"])
        self.closed_connection = False
        SumoSimulator._owner = weakref.ref(self)
        # ----- network bookkeeping (not visible to the junction controller) -----
        self.plans = {p.train: p for p in case.plans}
        self.queue = sorted((round(p.origin_time + 60 * p.start_delay), p.train) for p in case.plans)
        self.inserted: dict[str, float] = {}
        self.sink_arrival: dict[str, float] = {}
        self.calls = {p.train: {c.station: c for c in p.approach_calls + p.departure_calls} for p in case.plans}
        self.decided: dict[str, dict[str, int]] = {p.train: {} for p in case.plans}
        self.holders: dict[tuple[str, str, int], list[str]] = {
            (s.id, d, k): [] for s in self.layout.stations for d in "ew" for k in range(1, s.tracks + 1)}
        self.station_times: dict[str, dict[str, list[float | None]]] = {p.train: {} for p in case.plans}
        self.gate_out: dict[str, float] = {}
        self.road: dict[str, str] = {}
        self.track_changes = 0
        self.admission_speeds: list[float] = []
        self.upstream_locations: set[str] = set()   # kinds of place where queued trains were held
        self._decision = self.layout.decision_edges()
        self._corridors = EASTBOUND + WESTBOUND
        self._lines = {c: self.layout.line_edges(c) for c in self._corridors}
        self._feeds = {c: set(self.layout.feed_edges(c)) for c in self._corridors}
        self._gates = {c: self.layout.gate_edge(c) for c in self._corridors}
        self._first_out = {c: self.layout.first_departure_edge(c) for c in self._corridors}
        self._platform_edge = {name: p.id for p in case.junction.station.platforms
                               for name in self.layout.platform_edges(p.id)}
        self._bidi_pairs = [(f"{kind}{p.id}", f"-{kind}{p.id}") for p in case.junction.station.platforms
                            for kind in ("entry", "platform", "exit")]
        self._length: dict[str, float] = {}
        self._vehicles, self._speed = (), {}
        self._ready_for_estimates = False
        try:
            Simulator.__init__(self, case.junction)
            self.noise = {t.id: Estimate(0.0, d.dwell_extension, d.estimated_delay, d.forecast_error)
                          for t, d in zip(case.junction.trains, case.junction.disturbances)}
            self._ready_for_estimates = True
            self._events()
        except Exception:
            self.close()
            raise

    # ----- helpers ----------------------------------------------------------------------
    @property
    def seconds(self) -> float:
        return api.simulation.getTime()

    def length(self, edge: str) -> float:
        if edge not in self._length:
            self._length[edge] = api.lane.getLength(f"{edge}_0")
        return self._length[edge]

    def _remaining_to(self, vid: str, target: str) -> float:
        route, index = api.vehicle.getRoute(vid), api.vehicle.getRouteIndex(vid)
        if target not in route[index:]:
            return 0.0
        end = route.index(target, index)
        position = 0.0 if api.vehicle.getRoadID(vid).startswith(":") else api.vehicle.getLanePosition(vid)
        return (self.length(route[index]) - position) + sum(self.length(e) for e in route[index + 1:end + 1])

    # ----- interlocking -----------------------------------------------------------------------
    def route_free(self, route: str, start: int, duration: int) -> bool:
        end = start + duration + self.station.route_clearance
        return all(not (m.start < end and start < m.end and junction_conflict(self.station, route, m.route))
                   for m in self.active_movements)

    # ----- junction boundary ----------------------------------------------------------------
    def _fill_outer(self) -> None:
        """Outer-queue membership is physical (a train held at a gate), not a list transfer."""

    def _events(self) -> None:
        self.active_movements = [m for m in self.active_movements if m.end > self.time]
        if self._ready_for_estimates:
            self._update_queues()
            self._update_estimates()
        congested = bool(self.upstream)
        self.congestion_events += int(congested and not self._congested)
        self._congested = congested
        self._release_departures()

    def _update_queues(self) -> None:
        for corridor in self._corridors:
            gate, line = self._gates[corridor], self._lines[corridor]
            waiting = [(self._remaining_to(v, gate), v) for v in self._vehicles
                       if self.road.get(v) in line and self.states[v].status in ("future", "upstream", "outer")]
            ahead_queued = True
            for _, vid in sorted(waiting):
                state = self.states[vid]
                if state.status == "outer":
                    continue
                at_stop = api.vehicle.isStopped(vid)
                held = self.road[vid] == gate and at_stop
                blocked = self._speed.get(vid, 1.0) < STILL and not at_stop
                if ahead_queued and (held or blocked):
                    if state.signal_arrival is None:
                        state.signal_arrival = self.time
                        self.noise[vid].incoming_delay = max(0.0, self.time - self.trains[vid].scheduled_arrival)
                    if held:
                        if vid in self.upstream:
                            self.upstream.remove(vid)
                        state.status = "outer"
                        self.outer.append(vid)
                    else:
                        if vid not in self.upstream:
                            state.status = "upstream"
                            self.upstream.append(vid)
                        road = self.road[vid]
                        self.upstream_locations.add("station" if TRACK.match(road) else
                                                    "feeder" if road in self._feeds[corridor] else "line")
                else:
                    if vid in self.upstream:
                        self.upstream.remove(vid)
                        state.status = "future"
                    ahead_queued = False

    def _update_estimates(self) -> None:
        v, a, d = self.net["line_speed"], self.net["acceleration"], self.net["deceleration"]
        live = set(self._vehicles)
        for train in self.scenario.trains:
            if self.states[train.id].status != "future" or train.id not in live:
                continue  # not-yet-started trains keep the pre-sampled noisy start-delay forecast
            gate = self._gates[train.arrival_corridor]
            distance = self._remaining_to(train.id, gate) - HOLD_OFFSET
            pending = [c for c in self.plans[train.id].approach_calls
                       if c.stops and self.station_times[train.id].get(c.station, [None, None])[1] is None]
            # Remaining scheduled calls use nominal dwell only; extensions are not known in advance.
            eta = (self.seconds + max(0.0, distance) / v + v / (2 * d)
                   + sum(60 * c.dwell + v / (2 * a) + v / (2 * d) for c in pending))
            estimate = self.noise[train.id]
            estimate.estimated_delay = max(0.0, eta / 60 - train.scheduled_arrival)
            estimate.forecast_error = 0.0

    def assign(self, platform: int) -> None:
        train = self.current
        if train is None or not self.feasible(train, platform):
            raise ValueError("Unsafe or absent train-platform assignment")
        vid, plan = train.id, self.plans[train.id]
        gate = self._gates[train.arrival_corridor]
        if self.road.get(vid) != gate or not api.vehicle.isStopped(vid):
            raise RuntimeError(f"{vid} is not held at the junction gate")
        self.admission_speeds.append(api.vehicle.getSpeed(vid))
        tracks = {c.station: c.track for c in plan.departure_calls}
        route = ([gate] + self.layout.junction_path(train.arrival_corridor, platform, train.departure_corridor)
                 + self.layout.departure_path(train.departure_corridor, tracks))
        api.vehicle.setRoute(vid, route)
        stop_pos = self.station.platforms[platform - 1].length * 25 + 30  # as in the trained adapter
        api.vehicle.setStop(vid, route[3], pos=float(stop_pos), duration=1e9)  # platform{p} or -platform{p}
        api.vehicle.resume(vid)
        state = self.states[vid]
        state.status, state.entry, state.platform = "platform", self.time, platform
        state.ready = float("inf")
        self.occupants[platform] = vid
        self.outer.remove(vid)
        self._begin_movement(train, platform, "in")
        self.occupancy.append({"train": vid, "category": train.category, "platform": platform,
                               "start": self.time, "end": None})
        self.platform_changes += int(platform != train.planned_platform)

    def _junction_departure(self, vid: str) -> None:
        state = self.states[vid]
        if state.departure_start is None:
            raise RuntimeError(f"{vid} left the junction without a granted departure")
        state.departure, state.status = self.time, "departed"
        self.release[state.platform] = self.time + self.station.platforms[state.platform - 1].clearance
        del self.occupants[state.platform]
        self.stopped.discard(vid)
        self._end_movement(vid)
        for event in reversed(self.occupancy):
            if event["train"] == vid:
                event["end"] = self.time
                break
        self.gate_out[vid] = self.seconds

    # ----- minor stations (fixed local rule, identical for every junction policy) ---------------
    def _choose_track(self, vid: str, sid: str, direction: str) -> None:
        call = self.calls[vid][sid]
        tracks = list(range(1, self.layout.station(sid).tracks + 1))
        loops = ([call.track] + [k for k in tracks if k not in (1, call.track)]) if call.track != 1 else tracks[1:]
        order = (loops + [1]) if call.stops else ([1] + loops)
        free = [k for k in order if not self.holders[(sid, direction, k)]]
        track = free[0] if free else call.track
        route, index = list(api.vehicle.getRoute(vid)), api.vehicle.getRouteIndex(vid)
        prefix = f"{sid}_{direction}in"
        at = next(i for i in range(index, len(route)) if route[i].startswith(prefix))
        if int(route[at][len(prefix):]) != track:
            route[at:at + 3] = self.layout.track_edges(sid, direction, track)
            api.vehicle.setRoute(vid, route[index:])
        self.track_changes += int(track != call.track)
        self.holders[(sid, direction, track)].append(vid)
        self.decided[vid][sid] = track
        if call.stops:
            pos = self.net["station_track_length"] - STOP_OFFSET
            api.vehicle.setStop(vid, f"{sid}_{direction}pf{track}", pos=float(pos),
                                duration=float(60 * (call.dwell + call.extension)), until=float(call.departure))

    # ----- per-second physics -----------------------------------------------------------------
    def _insert_due(self) -> None:
        now = self.seconds
        while self.queue and self.queue[0][0] <= now:
            _, vid = self.queue.pop(0)
            train, plan = self.trains[vid], self.plans[vid]
            tracks = {c.station: c.track for c in plan.approach_calls}
            api.route.add(f"route_{vid}", self.layout.approach_path(train.arrival_corridor, tracks))
            api.vehicle.add(vid, f"route_{vid}", typeID="train", depart="now", departPos="base", departSpeed="max")
            api.vehicle.setLength(vid, train.length * 25.0)
            api.vehicle.setVehicleClass(vid, "rail_electric" if train.electric else "rail")
            gate = self._gates[train.arrival_corridor]
            api.vehicle.setStop(vid, gate, pos=self.length(gate) - HOLD_OFFSET, duration=1e9)
            self.inserted[vid] = now

    def _second(self) -> None:
        if api.simulation.getCollidingVehiclesNumber() or api.simulation.getStartingTeleportNumber():
            self.safety_violations += 1
            raise RuntimeError("SUMO reported a collision or teleport; episode is invalid")
        now = self.seconds
        for vid in api.simulation.getArrivedIDList():
            if self.states[vid].status != "departed":
                raise RuntimeError(f"{vid} reached the end of the line without passing the junction")
            self.sink_arrival[vid] = now
            self.road.pop(vid, None)
        self._vehicles = api.vehicle.getIDList()
        self._speed = {}
        occupied: set[str] = set()
        for vid in self._vehicles:
            self._speed[vid] = api.vehicle.getSpeed(vid)
            road = api.vehicle.getRoadID(vid)
            if road.startswith(":"):
                continue  # internal junction lane; transitions are tracked between normal edges
            occupied.add(road)
            previous = self.road.get(vid)
            if road != previous:
                self._entered(vid, road, previous)
                self.road[vid] = road
            state = self.states[vid]
            if road in self._decision:
                sid, direction = self._decision[road]
                if sid not in self.decided[vid]:
                    self._choose_track(vid, sid, direction)
            # Junction platform occupancy and stop detection, as in the trained adapter.
            if state.status == "platform" and self._platform_edge.get(road) == state.platform:
                if vid not in self._platform_clear_distance:
                    distance = api.vehicle.getDistance(vid) - api.vehicle.getLanePosition(vid)
                    self._platform_clear_distance[vid] = (distance + self.station.platforms[state.platform - 1].length * 25
                                                          + 40 + self.trains[vid].length * 25)
                    self.platform_occupancy.append({"train": vid, "category": self.trains[vid].category,
                                                    "platform": state.platform, "start": self.time, "end": None})
                if vid not in self.stopped and api.vehicle.isStopped(vid):
                    self.stopped.add(vid)
                    self._end_movement(vid)
                    train = self.trains[vid]
                    state.ready = max(train.scheduled_departure,
                                      self.time + train.dwell + self.noise[vid].dwell_extension)
            clear = self._platform_clear_distance.get(vid)
            if clear is not None and api.vehicle.getDistance(vid) >= clear:
                for interval in reversed(self.platform_occupancy):
                    if interval["train"] == vid and interval["end"] is None:
                        interval["end"] = self.time
                        break
                del self._platform_clear_distance[vid]
        # Independent check of the shared junction tracks (SUMO's bidi signalling also protects them).
        for forward, backward in self._bidi_pairs:
            if forward in occupied and backward in occupied:
                self.safety_violations += 1
                raise RuntimeError(f"Opposing trains on shared track {forward}")

    def _entered(self, vid: str, road: str, previous: str | None) -> None:
        now = self.seconds
        left = TRACK.match(previous or "")
        if left and left.group(3) == "pf" and left.group(1) in self.calls[vid]:
            self.station_times[vid].setdefault(left.group(1), [None, None])[1] = now
        if left and left.group(3) == "out":
            key = (left.group(1), left.group(2), int(left.group(4)))
            if vid in self.holders.get(key, []):
                self.holders[key].remove(vid)
        entered = TRACK.match(road)
        if entered and entered.group(3) == "pf" and entered.group(1) in self.calls[vid]:
            self.station_times[vid].setdefault(entered.group(1), [None, None])[0] = now
        departure = self.trains[vid].departure_corridor
        if previous == f"downstream{departure}" and road == self._first_out[departure]:
            self._junction_departure(vid)

    def tick(self) -> dict[str, float]:
        target = round(self.time) + 1
        costs = {"weighted_wait": 0.0, "outer": 0.0, "upstream": 0.0}
        quantum = 1 / 60
        for _ in range(60):
            self.active_movements = [m for m in self.active_movements if m.end > self.time]
            self._release_departures()
            for vid in self.outer + self.upstream:
                if self._speed.get(vid, 0.0) >= STILL:
                    continue  # rolling up the queue; not stationary this second
                state = self.states[vid]
                state.waiting += quantum
                state.upstream_waiting += quantum * (state.status == "upstream")
                costs["weighted_wait"] += quantum * self.trains[vid].priority
            for vid in self.occupants.values():
                state = self.states[vid]
                if state.ready <= self.time and state.departure_start is None:
                    state.departure_waiting += quantum
                    costs["weighted_wait"] += quantum * self.trains[vid].priority
            costs["outer"] += quantum * len(self.outer)
            costs["upstream"] += quantum * sum(self._speed.get(v, 0.0) < STILL for v in self.upstream)
            self._insert_due()
            api.simulationStep(round((self.time + quantum) * 60))
            self.time = api.simulation.getTime() / 60
            self._second()
        self.time = target
        self._events()
        return costs

    def drain(self, limit_minutes: float) -> list[str]:
        """After the junction finishes, run on until every train reaches the end of its line."""
        end = self.seconds + 60 * limit_minutes
        while self.seconds < end and (self.queue or api.vehicle.getIDList()
                                      or api.simulation.getMinExpectedNumber() > 0):
            self._insert_due()
            api.simulationStep(self.seconds + 1)
            self.time = self.seconds / 60
            self._second()
            if self.occupants:
                self.active_movements = [m for m in self.active_movements if m.end > self.time]
                self._release_departures()
        return [t.id for t in self.scenario.trains if t.id not in self.sink_arrival]

    # ----- safety ---------------------------------------------------------------------------------
    def assert_safety(self) -> None:
        assert self.safety_violations == 0
        assert len(self.outer) <= len(self._corridors), "one hold point per approach corridor"
        assert all(speed < STILL for speed in self.admission_speeds), "admitted trains start from rest"
        seen = self.outer + self.upstream + list(self.occupants.values())
        assert len(seen) == len(set(seen))
        assert len(seen) + sum(s.status in ("future", "departed") for s in self.states.values()) == len(self.states)
        for i, event in enumerate(self.occupancy):
            train, state = self.trains[event["train"]], self.states[event["train"]]
            assert self.station.platforms[event["platform"] - 1].compatible(train)
            assert event["start"] >= state.signal_arrival
            assert not any(c.platform == event["platform"] and c.start <= event["start"] < c.end
                           for c in self.scenario.closures)
            if state.departure is not None:
                assert state.departure >= state.departure_start >= state.entry
            for later in self.occupancy[i + 1:]:
                if later["platform"] == event["platform"]:
                    assert event["end"] is not None
                    assert later["start"] >= event["end"] + self.station.platforms[event["platform"] - 1].clearance
        movements = self.movements + list(self.pending.values())
        for i, first in enumerate(movements):
            for second in movements[i + 1:]:
                assert not (first.start < second.end and second.start < first.end
                            and junction_conflict(self.station, first.route, second.route))
        if not self.closed_connection:
            assert set(self.occupants.values()) <= set(api.vehicle.getIDList())
