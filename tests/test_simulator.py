from dataclasses import replace
import pytest
from railway_rl.station import Platform, Train, Station, Scenario, Disturbance, Closure
from railway_rl.simulator import Simulator


def scenario(trains=None, closures=(), station=None):
    station = station or Station(tuple(Platform(i) for i in range(1, 4)))
    trains = trains or (Train("T1", "express", 0, 7, 1),)
    return Scenario(station, trains, tuple(Disturbance() for _ in trains), closures)


def test_departure_and_clearance():
    sim = Simulator(scenario())
    sim.assign(1)
    with pytest.raises(ValueError):
        sim.assign(1)
    for _ in range(7):
        sim.tick()
    assert sim.states["T1"].departure == 7
    assert sim.release[1] == 9
    assert sim.complete
    sim.assert_safety()


def test_queue_conservation_and_waiting():
    trains = tuple(Train(f"T{i}", "express", 0, 7, 1) for i in range(5))
    sim = Simulator(scenario(trains))
    assert len(sim.outer) == 2 and len(sim.upstream) == 3
    costs = sim.tick()
    assert costs["weighted_wait"] == 15
    assert sum(s.waiting for s in sim.states.values()) == 5
    assert sum(s.upstream_waiting for s in sim.states.values()) == 3
    assert not sim.complete
    sim.assert_safety()


def test_routes_block_movements_not_occupancy():
    trains = (Train("T1", "express", 0, 7, 1), Train("T2", "express", 0, 9, 2))
    sim = Simulator(scenario(trains))
    sim.assign(1)
    assert not sim.feasible(sim.current, 2)
    sim.tick()
    assert not sim.feasible(sim.current, 2)
    sim.tick()
    assert sim.feasible(sim.current, 2)
    sim.assign(2)
    assert len(sim.occupants) == 2
    for _ in range(12):
        sim.tick()
    sim.assert_safety()


def test_compatibility_closures_and_clearance():
    platforms = (Platform(1, length=10), Platform(2, electrified=False), Platform(3, kind="terminal"))
    train = Train("T", "passenger", 0, 7, 1, length=8, electric=False)
    sim = Simulator(scenario((train,), (Closure(1, 0, 3),), Station(platforms)))
    assert not sim.feasible(train, 1)
    assert sim.feasible(train, 2)
    assert not sim.feasible(train, 3)
    assert not platforms[0].compatible(replace(train, length=20))
    sim.tick(); sim.tick(); sim.tick()
    assert sim.feasible(train, 1)


def test_determinism_and_invalid_action_no_mutation():
    a, b = Simulator(scenario()), Simulator(scenario())
    with pytest.raises(ValueError):
        a.assign(9)
    assert a.time == 0 and a.states == b.states
    for sim in (a, b):
        sim.tick(); sim.assign(1)
        for _ in range(12):
            sim.tick()
    assert a.states == b.states and a.movements == b.movements


def test_rake_constraint():
    trains = (Train("A", "express", 0, 7, 1),
              Train("B", "express", 10, 17, 2, rake_from="A", turnaround=5))
    sim = Simulator(scenario(trains))
    sim.assign(1)
    for _ in range(11):
        sim.tick()
    assert sim.states["B"].status == "future"
    sim.tick()
    assert sim.states["B"].signal_arrival == 12
