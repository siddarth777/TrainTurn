from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from railway_rl.baselines import POLICIES
from railway_rl.masking import action_mask
from railway_network.config import load_network_config
from railway_network.environment import NetworkTrainTurnEnv
from railway_network.evaluation import run_network_episode
from railway_network.layout import layout_from_config, trained_junction_speeds
from railway_network.scenario import (generate_network_scenario, save_network_scenario,
                                      load_network_scenario, junction_conflict)

SEED = 200000


def small(trains: int) -> dict:
    c = deepcopy(load_network_config())
    c["junction"]["scenario"]["trains"] = trains  # same infrastructure and rules, shorter day
    return c


@pytest.fixture(scope="module")
def config():
    return small(14)


def fcfs(obs, mask):
    return POLICIES["fcfs"](obs, mask, 8)


def test_layout_has_seven_stations_and_two_directions(config):
    layout = layout_from_config(config["network"])
    assert len(layout.stations) + 1 == 7  # six minor stations plus the controlled junction
    east = layout.approach_path("A", {"AW2": 1, "AW1": 2})
    assert east.index("AW2_epf1") < east.index("AW1_epf2") < east.index(layout.gate_edge("A"))
    west = layout.departure_path("C", {"AW2": 1, "AW1": 1})
    assert west.index("AW1_wpf1") < west.index("AW2_wpf1")  # westbound passes AW1 before AW2
    assert layout.junction_path("A", 3, "B") == ["approachA", "entry3", "platform3", "exit3", "downstreamB"]
    assert layout.junction_path("D", 3, "C") == ["approachD", "-exit3", "-platform3", "-entry3", "downstreamC"]


def test_throat_rule_serializes_shared_throats_only(config):
    station = generate_network_scenario(config, SEED).junction.station
    assert junction_conflict(station, "in:A:1", "out:C:2")       # both use the west throat
    assert junction_conflict(station, "in:C:1", "out:B:2")       # both use the east throat
    assert not junction_conflict(station, "in:A:1", "in:C:2")    # opposite throats, other platform
    assert junction_conflict(station, "in:A:1", "in:B:2")        # original eastbound rule kept


def test_junction_speeds_match_trained_network(config):
    case = generate_network_scenario(config, SEED)
    layout = layout_from_config(config["network"])
    root = ET.parse(layout.build(case.junction.station)).getroot()
    lanes = {lane.get("id"): float(lane.get("speed")) for e in root.iter("edge") for lane in e.iter("lane")}
    built = {(c.get("from"), c.get("to")): lanes[c.get("via")] for c in root.iter("connection") if c.get("via")}
    wanted = trained_junction_speeds(case.junction.station)
    assert wanted and all(built[k] == pytest.approx(v, abs=0.01) for k, v in wanted.items())
    assert sum(e.get("bidi") is not None and not e.get("id").startswith(":") for e in root.iter("edge")) == 48


def test_network_timetable_is_ordered_two_directional_and_reproducible(config, tmp_path):
    case = generate_network_scenario(config, SEED)
    assert case.plans == generate_network_scenario(config, SEED).plans
    assert {p.direction for p in case.plans} == {"e", "w"}
    for plan, train in zip(case.plans, case.junction.trains):
        assert (plan.direction == "e") == (train.arrival_corridor in "AB") == (train.departure_corridor in "AB")
        times = [plan.origin_time] + [t for c in plan.approach_calls for t in (c.arrival, c.departure)]
        times += [plan.gate_arrival, plan.junction_departure, plan.gate_out]
        times += [t for c in plan.departure_calls for t in (c.arrival, c.departure)] + [plan.sink_arrival]
        assert plan.origin_time >= 0
        assert all(b >= a for a, b in zip(times, times[1:])), plan.train
        for c in plan.approach_calls + plan.departure_calls:
            assert c.departure - c.arrival == pytest.approx(60 * c.dwell)
            assert c.stops or (c.dwell == 0 and c.extension == 0 and c.track == 1)
        if plan.terminal:
            assert not any(c.stops for c in plan.departure_calls)
    save_network_scenario(case, tmp_path)
    loaded = load_network_scenario(tmp_path, SEED)
    assert loaded.plans == case.plans and loaded.junction == case.junction


def test_observation_and_mask_contract_match_trained_junction(config):
    env = NetworkTrainTurnEnv(config, generate_network_scenario(config, SEED))
    try:
        obs, info = env.reset(seed=SEED)
        assert obs.shape == (229,) and obs.dtype == np.float32 and np.all(np.abs(obs) <= 1)
        assert env.action_space.n == 9
        assert np.array_equal(info["action_mask"], action_mask(env.sim, 8, config["junction"]["environment"]["hold_limit"]))
        start = env.sim.time
        _, _, _, _, info = env.step(0)  # holding advances time
        assert env.sim.time >= start + 1 and info["elapsed_minutes"] >= 1
        masked = np.flatnonzero(~info["action_mask"])
        if len(masked):
            with pytest.raises(ValueError):
                env.step(int(masked[0]))
    finally:
        env.close()


def test_full_small_day_completes_safely_and_deterministically(config):
    case = generate_network_scenario(config, SEED)
    first, rows, occupancy = run_network_episode(config, case, fcfs)
    second, _, _ = run_network_episode(config, case, fcfs)
    assert first == second
    assert first["terminated"] and not first["truncated"]
    assert first["j_safety_violations"] == 0 and first["j_processed_fraction"] == 1
    assert first["network_unfinished"] == 0 and first["reached_edge_fraction"] == 1
    assert first["eastbound_trains"] > 0 and first["westbound_trains"] > 0
    assert all(r["reached_edge"] and r["junction_station_delay"] >= 0 for r in rows)
    assert occupancy and all(o["end"] is not None and o["end"] > o["start"] for o in occupancy)


def test_admitted_trains_start_from_rest_at_the_gate(config):
    env = NetworkTrainTurnEnv(config, generate_network_scenario(config, SEED))
    try:
        obs, info = env.reset(seed=SEED)
        while True:
            obs, _, term, trunc, info = env.step(fcfs(obs, info["action_mask"]))
            if term or trunc:
                break
        assert len(env.sim.admission_speeds) == len(env.sim.scenario.trains)
        assert max(env.sim.admission_speeds) < 0.1
    finally:
        env.close()


def test_queue_spilling_into_minor_stations_is_charged_to_the_junction():
    config = small(128)  # the full default day; shorter days do not congest enough
    case = generate_network_scenario(config, SEED)

    def stall(obs, mask):  # hold whenever allowed so the junction queue backs up the lines
        return 0 if mask[0] else int(np.flatnonzero(mask)[0])
    metrics, _, _ = run_network_episode(config, case, stall)
    assert metrics["upstream_held_in_station"]
    assert metrics["j_upstream_minutes"] > 0
    assert metrics["network_unfinished"] == 0 and metrics["j_safety_violations"] == 0


def test_minor_station_calls_respect_timetable(config):
    case = generate_network_scenario(config, SEED)
    _, rows, _ = run_network_episode(config, case, fcfs)
    for row in rows:
        plan = case.plan(row["train"])
        calls = json.loads(row["minor_calls"])
        stopping = {c.station for c in plan.approach_calls + plan.departure_calls if c.stops}
        # No early departure from a scheduled stop (SUMO 'until'); one 1 s step tolerance.
        assert all(calls[s] >= -1 / 60 for s in stopping if s in calls)


def test_unstarted_train_forecasts_do_not_reveal_actual_start_delay(config):
    case = generate_network_scenario(config, SEED)
    last = case.plans[-1]
    changed = replace(case, plans=case.plans[:-1] + (replace(last, start_delay=last.start_delay + 90),))
    observations = []
    for variant in (case, changed):
        env = NetworkTrainTurnEnv(config, variant)
        try:
            obs, _ = env.reset(seed=SEED)
            trace = [obs]
            for _ in range(5):
                obs, *_ = env.step(int(np.flatnonzero(env.action_masks())[-1]))
                trace.append(obs)
            observations.append(np.stack(trace))
        finally:
            env.close()
    assert np.array_equal(observations[0], observations[1])
