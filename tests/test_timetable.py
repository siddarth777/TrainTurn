from railway_rl.config import load_config as _load_config

def load_config():
    config = _load_config()
    config["environment"]["backend"] = "reference"
    config["station"]["entry_minutes"] = 1
    config["station"]["departure_minutes"] = 1
    return config

from railway_rl.station import Scenario, Disturbance
from railway_rl.timetable import nominal_timetable, save_scenario, load_scenario
from railway_rl.simulator import Simulator


def test_nominal_timetable_has_zero_controlled_delay(tmp_path):
    for n in (3,):
        config = load_config()
        config["station"]["platform_count"] = n
        config["scenario"]["trains"] = 15
        station, trains = nominal_timetable(config, 13)
        case = Scenario(station, trains, tuple(Disturbance() for _ in trains))
        sim = Simulator(case)
        while not sim.complete and sim.time < 1800:
            if sim.current and sim.feasible(sim.current, sim.current.planned_platform):
                sim.assign(sim.current.planned_platform)
            sim.tick()
        sim.assert_safety()
        assert sim.complete
        assert sum(s.waiting + s.departure_waiting for s in sim.states.values()) == 0
        save_scenario(case, tmp_path / "case.json")
        assert load_scenario(tmp_path / "case.json") == case
