"""Record SUMO network episodes for the browser replay (positions, queues, decisions).

Observes the simulator without changing it: the recorded junction totals equal
those of ``evaluate_network.py`` for the same scenario and policy.

    python multistation/scripts/record_network_replay.py --checkpoint checkpoints/station8_128/best_model.pt \
        --seeds 200000 200001 200002 --output multistation/results/replay/replay.json
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import json
from pathlib import Path
import libsumo as api
import torch
from railway_network import simulator as network_simulator
from railway_network.config import load_network_config
from railway_network.environment import NetworkTrainTurnEnv
from railway_network.evaluation import junction_policies, network_metrics
from railway_network.scenario import generate_network_scenario
from railway_rl.evaluation.metrics import episode_metrics

SAMPLE_SECONDS = 10


class Recorder:
    def __init__(self):
        self.sim = None
        self.edges, self.cum, self.samples = {}, {}, {}

    def observe(self, sim) -> None:
        if sim is not self.sim:
            return
        t = round(sim.seconds)
        for vid in sim._vehicles:
            road = sim.road.get(vid)
            if road is None:
                continue
            path = self.edges.setdefault(vid, [])
            if not path or path[-1] != road:
                if path:
                    self.cum[vid] = self.cum.get(vid, 0.0) + sim.length(path[-1])
                path.append(road)
            if t % SAMPLE_SECONDS == 0:
                internal = api.vehicle.getRoadID(vid).startswith(":")
                offset = self.cum.get(vid, 0.0) + (sim.length(road) if internal else api.vehicle.getLanePosition(vid))
                self.samples.setdefault(vid, []).append([t, round(offset)])
        for vid in api.simulation.getArrivedIDList():
            path = self.edges.get(vid)
            if path:
                self.samples[vid].append([t, round(self.cum.get(vid, 0.0) + sim.length(path[-1]))])


RECORDER = Recorder()
_original_second = network_simulator.NetworkJunctionSimulator._second


def _observed_second(self):
    _original_second(self)
    RECORDER.observe(self)


def record(config, case, name, policy, agent=None) -> dict:
    global RECORDER
    RECORDER = Recorder()
    network_simulator.NetworkJunctionSimulator._second = _observed_second
    env = NetworkTrainTurnEnv(config, case)
    decisions, minutes = [], []
    try:
        # Attach before reset so the first trains' journeys are captured.
        original_init = network_simulator.NetworkJunctionSimulator.__init__

        def init(self, *args, **kwargs):
            RECORDER.sim = self
            original_init(self, *args, **kwargs)
        network_simulator.NetworkJunctionSimulator.__init__ = init
        obs, info = env.reset(seed=case.seed)
        network_simulator.NetworkJunctionSimulator.__init__ = original_init
        sim = env.sim
        lanes = {}
        for lane in api.lane.getIDList():
            if not lane.startswith(":"):
                lanes[lane.rsplit("_", 1)[0]] = [round(api.lane.getLength(lane), 1),
                                                 [[round(x, 1), round(y, 1)] for x, y in api.lane.getShape(lane)]]
        cum = 0.0
        tick = sim.tick

        def recorded_tick():
            nonlocal cum
            costs = tick()
            cum += costs["weighted_wait"]
            minutes.append([round(sim.time), list(sim.outer), list(sim.upstream),
                            [sim.occupants.get(p, "") for p in range(1, 9)], round(cum, 2)])
            return costs
        sim.tick = recorded_tick
        while True:
            mask = info["action_mask"]
            q = None
            if agent is not None:
                with torch.no_grad():
                    q = agent.online(torch.as_tensor(obs, device=agent.device).unsqueeze(0)).squeeze(0).cpu().tolist()
            action = policy(obs, mask)
            current = sim.current
            decisions.append([round(sim.time), current.id, int(action), int(sum(1 << i for i, m in enumerate(mask) if m)),
                              round(sim.states[current.id].waiting, 1), None if q is None else [round(v, 3) for v in q]])
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        sim.drain(config["network"]["drain_minutes"])
        junction, _ = episode_metrics(sim, config["junction"]["environment"]["horizon"])
        network, rows = network_metrics(sim)
        actual = {t.id: {"signal": sim.states[t.id].signal_arrival, "entry": sim.states[t.id].entry,
                         "platform": sim.states[t.id].platform, "departure": sim.states[t.id].departure,
                         "sink": sim.sink_arrival.get(t.id), "stations": sim.station_times[t.id],
                         "tracks": sim.decided[t.id]} for t in case.junction.trains}
    finally:
        env.close()
        network_simulator.NetworkJunctionSimulator._second = _original_second
    print(f"{case.seed} {name}: junction weighted delay {junction['weighted_station_delay']:.2f}, "
          f"network weighted lateness {network['network_weighted_lateness']:.1f}", flush=True)
    return {"lanes": lanes, "edges": RECORDER.edges, "samples": RECORDER.samples, "decisions": decisions,
            "minutes": minutes, "actual": actual,
            "metrics": {"j_weighted_station_delay": junction["weighted_station_delay"],
                        "j_platform_changes": junction["platform_changes"], **network},
            "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[200000])
    p.add_argument("--policies", nargs="+", default=["dqn", "priority", "fcfs"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    config = load_network_config(a.config)
    policies, digest = junction_policies(config, a.checkpoint)
    from railway_rl.agents.double_dqn import DoubleDQN
    agent = DoubleDQN.load(a.checkpoint)
    out = {"checkpoint_sha256": digest, "sample_seconds": SAMPLE_SECONDS, "days": []}
    lanes = None
    for seed in a.seeds:
        case = generate_network_scenario(config, seed)
        runs = {}
        for name in a.policies:
            run = record(config, case, name, policies[name], agent if name == "dqn" else None)
            lanes = lanes or run.pop("lanes")
            run.pop("lanes", None)
            runs[name] = run
        trains = [[t.id, t.category, t.priority, t.planned_platform, t.arrival_corridor, t.departure_corridor,
                   t.length, t.kind] for t in case.junction.trains]
        plans = {pl.train: {"origin": pl.origin_time, "start_delay": pl.start_delay, "gate": pl.gate_arrival,
                            "jdep": pl.junction_departure, "gout": pl.gate_out, "sink": pl.sink_arrival, "terminal": pl.terminal,
                            "calls": [[c.station, c.stops, c.arrival, c.departure] for c in pl.approach_calls + pl.departure_calls],
                            "direction": pl.direction}
                 for pl in case.plans}
        out["days"].append({"seed": seed, "trains": trains, "plans": plans, "runs": runs})
    net = config["network"]
    out["lanes"] = lanes
    out["stations"] = {sid: s["name"] for sid, s in net["stations"].items()}
    out["platforms"] = [[p.id, p.length] for p in case.junction.station.platforms]
    validation = Path(a.checkpoint).with_name("validation.csv")
    if validation.exists():
        out["validation"] = [[int(r["episode"]), round(float(r["mean_weighted_delay"]), 1)]
                             for r in csv.DictReader(validation.open())]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
