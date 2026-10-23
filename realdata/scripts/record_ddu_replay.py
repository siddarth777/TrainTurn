"""Record real DDU days in SUMO and build the animated replay page.

    python realdata/scripts/record_ddu_replay.py --scenarios realdata/data/ddu_sep2024 \
        --checkpoint checkpoints/station8_128/best_model.pt --days 2024-09-10:"Median day" ... \
        --output realdata/results/ddu_replay.html

Observes the simulator without changing it; totals equal those of replay_ddu.py.
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import json
from pathlib import Path
import libsumo as api
import torch
from railway_rl import sumo_simulator
from railway_rl.agents.double_dqn import DoubleDQN
from railway_rl.baselines import POLICIES
from railway_rl.environment import TrainTurnEnv
from railway_rl.evaluation.metrics import episode_metrics
from railway_rl.timetable import load_scenario
from railway_real.ddu import load_real_config

SAMPLE = 10
TEMPLATE = Path(__file__).resolve().parents[1] / "visualization" / "ddu_replay_template.html"
_FRAMES = {"rows": None}
_original = sumo_simulator.SumoSimulator._physical_events


def _observed(self):
    _original(self)
    t = round(api.simulation.getTime())
    if _FRAMES["rows"] is not None and t % SAMPLE == 0:
        _FRAMES["rows"].append([t, [[v, api.vehicle.getRoadID(v), round(api.vehicle.getLanePosition(v))]
                                    for v in api.vehicle.getIDList() if not api.vehicle.getRoadID(v).startswith(":")]])


def record(config, case, policy, agent=None):
    sumo_simulator.SumoSimulator._physical_events = _observed
    _FRAMES["rows"] = []
    env = TrainTurnEnv(config, case)
    decisions, minutes, lanes = [], [], {}
    try:
        obs, info = env.reset(seed=case.seed)
        sim = env.sim
        for lane in api.lane.getIDList():
            if not lane.startswith(":"):
                lanes[lane.rsplit("_", 1)[0]] = [round(api.lane.getLength(lane), 1),
                                                 [[round(x, 1), round(y, 1)] for x, y in api.lane.getShape(lane)]]
        cum, tick = [0.0], sim.tick

        def recorded():
            costs = tick()
            cum[0] += costs["weighted_wait"]
            minutes.append([round(sim.time), list(sim.outer), list(sim.upstream),
                            [sim.occupants.get(p, "") for p in range(1, 9)], round(cum[0], 2)])
            return costs
        sim.tick = recorded
        while True:
            mask = info["action_mask"]
            q = None
            if agent is not None:
                with torch.no_grad():
                    q = agent.online(torch.as_tensor(obs, device=agent.device).unsqueeze(0)).squeeze(0).cpu().tolist()
            action = policy(obs, mask)
            decisions.append([round(sim.time), sim.current.id, int(action), int(sum(1 << i for i, m in enumerate(mask) if m)),
                              round(sim.states[sim.current.id].waiting, 1), None if q is None else [round(v, 3) for v in q]])
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        metrics, trains = episode_metrics(sim, config["environment"]["horizon"])
    finally:
        env.close()
        sumo_simulator.SumoSimulator._physical_events = _original
    return {"lanes": lanes, "frames": _FRAMES["rows"], "decisions": decisions, "minutes": minutes,
            "trains": trains, "metrics": metrics}


def compact(case, runs, label):
    ids = [t.id for t in case.trains]
    ix = {t: i for i, t in enumerate(ids)}
    lanes = runs["dqn"]["lanes"]
    roads = sorted(lanes)
    rix = {r: i for i, r in enumerate(roads)}
    out = {}
    for name, run in runs.items():
        tracks = {}
        for t, rows in run["frames"]:
            for vid, road, pos in rows:
                tracks.setdefault(ix[vid], []).append([t, rix[road], pos])
        flat = []
        for i in range(len(ids)):
            tr = tracks.get(i, [])
            keep = [s for k, s in enumerate(tr) if k in (0, len(tr) - 1) or not (tr[k - 1][1:] == s[1:] == tr[k + 1][1:])]
            flat.append([v for s in keep for v in s])
        per = {ix[t["train"]]: [t["platform"], t["signal_arrival"], t["entry"], round(t["departure"], 2),
                                round(t["network_exit"], 2), round(t["station_delay"], 2), round(t["weighted_delay"], 2)]
               for t in run["trains"]}
        m = run["metrics"]
        out[name] = {"tracks": flat,
                     "minutes": [[x[0], [ix[v] for v in x[1]], [ix[v] for v in x[2]], [ix[v] if v else -1 for v in x[3]], x[4]]
                                 for x in run["minutes"]],
                     "decisions": [[d[0], ix[d[1]], *d[2:]] for d in run["decisions"]],
                     "per": [per[i] for i in range(len(ids))],
                     "metrics": {k: round(float(m[k]), 3) for k in ("weighted_station_delay", "mean_station_delay",
                                                                     "platform_changes", "holds", "safety_violations")}}
    trains = [[t.id, t.category, t.priority, t.planned_platform, t.scheduled_arrival, t.scheduled_departure,
               t.length, t.arrival_corridor, t.departure_corridor, t.dwell] for t in case.trains]
    return {"id": case.seed, "label": label, "trains": trains, "runs": out}, [[r, *lanes[r]] for r in roads]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ddu.yaml"))
    p.add_argument("--scenarios", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--days", nargs="+", required=True, help='DATE:"label" pairs')
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = load_real_config(a.config)
    config = cfg["junction"]
    best = DoubleDQN.load(a.checkpoint)
    initial = DoubleDQN.load(Path(a.checkpoint).with_name("initial_model.pt"))
    policies = {"dqn": (lambda o, m: best.act(o, m), best), "untrained": (lambda o, m: initial.act(o, m), initial),
                "priority": (lambda o, m: POLICIES["priority"](o, m, 8), None),
                "fcfs": (lambda o, m: POLICIES["fcfs"](o, m, 8), None)}
    days, lanes = [], None
    for spec in a.days:
        date, label = spec.split(":", 1)
        case = load_scenario(Path(a.scenarios) / "actual" / f"{date}.json")
        runs = {}
        for name, (policy, agent) in policies.items():
            runs[name] = record(config, case, policy, agent)
            print(f"{date} {name}: weighted station delay {runs[name]['metrics']['weighted_station_delay']:.2f}", flush=True)
        day, lane_list = compact(case, runs, f"{label} · {date}")
        days.append(day)
        lanes = lanes or lane_list
    data = {"lanes": lanes, "platforms": [[p.id, p.length, p.clearance] for p in case.station.platforms],
            "days": days, "validation": []}
    validation = Path(a.checkpoint).with_name("validation.csv")
    if validation.exists():
        data["validation"] = [[int(r["episode"]), round(float(r["mean_weighted_delay"]), 1)] for r in csv.DictReader(validation.open())]
    page = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(page)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
