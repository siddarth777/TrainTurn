"""Replay real DDU days with the trained junction controller and the heuristic baselines.

Step 1 ("planned"): each day's real timetable loaded from CSV with the original
loader, no delays. Step 2 ("actual"): the same days with the real incoming delays.
Every policy faces identical days (paired comparison).
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import libsumo
from tqdm import tqdm
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.evaluator import run_episode, write_csv
from railway_rl.evaluation.statistics import paired_difference, summarize
from railway_rl.station import Disturbance, Scenario
from railway_rl.timetable import load_scenario, load_timetable
from railway_real.ddu import load_real_config, station_from


def planned_scenario(cfg: dict, path: Path) -> Scenario:
    trains = load_timetable(path)
    seed = int(path.stem.replace("-", ""))
    return Scenario(station_from(cfg), trains, tuple(Disturbance() for _ in trains), (), seed,
                    f"real DDU timetable {path.stem} (RSTGCN Sep 2024), no delays")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ddu.yaml"))
    p.add_argument("--scenarios", required=True, help="directory written by build_ddu.py")
    p.add_argument("--checkpoint", help="trained junction checkpoint (CUDA); omit for baselines only")
    p.add_argument("--modes", nargs="+", default=["planned", "actual"])
    p.add_argument("--days", type=int, help="limit to the first N days")
    p.add_argument("--first-day", help="first day to replay (YYYY-MM-DD)")
    p.add_argument("--last-day", help="last day to replay (YYYY-MM-DD)")
    p.add_argument("--extra-checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="another checkpoint to replay alongside --checkpoint (repeatable)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds results; choose a new directory")
    cfg = load_real_config(a.config)
    junction = cfg["junction"]
    n = junction["environment"]["max_platforms"]
    policies = {k: (lambda o, m, fn=fn: fn(o, m, n)) for k, fn in POLICIES.items()}
    checkpoints = ({"dqn": a.checkpoint} if a.checkpoint else {}) | dict(e.split("=", 1) for e in a.extra_checkpoint)
    if a.extra_checkpoint and not a.checkpoint:
        raise SystemExit("--extra-checkpoint needs --checkpoint")
    digests, learned = {}, {}
    for name, path in checkpoints.items():
        from railway_rl.agents.double_dqn import DoubleDQN
        agent = DoubleDQN.load(path)
        if agent.config["environment"] != junction["environment"] or agent.config["reward"] != junction["reward"]:
            raise SystemExit(f"Checkpoint {name} settings differ from the junction config")
        learned[name] = lambda o, m, agent=agent: agent.act(o, m)
        digests[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    policies = {**learned, **policies}
    digest = digests.get("dqn")
    source = Path(a.scenarios)
    days = [d for d in json.loads((source / "manifest.json").read_text())["days"]
            if (a.first_day or d) <= d <= (a.last_day or d)][: a.days]
    cases = {}
    for mode in a.modes:
        for day in days:
            cases[(mode, day)] = (planned_scenario(cfg, source / "planned" / f"{day}.csv") if mode == "planned"
                                  else load_scenario(source / mode / f"{day}.json"))
    rows, trains = [], []
    with tqdm(total=len(cases) * len(policies), desc="DDU replay", unit="episode", dynamic_ncols=True) as bar:
        for (mode, day), case in cases.items():
            for name, policy in policies.items():
                bar.set_postfix(mode=mode, day=day, policy=name)
                metrics, per_train, _ = run_episode(junction, case, policy)
                tags = {"mode": mode, "day": day, "policy": name, "trains": len(case.trains)}
                rows.append({**tags, **metrics})
                trains += [{**tags, **r} for r in per_train]
                bar.update(1)
    summaries, paired = {}, {}
    for mode in a.modes:
        for name in policies:
            mine = [r for r in rows if r["mode"] == mode and r["policy"] == name]
            summaries[f"{mode}/{name}"] = {
                "complete": all(r["processed_fraction"] == 1 for r in mine),
                "safety_violations": sum(r["safety_violations"] for r in mine),
                "weighted_station_delay": summarize([r["weighted_station_delay"] for r in mine]),
                "mean_departure_lateness_completed": summarize([r["mean_departure_lateness_completed"] for r in mine])}
        names = list(learned)
        pairs = [(x, base) for x in names for base in POLICIES] + [(x, "dqn") for x in names if x != "dqn"]
        for mine_name, base in pairs:
            first = {r["day"]: r["weighted_station_delay"] for r in rows if r["mode"] == mode and r["policy"] == mine_name}
            second = {r["day"]: r["weighted_station_delay"] for r in rows if r["mode"] == mode and r["policy"] == base}
            paired[f"{mode}/{mine_name}-minus-{base}"] = {**paired_difference(first, second),
                                                          "lower_days": sum(first[d] < second[d] for d in first)}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episodes.csv", rows)
    write_csv(out / "trains.csv", trains)
    (out / "summary.json").write_text(json.dumps({"summaries": summaries, "paired_differences": paired}, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "config": cfg["source"], "scenarios": str(source), "days": days, "modes": a.modes,
        "policies": list(policies), "checkpoint_sha256": digest, "checkpoints": checkpoints,
        "checkpoint_sha256s": digests, "sumo": libsumo.getVersion()[1],
        "interpretation": "Replay of real September 2024 days at DDU (zero-shot for the DQN); "
                          "station layout and train lengths partly assumed"}, indent=2))
    print(json.dumps({"summaries": summaries, "paired_differences": paired}, indent=1))


if __name__ == "__main__":
    main()
