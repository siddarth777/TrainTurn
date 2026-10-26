"""Build DDU day scenarios from real September 2024 data.

Step 1 writes each day's real timetable as a CSV in the format read by
``railway_rl.timetable.load_timetable`` (planned times, no delays).
Step 2 writes each day's replay scenario (real incoming delays) as JSON.
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import hashlib
import json
from pathlib import Path
from railway_rl.timetable import save_scenario
from railway_real.ddu import build_runs, complete_days, day_scenario, describe, load_real_config, portable

FIELDS = ["id", "category", "scheduled_arrival", "scheduled_departure", "planned_platform", "length",
          "electric", "kind", "arrival_corridor", "departure_corridor", "dwell", "priority"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ddu.yaml"))
    p.add_argument("--output", required=True, help="new directory for the derived scenarios")
    a = p.parse_args()
    out = Path(a.output)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a build; choose a new directory")
    cfg = load_real_config(a.config)
    runs, skipped = build_runs(cfg)
    days = complete_days(runs)
    info = []
    for date in days:
        planned, planned_info = day_scenario(cfg, runs, date, "planned")
        path = out / "planned" / f"{date}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for t in planned.trains:
                writer.writerow({k: getattr(t, k) for k in FIELDS})
        info.append(planned_info)
        for mode in ("actual", "observed"):
            case, case_info = day_scenario(cfg, runs, date, mode)
            save_scenario(case, out / mode / f"{date}.json")
            info.append(case_info)
    raw = [Path(cfg["raw"]["rstgcn"]) / n for n in ("train_routes_Sep2024.csv", "train_routes_delays_Sep2024.csv")]
    raw += [Path(cfg["raw"]["datameet"]) / n for n in ("trains.json", "stations.json")]
    manifest = {"config": cfg["source"], "station": cfg["station"], "days": days, "skipped_runs": skipped,
                "summary": describe(runs), "per_day": info,
                "raw_sha256": {str(r.name): sha(r) for r in raw},
                "sources": {"rstgcn": "github.com/KoyenaChowdhury/RSTGCN (research use; cite IEEE T-ITS 2026)",
                            "datameet": "github.com/datameet/railways (CC0)"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=portable))
    print(f"{len(days)} days -> {out}")


if __name__ == "__main__":
    main()
