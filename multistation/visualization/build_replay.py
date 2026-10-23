"""Compact a recorded replay and inject it into the HTML template.

    python multistation/visualization/build_replay.py multistation/results/replay/replay.json \
        multistation/results/network_replay.html
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent


def compact(raw: dict) -> dict:
    names = sorted(raw["lanes"])
    index = {n: i for i, n in enumerate(names)}
    out = {"lanes": [[n, *raw["lanes"][n]] for n in names], "stations": raw["stations"],
           "platforms": raw["platforms"], "validation": raw.get("validation", []), "days": []}
    for day in raw["days"]:
        ids = [t[0] for t in day["trains"]]
        tix = {t: i for i, t in enumerate(ids)}
        plans = []
        for t in ids:
            p = day["plans"][t]
            plans.append([p["origin"], p["start_delay"], p["gate"], p["jdep"], p["gout"], p["sink"], int(p["terminal"]),
                          [[c[0], int(c[1]), c[2], c[3]] for c in p["calls"]]])
        runs = {}
        for name, run in day["runs"].items():
            paths, tracks = [], []
            for t in ids:
                paths.append([index[e] for e in run["edges"].get(t, [])])
                samples = run["samples"].get(t, [])
                keep = [s for i, s in enumerate(samples)
                        if i in (0, len(samples) - 1) or not (samples[i - 1][1] == s[1] == samples[i + 1][1])]
                tracks.append([v for s in keep for v in s])
            minutes = [[m[0], [tix[x] for x in m[1]], [tix[x] for x in m[2]],
                        [tix[x] if x else -1 for x in m[3]], m[4]] for m in run["minutes"]]
            decisions = [[d[0], tix[d[1]], *d[2:]] for d in run["decisions"]]
            actual = []
            for t in ids:
                a = run["actual"][t]
                actual.append([a["signal"], a["entry"], a["platform"], a["departure"], a["sink"],
                               {k: v for k, v in a["stations"].items()}])
            m = run["metrics"]
            runs[name] = {"paths": paths, "tracks": tracks, "minutes": minutes, "decisions": decisions,
                          "actual": actual,
                          "metrics": {k: round(m[k], 3) for k in ("j_weighted_station_delay", "network_weighted_lateness",
                                                                   "mean_public_lateness", "on_time_share",
                                                                   "j_platform_changes", "minor_track_deviations")}}
        out["days"].append({"seed": day["seed"], "trains": day["trains"], "plans": plans, "runs": runs})
    return out


def main(source: str, target: str) -> None:
    data = compact(json.loads(Path(source).read_text()))
    page = (HERE / "network_replay_template.html").read_text()
    Path(target).write_text(page.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {target} ({Path(target).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
