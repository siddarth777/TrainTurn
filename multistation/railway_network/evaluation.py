"""Paired evaluation on the network: junction metrics plus network-wide lateness.

Every policy controls only the junction. Scenarios (start delays, junction dwell
extensions, minor-station extensions, forecast noise) are materialized once and
shared by all policies (common random numbers).

Network metrics, all in minutes:
* ``public_lateness``: lateness at the end of the public journey, i.e. arrival
  at the network edge, or junction arrival for terminal services (their empty
  stock still runs on and is measured by ``sink_lateness``).
* ``network_weighted_lateness``: sum over trains of priority x max(0, public lateness).
* ``arrival_lateness_at_junction`` minus ``start_delay``: lateness gained on the
  western branches before the junction.
* ``departure_side_change``: line-end lateness minus lateness on leaving the junction.
* ``eastbound_*`` / ``westbound_*``: the same measures split by direction.
Lateness can be negative (early) because timetables include a running allowance.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import platform
from typing import Callable
import numpy as np
from tqdm import tqdm
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.evaluator import write_csv
from railway_rl.evaluation.metrics import episode_metrics
from railway_rl.evaluation.statistics import summarize, paired_difference
from .environment import NetworkTrainTurnEnv
from .scenario import NetworkScenario, generate_network_scenario, save_network_scenario

ON_TIME_MINUTES = 5.0


def network_metrics(sim) -> tuple[dict, list[dict]]:
    rows = []
    for train in sim.scenario.trains:
        plan, state = sim.plans[train.id], sim.states[train.id]
        reached = train.id in sim.sink_arrival
        arrival_j = None if state.signal_arrival is None else state.signal_arrival - train.scheduled_arrival
        gate_out = (sim.gate_out[train.id] - plan.gate_out) / 60 if train.id in sim.gate_out else None
        sink = (sim.sink_arrival[train.id] - plan.sink_arrival) / 60 if reached else None
        public = arrival_j if plan.terminal else sink
        calls = {}
        for call in plan.approach_calls + plan.departure_calls:
            times = sim.station_times[train.id].get(call.station)
            if times and times[1] is not None:
                calls[call.station] = (times[1] - call.departure) / 60
        rows.append({"train": train.id, "direction": plan.direction, "category": train.category,
                     "priority": train.priority,
                     "terminal": plan.terminal, "start_delay": plan.start_delay,
                     "inserted_lateness": (sim.inserted[train.id] - plan.origin_time) / 60 if train.id in sim.inserted else None,
                     "arrival_lateness_at_junction": arrival_j,
                     "junction_station_delay": state.waiting + state.departure_waiting,
                     "junction_platform": state.platform, "gate_out_lateness": gate_out,
                     "sink_lateness": sink, "public_lateness": public, "reached_edge": reached,
                     "minor_calls": json.dumps({k: round(v, 2) for k, v in calls.items()}),
                     "minor_tracks": json.dumps(sim.decided[train.id])})
    done = [r for r in rows if r["public_lateness"] is not None]
    weighted = sum(r["priority"] * max(0.0, r["public_lateness"]) for r in done)
    minor = [(r["priority"], v) for r in rows for v in json.loads(r["minor_calls"]).values()]
    pre = [r["arrival_lateness_at_junction"] - r["start_delay"] for r in rows if r["arrival_lateness_at_junction"] is not None]
    after = [r["sink_lateness"] - r["gate_out_lateness"] for r in rows
            if r["sink_lateness"] is not None and r["gate_out_lateness"] is not None]
    summary = {
        "network_weighted_lateness": weighted,
        "mean_public_lateness": float(np.mean([r["public_lateness"] for r in done])) if done else None,
        "max_public_lateness": max((r["public_lateness"] for r in done), default=None),
        "on_time_share": float(np.mean([r["public_lateness"] <= ON_TIME_MINUTES for r in done])) if done else None,
        "mean_sink_lateness": float(np.mean([r["sink_lateness"] for r in rows if r["reached_edge"]])) if any(r["reached_edge"] for r in rows) else None,
        "minor_weighted_lateness": sum(p * max(0.0, v) for p, v in minor),
        "mean_lateness_gained_before_junction": float(np.mean(pre)) if pre else None,
        "mean_departure_side_change": float(np.mean(after)) if after else None,
        "minor_track_deviations": sim.track_changes,
        "reached_edge_fraction": sum(r["reached_edge"] for r in rows) / len(rows),
        "network_unfinished": sum(not r["reached_edge"] for r in rows),
        "upstream_held_in_station": "station" in sim.upstream_locations,
    }
    for name, code in (("eastbound", "e"), ("westbound", "w")):
        mine = [r for r in done if r["direction"] == code]
        summary[f"{name}_trains"] = sum(r["direction"] == code for r in rows)
        summary[f"{name}_weighted_lateness"] = sum(r["priority"] * max(0.0, r["public_lateness"]) for r in mine)
        summary[f"{name}_junction_weighted_delay"] = sum(r["priority"] * r["junction_station_delay"]
                                                         for r in rows if r["direction"] == code)
    return summary, rows


def run_network_episode(config: dict, case: NetworkScenario, policy: Callable,
                        on_decision: Callable | None = None) -> tuple[dict, list[dict], list[dict]]:
    env = NetworkTrainTurnEnv(config, case)
    discretionary = 0
    try:
        observation, info = env.reset(seed=case.seed)
        while True:
            action = policy(observation, info["action_mask"])
            if on_decision is not None:
                on_decision(env.sim, observation, info["action_mask"], action)
            discretionary += int(action == 0 and info["action_mask"][1:].any())
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        unfinished = env.sim.drain(config["network"]["drain_minutes"])
        junction, _ = episode_metrics(env.sim, config["junction"]["environment"]["horizon"])
        network, rows = network_metrics(env.sim)
        metrics = {**{f"j_{k}": v for k, v in junction.items()}, **network,
                   "episode_reward": env.episode_reward, "terminated": terminated, "truncated": truncated,
                   "discretionary_holds": discretionary, "drain_unfinished": len(unfinished)}
        return metrics, rows, [dict(r) for r in env.sim.platform_occupancy]
    finally:
        env.close()


def junction_policies(config: dict, checkpoint: str | Path | None) -> tuple[dict, str | None]:
    n = config["junction"]["environment"]["max_platforms"]
    policies = {name: (lambda obs, mask, fn=fn: fn(obs, mask, n)) for name, fn in POLICIES.items()}
    digest = None
    if checkpoint:
        from railway_rl.agents.double_dqn import DoubleDQN
        agent = DoubleDQN.load(checkpoint)
        junction = config["junction"]
        if agent.config["environment"] != junction["environment"] or agent.config["reward"] != junction["reward"]:
            raise ValueError("Checkpoint observation/time/reward settings differ from the junction config")
        policies["dqn"] = lambda obs, mask: agent.act(obs, mask)
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return policies, digest


def evaluate_network(config: dict, episodes: int, seed: int, output: str | Path,
                     checkpoint: str | Path | None = None) -> dict:
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    if seed < config["junction"]["training"]["test_seed"]:
        raise ValueError("Use held-out test seeds, not training or validation seeds")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Experiment already exists: {output}. Choose a new output directory.")
    policies, digest = junction_policies(config, checkpoint)
    cases = [generate_network_scenario(config, seed + i) for i in range(episodes)]
    hashes = {}
    for case in cases:
        for file in save_network_scenario(case, output / "scenarios"):
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    rows, trains, traces = [], [], {}
    with tqdm(total=len(policies) * len(cases), desc="Network evaluation", unit="episode", dynamic_ncols=True) as bar:
        for name, policy in policies.items():
            for case in cases:
                bar.set_postfix(policy=name, scenario=str(case.seed))
                metrics, per_train, occupancy = run_network_episode(config, case, policy)
                rows.append({"policy": name, "scenario": case.seed, **metrics})
                trains.extend({"policy": name, "scenario": case.seed, **r} for r in per_train)
                if case.seed == seed:
                    traces[name] = occupancy
                bar.update(1)
    keys = ("j_weighted_station_delay", "network_weighted_lateness", "mean_public_lateness",
            "on_time_share", "minor_weighted_lateness")
    summaries, paired = {}, {}
    for name in policies:
        mine = [r for r in rows if r["policy"] == name]
        complete = all(r["j_processed_fraction"] == 1 and r["network_unfinished"] == 0 for r in mine)
        summaries[name] = {"complete": complete, **{k: summarize([r[k] for r in mine]) for k in keys}}
    if "dqn" in policies:
        for baseline in POLICIES:
            if summaries["dqn"]["complete"] and summaries[baseline]["complete"]:
                for k in ("j_weighted_station_delay", "network_weighted_lateness"):
                    first = {r["scenario"]: r[k] for r in rows if r["policy"] == "dqn"}
                    second = {r["scenario"]: r[k] for r in rows if r["policy"] == baseline}
                    paired[f"{k}/dqn-minus-{baseline}"] = paired_difference(first, second)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "trains.csv", trains)
    (output / "occupancy.json").write_text(json.dumps(traces))
    result = {"summaries": summaries, "paired_differences": paired, "preliminary": episodes < 100}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    import libsumo
    manifest = {"network_config": config["source"], "config": deepcopy({k: config[k] for k in ("junction", "network")}),
                "episodes": episodes, "seed": seed, "policies": list(policies), "checkpoint_sha256": digest,
                "scenario_sha256": hashes, "python": platform.python_version(), "sumo": libsumo.getVersion()[1],
                "provenance": "synthetic; uncalibrated",
                "interpretation": ("Zero-shot transfer of a junction controller trained in isolation; "
                                   + ("preliminary small sample" if episodes < 100 else "held-out evaluation"))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result
