"""Sequential libsumo evaluation on immutable, shared exogenous scenarios."""
from pathlib import Path
from typing import Callable
import csv
import hashlib
import json
import platform
from tqdm import tqdm
from ..baselines import POLICIES
from ..delays import generate_scenario
from ..environment import TrainTurnEnv
from ..station import Scenario
from ..timetable import save_scenario
from .metrics import episode_metrics
from .statistics import summarize, paired_difference


def run_episode(config: dict, scenario: Scenario, policy: Callable) -> tuple[dict, list[dict], list[dict]]:
    env = TrainTurnEnv(config, scenario)
    discretionary_holds = 0
    try:
        observation, info = env.reset(seed=scenario.seed)
        while True:
            action = policy(observation, info["action_mask"])
            discretionary_holds += int(action == 0 and info["action_mask"][1:].any())
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        metrics, trains = episode_metrics(env.sim, config["environment"]["horizon"])
        metrics.update({"episode_reward": env.episode_reward, "terminated": terminated, "truncated": truncated,
                        "discretionary_holds": discretionary_holds})
        occupancy = [dict(row) for row in getattr(env.sim, "platform_occupancy", env.sim.occupancy)]
        return metrics, trains, occupancy
    finally:
        env.close()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config: dict, episodes: int, seed: int, output: str | Path,
             checkpoint: str | Path | None = None, stress: bool = False) -> dict:
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Experiment already exists: {output}. Choose a new output directory.")
    n = config["environment"]["max_platforms"]
    policies = {name: (lambda obs, mask, fn=fn: fn(obs, mask, n)) for name, fn in POLICIES.items()}
    checkpoint_hash = None
    if checkpoint:
        from ..agents.double_dqn import DoubleDQN
        agent = DoubleDQN.load(checkpoint)
        if agent.config["environment"] != config["environment"] or agent.config["reward"] != config["reward"]:
            raise ValueError("Checkpoint observation, time, and reward settings must match evaluation")
        if seed < config["training"]["test_seed"]:
            raise ValueError("Use held-out test seeds, not training or validation seeds")
        policies["dqn"] = lambda obs, mask: agent.act(obs, mask)
        checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    from copy import deepcopy
    settings = [("normal", 1.0, "normal")]
    if stress:
        settings += [("severity1.5", 1.5, "normal"), ("severity2", 2.0, "normal"),
                     ("closure", 1.0, "closure"), ("peak", 1.0, "peak"), ("corridor", 1.0, "corridor")]
    all_rows, train_rows, traces, case_hashes = [], [], {}, {}
    with tqdm(total=len(settings) * len(policies) * episodes,
              desc="Evaluation", unit="episode", dynamic_ncols=True) as progress:
        for setting, multiplier, mode in settings:
            setting_config = deepcopy(config)
            setting_config["scenario"]["severity"] *= multiplier
            setting_config["scenario"]["mode"] = mode if stress else config["scenario"]["mode"]
            # Materialize every scenario before any policy sees it.
            scenarios = [generate_scenario(setting_config, seed + i) for i in range(episodes)]
            for case in scenarios:
                file = output / "scenarios" / f"{setting}_{case.seed}.json"
                save_scenario(case, file)
                case_hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
            for name, policy in policies.items():
                for case in scenarios:
                    progress.set_postfix(setting=setting, policy=name, scenario=str(case.seed))
                    metrics, rows, occupancy = run_episode(setting_config, case, policy)
                    tags = {"policy": name, "scenario": case.seed, "setting": setting}
                    all_rows.append({**tags, **metrics})
                    train_rows.extend({**tags, **row} for row in rows)
                    if case.seed == seed:
                        traces[f"{setting}_{name}"] = occupancy
                    progress.update(1)
    summaries, paired = {}, {}
    for setting, _, _ in settings:
        for name in policies:
            rows = [r for r in all_rows if r["policy"] == name and r["setting"] == setting]
            complete = all(r["processed_fraction"] == 1 for r in rows)
            summaries[f"{setting}/{name}"] = {"complete": complete,
                "weighted_station_delay": summarize([r["weighted_station_delay"] for r in rows]),
                "processed_fraction": summarize([r["processed_fraction"] for r in rows])}
        if "dqn" in policies:
            dqn = {r["scenario"]: r["weighted_station_delay"] for r in all_rows if r["policy"] == "dqn" and r["setting"] == setting}
            for baseline in POLICIES:
                base = {r["scenario"]: r["weighted_station_delay"] for r in all_rows if r["policy"] == baseline and r["setting"] == setting}
                if summaries[f"{setting}/dqn"]["complete"] and summaries[f"{setting}/{baseline}"]["complete"]:
                    paired[f"{setting}/dqn-minus-{baseline}"] = paired_difference(dqn, base)
    write_csv(output / "episodes.csv", all_rows)
    write_csv(output / "trains.csv", train_rows)
    (output / "occupancy.json").write_text(json.dumps(traces, indent=2))
    result = {"summaries": summaries, "paired_differences": paired, "preliminary": episodes < 100}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    import libsumo
    manifest = {"config": config, "episodes": episodes, "seed": seed, "policies": list(policies),
                "checkpoint_sha256": checkpoint_hash, "scenario_sha256": case_hashes,
                "python": platform.python_version(), "sumo": libsumo.getVersion()[1],
                "provenance": "synthetic; uncalibrated", "stress": stress,
                "interpretation": "Preliminary smoke evaluation" if episodes < 100 else "Held-out evaluation; assess training-seed uncertainty separately"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result
