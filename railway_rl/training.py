"""Small-by-default CUDA training with fixed, disjoint validation scenarios."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from .agents.double_dqn import DoubleDQN, require_cuda
from .delays import generate_scenario
from .environment import TrainTurnEnv
from .observations import observation_size
from .evaluation.evaluator import run_episode, write_csv
from .evaluation.metrics import episode_metrics
from .baselines import POLICIES


def validation_metrics(config: dict, cases: list, policy) -> dict:
    rows = [run_episode(config, case, policy)[0] for case in cases]
    complete = all(row["processed_fraction"] == 1 for row in rows)
    return {"mean_weighted_delay": float(np.mean([r["weighted_station_delay"] for r in rows])) if complete else None,
            "mean_reward": float(np.mean([r["episode_reward"] for r in rows])),
            "std_reward": float(np.std([r["episode_reward"] for r in rows])),
            "mean_discretionary_holds": float(np.mean([r["discretionary_holds"] for r in rows])),
            "complete": complete}


def train(config: dict, seed: int, output: str | Path, warm_start: str | None = None) -> dict:
    require_cuda()  # Fail before creating outputs or running SUMO if the driver is unavailable.
    c, t = config["environment"], config["training"]
    if config["environment"]["backend"] != "sumo":
        raise ValueError("Production training must use SUMO")
    if seed < 0 or seed + t["episodes"] >= t["validation_seed"] or t["validation_seed"] + t["validation_episodes"] >= t["test_seed"]:
        raise ValueError("Training, validation and test seed ranges must be disjoint")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new training output directory to preserve existing runs")
    agent = DoubleDQN(observation_size(c["max_platforms"], c["forecast_count"]), c["max_platforms"] + 1, config, seed)
    if warm_start:
        previous = DoubleDQN.load(warm_start)
        if previous.config["environment"] != config["environment"]:
            raise ValueError("Warm start requires identical observation and environment settings")
        agent.online.load_state_dict(previous.online.state_dict())
        agent.target.load_state_dict(previous.online.state_dict())
    validation_cases = [generate_scenario(config, t["validation_seed"] + i) for i in range(t["validation_episodes"])]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    history, validation_history, best, started = [], [], float("inf"), time.monotonic()
    initial = validation_metrics(config, validation_cases, lambda o, m: agent.act(o, m))
    validation_history.append({"episode": 0, **initial})
    agent.save(output / "initial_model.pt", {"episode": 0, "untrained": warm_start is None})
    references = {name: validation_metrics(config, validation_cases,
                  lambda o, m, fn=fn: fn(o, m, c["max_platforms"])) for name, fn in POLICIES.items()}
    (output / "validation_references.json").write_text(json.dumps(references, indent=2))
    print(f"episode=0 validation_delay={initial['mean_weighted_delay']} (before training)", flush=True)
    for episode in range(t["episodes"]):
        episode_config = deepcopy(config)
        offset = episode
        for stage in config["curriculum"]:
            if offset < stage["episodes"]:
                episode_config["station"]["platform_count"] = stage["platforms"]
                episode_config["scenario"].update(trains=stage["trains"], severity=stage["severity"])
                break
            offset -= stage["episodes"]
        env = TrainTurnEnv(episode_config)
        losses = []
        try:
            obs, info = env.reset(seed=seed + episode)
            while True:
                action = agent.act(obs, info["action_mask"], agent.epsilon)
                next_obs, reward, term, trunc, next_info = env.step(action)
                agent.replay.add(obs, action, reward, next_obs, term, trunc,
                                 next_info["elapsed_minutes"], next_info["action_mask"])
                agent.steps += 1
                update = agent.learn()
                if update is not None:
                    losses.append(update)
                obs, info = next_obs, next_info
                if term or trunc:
                    break
            metrics, _ = episode_metrics(env.sim, c["horizon"])
            history.append({"episode": episode + 1, "seed": seed + episode, "steps": agent.steps,
                            "updates": agent.updates, "epsilon": agent.epsilon, "reward": env.episode_reward,
                            "loss": float(np.mean([x["loss"] for x in losses])) if losses else None,
                            "max_abs_q": max((x["max_abs_q"] for x in losses), default=None), **metrics})
        finally:
            env.close()
        if (episode + 1) % t["validate_every"] == 0 or episode + 1 == t["episodes"]:
            measured = validation_metrics(config, validation_cases, lambda o, m: agent.act(o, m))
            score = measured["mean_weighted_delay"] if measured["complete"] else float("inf")
            validation_history.append({"episode": episode + 1, **measured})
            if score < best:
                best = score
                agent.save(output / "best_model.pt", {"episode": episode + 1, "validation_weighted_delay": best})
            print(f"episode={episode+1} updates={agent.updates} validation_delay={score:.2f} "
                  f"discretionary_holds={measured['mean_discretionary_holds']:.1f}", flush=True)
        write_csv(output / "training.csv", history)
        write_csv(output / "validation.csv", validation_history)
        agent.save(output / "final_model.pt", {"episode": episode + 1, "preliminary": True})
    result = {"episodes": t["episodes"], "optimizer_updates": agent.updates, "elapsed_seconds": time.monotonic() - started,
              "best_validation_delay": best if np.isfinite(best) else None,
              "training_seed": seed, "device": "cuda", "preliminary": True,
              "initial_validation_delay": initial["mean_weighted_delay"],
              "improved_over_initial": bool(initial["complete"] and best < initial["mean_weighted_delay"]),
              "cuda_memory": agent.memory_usage(),
              "warm_start": warm_start, "resume_semantics": "Weights-only warm start; replay and optimizer restart"}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result
