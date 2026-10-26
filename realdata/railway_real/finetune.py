"""Fine-tune the synthetic-trained Double DQN on real DDU days.

The loop mirrors ``railway_rl.training.train`` (same agent, replay, masked Double
DQN targets and greedy validation), but its episodes are real days instead of
generated scenarios:

* training: each training day ``passes`` times, in a seeded shuffled order;
* validation: greedy episodes on the validation days, before training (the
  unchanged synthetic checkpoint) and every ``validate_every`` episodes. The best
  of these, including the unchanged checkpoint, is saved as ``best_model.pt``;
* test days are not touched here.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from railway_rl.agents.double_dqn import DoubleDQN, require_cuda
from railway_rl.environment import TrainTurnEnv
from railway_rl.evaluation.evaluator import run_episode, write_csv
from railway_rl.evaluation.metrics import episode_metrics
from railway_rl.observations import observation_size
from railway_rl.station import Scenario
from railway_rl.timetable import load_scenario
from .ddu import portable


def days_between(available: list[str], span: list[str]) -> list[str]:
    return [d for d in available if span[0] <= d <= span[1]]


def load_days(source: Path, mode: str, days: list[str]) -> dict[str, Scenario]:
    return {d: load_scenario(source / mode / f"{d}.json") for d in days}


def greedy_validation(config: dict, cases: dict[str, Scenario], agent: DoubleDQN) -> dict:
    rows = {d: run_episode(config, case, lambda o, m: agent.act(o, m))[0] for d, case in cases.items()}
    complete = all(r["processed_fraction"] == 1 for r in rows.values())
    return {"mean_weighted_delay": float(np.mean([r["weighted_station_delay"] for r in rows.values()])) if complete else None,
            "mean_discretionary_holds": float(np.mean([r["discretionary_holds"] for r in rows.values()])),
            "safety_violations": int(sum(r["safety_violations"] for r in rows.values())),
            "complete": complete, **{f"delay_{d}": r["weighted_station_delay"] for d, r in rows.items()}}


def finetune(cfg: dict, source: Path, output: Path) -> dict:
    require_cuda()
    f, junction = cfg["finetune"], cfg["junction"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory")
    available = json.loads((source / "manifest.json").read_text())["days"]
    train_days = days_between(available, f["train_days"])
    validation_days = days_between(available, f["validation_days"])
    if not train_days or not validation_days:
        raise ValueError("Empty training or validation split")
    train_cases = load_days(source, f["mode"], train_days)
    validation_cases = load_days(source, f["mode"], validation_days)

    previous = DoubleDQN.load(f["warm_start"])
    if previous.config["environment"] != junction["environment"] or previous.config["reward"] != junction["reward"]:
        raise ValueError("Warm-start checkpoint settings differ from the junction config")
    config = {**previous.config, "agent": {**previous.config["agent"], **f["agent"]}}
    env_cfg = config["environment"]
    agent = DoubleDQN(observation_size(env_cfg["max_platforms"], env_cfg["forecast_count"]),
                      env_cfg["max_platforms"] + 1, config, f["seed"])
    agent.online.load_state_dict(previous.online.state_dict())
    agent.target.load_state_dict(previous.online.state_dict())
    del previous

    rng = np.random.default_rng(f["seed"])
    schedule = [d for _ in range(f["passes"]) for d in rng.permutation(train_days)]
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps({**cfg, "junction": config}, indent=2, default=portable))
    history, validation_history, started = [], [], time.monotonic()
    initial = greedy_validation(config, validation_cases, agent)
    validation_history.append({"episode": 0, **initial})
    best = initial["mean_weighted_delay"] if initial["complete"] else float("inf")
    best_episode = 0
    agent.save(output / "best_model.pt", {"episode": 0, "validation_weighted_delay": best,
                                          "note": "unchanged warm-start weights"})
    print(f"episode=0 validation_delay={best:.2f} holds={initial['mean_discretionary_holds']:.1f} "
          f"(synthetic checkpoint, before fine-tuning)", flush=True)
    for episode, day in enumerate(schedule, start=1):
        env = TrainTurnEnv(config, train_cases[day])
        losses = []
        try:
            obs, info = env.reset(seed=train_cases[day].seed)
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
            metrics, _ = episode_metrics(env.sim, env_cfg["horizon"])
        finally:
            env.close()
        history.append({"episode": episode, "day": day, "steps": agent.steps, "updates": agent.updates,
                        "epsilon": agent.epsilon, "reward": env.episode_reward,
                        "loss": float(np.mean([x["loss"] for x in losses])) if losses else None, **metrics})
        if episode % f["validate_every"] == 0 or episode == len(schedule):
            measured = greedy_validation(config, validation_cases, agent)
            score = measured["mean_weighted_delay"] if measured["complete"] else float("inf")
            validation_history.append({"episode": episode, **measured})
            if score < best:
                best, best_episode = score, episode
                agent.save(output / "best_model.pt", {"episode": episode, "validation_weighted_delay": best})
            print(f"episode={episode} updates={agent.updates} eps={agent.epsilon:.3f} "
                  f"train_delay={metrics['weighted_station_delay']:.1f} validation_delay={score:.2f} "
                  f"holds={measured['mean_discretionary_holds']:.1f} best={best:.2f}@{best_episode}", flush=True)
        write_csv(output / "training.csv", history)
        write_csv(output / "validation.csv", validation_history)
    agent.save(output / "final_model.pt", {"episode": len(schedule), "preliminary": True})
    result = {"episodes": len(schedule), "optimizer_updates": agent.updates,
              "elapsed_seconds": time.monotonic() - started, "mode": f["mode"],
              "train_days": train_days, "validation_days": validation_days, "test_days_untouched": f["test_days"],
              "seed": f["seed"], "agent_overrides": f["agent"],
              "warm_start": portable(f["warm_start"]),
              "warm_start_sha256": hashlib.sha256(Path(f["warm_start"]).read_bytes()).hexdigest(),
              "initial_validation_delay": initial["mean_weighted_delay"],
              "best_validation_delay": best if np.isfinite(best) else None, "best_episode": best_episode,
              "improved_over_warm_start": best_episode > 0, "cuda_memory": agent.memory_usage(),
              "device": "cuda", "resume_semantics": "Weights-only warm start; replay and optimizer restart"}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result
