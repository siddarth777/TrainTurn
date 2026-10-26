"""Headless figures from recorded runs only. Missing experiments are not synthesized."""
from pathlib import Path
import json
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "railway-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

COLORS = {"premium": "#ba3b46", "superfast": "#e6a23c", "express": "#4477aa", "passenger": "#228866", "freight": "#8866aa"}


def generate_plots(results: str | Path, training: str | Path | None = None) -> list[str]:
    root = Path(results)
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = pd.read_csv(root / "episodes.csv")
    trains = pd.read_csv(root / "trains.csv")
    traces = json.loads((root / "occupancy.json").read_text())
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    written = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            # Omit the PDF creation timestamp so figures depend only on the data.
            metadata = {"CreationDate": None} if extension == "pdf" else None
            fig.savefig(path, bbox_inches="tight", metadata=metadata)
            written.append(str(path))
        plt.close(fig)
    policies = manifest["policies"]
    values = [summary["summaries"][f"normal/{p}"]["weighted_station_delay"] for p in policies]
    means = np.array([v["mean"] for v in values])
    errors = np.array([means - [v["ci95_low"] for v in values], [v["ci95_high"] for v in values] - means])
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(policies, means, color=["#ba3b46" if p == "dqn" else "#4477aa" for p in policies], yerr=np.maximum(0, errors), capsize=4)
    ax.set_ylabel("Priority-weighted station waiting (train-min)")
    status = "Preliminary" if summary["preliminary"] else "Held-out"
    ax.set_title(f"{status}: {manifest['episodes']} scenarios; 95% scenario-bootstrap intervals")
    save(fig, "policy_comparison")
    classes = [c for c in COLORS if c in set(trains["category"])]
    grouped = trains[trains["setting"] == "normal"].groupby(["category", "policy"])["station_delay"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    width = 0.8 / len(policies)
    for index, policy in enumerate(policies):
        ax.bar(np.arange(len(classes)) + index * width, [grouped.get((c, policy), np.nan) for c in classes], width, label=policy)
    ax.set_xticks(np.arange(len(classes)) + width * (len(policies) - 1) / 2, classes)
    ax.set_ylabel("Mean station-induced waiting (min/train)")
    station, scenario = manifest["config"]["station"], manifest["config"]["scenario"]
    ax.set_title(f"Per-class delay; synthetic {station['platform_count']}-platform, {scenario['trains']}-train scenarios")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "class_delays")
    comparison = ["fcfs", "dqn"] if "dqn" in policies else ["fcfs", "timetable"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    for ax, policy in zip(axes, comparison):
        for event in traces[f"normal_{policy}"]:
            if event["end"] is None:
                continue
            ax.broken_barh([(event["start"], event["end"] - event["start"])], (event["platform"] - 0.35, 0.7),
                           facecolors=COLORS[event["category"]])
        ax.set_title(f"{policy} | scenario {manifest['seed']}", loc="left")
        n = manifest["config"]["station"]["platform_count"]
        ax.set_yticks(range(1, n + 1));ax.set_ylabel("Platform")
    axes[-1].set_xlabel("Simulation time (min); actual front-entry to rear-clear occupancy")
    axes[0].legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()], ncol=3, fontsize=7)
    save(fig, "occupancy_gantt")
    if episodes["setting"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.2, 3.5))
        settings = list(episodes["setting"].unique())
        for p in policies:
            means = [episodes[(episodes.policy == p) & (episodes.setting == s)].weighted_station_delay.mean() for s in settings]
            ax.plot(settings, means, marker="o", label=p)
        ax.set_ylabel("Priority-weighted station waiting (train-min)")
        ax.set_title("Stress scenarios (paired within each setting)");ax.legend()
        ax.tick_params(axis="x", rotation=20)
        save(fig, "robustness")
    if training is not None:
        path = Path(training) / "validation.csv"
        if not path.exists():
            raise FileNotFoundError(f"No actual validation history: {path}")
        data = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].plot(data.episode, data.mean_reward)
        axes[0].fill_between(data.episode, data.mean_reward - data.std_reward, data.mean_reward + data.std_reward, alpha=0.2)
        axes[0].set_ylabel("Evaluation reward (band: scenario SD)")
        axes[1].plot(data.episode, data.mean_weighted_delay)
        axes[1].set_ylabel("Priority-weighted station waiting")
        for ax in axes:ax.set_xlabel("Training episodes")
        save(fig, "learning_curve")
    return written
