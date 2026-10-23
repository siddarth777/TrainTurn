"""Evaluate junction policies inside the seven-station network on shared scenarios."""
import _bootstrap  # noqa: F401
import argparse
import json
from railway_network.config import load_network_config
from railway_network.evaluation import evaluate_network


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="network YAML (default: multistation/configs/network.yaml)")
    p.add_argument("--checkpoint", help="junction Double DQN checkpoint (CUDA required); omit for baselines only")
    p.add_argument("--episodes", type=int, help="scenarios (default from config: 3)")
    p.add_argument("--seed", type=int, help="first held-out scenario seed (default 200000)")
    p.add_argument("--output", required=True, help="new results directory")
    a = p.parse_args()
    config = load_network_config(a.config)
    result = evaluate_network(config, a.episodes or config["evaluation"]["episodes"],
                              a.seed if a.seed is not None else config["evaluation"]["test_seed"],
                              a.output, a.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
