"""CUDA-only original Double DQN, including masks and elapsed-time targets."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .network import QNetwork
from .replay_buffer import ReplayBuffer


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This project requires GPU-only neural execution; no CPU fallback. "
                           "Check nvidia-smi and the NVIDIA driver, then rerun scripts/check_runtime.py.")
    return torch.device("cuda", torch.cuda.current_device())


@torch.no_grad()
def bellman_targets(rewards: torch.Tensor, terminated: torch.Tensor, elapsed: torch.Tensor,
                    next_masks: torch.Tensor, online_values: torch.Tensor,
                    target_values: torch.Tensor, gamma: float) -> torch.Tensor:
    targets = rewards.clone()
    continuing = ~terminated.bool()
    if torch.any(continuing):
        masks = next_masks[continuing].bool()
        if not torch.all(masks.any(dim=1)):
            raise ValueError("All next actions are masked in a nonterminal transition")
        choices = online_values[continuing].masked_fill(~masks, -torch.inf).argmax(dim=1)
        bootstrap = target_values[continuing].gather(1, choices[:, None]).squeeze(1)
        targets[continuing] += gamma ** elapsed[continuing] * bootstrap
    return targets


class DoubleDQN:
    def __init__(self, observation_dim: int, actions: int, config: dict, seed: int) -> None:
        self.device = require_cuda()
        c = config["agent"]
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        # Limit the caching allocator, not driver/display allocations outside PyTorch.
        self.vram_limit_gib = min(c.get("vram_limit_gib", 5.5), total_memory / 2**30)
        torch.cuda.memory.set_per_process_memory_fraction(self.vram_limit_gib * 2**30 / total_memory,
                                                        self.device)
        self.config, self.seed = config, seed
        self.observation_dim, self.actions = observation_dim, actions
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(c["threads"])
        torch.use_deterministic_algorithms(True)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(observation_dim, actions, c["hidden"]).to(self.device)
        self.target = QNetwork(observation_dim, actions, c["hidden"]).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c["learning_rate"])
        self.replay = ReplayBuffer(c["replay_capacity"], observation_dim, actions, seed + 1)
        self.steps, self.updates = 0, 0

    def memory_usage(self) -> dict[str, float]:
        return {"allocator_limit_gib": self.vram_limit_gib,
                "allocated_gib": torch.cuda.memory_allocated(self.device) / 2**30,
                "reserved_gib": torch.cuda.memory_reserved(self.device) / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30}

    @property
    def epsilon(self) -> float:
        c = self.config["agent"]
        fraction = min(1.0, self.steps / c["epsilon_steps"])
        return c["epsilon_start"] + fraction * (c["epsilon_end"] - c["epsilon_start"])

    @torch.no_grad()
    def act(self, observation: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("No feasible action")
        # Greedy validation must not alter subsequent training exploration.
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        obs = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        values = self.online(obs).squeeze(0)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite Q-values")
        values = values.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
        return int(values.argmax().item())

    def learn(self) -> dict | None:
        c = self.config["agent"]
        if self.replay.size < max(c["warmup"], c["batch_size"]) or self.steps % c["train_every"]:
            return None
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in self.replay.sample(c["batch_size"]).items()}
        with torch.no_grad():
            targets = bellman_targets(batch["rewards"], batch["terminated"], batch["elapsed"], batch["next_masks"],
                                      self.online(batch["next_observations"]), self.target(batch["next_observations"]),
                                      self.config["environment"]["gamma"])
        values = self.online(batch["observations"]).gather(1, batch["actions"][:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(values, targets)
        if not torch.isfinite(loss) or not torch.isfinite(targets).all():
            raise FloatingPointError("Nonfinite loss or Bellman target")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), c["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c["target_update"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.item()), "max_abs_q": float(values.detach().abs().max().item()),
                "gradient_norm": float(norm.item())}

    def save(self, path: str | Path, metadata: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "online": self.online.state_dict(), "target": self.target.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "config": self.config, "seed": self.seed,
                   "observation_dim": self.observation_dim, "actions": self.actions,
                   "steps": self.steps, "updates": self.updates, "metadata": metadata,
                   "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "DoubleDQN":
        device = require_cuda()
        data = torch.load(path, map_location=device, weights_only=True)
        if data["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        agent = cls(data["observation_dim"], data["actions"], data["config"], data["seed"])
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.steps, agent.updates = data["steps"], data["updates"]
        return agent
