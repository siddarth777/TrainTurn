"""Fixed-capacity NumPy ring buffer with an explicitly seeded sampler."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, actions: int, seed: int) -> None:
        if min(capacity, observation_dim, actions) < 1:
            raise ValueError("Replay dimensions must be positive")
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {
            "observations": np.empty((capacity, observation_dim), np.float32),
            "next_observations": np.empty((capacity, observation_dim), np.float32),
            "actions": np.empty(capacity, np.int64), "rewards": np.empty(capacity, np.float32),
            "terminated": np.empty(capacity, bool), "truncated": np.empty(capacity, bool),
            "elapsed": np.empty(capacity, np.float32), "next_masks": np.empty((capacity, actions), bool),
        }

    def add(self, observation, action: int, reward: float, next_observation,
            terminated: bool, truncated: bool, elapsed: float, next_mask) -> None:
        if elapsed <= 0 or not np.isfinite(reward) or not np.isfinite(elapsed):
            raise ValueError("Replay needs finite rewards and positive elapsed time")
        if not 0 <= action < self.arrays["next_masks"].shape[1]:
            raise ValueError("Action outside replay action space")
        if not terminated and not np.any(next_mask):
            raise ValueError("Nonterminal transition requires a feasible next action")
        values = (observation, next_observation, action, reward, terminated, truncated, elapsed, next_mask)
        for key, value in zip(self.arrays, values):
            if np.shape(value) != self.arrays[key].shape[1:]:
                raise ValueError(f"Replay shape mismatch: {key}")
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all():
            raise ValueError("Nonfinite observation")
        for key, value in zip(self.arrays, values):
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 1 <= batch_size <= self.size:
            raise ValueError("Insufficient replay samples")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return {key: value[indices] for key, value in self.arrays.items()}
