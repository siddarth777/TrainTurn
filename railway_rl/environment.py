"""Event-decision Gymnasium interface over the one-minute simulator."""
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import load_config, validate
from .delays import generate_scenario
from .station import Scenario
from .simulator import Simulator
from .masking import action_mask
from .observations import controller_view, encode, observation_size
from .rewards import minute_reward


class TrainTurnEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(self, config: dict | None = None, scenario: Scenario | None = None,
                 render_mode: str | None = None) -> None:
        self.config = validate(deepcopy(config)) if config is not None else load_config()
        self.fixed_scenario = scenario
        self.render_mode = render_mode
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("Unsupported render mode")
        c = self.config["environment"]
        self.action_space = spaces.Discrete(c["max_platforms"] + 1)
        self.observation_space = spaces.Box(-1, 1, (observation_size(c["max_platforms"], c["forecast_count"]),), np.float32)
        self.sim: Simulator | None = None
        self.ended = True
        self.episode_reward = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario", self.fixed_scenario)
        if scenario is None:
            scenario_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
            scenario = generate_scenario(self.config, scenario_seed)
        if len(scenario.station.platforms) > self.config["environment"]["max_platforms"]:
            raise ValueError("Scenario exceeds observation/action padding")
        if min(t.scheduled_arrival for t in scenario.trains) >= self.config["environment"]["horizon"]:
            raise ValueError("Horizon must extend beyond the first scheduled arrival")
        self.close()
        if self.config["environment"]["backend"] == "sumo":
            from .sumo_simulator import SumoSimulator
            self.sim = SumoSimulator(scenario)
        else:
            self.sim = Simulator(scenario)
        self.ended, self.episode_reward = False, 0.0
        # No controllable queue costs precede the first allocation opportunity.
        while self.sim.current is None and self.sim.time < self.config["environment"]["horizon"]:
            self.sim.tick()
        return self._observation(), self._info(0)

    def action_masks(self) -> np.ndarray:
        if self.sim is None:
            raise RuntimeError("reset() must precede action_masks()")
        c = self.config["environment"]
        return action_mask(self.sim, c["max_platforms"], c["hold_limit"])

    def _observation(self) -> np.ndarray:
        return encode(controller_view(self.sim, self.config["environment"]), self.config["environment"])

    def _info(self, elapsed: int) -> dict:
        return {"action_mask": self.action_masks(), "elapsed_minutes": elapsed,
                "view": controller_view(self.sim, self.config["environment"]),
                "episode_reward": self.episode_reward,
                "unfinished_trains": tuple(t for t, s in self.sim.states.items() if s.status != "departed") if self.ended else ()}

    def step(self, action: int):
        if self.ended or self.sim is None:
            raise RuntimeError("Call reset() before stepping or after an episode ends")
        if not self.action_space.contains(action) or not self.action_masks()[action]:
            raise ValueError(f"Action {action!r} is masked or out of bounds")
        action = int(action)
        mask = self.action_masks()
        current = self.sim.current
        change = action != 0 and action != current.planned_platform
        unnecessary_hold = action == 0 and bool(mask[1:].any())
        if action:
            self.sim.assign(action)
        else:
            self.sim.holds += 1
            if current and self.sim.states[current.id].waiting >= self.config["environment"]["hold_limit"]:
                self.sim.forced_waits += 1
        reward, elapsed, undiscounted = 0.0, 0, 0.0
        c = self.config["environment"]
        while True:
            costs = self.sim.tick()
            r = minute_reward(costs, self.config["reward"], platform_change=change and elapsed == 0,
                              unnecessary_hold=unnecessary_hold and elapsed == 0)
            reward += c["gamma"] ** elapsed * r
            undiscounted += r
            elapsed += 1
            if self.sim.current is not None or self.sim.complete or self.sim.time >= c["horizon"]:
                break
        terminated = self.sim.complete
        truncated = self.sim.time >= c["horizon"] and not terminated
        self.ended = terminated or truncated
        self.episode_reward += undiscounted
        info = self._info(elapsed)
        info["undiscounted_reward"] = undiscounted
        info["forced_wait"] = bool(current and action == 0 and self.sim.states[current.id].waiting >= c["hold_limit"] and not mask[1:].any())
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.sim is None:
            return "Environment not reset"
        text = (f"minute={self.sim.time} platforms={self.sim.occupants} "
                f"outer={self.sim.outer} upstream={self.sim.upstream}")
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        if self.sim is not None and hasattr(self.sim, "close"):
            self.sim.close()
