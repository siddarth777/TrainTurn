"""Gymnasium environment for the controlled junction inside the seven-station network."""
from railway_rl.environment import TrainTurnEnv
from .scenario import NetworkScenario, generate_network_scenario
from .simulator import NetworkJunctionSimulator


class NetworkTrainTurnEnv(TrainTurnEnv):
    """Same observation (229 features), action space and reward as ``TrainTurnEnv``.

    Only the junction's decisions are exposed. Minor stations follow their fixed
    rule inside the simulator, and network-wide outcomes are measured separately
    (see ``evaluation.network_metrics``); they do not enter the reward.
    """

    def __init__(self, config: dict, case: NetworkScenario | None = None, render_mode: str | None = None) -> None:
        super().__init__(config["junction"], None, render_mode)
        self.network_config = config
        self.fixed_case = case

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super(TrainTurnEnv, self).reset(seed=seed)
        case = (options or {}).get("case", self.fixed_case)
        if case is None:
            case_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
            case = generate_network_scenario(self.network_config, case_seed)
        self.close()
        self.sim = NetworkJunctionSimulator(case, self.network_config)
        self.ended, self.episode_reward = False, 0.0
        while self.sim.current is None and self.sim.time < self.config["environment"]["horizon"]:
            self.sim.tick()
        return self._observation(), self._info(0)
