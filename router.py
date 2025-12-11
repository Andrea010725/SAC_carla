# planners/router.py
import numpy as np
from typing import Dict, Any, Tuple
from planners.base import PlannerBase, LowLevelAction
from .il_planner import ILPlanner
from .rl_planner import RLPlanner
from .rule_planner import RulePlanner

PLANNER_IDS = {
    0: "RULE",
    1: "IL",
    2: "RL",
}

class PlannerRouter:
    """
    将 selector 的连续动作 a∈[-1,1] -> {RULE, IL, RL}，并调度到对应 planner 产生低层控制。
    """
    def __init__(self, config):
        self.config = config
        # 你可以在这里注入真实模型路径
        self.rule = RulePlanner()
        self.il = ILPlanner(model_path=None, scaler_path=None)
        self.rl = RLPlanner(checkpoint=None)

    def reset(self):
        self.rule.reset()
        self.il.reset()
        self.rl.reset()

    def _bucketize(self, a: float) -> int:
        """
        [-1, -1/3) -> 0(RULE)
        [-1/3, 1/3) -> 1(IL)
        [1/3, 1] -> 2(RL)
        """
        a = float(np.clip(a, -1.0, 1.0))
        bins = self.config.planner_bins  # e.g., [-1/3, 1/3]
        if a < bins[0]:
            return 0
        elif a < bins[1]:
            return 1
        else:
            return 2

    def plan_low_level(self, selector_action_scalar: float, obs, info: Dict[str, Any] = None) -> Tuple[LowLevelAction, int, str]:
        pid = self._bucketize(selector_action_scalar)
        if pid == 0:
            control = self.rule.plan(obs, info)
        elif pid == 1:
            control = self.il.plan(obs, info)
        else:
            control = self.rl.plan(obs, info)
        name = PLANNER_IDS[pid]
        return control, pid, name
