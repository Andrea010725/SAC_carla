# SAC_Carla/planners/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

# 统一低层控制：(throttle, brake, steer)
LowLevelAction = Tuple[float, float, float]

class PlannerBase(ABC):
    """所有 planner 的统一接口"""

    def attach_context(self, world, ego, lane_ref=None):
        """可选：在 reset 后注入 CARLA 上下文（world/ego/参考线等）"""
        pass

    def reset(self):
        """可选：episode 开始时清理内部状态"""
        pass

    @abstractmethod
    def plan(self, obs, info: Dict[str, Any] = None) -> LowLevelAction:
        """返回低层控制 (throttle, brake, steer)"""
        raise NotImplementedError
