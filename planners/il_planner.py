from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import numpy as np
import carla

from planners.base import PlannerBase, LowLevelAction

try:
    # 0.9.15 自带
    from agents.navigation.basic_agent import BasicAgent
except Exception as e:
    BasicAgent = None
    _basic_agent_import_error = e


class ILPlanner(PlannerBase):
    """
    极简 IL planner 占位实现：
    - 使用 BasicAgent.run_step() 作为“专家/模仿”低层控制生成器
    - 不开启 autopilot；只在 plan() 时拿控制量。
    - attach_context() 里拿到 world/ego/lane_ref
    """

    def __init__(self, v_ref_kmh: float = 36.0):
        """
        Args:
            v_ref_kmh: 期望速度（km/h），默认约 10 m/s
        """
        self.v_ref_kmh = float(v_ref_kmh)
        self._world: Optional[carla.World] = None
        self._ego: Optional[carla.Vehicle] = None
        self._lane_ref = None
        self._agent: Optional[BasicAgent] = None

    # ---------- 生命周期 ----------
    def attach_context(self, world: carla.World, ego: carla.Vehicle, lane_ref=None):
        if BasicAgent is None:
            raise RuntimeError(f"BasicAgent 导入失败：{_basic_agent_import_error}")

        self._world = world
        self._ego = ego
        self._lane_ref = lane_ref

        # 目标速度 km/h
        self._agent = BasicAgent(self._ego, target_speed=self.v_ref_kmh)

        # 可选：把目标点设定为车辆前方若干米的路点（这里先不强制设置，保持 BasicAgent 内部默认）
        # 若你需要可以在这里用 lane_ref 决定一个短前视目标点：
        # if self._lane_ref is not None:
        #     x, y = ...
        #     self._agent.set_destination(carla.Location(x=x, y=y, z=self._ego.get_location().z))

    def reset(self):
        # BasicAgent 没有必须的 reset，这里留空即可
        pass

    # ---------- 主接口（必须） ----------
    def plan(self, obs, info: Dict[str, Any] = None) -> LowLevelAction:
        """
        输入:
          - obs: 与环境暴露一致（未强依赖）
          - info: 可选
        输出:
          - (throttle, brake, steer)
        """
        assert self._ego is not None and self._agent is not None, \
            "ILPlanner 未 attach_context 或 ego/agent 未就绪"

        # BasicAgent 输出的是 VehicleControl(throttle, brake, steer,...)
        ctrl: carla.VehicleControl = self._agent.run_step()

        # 规范化到通用低层控制三元组
        throttle = float(np.clip(ctrl.throttle, 0.0, 1.0))
        brake    = float(np.clip(ctrl.brake,    0.0, 1.0))
        steer    = float(np.clip(ctrl.steer,   -1.0, 1.0))

        return throttle, brake, steer