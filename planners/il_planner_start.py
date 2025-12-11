"""
热启动：attach_context() 时，把内部目标速度初始化为“上一刻速度”（如果 info 里给了就用 info，否则直接读 ego 的实时速度），绝不掉到 0。

速率限制：每步只允许目标速度按上/下限速率变化（km/h 每秒），避免突变造成急加速/急刹。

上升速率 accel_kmhps，下降速率 decel_kmhps（给一个较小的下降速率可避免刚切换就猛刹）。

随时接收外部参考速度：plan(..., info) 可读 info["ref_speed_kmh"]（或 ref_speed_ms），按速率限制逐步逼近它；如果没给，就用构造时的 v_ref_kmh。

不强制 set_destination（避免一些地图切换时的突发行为），如需可自行开启注释行。

"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import carla

from planners.base import PlannerBase, LowLevelAction

try:
    from agents.navigation.basic_agent import BasicAgent
except Exception as e:
    BasicAgent = None
    _basic_agent_import_error = e


def _speed_ms_to_kmh(v_ms: float) -> float:
    return float(v_ms * 3.6)


def _ego_speed_ms(veh: carla.Vehicle) -> float:
    v = veh.get_velocity()
    return float(np.sqrt(v.x**2 + v.y**2 + v.z**2))


class ILPlanner(PlannerBase):
    """
    IL planner（BasicAgent 驱动）——支持热启动与速率限制：
    - 切换过来时以“当前速度”为初值，不发生掉速到 0 的现象
    - 每步用速率限制把 target_speed 向 ref_speed 平滑跟随
    - 可在 plan(info) 中传入 ref_speed_kmh / ref_speed_ms
    """

    def __init__(
        self,
        v_ref_kmh: float = 36.0,     # 缺省参考速度（没传 info 时使用）
        accel_kmhps: float = 20.0,   # 目标速度上升最大斜率 (km/h 每秒)
        decel_kmhps: float = 10.0,   # 目标速度下降最大斜率 (km/h 每秒) ——小一些避免猛刹
    ):
        self.v_ref_kmh = float(v_ref_kmh)
        self.accel_kmhps = float(accel_kmhps)
        self.decel_kmhps = float(decel_kmhps)

        self._world: Optional[carla.World] = None
        self._ego: Optional[carla.Vehicle] = None
        self._agent: Optional[BasicAgent] = None

        self._dt: float = 0.05  # 从 world settings 读；兜底 20Hz
        self._target_kmh: float = 0.0  # 内部“当前目标速度”，随时间平滑更新
        self._attached: bool = False

    # ---------- 生命周期 ----------
    def attach_context(self, world: carla.World, ego: carla.Vehicle, lane_ref=None):
        if BasicAgent is None:
            raise RuntimeError(f"BasicAgent 导入失败：{_basic_agent_import_error}")

        self._world = world
        self._ego = ego

        # dt 估计
        try:
            s = world.get_settings()
            if s.synchronous_mode and s.fixed_delta_seconds:
                self._dt = float(s.fixed_delta_seconds)
        except Exception:
            pass

        # 以当前车速作为“热启动”的初始目标速度
        cur_ms = _ego_speed_ms(ego)
        self._target_kmh = _speed_ms_to_kmh(cur_ms)

        # 初始化 BasicAgent，直接用“当前目标速度”作为初始 target
        self._agent = BasicAgent(self._ego, target_speed=self._target_kmh)

        # 如需明确一个短前视目的地，放开下面三行（可选）：
        # tf = ego.get_transform()
        # dest = carla.Location(tf.location.x + 30*np.cos(np.deg2rad(tf.rotation.yaw)),
        #                       tf.location.y + 30*np.sin(np.deg2rad(tf.rotation.yaw)),
        #                       tf.location.z)
        # self._agent.set_destination(dest)

        self._attached = True

    def reset(self):
        # 不清 target（避免切换/重置瞬间掉速）；只同步一次当前车速到内部目标更稳妥
        if self._ego is not None:
            self._target_kmh = _speed_ms_to_kmh(_ego_speed_ms(self._ego))
            if self._agent is not None:
                self._agent.set_target_speed(self._target_kmh)

    # ---------- 主接口 ----------
    def plan(self, obs, info: Dict[str, Any] = None) -> LowLevelAction:
        assert self._attached and self._ego is not None and self._agent is not None, \
            "ILPlanner 未 attach_context 或未就绪"

        info = info or {}

        # 1) 读取参考速度（支持外部传入；否则用默认 v_ref_kmh）
        if "ref_speed_kmh" in info:
            ref_kmh = float(info["ref_speed_kmh"])
        elif "ref_speed_ms" in info:
            ref_kmh = _speed_ms_to_kmh(float(info["ref_speed_ms"]))
        else:
            ref_kmh = self.v_ref_kmh

        # 2) 若外部还提供了“上一刻速度”，也可用于修正首次调用时的初值（可选）
        #    仅在 target_kmh 还没被使用过或需要强制对齐时采用
        if "prev_speed_kmh" in info:
            prev_kmh = float(info["prev_speed_kmh"])
            # 如果二者差太大，第一次可直接对齐到 prev，避免刚切换的刹/冲突
            if abs(self._target_kmh - prev_kmh) > 1.0:
                self._target_kmh = prev_kmh
        elif "prev_speed_ms" in info:
            prev_kmh = _speed_ms_to_kmh(float(info["prev_speed_ms"]))
            if abs(self._target_kmh - prev_kmh) > 1.0:
                self._target_kmh = prev_kmh

        # 3) 速率限制更新内部目标速度（slew-rate limiter）
        up_step  = self.accel_kmhps * self._dt
        dn_step  = self.decel_kmhps * self._dt

        if ref_kmh > self._target_kmh:
            self._target_kmh = min(self._target_kmh + up_step, ref_kmh)
        else:
            self._target_kmh = max(self._target_kmh - dn_step, ref_kmh)

        # 将目标速度下发给 BasicAgent
        self._agent.set_target_speed(self._target_kmh)

        # 4) 求控制量
        ctrl: carla.VehicleControl = self._agent.run_step()

        throttle = float(np.clip(ctrl.throttle, 0.0, 1.0))
        brake    = float(np.clip(ctrl.brake,    0.0, 1.0))
        steer    = float(np.clip(ctrl.steer,   -1.0, 1.0))

        return throttle, brake, steer
