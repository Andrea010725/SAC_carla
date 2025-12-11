# planners/rule_planner_adapter.py
from typing import Dict, Any, Tuple, Optional
import numpy as np

# 直接复用你给的实现
import sys
# 从项目内部的 agent_base 目录导入
sys.path.insert(0, "/home/ajifang/SAC_carla/agent_base")
from rule_based_agent import LaneRef, RuleBasedPlanner

LowLevelAction = Tuple[float, float, float]  # (throttle, brake, steer)

class RulePlannerAdapter:
    """
    将你在 agents/rule_based/agent.py 里的 RuleBasedPlanner 封装为一个“可插拔 planner”：
    - 在 attach_context() 里注入 world / ego / ref
    - 在 plan() 里调用 update_corridor_simplified() + compute_control()
    """
    def __init__(self, v_ref_base: float = 12.0):
        self.v_ref_base = float(v_ref_base)
        self._planner: Optional[RuleBasedPlanner] = None
        self._world = None
        self._ego = None
        self._ref: Optional[LaneRef] = None

    def attach_context(self, world, ego, ref: LaneRef):
        """在 episode reset 后由远程环境调用一次"""
        self._world = world
        self._ego = ego
        self._ref = ref
        self._planner = RuleBasedPlanner(ref, v_ref_base=self.v_ref_base)

    def reset(self):
        # 走廊等状态会在下一次 plan() 时自动更新
        if self._planner:
            self._planner.corridor = None

    def plan(self, obs: np.ndarray, info: Dict[str, Any] = None) -> LowLevelAction:
        """
        输入:
          - obs: 9 维观测 (x,y,z,pitch,yaw,roll,acc,ang_v,vel) —— 不直接用，但保留接口一致性
          - info: 可选 (此处不强依赖)
        输出:
          - (throttle, brake, steer)
        """
        assert self._planner is not None and self._world is not None and self._ego is not None and self._ref is not None, \
            "RulePlannerAdapter 未 attach_context()"

        try:
            # 1) 更新走廊（会用到 world/ego/ref）
            self._planner.update_corridor_simplified(
                world=self._world, ego=self._ego,
                s_ahead=30.0,        # 前视距离（米）- 增大可以看到更远
                ds=1.0,              # 纵向采样间隔（米）- 减小更精细但计算量大
                ey_range=8.0,        # 横向搜索范围（米）- 增大可以探索更宽
                dey=0.15,            # 横向采样间隔（米）- 减小更精细但计算量大
                horizon_T=2.0,       # 预测时间范围（秒）
                dt=0.2,              # 预测时间步长（秒）
                debug_draw=True      # 改为True可在CARLA中绘制规划路径
            )
            # 2) 计算控制
            throttle, steer, brake, _dbg = self._planner.compute_control(self._ego, dt=0.05)
            # 3) 与 CarlaEnv.step 的接口一致：返回 (throttle, brake, steer)
            return float(throttle), float(brake), float(steer)
        except Exception as e:
            # 如果优化失败（数值问题、边界约束不可行等），返回一个安全的默认控制
            print(f"[RulePlannerAdapter] 优化失败: {e}, 使用安全默认控制")
            # 🔧 提高默认油门 (原0.3 -> 0.6) - 确保车辆能够前进
            return 0.6, 0.0, 0.0  # (throttle, brake, steer)

