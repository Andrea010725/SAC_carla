# planners/router_carla.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from rule_planner_adapter import RulePlannerAdapter
# 如果你的 ILPlanner 不在同级目录，请把下面这行改成实际路径：
# 例如: from agents.il_based.il_planner import ILPlanner
import sys
sys.path.append("/home/ajifang/czw/SAC_Carla/planners")
from planners.il_planner import ILPlanner  # <- 改为从本工程 planners 包导入

# ✅ 新增: 导入PPO适配器
from planners.rl_ppo_adapter import RLPPOAdapter


LowLevelAction = Tuple[float, float, float]  # (throttle, brake, steer)
PLANNER_IDS = {0: "RULE", 1: "IL", 2: "RL"}


class PlannerRouterCarla:
    """
    路由器：
      - reset() 之后由 env_wrapper 在每个远程进程内调用 attach_context(world, ego, lane_ref)
      - 训练循环中每步调用 plan_low_level(a_scalar, obs, info) 获取低层控制
    """

    def __init__(self, planner_bins = (-1/3, 1/3), v_ref_base: float = 12.0,
                 use_ppo: bool = True, ppo_model_path: str = None, ppo_action_dim: int = 2):
        """
        Args:
            planner_bins: 两个分界点 (b1, b2)，把 a∈[-1,1] 分成三段：
                          a < b1 -> RULE,  b1 <= a < b2 -> IL,  a >= b2 -> RL
            v_ref_base:   规则/IL 的目标速度（m/s）；IL 内部会转成 km/h
            use_ppo:      是否使用PPO作为RL planner (默认True)
            ppo_model_path: PPO模型路径 (可选,无模型时使用默认控制)
            ppo_action_dim: PPO动作维度 (2或3)
        """
        self.bins = tuple(planner_bins)
        assert len(self.bins) == 2 and self.bins[0] < self.bins[1], "planner_bins 必须是递增的两个分界点"

        # 规则 planner（你现有的适配器）
        self.rule = RulePlannerAdapter(v_ref_base=v_ref_base)

        # IL planner：当前用 BasicAgent 生成低层控制，不接管车辆
        self.il = ILPlanner(v_ref_kmh=v_ref_base * 3.6)

        # ✅ RL planner: 使用PPO适配器
        if use_ppo:
            print(f"[Router] 🎯 使用PPO作为RL planner (action_dim={ppo_action_dim})")
            self.rl = RLPPOAdapter(
                model_path=ppo_model_path,
                use_ppo_with_trd=True,
                action_dim=ppo_action_dim
            )
            self._use_ppo = True
        else:
            print("[Router] 使用IL作为RL planner (占位符)")
            self.rl = self.il
            self._use_ppo = False

        self._ctx_attached = False

    # ---- 生命周期 ----
    def attach_context(self, world, ego, lane_ref):
        """
        注意：**不要**加 dt 参数；env_wrapper 就是按这个签名调用的。
        把上下文传给各个 planner（谁需要谁用）。
        """
        # 规则
        try:
            self.rule.attach_context(world, ego, lane_ref)
        except Exception:
            pass

        # IL
        try:
            self.il.attach_context(world, ego, lane_ref)
        except Exception:
            pass

        # RL（使用PPO或IL）
        try:
            if self.rl is not self.il:  # 如果使用PPO
                self.rl.attach_context(world, ego, lane_ref)
        except Exception as e:
            print(f"[Router] ⚠️ RL planner attach_context失败: {e}")

        self._ctx_attached = True

    def reset(self):
        for p in (self.rule, self.il, self.rl):
            if hasattr(p, "reset"):
                try:
                    p.reset()
                except Exception:
                    pass

    # ---- 内部：把标量动作分桶 ----
    def _bucketize(self, a: float) -> int:
        a = float(np.clip(a, -1.0, 1.0))
        b1, b2 = self.bins
        if a < b1:
            return 0  # RULE
        elif a < b2:
            return 1  # IL
        else:
            return 2  # RL

    # ---- 主接口：把选择器输出路由到某个 planner，产出低层控制 ----
    def plan_low_level(self, selector_action_scalar: float, obs: np.ndarray,
                       info: Dict[str, Any] = None) -> Tuple[LowLevelAction, int, str]:
        assert self._ctx_attached, "Router 未 attach_context()"
        pid = self._bucketize(selector_action_scalar)

        if pid == 0:
            control = self.rule.plan(obs, info or {})
        elif pid == 1:
            control = self.il.plan(obs, info or {})
        else:
            # ✅ 使用PPO planner
            control = self.rl.plan(obs, info or {})

        return control, pid, PLANNER_IDS[pid]
