# env_wrapper.py

import sys
import ray
import carla
import numpy as np
import gym

from carla_base.carla_env import CarlaEnv

# ===== 路由器 & LaneRef 导入 =====
from router_carla import PlannerRouterCarla  # 你的 Router（选择/融合 IL / RL / Rule）
# 导入 LaneRef from agent_base
sys.path.insert(0, "/home/ajifang/SAC_carla/agent_base")
try:
    from rule_based_agent import LaneRef  # 从 agent_base 导入 LaneRef
except Exception as e:
    print(f"[WARNING] LaneRef import failed: {e}")
    LaneRef = None  # 若不存在，也能先跑（Rule-based 会降级）


# ========== 工具：稳健构建 LaneRef ==========
def _build_safe_lane_ref(amap: carla.Map,
                         seed_wp: carla.Waypoint,
                         step: float = 1.0,
                         max_len: float = 500.0):
    """
    稳健构建 LaneRef：保证至少 3 个点；不跨 lane；失败则多策略兜底。
    返回 (LaneRef实例 or None, errmsg or None)
    """
    if LaneRef is None:
        return None, "LaneRef class not available (import failed)"

    if seed_wp is None:
        return None, "seed waypoint is None"

    def try_once(st: float):
        try:
            ref = LaneRef(amap, seed_wp=seed_wp, step=st, max_len=max_len)
            # 最少 3 个采样点（P 长度 >= 3）
            P = getattr(ref, "P", None)
            if P is None or len(P) < 3:
                return None, f"insufficient points ({0 if P is None else len(P)})"
            return ref, None
        except Exception as e:
            return None, str(e)

    # 先用默认 step
    ref, err = try_once(step)
    if ref is not None:
        return ref, None

    # 缩小步长重试
    for st in (0.5, 0.25):
        ref2, err2 = try_once(st)
        if ref2 is not None:
            return ref2, None
        err = err2  # 记录最后一次错误

    return None, err or "unknown error while building LaneRef"


# ========== 并行环境封装 ==========
class ParallelEnv(object):
    def __init__(self, config):
        # 干净重启 Ray
        ray.shutdown()
        ray.init()

        # 创建远程环境列表（每个使用不同的 Carla/TM 端口）
        self.env_list = [
            CarlaRemoteEnv.remote(config=config,
                                  carla_port=carla_port,
                                  tm_port=config.carla_tm_ports[index])
            for index, carla_port in enumerate(config.carla_ports)
        ]

        self.env_num = config.num_parallel_envs
        self.episode_reward_list = [0.0] * self.env_num
        self.episode_steps_list = [0] * self.env_num
        self._max_episode_steps = config.max_episode_steps
        self.total_steps = 0

        # 调试统计：记录 planner 选择次数
        self.planner_stats = {"RULE": 0, "IL": 0, "RL": 0}

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self):
        obs_list = [env.reset.remote() for env in self.env_list]
        obs_list = [ray.get(obs) for obs in obs_list]
        self.obs_list = np.array(obs_list)

        # 初始化占位，避免 get_obs 首次访问空属性
        self.reward_list = np.zeros(self.env_num, dtype=float)
        self.done_list = np.zeros(self.env_num, dtype=bool)
        self.next_obs_list = self.obs_list.copy()
        self.info_list = np.array([{} for _ in range(self.env_num)], dtype=object)
        return self.obs_list

    def step(self, action_list):
        """
        action_list:
          - 当 config.planner_selection_mode=True 时，每个元素为标量/shape=(1,) 的 a∈[-1,1]
          - 否则，每个元素 shape=(2,) 为原直接控制动作
        """
        fut_list = [self.env_list[i].step.remote(action_list[i]) for i in range(self.env_num)]
        return_list = ray.get(fut_list)  # list(tuple(next_obs, reward, done, info))
        return_list = np.array(return_list, dtype=object)

        self.next_obs_list = return_list[:, 0]  # list[np.ndarray]
        self.reward_list = return_list[:, 1].astype(float)
        self.done_list = return_list[:, 2].astype(bool)
        self.info_list = return_list[:, 3]

        # 调试：统计并打印 planner 选择
        for info in self.info_list:
            if isinstance(info, dict) and "planner_name" in info:
                pname = info["planner_name"]
                if pname in self.planner_stats:
                    self.planner_stats[pname] += 1

        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        """
        维护累计步数/回合统计，并在回合结束时自动 reset 远程环境。
        """
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += float(self.reward_list[i])

            self.obs_list[i] = self.next_obs_list[i]

            if self.done_list[i] or self.episode_steps_list[i] >= self._max_episode_steps:
                # 回合结束 -> 重置统计
                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0.0
                # 远程 reset
                obs_i = ray.get(self.env_list[i].reset.remote())
                self.obs_list[i] = np.array(obs_i)
        return self.obs_list


# ========== 远程单环境（Ray Actor） ==========
@ray.remote
class CarlaRemoteEnv(object):
    """
    远程封装单个 CARLA 环境：
    - 与原实现兼容；
    - 新增：reset() 中构建 LaneRef，并 attach 到 PlannerRouterCarla；
           step() 中若启用 planner_selection_mode，则执行“选择→低层控制→映射→env.step()”链路。
    """

    def __init__(self, config, carla_port, tm_port):
        class ActionSpace(object):
            def __init__(self, action_space=None, low=None, high=None, shape=None, n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n
                # ……其它保持不变……
                from copy import deepcopy

                cfg = deepcopy(config)
                # 训练用的远程环境一律关闭渲染与HUD（相机/pygame）
                cfg.render = False
                cfg.planner_mode = "RL"  # 这里只是显示用途，训练无所谓
                # 记录一个标记，让 CarlaEnv 如果需要也能知道自己在 Ray 里
                cfg.under_ray = True

                self.config = cfg
                self.env = CarlaEnv(cfg, carla_port, tm_port)
                # ……后续保持不变……

            def sample(self):
                return self.action_space.sample()

        self.config = config
        self.env = CarlaEnv(config, carla_port, tm_port)

        # 记录动作边界（用于“旧模式”的线性映射；你的 action_space 本身是[-1,1] 也无妨）
        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]

        self._max_episode_steps = self.config.max_episode_steps
        self.action_space = ActionSpace(self.env.action_space,
                                        self.env.action_space.low,
                                        self.env.action_space.high,
                                        self.env.action_space.shape)

        # NEW: 远程进程内创建 Router
        bins = getattr(config, "planner_bins", (-1 / 3, 1 / 3))
        self._router = PlannerRouterCarla(planner_bins=bins, v_ref_base=12.0)
        self._router_ctx_ready = False  # 是否已 attach_context

        self._last_obs = None
        self._lane_ref = None  # 保存 LaneRef 以便规则规划使用

        import inspect, carla_base
        print("[DEBUG] carla_base loaded from:", getattr(carla_base, "__file__", "<unknown>"))
        print("[DEBUG] CarlaEnv defined in:", inspect.getsourcefile(CarlaEnv))
        print("[DEBUG] CarlaEnv module:", CarlaEnv.__module__)
        assert hasattr(self.env, "action_space"), "[FATAL] CarlaEnv has no action_space (wrong import or early crash?)"
        print("[DEBUG] action_space type:", type(self.env.action_space))

    # ---------- 构建 LaneRef（基于当前 EGO） ----------
    def _build_lane_ref_from_current_pose(self):
        world = self.env.world
        amap = world.get_map()
        ego = getattr(self.env, "ego", None)
        if ego is None:
            ego = getattr(self.env, "vehicle", None)
        if ego is None:
            raise RuntimeError("CarlaRemoteEnv: ego actor 不可用，无法构建 LaneRef")

        ego_wp = amap.get_waypoint(ego.get_location(),
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            raise RuntimeError("CarlaRemoteEnv: 无法在 ego 位置找到 Driving waypoint")

        ref, err = _build_safe_lane_ref(amap, ego_wp, step=1.0, max_len=500.0)
        if ref is None:
            raise RuntimeError(err or "LaneRef 构建失败")
        return ref

    def _try_attach_router_context(self):
        """给 Router 注入 world/ego/ref/dt，上下文缺失时抛异常。"""
        world = self.env.world
        ego = getattr(self.env, "ego", None) or getattr(self.env, "vehicle", None)
        if ego is None:
            raise RuntimeError("ego 不存在（尚未 spawn？）")

        amap = world.get_map()
        # ref 可能已经有了；没有则现建（允许失败时降级）
        if self._lane_ref is None and LaneRef is not None:
            try:
                self._lane_ref = self._build_lane_ref_from_current_pose()
            except Exception as e:
                print(f"[CarlaRemoteEnv] LaneRef 构建失败：{e}")
                self._lane_ref = None  # 允许先无参考线运行（Rule-based 会降级）

        # attach
        self._router.attach_context(
            world=world,
            ego=ego,
            lane_ref=self._lane_ref

        )
        self._router.reset()
        self._router_ctx_ready = True

    # ---------- 外部接口 ----------
    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs
        self._router_ctx_ready = False  # 每次 reset 后重新 attach（地图/ego可能变化）

        # 尝试 attach（失败不终止；在 step() 再兜底一次）
        try:
            self._try_attach_router_context()
        except Exception as e:
            print(f"[CarlaRemoteEnv] attach_context 失败（延后到 step 再试）：{e}")

        return obs

    def step(self, model_output_act: np.ndarray):
        """
        Args:
            model_output_act:
              - planner_selection_mode=True: 标量/shape=(1,) a∈[-1,1]（选择器输出）
              - planner_selection_mode=False: shape=(2,) 旧模式（直接控制）
        Returns:
            next_obs, reward, done, info
        """
        # 输入范围校验
        arr = np.asarray(model_output_act)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        assert np.all((arr <= 1.0 + 1e-3) & (arr >= -1.0 - 1e-3)), \
            'the action should be in range [-1.0, 1.0]'

        if getattr(self.config, "planner_selection_mode", False):
            # 选择器模式：a_scalar -> Router -> 低层控制
            # 兜底：若还没 attach，这里再尝试一次，避免断言
            if not self._router_ctx_ready:
                try:
                    self._try_attach_router_context()
                except Exception as e:
                    print(f"[CarlaRemoteEnv] late attach_context 失败：{e}")
                    # === 安全兜底（不走旧模式线性映射，以免 shape 错）===
                    # 把选择器标量 a∈[-1,1] 当作方向盘，给一个小油门让车动起来
                    a_scalar = float(arr[0])
                    throttle_brake = 0.2  # 小正油门
                    steer = float(np.clip(a_scalar, -1.0, 1.0))
                    safe_action = np.array([throttle_brake, steer], dtype=np.float32)
                    next_obs, reward, done, info = self.env.step(safe_action)
                    self._last_obs = next_obs
                    return next_obs, reward, done, info

            a_scalar = float(arr[0])
            obs_now = self._last_obs if self._last_obs is not None else self.env._get_state_obs()

            # Router 返回可为 VehicleControl 或 (throttle, brake, steer)
            control, pid, pname = self._router.plan_low_level(a_scalar, obs_now, info={})

            if hasattr(control, "throttle") and hasattr(control, "brake") and hasattr(control, "steer"):
                throttle = float(np.clip(control.throttle, 0.0, 1.0))
                brake = float(np.clip(control.brake, 0.0, 1.0))
                steer = float(np.clip(control.steer, -1.0, 1.0))
            else:
                # 假定三元组
                throttle, brake, steer = control
                throttle = float(np.clip(throttle, 0.0, 1.0))
                brake = float(np.clip(brake, 0.0, 1.0))
                steer = float(np.clip(steer, -1.0, 1.0))

            # 合成为 env 的 [throttle_brake, steer]
            throttle_brake = throttle if brake <= 1e-6 else -brake
            mapped_action = np.array([throttle_brake, steer], dtype=np.float32)

            next_obs, reward, done, info = self.env.step(mapped_action)
            info = info or {}
            info.update({"planner_id": pid, "planner_name": pname})

            # 更新 pygame 显示的 planner 名
            if hasattr(self.env, 'set_planner_mode'):
                self.env.set_planner_mode(pname)

        else:
            # ====== 旧模式：支持 2维 or 3维（第3维 y_ref 只透传）======
            if arr.size == 1:
                # 若误传标量，扩展成 [throttle_brake, steer] = [0.0, a]
                arr = np.array([0.0, float(arr[0])], dtype=np.float32)

            # ✅ 允许 2维 或 3维
            if arr.size >= 3:
                y_ref = float(arr[2])
            else:
                y_ref = 0.0

            # ✅ 只取前两维用于控制
            arr_ctrl = arr[:2].astype(np.float32)

            # 线性映射原样保留（你 CarlaEnv 本身 action_space 是[-1,1]，这步等价于不变）
            mapped_action = self.low_bound + (arr_ctrl - (-1.0)) * (
                    (self.high_bound - self.low_bound) / 2.0
            )
            mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)

            next_obs, reward, done, info = self.env.step(mapped_action)
            info = info or {}
            info["y_ref"] = y_ref

        self._last_obs = next_obs
        return next_obs, reward, done, info


# ========== 本地评估环境 ==========
class LocalEnv(object):
    def __init__(self, config):
        """
        本地单进程环境（评测用）。保持原有逻辑；
        如需在本地也跑“选择器模式”，可仿照 CarlaRemoteEnv.step 的实现改造。
        """
        self.env = CarlaEnv(config, config.eval_carla_port, config.eval_carla_tm_port)
        self._max_episode_steps = config.max_episode_steps
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_dim = 3

    def reset(self):
        return self.env.reset()

    def step(self, model_output_act):
        arr = np.asarray(model_output_act, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        assert np.all((arr <= 1.0 + 1e-3) & (arr >= -1.0 - 1e-3)), \
            'the action should be in range [-1.0, 1.0]'

        # ✅ y_ref 透传
        y_ref = float(arr[2]) if arr.size >= 3 else 0.0

        # ✅ 控制只用前两维
        if arr.size == 1:
            arr_ctrl = np.array([0.0, float(arr[0])], dtype=np.float32)
        else:
            arr_ctrl = arr[:2].astype(np.float32)

        mapped_action = self.low_bound + (arr_ctrl - (-1.0)) * (
                (self.high_bound - self.low_bound) / 2.0
        )
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)

        obs, reward, done, info = self.env.step(mapped_action)
        info = info or {}
        info["y_ref"] = y_ref
        return obs, reward, done, info
