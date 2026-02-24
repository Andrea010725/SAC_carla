"""
单独训练PPO Agent - 带Wandb监控
（修正版：稳定reward分解 + 不篡改动作 + ✅W&B step对齐 + ✅episode横轴
 + ✅y_ref 映射版/不映射版开关）
"""
import sys
import os
import time

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 添加当前项目路径
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 添加 planners 目录
planners_path = os.path.join(script_dir, "planners")
if planners_path not in sys.path:
    sys.path.insert(0, planners_path)

# 添加 CARLA PythonAPI 路径（用于导入agents模块和carla模块）
carla_api_path = "/home/ajifang/carla/PythonAPI/carla/"
carla_egg_path = "/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
if carla_api_path not in sys.path:
    sys.path.insert(0, carla_api_path)
if carla_egg_path not in sys.path:
    sys.path.insert(0, carla_egg_path)

import gym
import numpy as np

from config import Config
from carla_base.carla_env import CarlaEnv
from training_logger import TrainingLogger

# Wandb导入（优雅处理）
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ Wandb已导入")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb未安装，将跳过在线监控")
    print("   安装命令: pip install wandb")


# ============================================================
# ✅ 观测归一化类（修复P0级问题：缺少观测归一化）
# ============================================================
class RunningMeanStd:
    """在线计算均值和标准差（Welford算法）

    用于归一化观测，解决不同维度尺度差异巨大的问题：
    - x, y坐标: ±500米
    - yaw: ±180度
    - speed: 0-10 m/s
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        """更新统计量（Welford在线算法）"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        """归一化观测到均值0、标准差1"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# ============================================================
# ✅ 训练课程策略（Curriculum Learning）
# 目标：先学“能走、少撞”，再逐步增加场景难度
# ============================================================
def apply_curriculum(episode: int, env: "CarlaGymEnv", succ_rate: float = 0.0, window_ready: bool = False):
    cfg = env.config

    # ✅ 成功率门控课程（优先于固定轮数）
    # 规则：窗口未就绪时，固定在 Stage1
    if not window_ready:
        stage = 1
    else:
        s1_thr = float(getattr(cfg, "stage1_success_threshold", 0.70))
        s2_thr = float(getattr(cfg, "stage2_success_threshold", 0.85))
        if succ_rate < s1_thr:
            stage = 1
        elif succ_rate < s2_thr:
            stage = 2
        else:
            stage = 3

    # 避免长期卡在低阶段：到达指定episode后强制进入更高阶段
    force_s2_ep = int(getattr(cfg, "force_stage2_episode", 80))
    force_s3_ep = int(getattr(cfg, "force_stage3_episode", 160))
    if episode >= force_s3_ep:
        stage = max(stage, 3)
    elif episode >= force_s2_ep:
        stage = max(stage, 2)

    # --- Stage 1: 先练换道/让行基础（不含 jaywalker） ---
    if stage == 1:
        cfg.scenario_pool = ["cones", "trimma"]
        cfg.scenario_weights = {
            "cones": 0.55,
            "trimma": 0.45,
        }
        cfg.cone_num = 6
        # 先关闭 y_ref，减少横纵耦合难度
        cfg.use_yref_in_steer = False
        cfg.yref_steer_gain = 0.0

    # --- Stage 2: 加入施工变道，仍不含 jaywalker ---
    elif stage == 2:
        cfg.scenario_pool = ["cones", "trimma", "construction_lane_change"]
        cfg.scenario_weights = {
            "cones": 0.40,
            "trimma": 0.35,
            "construction_lane_change": 0.25,
        }
        cfg.cone_num = 8
        cfg.use_yref_in_steer = False
        cfg.yref_steer_gain = 0.0

    # --- Stage 3: 加入 jaywalker（低权重，目标是停车避让） ---
    else:
        cfg.scenario_pool = ["cones", "trimma", "construction_lane_change", "jaywalker"]
        cfg.scenario_weights = {
            "cones": 0.30,
            "trimma": 0.25,
            "construction_lane_change": 0.25,
            "jaywalker": 0.20,
        }
        cfg.cone_num = 8
        # ✅ Stage3 是否开启 y_ref（做成开关，默认关闭更稳）
        stage3_use_yref = bool(getattr(cfg, "stage3_use_yref", False))
        if stage3_use_yref:
            cfg.use_yref_in_steer = True
            warm_end = int(getattr(cfg, "stage3_yref_warmup_end", 450))
            gain_warm = float(getattr(cfg, "stage3_yref_gain_warmup", 0.01))
            gain_full = float(getattr(cfg, "stage3_yref_gain_full", 0.03))
            cfg.yref_steer_gain = gain_warm if episode <= warm_end else gain_full
        else:
            cfg.use_yref_in_steer = False
            cfg.yref_steer_gain = 0.0


# ============================================================
# ✅ 熵系数调度：先探索，后收敛
# ============================================================
def update_entropy_coeff(episode: int, agent, start: float = 0.02, end: float = 0.005, decay_ep: int = 600):
    if not hasattr(agent, "entropy_strength"):
        return
    frac = min(1.0, float(episode) / float(decay_ep))
    coeff = start + (end - start) * frac
    if hasattr(agent.entropy_strength, "value"):
        agent.entropy_strength.value = float(coeff)

class CarlaGymEnv(gym.Env):
    """
    CARLA Gym wrapper for PPO

    ✅ 关键点（与你的 CarlaEnv.step 对齐）：
    - PPO 输出 3维: [throttle_brake, steer_raw, y_ref] in [-1,1]^3
    - CarlaEnv.step() 也吃 3维，并在内部用开关决定：
        steer = steer_raw (+ gain * y_ref)  或  steer = steer_raw
    - wrapper 不再做 steer 映射/裁2维，只做：
        1) action 安全处理
        2) (可选) 纵向 bias/forced throttle
        3) (可选) y_ref 幅度惩罚（加到 reward + reward分解）
        4) 日志对齐（W&B/TrainingLogger）
    """

    def __init__(self, config, logger=None, wandb_run=None):
        super().__init__()
        self.config = config
        # ✅ 端口使用配置，避免硬编码导致多实例/端口冲突
        carla_port = int(getattr(config, "carla_port", 2000))
        tm_port = int(getattr(config, "carla_tm_port", 8000))
        self.carla_port = carla_port
        self.carla_tm_port = tm_port
        self.carla_env = CarlaEnv(config, carla_port, tm_port)
        self.logger = logger
        self.wandb_run = wandb_run
        self.reset_retry_times = int(getattr(config, "reset_retry_times", 2))
        self.reset_rebuild_sleep_sec = float(getattr(config, "reset_rebuild_sleep_sec", 1.0))
        self.rebuild_env_retry_times = int(getattr(config, "rebuild_env_retry_times", 15))
        self.rebuild_env_sleep_sec = float(getattr(config, "rebuild_env_sleep_sec", 2.0))

        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        # )
        # ✅ action 维度开关（2维/3维）
        # 说明：PPO当前网络固定输出3维，这里用开关仅用于“屏蔽y_ref”
        self.use_action_dim2 = bool(getattr(config, "use_action_dim2", False))
        # ✅ wrapper 直接复用底层 env 的 space，避免维度写死
        self.observation_space = self.carla_env.observation_space
        self.action_space = self.carla_env.action_space

        self.max_episode_steps = int(getattr(config, "max_episode_steps", 512))

        # ===== ✅ y_ref 开关交给 CarlaEnv（字段名兼容旧/新配置）=====
        # 优先使用 use_yref_in_steer / yref_steer_gain，若没有则兼容旧字段
        self._sync_runtime_switches_to_carla_env()

        # y_ref 惩罚：wrapper层加（也可以以后下放到 CarlaEnv）
        self.yref_penalty = float(getattr(config, "yref_penalty", 0.0))

        # 饱和率监控阈值（仅用于日志）
        self.log_steer_saturation = bool(getattr(config, "log_steer_saturation", True))
        self.steer_sat_threshold = float(getattr(config, "steer_sat_threshold", 0.999))

        # ===== 可选：action bias / forced throttle（默认关闭）=====
        self.current_episode = 0
        self.use_action_bias = bool(getattr(config, "use_action_bias", False))
        self.use_forced_throttle = bool(getattr(config, "use_forced_throttle", False))
        self.action_bias_strength = float(getattr(config, "action_bias_strength", 0.7))
        self.action_bias_decay = float(getattr(config, "action_bias_decay", 0.999))

        self.episode_collision = False
        self.global_env_step = 0
        self.current_episode = 0
        self._step_count = 0
        self.step_reward_components = {}

        # reward分解（每step）
        # self.step_reward_components = {
        #     "base_reward": 0.0,
        #     "r_wp": 0.0,
        #     "r_progress": 0.0,
        #     "r_speed": 0.0,
        #     "r_lane": 0.0,
        #     "r_offroad": 0.0,
        #     "r_smooth": 0.0,
        #     "r_mag": 0.0,
        #     "r_collision": 0.0,
        #     "r_idle": 0.0,
        #     "r_yref": 0.0,   # wrapper额外项
        # }

        self.wandb_step_log_interval = int(getattr(config, "wandb_step_log_interval", 10))

        # ✅ 观测归一化（修复P0级问题）
        self.obs_normalizer = RunningMeanStd(shape=(self.observation_space.shape[0],))
        self.normalize_obs = True  # 可以通过config控制
        self.obs_clip = 10.0  # 归一化后裁剪到[-10, 10]

        print("\n[DEBUG] CarlaEnv y_ref switch after wrapper init:")
        print("  carla_env.use_yref_in_steer =", getattr(self.carla_env, "use_yref_in_steer", None))
        print("  ✅ 观测归一化已启用: normalize_obs =", self.normalize_obs)
        print("  carla_env.yref_steer_gain   =", getattr(self.carla_env, "yref_steer_gain", None))

    @staticmethod
    def _is_timeout_error(err: Exception) -> bool:
        msg = str(err)
        return ("time-out of" in msg) or ("TimeoutException" in msg) or ("simulator" in msg)

    def _sync_runtime_switches_to_carla_env(self):
        use_yref_flag = getattr(self.config, "use_yref_in_steer", getattr(self.config, "use_yref_mapping", False))
        yref_gain = getattr(self.config, "yref_steer_gain", getattr(self.config, "yref_gain", 0.0))
        self.carla_env.use_yref_in_steer = bool(use_yref_flag)
        self.carla_env.yref_steer_gain = float(yref_gain)

    def _rebuild_carla_env(self, reason: str):
        print(f"[CarlaGymEnv] ⚠️ 重建 CarlaEnv: {reason}")
        try:
            self.carla_env.close()
        except Exception:
            pass
        last_err = None
        max_retries = max(0, int(self.rebuild_env_retry_times))
        for attempt in range(max_retries + 1):
            try:
                self.carla_env = CarlaEnv(self.config, self.carla_port, self.carla_tm_port)
                self._sync_runtime_switches_to_carla_env()
                self.observation_space = self.carla_env.observation_space
                self.action_space = self.carla_env.action_space
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    print(
                        f"[CarlaGymEnv] ⚠️ CarlaEnv重建失败，等待重试 {attempt + 1}/{max_retries}: {e}"
                    )
                    time.sleep(max(0.0, float(self.rebuild_env_sleep_sec)))
                    continue
                break
        raise RuntimeError(f"[CarlaGymEnv] 重建 CarlaEnv 最终失败: {last_err}")

    def reset(self):
        obs = None
        reset_tries = max(0, int(self.reset_retry_times))
        for attempt in range(reset_tries + 1):
            try:
                obs = self.carla_env.reset()
                break
            except Exception as e:
                if (attempt < reset_tries) and self._is_timeout_error(e):
                    print(f"[GymReset] ⚠️ reset超时，重试 {attempt + 1}/{reset_tries}: {e}")
                    self._rebuild_carla_env(reason="reset timeout")
                    time.sleep(max(0.0, float(self.reset_rebuild_sleep_sec)))
                    continue
                raise

        if obs is None:
            raise RuntimeError("[GymReset] reset failed with no observation.")

        self.current_episode += 1
        self._step_count = 0
        self.episode_collision = False

        # ✅ 归一化观测
        obs = np.asarray(obs, dtype=np.float32)
        if self.normalize_obs:
            self.obs_normalizer.update(obs.reshape(1, -1))
            obs = self.obs_normalizer.normalize(obs)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        return obs

    def step(self, action):
        step_id = int(self.global_env_step)

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # ✅ 动作维度兼容：use_action_dim2 时强制 y_ref=0（屏蔽第三维）
        if action.size < 3:
            raise ValueError(f"action must have 3 dims [a0,a1,a2], got shape={action.shape}")

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"⚠️ NaN/Inf action: {action}, fallback to [0.3,0.0,0.0]")
            action = np.array([0.3, 0.0, 0.0], dtype=np.float32)

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a0 = float(action[0])
        a1 = float(action[1])
        a2 = 0.0 if self.use_action_dim2 else float(action[2])

        # ----- bias / forced throttle -----
        applied_bias = 0.0
        forced_throttle = False
        a0_applied = a0

        if self.use_action_bias and self.current_episode < 500:
            applied_bias = self.action_bias_strength * (self.action_bias_decay ** self.current_episode)
            a0_applied = float(np.clip(a0_applied + applied_bias, -1.0, 1.0))

        if self.use_forced_throttle and (-0.1 <= a0_applied <= 0.1):
            a0_applied = 0.2
            forced_throttle = True

        action3 = np.array([a0_applied, a1, a2], dtype=np.float32)
        try:
            obs, reward, done, info = self.carla_env.step(action3)
        except Exception as e:
            if not self._is_timeout_error(e):
                raise
            print(f"[GymStep] ⚠️ step超时，结束当前episode并重建环境: {e}")
            self._rebuild_carla_env(reason="step timeout")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -120.0
            done = True
            info = {
                "done_reason": "sim_timeout",
                "timeout": 1.0,
                "collision": 0.0,
                "speed": 0.0,
            }
        if info is None:
            info = {}

        # 四场景动态逻辑：确保 jaywalker 等场景在训练中真实触发
        if hasattr(self.carla_env, "scenario_instance") and self.carla_env.scenario_instance:
            scenario = self.carla_env.scenario_instance

            if hasattr(scenario, "check_and_trigger"):
                try:
                    ego_loc = self.carla_env.ego.get_location()
                    scenario.check_and_trigger(ego_loc)
                except Exception as e:
                    print(f"⚠️ check_and_trigger failed: {e}")

            if hasattr(scenario, "tick_update"):
                try:
                    scenario.tick_update()
                except Exception as e:
                    print(f"⚠️ tick_update failed: {e}")

        # ✅ 归一化观测
        obs = np.asarray(obs, dtype=np.float32)
        if self.normalize_obs:
            self.obs_normalizer.update(obs.reshape(1, -1))
            obs = self.obs_normalizer.normalize(obs)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        # ----- y_ref penalty -----
        r_yref = 0.0
        if self.yref_penalty > 0.0:
            r_yref = -float(self.yref_penalty * abs(a2))
            reward = float(reward) + r_yref

        # ----- reward components -----
        comps = {}
        if isinstance(getattr(self.carla_env, "last_reward_components", None), dict):
            comps = self.carla_env.last_reward_components.copy()
        comps["r_yref"] = float(r_yref)
        self.step_reward_components = comps

        # ----- info -----
        info.update({
            "wrapped_a0_raw": float(a0),
            "wrapped_a0_applied": float(a0_applied),
            "wrapped_applied_bias": float(applied_bias),
            "wrapped_forced_throttle": int(forced_throttle),
            "wrapper_yref_penalty": float(self.yref_penalty),
            "env_step": float(step_id),
        })

        # ----- logger -----
        if self.logger is not None:
            speed = float(info.get("speed", 0.0))
            collision = bool(info.get("collision", 0.0))
            if collision:
                self.episode_collision = True
            self.logger.log_step(speed=speed, collision=collision)

        # ----- W&B log -----
        if self.wandb_run is not None and (step_id % max(1, self.wandb_step_log_interval) == 0):
            speed = float(info.get("speed", 0.0))
            steer_raw = float(info.get("raw_steer", a1))
            y_ref = float(info.get("raw_y_ref", a2))
            steer_applied = float(info.get("applied_steer", steer_raw))

            yref_used = int(info.get("yref_used", 1))   #  0
            delta_total = float(steer_applied - steer_raw)
            delta_from_yref = float(info.get("steer_delta_from_yref", 0.0)) if yref_used else 0.0

            steer_saturated = 0
            if self.log_steer_saturation:
                steer_saturated = int(abs(steer_applied) >= self.steer_sat_threshold)

            step_log = {
                "env_step": int(step_id),
                "step/reward": float(reward),
                "step/speed": float(speed),

                "step/action_throttle_brake_raw": float(a0),
                "step/action_throttle_brake_applied": float(a0_applied),
                "step/action_steer_raw": float(steer_raw),
                "step/y_ref": float(y_ref),

                "step/action_steer_applied": float(steer_applied),
                "debug/steer_postprocess_delta": float(delta_total),
                "debug/steer_delta_from_yref": float(delta_from_yref),
                "debug/steer_saturated": int(steer_saturated),

                "step/applied_bias": float(applied_bias),
                "step/forced_throttle": int(forced_throttle),

                "debug/yref_used": float(yref_used),
                "debug/yref_steer_gain": float(info.get("yref_steer_gain",
                                                        getattr(self.carla_env, "yref_steer_gain", 0.0))),
            }

            sum_components = float(sum(self.step_reward_components.values())) if self.step_reward_components else 0.0
            total_reward = float(reward)
            step_log.update({
                "debug_reward/sum_components": sum_components,
                "debug_reward/total_reward": total_reward,
                "debug_reward/components_diff": float(total_reward - sum_components),
            })

            for k, v in self.step_reward_components.items():
                step_log[f"debug_reward/{k}"] = float(v)

            self.wandb_run.log(step_log, step=int(step_id), commit=True)

        self._step_count += 1
        self.global_env_step += 1
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def close(self):
        if hasattr(self, "carla_env") and self.carla_env is not None:
            self.carla_env.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode="human"):
        pass

def train_with_logging(agent, env, logger, wandb_run=None, episodes=1500, timesteps=512, save_every=100):
    """
    带日志记录和Wandb监控的PPO训练循环 + ✅自动早停/收敛判定
    """
    import time
    import os
    import numpy as np
    from collections import deque
    from planners.rl_agent_only import utils

    assert episodes % agent.update_frequency == 0
    agent.memory = agent.get_memory()

    policy_loss_tracker = 0.0
    value_loss_tracker = 0.0
    entropy_tracker = 0.0
    update_count = 0

    # ============================================================
    # ✅ Early Stop / Convergence 规则（可通过 config.enable_early_stop 关闭）
    # ============================================================
    enable_early_stop = bool(getattr(getattr(env, "config", None), "enable_early_stop", True))
    W = 20  # 滑动窗口长度（建议 20~50）
    MIN_EP = 30  # 至少跑到这个 episode 才开始早停判断

    # ---- 收敛判据（满足就停）----
    TARGET_SUCCESS = 0.80       # success率 >= 0.80
    TARGET_RAN_FULL = 0.80      # ran_full率 >= 0.80
    MAX_COLLISION = 0.10        # collision率 <= 0.10
    NEED_STABLE_WINDOWS = 3     # 连续多少个窗口满足才算“稳定收敛”

    # ---- 平台期判据（长时间没提升就停）----
    PLATEAU_PATIENCE = 8        # 连续多少个窗口没提升就停（8个窗口≈8次检查）
    IMPROVE_EPS = 0.01          # success_rate 提升小于该值视为“没提升”

    # ---- 崩溃判据（明显坏掉就停）----
    BAD_PATIENCE = 3            # 连续 BAD_PATIENCE 个窗口崩 → 停
    BAD_COLLISION = 0.80        # collision率 > 0.80 认为崩
    BAD_RAN_FULL = 0.10         # ran_full率 < 0.10 认为崩

    # 统计窗口
    dq_success = deque(maxlen=W)
    dq_ran_full = deque(maxlen=W)
    dq_no_collision = deque(maxlen=W)
    dq_avg_rps = deque(maxlen=W)  # 可选：avg_reward_per_step
    scenario_window = deque(maxlen=max(200, W * 10))

    stable_good_count = 0
    bad_count = 0

    best_succ_rate = -1.0
    best_succ_episode = 0
    plateau_count = 0

    def _mean(dq):
        return float(sum(dq) / max(1, len(dq)))

    for episode in range(1, episodes + 1):
        done_reason = "unknown"
        last_info = {}  # ✅ 保存本 episode 最后一步 info，避免作用域/未定义问题

        episode_std_mean_sum = 0.0
        episode_std_mean_count = 0

        logger.start_episode(episode)

        # ============================================================
        # ✅ 课程学习：动态调整场景 + y_ref 难度（success_rate 门控）
        # ============================================================
        try:
            succ_rate_prev = _mean(dq_success) if len(dq_success) > 0 else 0.0
            window_ready = (len(dq_success) == W and episode >= MIN_EP)
            apply_curriculum(episode, env, succ_rate=succ_rate_prev, window_ready=window_ready)
        except Exception as e:
            print(f"[Curriculum] ⚠️ apply_curriculum failed: {e}")

        # ============================================================
        # ✅ 熵系数调度：先探索，后收敛
        # ============================================================
        try:
            ent_start = float(getattr(env.config, "entropy_start", 0.02))
            ent_end = float(getattr(env.config, "entropy_end", 0.005))
            ent_decay_ep = int(getattr(env.config, "entropy_decay_ep", 600))
            update_entropy_coeff(episode, agent, start=ent_start, end=ent_end, decay_ep=ent_decay_ep)
        except Exception as e:
            print(f"[EntropySchedule] ⚠️ update_entropy_coeff failed: {e}")

        agent.seed_regularization()
        agent.on_episode_start()
        agent.reset()

        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        t0 = time.time()

        episode_speed_sum = 0.0
        episode_collision_count = 0
        episode_reward_components = {}
        episode_duration = 0.0

        # ---------------------- rollout ----------------------
        for t in range(1, timesteps + 1):
            if isinstance(state, dict):
                state = {f"state_{k}": v for k, v in state.items()}
            state_t = utils.to_tensor(state)

            action, mean, std, log_prob, value = agent.predict(state_t)

            # ===== rollout randomness proxy (NO distribution assumption) =====
            std_np = std.numpy() if hasattr(std, "numpy") else np.asarray(std)
            std_np = np.asarray(std_np).reshape(-1)

            # 仅记录 std 的均值作为“随机性 proxy”，不叫 entropy
            episode_std_mean_sum += float(np.mean(std_np)) if std_np.size > 0 else 0.0
            episode_std_mean_count += 1

            # ===== W&B debug std（按 env_step 对齐）=====
            if wandb_run is not None:
                step_id = int(env.global_env_step)
                if (step_id % max(1, env.wandb_step_log_interval)) == 0:
                    std_np2 = std.numpy() if hasattr(std, "numpy") else np.asarray(std)
                    std_np2 = np.asarray(std_np2).reshape(-1)
                    std_th = float(std_np2[0]) if std_np2.size > 0 else 0.0
                    std_st = float(std_np2[1]) if std_np2.size > 1 else 0.0
                    std_yr = float(std_np2[2]) if std_np2.size > 2 else 0.0
                    std_lon_mean = std_th
                    std_lat_mean = float(np.mean([std_st, std_yr]))

                    wandb_run.log(
                        {
                            "env_step": int(step_id),
                            "debug/std_lon_mean": std_lon_mean,
                            "debug/std_lat_mean": std_lat_mean,
                            "debug/std_throttle": std_th,
                            "debug/std_steer": std_st,
                            "debug/std_y_ref": std_yr,
                        },
                        step=step_id,
                        commit=False,
                    )

            action_env = agent.convert_action(action)
            try:
                next_state, reward, done, info = env.step(action_env)
            except Exception as e:
                msg = str(e)
                timeout_like = ("time-out of" in msg) or ("TimeoutException" in msg) or ("simulator" in msg)
                if timeout_like:
                    print(f"[EnvStep] ⚠️ simulator timeout at episode={episode}, t={t}: {msg}")
                    next_state = state
                    reward = -80.0
                    done = True
                    info = {
                        "done_reason": "sim_timeout",
                        "timeout": 1.0,
                        "collision": 0.0,
                        "speed": 0.0,
                    }
                else:
                    raise
            last_info = info or {}

            episode_reward += float(reward)
            episode_steps = t

            # ✅ 累积 reward 分解（第一次step后初始化keys）
            if (not episode_reward_components) and getattr(env, "step_reward_components", None):
                episode_reward_components = {k: 0.0 for k in env.step_reward_components.keys()}

            for k in episode_reward_components.keys():
                episode_reward_components[k] += float(env.step_reward_components.get(k, 0.0))

            # 速度统计：优先使用 info 中的真实速度（obs 已被归一化）
            episode_speed_sum += float(last_info.get("speed", 0.0))

            # 碰撞统计
            if last_info.get("collision", 0.0):
                episode_collision_count += 1

            # 存经验
            a = action.numpy().reshape(-1)
            m = mean.numpy().reshape(-1)
            s = std.numpy().reshape(-1)

            def _pad3(x):
                x = np.asarray(x).reshape(-1)
                if x.size >= 3:
                    return x[:3]
                if x.size == 2:
                    return np.array([x[0], x[1], 0.0], dtype=np.float32)
                if x.size == 1:
                    return np.array([x[0], 0.0, 0.0], dtype=np.float32)
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

            a3 = _pad3(a)
            m3 = _pad3(m)
            s3 = _pad3(s)

            agent.log(
                a_th=a3[0], a_steer=a3[1], a_yref=a3[2],
                std_th=s3[0], std_steer=s3[1], std_yref=s3[2],
                mean_th=m3[0], mean_steer=m3[1], mean_yref=m3[2],
                reward=float(reward),
            )

            agent.memory.append(state_t, action, reward, value, log_prob)

            state = next_state

            if done or (t == timesteps):
                episode_duration = time.time() - t0
                done_reason = str(last_info.get("done_reason", "unknown"))

                print(
                    f"Episode {episode} terminated after {t} timesteps in {round(episode_duration, 3)}s "
                    f"with reward {round(episode_reward, 3)}."
                )
                agent.log(timestep=t)

                if isinstance(state, dict):
                    state = {f"state_{k}": v for k, v in state.items()}
                state_tt = utils.to_tensor(state)

                last_value = agent.network.predict_last_value(
                    state_tt, timestep=(t + 1) / timesteps, is_terminal=done
                )
                agent.end_episode(last_value, append=agent.update_frequency > 1)
                break

        # ---------------------- episode stats ----------------------
        avg_speed = episode_speed_sum / episode_steps if episode_steps > 0 else 0.0
        logger.log_episode_reward(episode_reward)

        # ---------------------- update ----------------------
        if episode % agent.update_frequency == 0:
            temp_policy_loss, temp_value_loss, temp_entropy = [], [], []

            value_batches = agent.get_value_batches()
            policy_batches = agent.get_policy_batches()

            for _ in range(agent.optimization_steps["policy"]):
                for data_batch in policy_batches:
                    agent.seed_regularization()
                    total_loss, policy_grads = agent.get_policy_gradients(data_batch)

                    policy_loss_val = float(total_loss[0].numpy()) if isinstance(total_loss, tuple) else float(total_loss.numpy())
                    if not (np.isnan(policy_loss_val) or np.isinf(policy_loss_val)):
                        temp_policy_loss.append(policy_loss_val)

                    if hasattr(agent, "_logs") and "entropy" in agent._logs:
                        entropy_val = agent._logs["entropy"]
                        if isinstance(entropy_val, (list, tuple)):
                            entropy_val = entropy_val[-1] if entropy_val else 0.0
                        if not (np.isnan(entropy_val) or np.isinf(entropy_val)):
                            temp_entropy.append(float(entropy_val))

                    agent.update_policy(policy_grads)

            for _ in range(agent.optimization_steps["value"]):
                for data_batch in value_batches:
                    agent.seed_regularization()
                    value_loss, value_grads = agent.get_value_gradients(data_batch)
                    value_loss_val = float(value_loss.numpy())
                    if not (np.isnan(value_loss_val) or np.isinf(value_loss_val)):
                        temp_value_loss.append(value_loss_val)
                    agent.update_value(value_grads)

            if temp_policy_loss:
                policy_loss_tracker = sum(temp_policy_loss) / len(temp_policy_loss)
            if temp_value_loss:
                value_loss_tracker = sum(temp_value_loss) / len(temp_value_loss)
            if temp_entropy:
                entropy_tracker = sum(temp_entropy) / len(temp_entropy)
            else:
                # ✅ fallback：用 rollout std proxy 代替，避免日志一直 0
                if episode_std_mean_count > 0:
                    entropy_tracker = float(episode_std_mean_sum / episode_std_mean_count)

            agent.memory.delete()
            agent.memory = agent.get_memory()
            update_count += 1

        elif agent.update_frequency > 1:
            agent.memory.rewards = agent.memory.rewards[:-1]
            agent.memory.values = agent.memory.values[:-1]
            # ✅ 重要：去掉 dummy 后同步 index，避免下一回合切片错位
            if hasattr(agent.memory, "index"):
                agent.memory.index = max(0, agent.memory.index - 1)

        logger.log_training_metrics(
            policy_loss=policy_loss_tracker,
            value_loss=value_loss_tracker,
            entropy=entropy_tracker
        )
        logger.end_episode()

        # ---------------------- success / ran_full / no_collision ----------------------
        timeout_flag = bool(float(last_info.get("timeout", 0.0)) > 0.5)
        ran_full = bool(timeout_flag or (episode_steps >= timesteps))

        bad_reasons = {"collision", "offroad", "no_progress"}  # ✅ 不把 time_limit 当失败
        no_collision = bool(episode_collision_count == 0)
        scenario_name = str(getattr(env.carla_env, "scenario", "")).lower()
        is_jaywalker = ("jaywalker" in scenario_name)
        success_speed_min = float(
            getattr(env.config, "success_min_avg_speed_jaywalker", 0.9)
            if is_jaywalker else
            getattr(env.config, "success_min_avg_speed_lane", 1.9)
        )
        speed_ok = bool(avg_speed >= success_speed_min)
        success = int(ran_full and no_collision and (done_reason not in bad_reasons) and speed_ok)
        scenario_window.append((scenario_name, int(success), int(no_collision), int(ran_full)))

        # ============================================================
        # ✅ Early Stop 统计（每个episode都更新一次）
        # ============================================================
        dq_success.append(int(success))
        dq_ran_full.append(int(ran_full))
        dq_no_collision.append(int(no_collision))
        dq_avg_rps.append(float(episode_reward / max(1, episode_steps)))

        succ_rate = _mean(dq_success)
        ran_full_rate = _mean(dq_ran_full)
        no_collision_rate = _mean(dq_no_collision)
        collision_rate = 1.0 - no_collision_rate

        # ---- 收敛判据：连续满足 NEED_STABLE_WINDOWS 次 ----
        if (episode >= MIN_EP and len(dq_success) == W and
            succ_rate >= TARGET_SUCCESS and
            ran_full_rate >= TARGET_RAN_FULL and
            collision_rate <= MAX_COLLISION):
            stable_good_count += 1
        else:
            stable_good_count = 0

        # ---- 崩溃判据：连续 BAD_PATIENCE 次 ----
        if (episode >= MIN_EP and len(dq_success) == W and
            (collision_rate > BAD_COLLISION or ran_full_rate < BAD_RAN_FULL)):
            bad_count += 1
        else:
            bad_count = 0

        # ---- 平台期判据：success_rate 长期没提升 ----
        if episode >= MIN_EP and len(dq_success) == W:
            if succ_rate > best_succ_rate + IMPROVE_EPS:
                best_succ_rate = succ_rate
                best_succ_episode = episode
                plateau_count = 0
            else:
                plateau_count += 1

        # ---------------------- W&B episode log ----------------------
        if wandb_run is not None:
            def _scenario_metric(metric_index: int, name: str) -> float:
                values = [item[metric_index] for item in scenario_window if item[0] == name]
                if not values:
                    return -1.0
                return float(sum(values) / len(values))

            avg_reward_components = {
                f"reward/{k}": (v / episode_steps if episode_steps > 0 else 0.0)
                for k, v in episode_reward_components.items()
            }

            wandb_metrics = {
                "episode_num": int(episode),
                "episode/ran_full": int(ran_full),
                "episode/no_collision": int(no_collision),
                "episode/success": int(success),
                "episode/done_reason": done_reason,
                "episode/speed_ok_for_success": int(speed_ok),
                "episode/success_speed_min": float(success_speed_min),
                "episode/scenario_is_jaywalker": int(is_jaywalker),

                "episode/total_reward": float(episode_reward),
                "episode/avg_reward_per_step": float(episode_reward / max(1, episode_steps)),
                "episode/steps": int(episode_steps),
                "episode/duration": float(episode_duration),
                "episode/avg_speed": float(avg_speed),
                "episode/collision_count": int(episode_collision_count),

                "training/policy_loss": float(policy_loss_tracker),
                "training/value_loss": float(value_loss_tracker),
                "training/entropy": float(entropy_tracker),
                "training/update_count": int(update_count),
                "training/std_rollout_mean": float(episode_std_mean_sum / max(1, episode_std_mean_count)),

                # ✅ early-stop 观测指标（非常建议记录，方便你看是不是在收敛）
                "early/success_rate_W": float(succ_rate),
                "early/ran_full_rate_W": float(ran_full_rate),
                "early/collision_rate_W": float(collision_rate),
                "early/stable_good_count": int(stable_good_count),
                "early/bad_count": int(bad_count),
                "early/plateau_count": int(plateau_count),
                "early/best_succ_rate": float(best_succ_rate),
                "early/best_succ_episode": int(best_succ_episode),

                "scenario_success/cones": _scenario_metric(1, "cones"),
                "scenario_success/trimma": _scenario_metric(1, "trimma"),
                "scenario_success/construction_lane_change": _scenario_metric(1, "construction_lane_change"),
                "scenario_success/jaywalker": _scenario_metric(1, "jaywalker"),
            }
            wandb_metrics.update(avg_reward_components)

            last_step = int(env.global_env_step) - 1
            wandb_metrics["env_step"] = last_step
            wandb_run.log(wandb_metrics, step=last_step, commit=True)

        # ---------------------- summaries / record ----------------------
        agent.log(episode_rewards=episode_reward)
        agent.write_summaries()

        if agent.should_record:
            agent.record(episode)

        agent.on_episode_end()

        # ---------------------- periodic save ----------------------
        if episode % save_every == 0:
            agent.save()
            if wandb_run is not None:
                checkpoint_path = os.path.join(agent.base_path, "policy_net.index")
                if os.path.exists(checkpoint_path):
                    wandb_run.save(os.path.join(agent.base_path, "policy_net*"))
                    wandb_run.save(os.path.join(agent.base_path, "value_net*"))

        if episode % 10 == 0:
            logger.print_summary(window=10)

        # ============================================================
        # ✅ Early Stop 触发：收敛 / 平台期 / 崩溃
        # ============================================================
        if enable_early_stop and episode >= MIN_EP and len(dq_success) == W:
            if stable_good_count >= NEED_STABLE_WINDOWS:
                print(
                    f"\n✅ EarlyStop: CONVERGED. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}\n"
                )
                agent.save()
                break

            if bad_count >= BAD_PATIENCE:
                print(
                    f"\n�� EarlyStop: DIVERGED/BAD. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}\n"
                )
                agent.save()
                break

            if plateau_count >= PLATEAU_PATIENCE:
                print(
                    f"\n�� EarlyStop: PLATEAU. "
                    f"best_succ_rate={best_succ_rate:.3f} at ep {best_succ_episode}, "
                    f"current succ_rate(W)={succ_rate:.3f}\n"
                )
                agent.save()
                break


def train_ppo():
    print("=" * 70)
    print("PPO单独训练 - 带Wandb实时监控（修正版：✅W&B step对齐 + ✅episode横轴 + ✅y_ref开关）")
    print("=" * 70)

    # =========================
    # ✅ 可视化诊断短跑开关
    # 说明：用于快速观察碰撞发生位置/时刻
    # =========================
    debug_vis = False
    debug_vis_episodes = 10
    debug_vis_timesteps = 256

    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project="SAC-CARLA-PPO",
                name=f"ppo-standalone-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "algorithm": "PPO",
                    "env": "CARLA",
                    "scenario": "4scenarios_lane_yield_plus_jaywalker",
                    "scenario_pool": ["cones", "trimma", "construction_lane_change", "jaywalker"],
                    # ✅ P3: 修正超参数配置，与实际agent参数一致
                    "policy_lr": 1e-4,           # 修正：从5e-5改为1e-4
                    "value_lr": 2e-4,            # 修正：从1e-4改为2e-4
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "clip_ratio": 0.2,           # 修正：与agent一致
                    "entropy_reg": 0.02,
                    "entropy_start": 0.02,
                    "entropy_end": 0.005,
                    "entropy_decay_ep": 600,
                    "batch_size": 128,           # 修正：从256改为128
                    "update_frequency": 2,       # 新增：与agent一致
                    "optimization_steps": (4, 4),
                    "trd_loss_coef": 0.0,
                    "episodes": 1500,
                    "max_steps_per_episode": 256,
                    "wandb_step_log_interval": 50,

                    # 不篡改动作：保持 False（开启会破坏严格PPO假设）
                    "use_action_bias": False,
                    "use_forced_throttle": False,
                    "action_bias_strength": 0.7,
                    "action_bias_decay": 0.999,

                    # ✅ 两版本开关
                    # 兼容旧字段 + 新字段（训练中会用 curriculum 动态开关）
                    "use_yref_mapping": False,   # 兼容旧字段
                    "use_yref_in_steer": False,  # 新字段
                    "yref_gain": 0.0,
                    "yref_steer_gain": 0.0,
                    "yref_penalty": 0.0,
                    "obs_control_lat_tol": 2.8,
                    "low_speed_brake_cut_speed": 0.8,
                    "low_speed_throttle_floor_speed": 1.0,
                    "low_speed_throttle_floor": 0.18,
                    "min_throttle_when_stuck": 0.28,

                    "log_steer_saturation": True,
                    "steer_sat_threshold": 0.999,

                },
                tags=["ppo", "carla", "standalone", "4scenarios", "lane-yield", "jaywalker-stop"],
                notes="PPO standalone training with reward decomposition monitoring"
            )
            print("\n✅ Wandb已初始化")
            print(f"   Run名称: {wandb_run.name}")
            print(f"   Run URL: {wandb_run.url}")

            # 指标横轴
            wandb.define_metric("env_step")
            wandb.define_metric("episode_num")

            wandb.define_metric("step/*", step_metric="env_step")
            wandb.define_metric("debug/*", step_metric="env_step")
            wandb.define_metric("debug_reward/*", step_metric="env_step")

            wandb.define_metric("episode/*", step_metric="episode_num")
            wandb.define_metric("reward/*", step_metric="episode_num")
            wandb.define_metric("training/*", step_metric="episode_num")
            wandb.define_metric("scenario_success/*", step_metric="episode_num")

        except Exception as e:
            print(f"\n⚠️  Wandb初始化失败: {e}")
            wandb_run = None

    config = Config()

    # ✅ 端口显式设置：保持与旧脚本一致（原来固定 2000/8000）
    # 如果你的 TM/Server 端口不同，直接改这里即可
    config.carla_port = 2000
    config.carla_tm_port = 8000
    config.tm_port = config.carla_tm_port

    # ✅ 启用随机场景训练
    config.random_scenario = True  # 开启随机场景
    # ✅ 只保留当前 ScenarioFactory 真正支持的场景
    # pedestrian_crossing 目前未注册，会走 fallback 导致“无障碍”场景，容易误判收敛
    config.scenario_pool = [
        "cones",
        "trimma",
        "construction_lane_change",
        "jaywalker",             # 默认低权重
    ]
    # ✅ 四场景权重：前三个偏“换道让行”，jaywalker 低权重偏“停车避让”
    config.scenario_weights = {
        "cones": 0.30,
        "trimma": 0.25,
        "construction_lane_change": 0.25,
        "jaywalker": 0.20,
    }

    # ✅ y_ref 新字段默认值（训练中会被 curriculum 动态覆盖）
    config.use_yref_in_steer = False
    config.yref_steer_gain = 0.0
    # 兼容旧字段
    config.use_yref_mapping = False
    config.yref_gain = 0.0

    # ✅ 训练阶段关闭可视化（减少CARLA渲染负载，降低断连/内存占用）
    # 评估阶段再打开 render + spectator_mode 即可
    config.render = False
    config.spectator_mode = "none"

    # ✅ 训练阶段关闭调试绘制（debug线段也会增加server负载）
    config.enable_debug_drawing = False
    config.debug_draw_interval = 50  # 即使打开也降低频率
    config.draw_detection_range = False
    config.draw_ego_direction = False
    config.draw_obstacle_boxes = False
    config.draw_lane_center = False
    # ✅ 奖励缩放/裁剪（稳定 value loss）
    config.reward_scale = 1.0
    config.reward_clip = 0.0

    # ✅ 低速参数回到保守区间，避免“低速强推+刹车被削弱”
    config.low_speed_throttle_floor_speed = 1.1
    config.low_speed_throttle_floor = 0.16
    config.low_speed_brake_cut_speed = 0.8
    config.min_throttle_when_stuck = 0.26
    config.obs_control_lat_tol = 2.8

    # ✅ 速度保护与近障碍限油（防爆冲）
    config.speed_governor_speed = 8.0
    config.speed_governor_brake_gain = 0.25
    config.obs_throttle_cap_dist = 10.0
    config.obs_throttle_cap = 0.38
    config.obs_brake_dist = 6.0
    config.obs_brake_value = 0.4
    config.obs_brake_hard_dist = 3.0
    config.obs_brake_hard_value = 0.6
    # reset/cleanup 阶段降低阻塞风险
    config.cleanup_post_ticks = 1

    # 控制侧场景分档：非 jaywalker 更积极，jaywalker 更保守
    config.obs_throttle_cap_dist_lane = 8.5
    config.obs_throttle_cap_lane = 0.42
    config.obs_throttle_cap_dist_jaywalker = 12.0
    config.obs_throttle_cap_jaywalker = 0.25
    config.obs_control_lat_tol_lane = 3.0
    config.obs_control_lat_tol_jaywalker = 3.0
    # obstacle brake shield 分场景：lane 更少“过度刹停”，jaywalker 继续保守
    config.obs_brake_dist_lane = 5.8
    config.obs_brake_value_lane = 0.38
    config.obs_brake_hard_dist_lane = 2.8
    config.obs_brake_hard_value_lane = 0.62
    config.obs_brake_dist_jaywalker = 7.0
    config.obs_brake_value_jaywalker = 0.45
    config.obs_brake_hard_dist_jaywalker = 3.2
    config.obs_brake_hard_value_jaywalker = 0.70
    config.low_speed_brake_cut_speed_lane = 0.9
    config.low_speed_brake_cut_speed_jaywalker = 0.5
    config.low_speed_throttle_floor_speed_lane = 1.2
    config.low_speed_throttle_floor_speed_jaywalker = 0.7
    config.low_speed_throttle_floor_lane = 0.20
    config.low_speed_throttle_floor_jaywalker = 0.10
    config.min_throttle_when_stuck_lane = 0.28
    config.min_throttle_when_stuck_jaywalker = 0.16

    # ✅ 场景感知速度策略（非 jaywalker 更积极，jaywalker 更保守）
    config.target_speed_lane = 5.2
    config.target_speed_jaywalker = 4.0
    config.v_max_lane = 6.8
    config.v_max_jaywalker = 5.5
    config.overspeed_start_lane = 5.1
    config.overspeed_start_jaywalker = 4.5
    config.v_cap_near_lane = 3.8
    config.v_cap_near_jaywalker = 2.2
    config.k_speed_lane = 1.35
    config.k_speed_jaywalker = 1.1
    config.k_overspeed_lane = 0.20
    config.k_overspeed_jaywalker = 0.15

    config.low_speed_th_lane = 2.0
    config.low_speed_th_jaywalker = 0.6
    config.k_low_speed_lane = 0.30
    config.k_low_speed_jaywalker = 0.04

    config.success_bonus_lane = 120.0
    config.success_bonus_jaywalker = 150.0
    config.success_speed_th_lane = 2.4
    config.success_speed_th_jaywalker = 1.0
    config.success_prog_th_lane = 0.05
    config.success_prog_th_jaywalker = 0.02

    # 训练侧 success 判据的最低平均速度门槛（用于课程/早停）
    config.success_min_avg_speed_lane = 2.0
    config.success_min_avg_speed_jaywalker = 0.9
    config.progress_full_speed_lane = 3.0
    config.progress_min_gate_lane = 0.05
    config.clear_dist_lane = 24.0
    config.clear_speed_target_lane = 2.3
    config.k_slow_clear_lane = 0.24
    # reward 与控制的障碍横向门槛统一到同一量级，降低“奖励判危险但控制不刹”的冲突
    config.obs_reward_lat_tol_lane = 3.2
    config.obs_reward_lat_tol_jaywalker = 3.0
    config.obs_reward_gate_floor_lane = 0.05
    config.obs_reward_gate_floor_jaywalker = 0.20
    config.k_avoid_lat_lane = 0.22
    config.k_avoid_lat_jaywalker = 0.45
    # construction 场景负载控制（防 timeout）
    config.construction_num_cones = 8
    config.construction_cone_interval = 3
    config.construction_num_garbage = 10
    config.construction_num_workers = 1
    config.construction_debug_scan = False
    config.construction_setup_stabilize_ticks = 2
    config.traffic_density = 1.5
    config.flow_range = 50.0
    config.front_speed_diff_pct = -20.0
    config.side_speed_diff_pct = 30.0
    # simulator重启恢复参数：给 CARLA 充分拉起时间，避免一次崩溃带崩整训练
    config.reset_retry_times = 8
    config.reset_rebuild_sleep_sec = 2.0
    config.rebuild_env_retry_times = 20
    config.rebuild_env_sleep_sec = 3.0
    # ✅ 障碍物门控调试（大量打印会拖慢训练，默认关闭）
    config.debug_obstacle_gate = False
    config.wandb_step_log_interval = 50
    config.observations_type = "state_lane_obstacles"
    config.obs_obstacle_k = 5
    config.obs_obstacle_range = 50.0

    # ✅ 成功率门控课程阈值
    config.stage1_success_threshold = 0.65
    config.stage2_success_threshold = 0.82
    config.force_stage2_episode = 80
    config.force_stage3_episode = 160

    # ✅ 动作降维开关（False=完整3维动作：throttle/steer/y_ref）
    config.use_action_dim2 = False

    # ✅ Stage3 是否启用 y_ref（默认关闭更稳）
    config.stage3_use_yref = False
    # ✅ Stage3 y_ref 渐进增益
    config.stage3_yref_warmup_end = 450
    config.stage3_yref_gain_warmup = 0.01
    config.stage3_yref_gain_full = 0.03

    # ✅ 统一episode步长：训练timesteps 与 env.max_episode_steps 对齐
    # 防止time_limit截断导致统计/收敛判断偏移
    config.max_episode_steps = 256
    config.train_episodes = 1500
    config.entropy_start = 0.02
    config.entropy_end = 0.005
    config.entropy_decay_ep = 600
    config.optimization_steps = (4, 4)
    config.trd_loss_coef = 0.0
    config.policy_lr = 1e-4
    config.value_lr = 2e-4
    config.gamma = 0.99
    config.lambda_ = 0.95
    config.clip_ratio = 0.2
    config.batch_size = 128
    config.update_frequency = 2

    # ✅ 关闭 early-stop（先保证稳定训练）
    # 如果你想恢复早停，把这个改回 True
    config.enable_early_stop = False

    # Parked obstacles场景参数
    config.num_parked_cars = 4
    config.parked_car_spacing = 8.0
    config.parked_car_start_distance_min = 12.0
    config.parked_car_start_distance_max = 20.0

    # Cones场景参数
    config.cone_num = 15                    # 锥桶数量
    config.cone_step_behind = 3.0           # 纵向间距（米）
    config.cone_step_lateral = 0.4          # 横向递进（米）
    config.spawn_min_gap_from_cone = 20.0   # 自车距第一个锥桶距离

    # ===== 把（W&B或默认）版本开关写回 config，保证 env 读到一致的值 =====
    if wandb_run is not None:
        cfg = wandb_run.config
        config.use_action_bias = bool(cfg.get("use_action_bias", False))
        config.use_forced_throttle = bool(cfg.get("use_forced_throttle", False))
        config.action_bias_strength = float(cfg.get("action_bias_strength", 0.7))
        config.action_bias_decay = float(cfg.get("action_bias_decay", 0.999))
        config.train_episodes = int(cfg.get("episodes", 1500))
        config.max_episode_steps = int(cfg.get("max_steps_per_episode", 256))
        config.policy_lr = float(cfg.get("policy_lr", 1e-4))
        config.value_lr = float(cfg.get("value_lr", 2e-4))
        config.gamma = float(cfg.get("gamma", 0.99))
        config.lambda_ = float(cfg.get("lambda", 0.95))
        config.clip_ratio = float(cfg.get("clip_ratio", 0.2))
        config.batch_size = int(cfg.get("batch_size", 128))
        config.update_frequency = int(cfg.get("update_frequency", 2))
        config.entropy_reg = float(cfg.get("entropy_reg", 0.02))
        config.entropy_start = float(cfg.get("entropy_start", 0.02))
        config.entropy_end = float(cfg.get("entropy_end", 0.005))
        config.entropy_decay_ep = int(cfg.get("entropy_decay_ep", 600))
        _opt_steps = cfg.get("optimization_steps", (4, 4))
        if isinstance(_opt_steps, (list, tuple)) and len(_opt_steps) == 2:
            config.optimization_steps = (int(_opt_steps[0]), int(_opt_steps[1]))
        else:
            config.optimization_steps = (4, 4)
        config.trd_loss_coef = float(cfg.get("trd_loss_coef", 0.0))

        # ✅ y_ref 新旧字段统一：训练内部都用 use_yref_in_steer / yref_steer_gain
        _use_yref = bool(cfg.get("use_yref_in_steer", cfg.get("use_yref_mapping", False)))
        _yref_gain = float(cfg.get("yref_steer_gain", cfg.get("yref_gain", 0.0)))
        config.use_yref_in_steer = _use_yref
        config.yref_steer_gain = _yref_gain
        # 兼容旧字段（防止其他模块仍在读取旧名）
        config.use_yref_mapping = _use_yref
        config.yref_gain = _yref_gain
        config.yref_penalty = float(cfg.get("yref_penalty", 0.0))
        config.log_steer_saturation = bool(cfg.get("log_steer_saturation", True))
        config.steer_sat_threshold = float(cfg.get("steer_sat_threshold", 0.999))
        config.wandb_step_log_interval = int(cfg.get("wandb_step_log_interval", 50))
    else:
        # wandb未开启时的默认值（你也可以手动改这里测试两版）
        config.use_action_bias = False
        config.use_forced_throttle = False
        config.action_bias_strength = 0.7
        config.action_bias_decay = 0.999
        config.train_episodes = int(getattr(config, "train_episodes", 1500))
        config.max_episode_steps = int(getattr(config, "max_episode_steps", 256))
        config.policy_lr = float(getattr(config, "policy_lr", 1e-4))
        config.value_lr = float(getattr(config, "value_lr", 2e-4))
        config.gamma = float(getattr(config, "gamma", 0.99))
        config.lambda_ = float(getattr(config, "lambda_", 0.95))
        config.clip_ratio = float(getattr(config, "clip_ratio", 0.2))
        config.batch_size = int(getattr(config, "batch_size", 128))
        config.update_frequency = int(getattr(config, "update_frequency", 2))
        config.entropy_reg = float(getattr(config, "entropy_reg", 0.02))
        config.entropy_start = float(getattr(config, "entropy_start", 0.02))
        config.entropy_end = float(getattr(config, "entropy_end", 0.005))
        config.entropy_decay_ep = int(getattr(config, "entropy_decay_ep", 600))
        config.optimization_steps = tuple(getattr(config, "optimization_steps", (4, 4)))
        config.trd_loss_coef = float(getattr(config, "trd_loss_coef", 0.0))

        # ✅ y_ref 新旧字段统一
        config.use_yref_in_steer = False
        config.yref_steer_gain = 0.0
        config.use_yref_mapping = False
        config.yref_gain = 0.0
        config.yref_penalty = 0.0
        config.log_steer_saturation = True
        config.steer_sat_threshold = 0.999
        config.wandb_step_log_interval = int(getattr(config, "wandb_step_log_interval", 50))

    # ===== 根据版本自动设置 yref_penalty =====
    # 不映射版：给一个小惩罚把 y_ref 压到 0
    # 映射版：不惩罚（或很小）
    if bool(getattr(config, "use_yref_in_steer", config.use_yref_mapping)):
        config.yref_penalty = 0.0  # 或 0.01（看你是否希望更平滑）
    else:
        config.yref_penalty = 0.05  # 推荐从 0.05 开始（范围 0.02~0.1）

    # =========================
    # ✅ Debug 可视化模式覆盖（短跑诊断）
    # =========================
    if debug_vis:
        print("\n[VIS DEBUG] ✅ 启用可视化短跑诊断模式")
        config.render = True
        config.spectator_mode = "none"

        config.enable_debug_drawing = True
        config.debug_draw_interval = 5
        config.draw_detection_range = True
        config.draw_ego_direction = True
        config.draw_obstacle_boxes = True
        config.draw_lane_center = True

        config.random_scenario = False
        config.scenario = "cones"
        config.scenario_pool = ["cones"]

        config.max_episode_steps = int(debug_vis_timesteps)

    # reward config（保持你的逻辑）
    print("\n[0] 加载激进版Reward配置...")
    try:
        from reward_config_aggressive import get_aggressive_config
        config.reward_config = get_aggressive_config()
        print("✅ 激进版Reward已加载")
    except Exception as e:
        print(f"⚠️  无法加载激进版reward，使用默认配置: {e}")

    print("\n[1] 创建训练日志记录器...")
    logger = TrainingLogger(log_file="training_log.json", auto_save_interval=5)
    print("✅ 日志记录器创建成功")

    print("\n[DEBUG] Config y_ref switch:")
    print("  use_yref_in_steer =", getattr(config, "use_yref_in_steer", None))
    print("  yref_steer_gain   =", getattr(config, "yref_steer_gain", None))
    print("  use_yref_mapping  =", getattr(config, "use_yref_mapping", None))
    print("  yref_gain         =", getattr(config, "yref_gain", None))
    print("  yref_penalty     =", getattr(config, "yref_penalty", None))

    print("\n[2] 创建CARLA Gym环境...")
    env = CarlaGymEnv(config, logger=logger, wandb_run=wandb_run)
    print("✅ 环境创建成功")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    print(
        "  - yref_in_steer=",
        getattr(env.carla_env, "use_yref_in_steer", None),
        ", yref_steer_gain=",
        getattr(env.carla_env, "yref_steer_gain", None),
    )

    print("\n[3] 创建PPO Agent...")
    from rl_agent_only.agents.ppo import PPOAgent

    # ✅ obs_dim 改了（9 -> 30）后，旧权重一定不兼容，必须从头训
    weights_dir = "./weights/ppo-carla-obs30"  # 换个新目录
    load_existing = False
    resume_policy_only = False
    print("\n✅ obs_dim=30：默认从头训练（不恢复旧 policy/value，避免历史坏策略污染）")

    # ✅ 修复P1级和P2级问题：优化超参数
    agent = PPOAgent(
        env=env,
        seed=int(getattr(config, "seed", 10)),
        policy_lr=float(config.policy_lr),
        value_lr=float(config.value_lr),
        gamma=float(config.gamma),
        lambda_=float(config.lambda_),
        clip_ratio=float(config.clip_ratio),
        entropy_regularization=float(config.entropy_reg),
        optimization_steps=tuple(config.optimization_steps),
        batch_size=int(config.batch_size),
        update_frequency=int(config.update_frequency),
        trd_loss_coef=float(getattr(config, "trd_loss_coef", 0.0)),
        name="ppo-carla-obs30",
        load=load_existing
    )

    print("✅ PPO Agent创建成功")
    print(f"  - 模型保存路径: {agent.base_path}")

    # ✅ 仅恢复 policy，避免 reward 变化导致 value 失配
    if resume_policy_only:
        import os
        lat_path = os.path.join(weights_dir, "policy_net_lat")
        lon_path = os.path.join(weights_dir, "policy_net_lon")
        try:
            agent.network.policy_lat.load_weights(lat_path)
            agent.network.policy_lon.load_weights(lon_path)
            agent.network.update_old_policy()
            print(f"✅ 已加载 policy_lat: {lat_path}")
            print(f"✅ 已加载 policy_lon: {lon_path}")
            print("✅ value 网络已重置（未加载旧权重）")
        except Exception as e:
            print(f"⚠️ policy 权重加载失败: {e}")

    print("\n[4] 开始训练...")
    print("-" * 70)

    try:
        train_with_logging(
            agent, env, logger,
            wandb_run=wandb_run,
            episodes=(debug_vis_episodes if debug_vis else int(getattr(config, "train_episodes", 1500))),
            # ✅ 和 env.max_episode_steps 保持一致，避免time_limit截断影响统计
            timesteps=int(getattr(config, "max_episode_steps", 512)),
            save_every=100
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练异常中止: {e}")
    finally:
        logger.close()
        env.close()
        if wandb_run:
            wandb_run.finish()
            print("\n✅ Wandb run已结束")

    print("\n" + "=" * 70)
    print("✅ PPO训练完成!")
    print("=" * 70)
    print(f"\n模型已保存到: {agent.base_path}")
    return agent.base_path


if __name__ == "__main__":
    model_path = train_ppo()
    print(f"\n下一步: 在router中使用训练好的模型")
    print(f"  ppo_model_path='{model_path}'")
