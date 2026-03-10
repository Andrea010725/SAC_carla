"""
单独训练PPO Agent - 带Wandb监控
（修正版：稳定reward分解 + 不篡改动作 + ✅W&B step对齐 + ✅episode横轴
 + ✅y_ref 映射版/不映射版开关）
"""
import sys
import os
import time
from collections import deque

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
import tensorflow as tf

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
        s1_thr = float(getattr(cfg, "stage1_success_threshold", 0.30))
        s2_thr = float(getattr(cfg, "stage2_success_threshold", 0.60))
        if succ_rate < s1_thr:
            stage = 1
        elif succ_rate < s2_thr:
            stage = 2
        else:
            stage = 3

    # 可选强制升阶段：默认关闭，避免“还没学会就被推高难”
    force_curriculum = bool(getattr(cfg, "enable_forced_curriculum", True))
    force_s2_ep = int(getattr(cfg, "force_stage2_episode", 140))
    force_s3_ep = int(getattr(cfg, "force_stage3_episode", 360))
    force_s2_min_succ = float(getattr(cfg, "force_stage2_min_success", 0.0))
    force_s3_min_succ = float(getattr(cfg, "force_stage3_min_success", 0.0))
    if force_curriculum and window_ready:
        if episode >= force_s3_ep and succ_rate >= force_s3_min_succ:
            stage = max(stage, 3)
        elif episode >= force_s2_ep and succ_rate >= force_s2_min_succ:
            stage = max(stage, 2)

    # 最短阶段约束：保证前期有足够多的 cones 训练轮次
    # - stage1_min_episode 前：强制只训 stage1(cones)
    # - stage2_min_episode 前：最多 stage2，不进入 stage3
    stage1_min_episode = int(getattr(cfg, "stage1_min_episode", 800))
    stage2_min_episode = int(getattr(cfg, "stage2_min_episode", 950))
    if episode <= stage1_min_episode:
        stage = 1
    elif episode <= stage2_min_episode:
        stage = min(stage, 2)

    # --- Stage 1: 只学 cones（先把最基础的直行+绕障学会） ---
    if stage == 1:
        cfg.scenario_pool = ["cones"]
        cfg.scenario_weights = {
            "cones": 1.00,
        }
        cfg.cone_num = 5
        # Stage1 仍保留很小的 y_ref 引导，避免“横纵策略完全解耦”导致后续不稳定
        cfg.use_yref_in_steer = True
        cfg.yref_steer_gain = float(getattr(cfg, "stage1_yref_gain", 0.0015))

    # --- Stage 2: 加入 trimma（仍不含 jaywalker） ---
    elif stage == 2:
        cfg.scenario_pool = ["cones", "trimma"]
        cfg.scenario_weights = {
            "cones": 0.75,
            "trimma": 0.25,
        }
        cfg.cone_num = 6
        cfg.use_yref_in_steer = True
        cfg.yref_steer_gain = float(getattr(cfg, "stage2_yref_gain", 0.003))

    # --- Stage 3: 加入 jaywalker（低权重，目标是停车避让） ---
    else:
        cfg.scenario_pool = ["cones", "trimma", "construction_lane_change", "jaywalker"]
        cfg.scenario_weights = {
            "cones": 0.35,
            "trimma": 0.25,
            "construction_lane_change": 0.25,
            "jaywalker": 0.15,
        }
        cfg.cone_num = 8
        # ✅ Stage3 是否开启 y_ref（做成开关，默认关闭更稳）
        stage3_use_yref = bool(getattr(cfg, "stage3_use_yref", True))
        if stage3_use_yref:
            cfg.use_yref_in_steer = True
            warm_end = int(getattr(cfg, "stage3_yref_warmup_end", 520))
            gain_warm = float(getattr(cfg, "stage3_yref_gain_warmup", 0.004))
            gain_full = float(getattr(cfg, "stage3_yref_gain_full", 0.008))
            cfg.yref_steer_gain = gain_warm if episode <= warm_end else gain_full
        else:
            cfg.use_yref_in_steer = False
            cfg.yref_steer_gain = 0.0


# ============================================================
# ✅ 熵系数调度：先探索，后收敛
# ============================================================
def update_entropy_coeff(episode: int, agent, start: float = 0.028, end: float = 0.010, decay_ep: int = 2600):
    if not hasattr(agent, "entropy_strength"):
        return
    frac = min(1.0, float(episode) / float(decay_ep))
    coeff = start + (end - start) * frac
    if hasattr(agent.entropy_strength, "value"):
        agent.entropy_strength.value = float(coeff)


def _resume_state_path(base_path: str) -> str:
    return os.path.join(base_path, "resume_state.npz")


def has_compatible_checkpoint(base_path: str) -> bool:
    required = [
        os.path.join(base_path, "policy_net_lat.index"),
        os.path.join(base_path, "policy_net_lon.index"),
        os.path.join(base_path, "value_net.index"),
    ]
    return all(os.path.exists(p) for p in required)


def load_resume_state(base_path: str) -> dict:
    state = {
        "global_episode": 0,
        "global_env_step": 0,
        "obs_norm_mean": None,
        "obs_norm_var": None,
        "obs_norm_count": None,
        "loaded": False,
    }
    path = _resume_state_path(base_path)
    if not os.path.exists(path):
        return state

    try:
        with np.load(path, allow_pickle=False) as data:
            state["global_episode"] = int(data["global_episode"]) if "global_episode" in data else 0
            state["global_env_step"] = int(data["global_env_step"]) if "global_env_step" in data else 0
            state["obs_norm_mean"] = np.asarray(data["obs_norm_mean"], dtype=np.float32) if "obs_norm_mean" in data else None
            state["obs_norm_var"] = np.asarray(data["obs_norm_var"], dtype=np.float32) if "obs_norm_var" in data else None
            state["obs_norm_count"] = float(data["obs_norm_count"]) if "obs_norm_count" in data else None
            state["loaded"] = True
    except Exception as e:
        print(f"[Resume] ⚠️ 读取续训状态失败，忽略并从默认状态继续: {e}")
    return state


def save_resume_state(base_path: str, global_episode: int, global_env_step: int, obs_norm_state: dict = None):
    path = _resume_state_path(base_path)
    os.makedirs(base_path, exist_ok=True)
    payload = {
        "global_episode": np.int64(max(0, int(global_episode))),
        "global_env_step": np.int64(max(0, int(global_env_step))),
        "obs_norm_mean": np.array([], dtype=np.float32),
        "obs_norm_var": np.array([], dtype=np.float32),
        "obs_norm_count": np.float64(1e-4),
    }

    if isinstance(obs_norm_state, dict):
        if obs_norm_state.get("mean") is not None:
            payload["obs_norm_mean"] = np.asarray(obs_norm_state["mean"], dtype=np.float32)
        if obs_norm_state.get("var") is not None:
            payload["obs_norm_var"] = np.asarray(obs_norm_state["var"], dtype=np.float32)
        if obs_norm_state.get("count") is not None:
            payload["obs_norm_count"] = np.float64(max(1e-4, float(obs_norm_state["count"])))

    np.savez(path, **payload)


def read_best_ckpt_metrics(base_path: str) -> dict:
    """
    读取 best_ckpt/best_metrics.txt（若存在）用于判断续训权重是否具备基本质量。
    """
    metrics = {}
    meta_path = os.path.join(base_path, "best_ckpt", "best_metrics.txt")
    if not os.path.exists(meta_path):
        return metrics
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    metrics[k] = float(v)
                except Exception:
                    metrics[k] = v
    except Exception:
        return {}
    return metrics


def _to_jsonable_config_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable_config_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable_config_value(val) for k, val in v.items()}
    return str(v)


def dump_effective_config(config_obj, out_path: str):
    """
    将最终生效的 config 快照写盘，避免“改了参数但实际没生效”。
    """
    try:
        cfg_dict = {}
        for k, v in vars(config_obj).items():
            if str(k).startswith("_"):
                continue
            if callable(v):
                continue
            cfg_dict[str(k)] = _to_jsonable_config_value(v)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[Config] ✅ effective config 已保存: {out_path}")
    except Exception as e:
        print(f"[Config] ⚠️ 保存 effective config 失败: {e}")


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
        # 防“低速苟活”惩罚：按场景分阈值，直接作用训练reward
        self.anti_crawl_enable = bool(getattr(config, "anti_crawl_enable", True))
        # 连续低速惩罚默认关闭，避免与 CarlaEnv 内部低速惩罚重复叠加。
        # 低速约束主要交给 CarlaEnv reward 内部项（r_low_speed / r_slow_clear）。
        self.wrapper_crawl_penalty_enable = bool(getattr(config, "wrapper_crawl_penalty_enable", False))
        self.anti_crawl_speed_lane = float(getattr(config, "anti_crawl_speed_lane", 3.0))
        self.anti_crawl_speed_jaywalker = float(getattr(config, "anti_crawl_speed_jaywalker", 1.0))
        self.anti_crawl_penalty_k_lane = float(getattr(config, "anti_crawl_penalty_k_lane", 0.32))
        self.anti_crawl_penalty_k_jaywalker = float(getattr(config, "anti_crawl_penalty_k_jaywalker", 0.08))
        self.anti_crawl_penalty_cap_lane = float(getattr(config, "anti_crawl_penalty_cap_lane", 1.3))
        self.anti_crawl_penalty_cap_jaywalker = float(getattr(config, "anti_crawl_penalty_cap_jaywalker", 0.6))
        self.anti_crawl_terminate_enable = bool(getattr(config, "anti_crawl_terminate_enable", True))
        self.anti_crawl_warmup_steps_lane = int(getattr(config, "anti_crawl_warmup_steps_lane", 140))
        self.anti_crawl_warmup_steps_jaywalker = int(getattr(config, "anti_crawl_warmup_steps_jaywalker", 120))
        self.anti_crawl_min_avg_speed_lane = float(getattr(config, "anti_crawl_min_avg_speed_lane", 3.0))
        self.anti_crawl_min_avg_speed_jaywalker = float(getattr(config, "anti_crawl_min_avg_speed_jaywalker", 0.9))
        # 低速终止改为“近期速度窗口”判定，避免前期起步速度拖低全程平均导致误杀
        self.anti_crawl_recent_window_lane = int(getattr(config, "anti_crawl_recent_window_lane", 48))
        self.anti_crawl_recent_window_jaywalker = int(getattr(config, "anti_crawl_recent_window_jaywalker", 36))
        self.anti_crawl_min_recent_speed_lane = float(getattr(config, "anti_crawl_min_recent_speed_lane", 3.0))
        self.anti_crawl_min_recent_speed_jaywalker = float(getattr(config, "anti_crawl_min_recent_speed_jaywalker", 0.9))
        self.anti_crawl_terminate_penalty_lane = float(getattr(config, "anti_crawl_terminate_penalty_lane", 45.0))
        self.anti_crawl_terminate_penalty_jaywalker = float(getattr(config, "anti_crawl_terminate_penalty_jaywalker", 25.0))
        self.anti_crawl_terminate_start_episode = int(getattr(config, "anti_crawl_terminate_start_episode", 420))
        self.anti_crawl_terminate_speed_ratio = float(getattr(config, "anti_crawl_terminate_speed_ratio", 0.78))
        # 低速终止前至少行驶这么远，避免“还没到障碍区就被终止”
        self.anti_crawl_min_travel_before_terminate_lane = float(
            getattr(config, "anti_crawl_min_travel_before_terminate_lane", 16.0)
        )
        self.anti_crawl_min_travel_before_terminate_jaywalker = float(
            getattr(config, "anti_crawl_min_travel_before_terminate_jaywalker", 6.0)
        )
        self.anti_crawl_terminate_obs_exempt_dist_lane = float(
            getattr(config, "anti_crawl_terminate_obs_exempt_dist_lane", 12.0)
        )
        self.anti_crawl_terminate_obs_exempt_dist_jaywalker = float(
            getattr(config, "anti_crawl_terminate_obs_exempt_dist_jaywalker", 18.0)
        )
        self.anti_crawl_penalty_near_obs_scale_lane = float(
            getattr(config, "anti_crawl_penalty_near_obs_scale_lane", 0.35)
        )
        self.anti_crawl_penalty_near_obs_scale_jaywalker = float(
            getattr(config, "anti_crawl_penalty_near_obs_scale_jaywalker", 0.60)
        )
        # 低速且无近障碍时，惩罚不必要刹车，避免策略退化为“踩刹车苟活”
        self.bad_brake_penalty_enable = bool(getattr(config, "bad_brake_penalty_enable", False))
        self.bad_brake_k_lane = float(getattr(config, "bad_brake_k_lane", 0.35))
        self.bad_brake_k_jaywalker = float(getattr(config, "bad_brake_k_jaywalker", 0.10))
        self.bad_brake_apply_speed_min_lane = float(
            getattr(config, "bad_brake_apply_speed_min_lane", 3.0)
        )
        self.bad_brake_apply_speed_min_jaywalker = float(
            getattr(config, "bad_brake_apply_speed_min_jaywalker", 1.2)
        )
        self.bad_brake_apply_obs_dist_min_lane = float(
            getattr(config, "bad_brake_apply_obs_dist_min_lane", 16.0)
        )
        self.bad_brake_apply_obs_dist_min_jaywalker = float(
            getattr(config, "bad_brake_apply_obs_dist_min_jaywalker", 10.0)
        )

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
        self._episode_start_location = None
        max_recent_window = max(
            1,
            int(self.anti_crawl_recent_window_lane),
            int(self.anti_crawl_recent_window_jaywalker),
        )
        self._recent_speed_hist = deque(maxlen=max_recent_window)

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
        self.normalize_obs = bool(getattr(config, "normalize_obs", True))
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

    def get_obs_norm_state(self) -> dict:
        if not hasattr(self, "obs_normalizer") or self.obs_normalizer is None:
            return {}
        return {
            "mean": np.asarray(self.obs_normalizer.mean, dtype=np.float32).copy(),
            "var": np.asarray(self.obs_normalizer.var, dtype=np.float32).copy(),
            "count": float(self.obs_normalizer.count),
        }

    def set_obs_norm_state(self, state: dict) -> bool:
        if not isinstance(state, dict):
            return False
        mean = state.get("mean", None)
        var = state.get("var", None)
        count = state.get("count", None)
        if mean is None or var is None or count is None:
            return False

        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        var = np.asarray(var, dtype=np.float32).reshape(-1)
        obs_dim = int(self.observation_space.shape[0])
        if mean.size != obs_dim or var.size != obs_dim:
            print(
                f"[Resume] ⚠️ obs_normalizer 维度不匹配，跳过恢复: "
                f"mean={mean.size}, var={var.size}, obs_dim={obs_dim}"
            )
            return False

        self.obs_normalizer.mean = mean
        self.obs_normalizer.var = np.maximum(var, 1e-8)
        self.obs_normalizer.count = max(1e-4, float(count))
        return True

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
        self._episode_speed_sum = 0.0
        self._recent_speed_hist.clear()

        # ✅ 归一化观测
        obs = np.asarray(obs, dtype=np.float32)
        if self.normalize_obs:
            self.obs_normalizer.update(obs.reshape(1, -1))
            obs = self.obs_normalizer.normalize(obs)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        # 记录 episode 起点位置，用于 anti-crawl 终止门控
        try:
            ego = getattr(self.carla_env, "ego", None)
            self._episode_start_location = ego.get_location() if ego is not None else None
        except Exception:
            self._episode_start_location = None

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

        # ----- anti-crawl penalty + terminate gate -----
        scenario_name = str(getattr(self.carla_env, "scenario", "")).lower()
        is_jaywalker = ("jaywalker" in scenario_name)
        speed_now = float(info.get("speed", 0.0))
        self._episode_speed_sum += speed_now
        steps_now = int(self._step_count) + 1
        avg_speed_so_far = float(self._episode_speed_sum / max(1, steps_now))
        self._recent_speed_hist.append(speed_now)
        traveled_dist = 0.0
        try:
            ego = getattr(self.carla_env, "ego", None)
            if (ego is not None) and (self._episode_start_location is not None):
                traveled_dist = float(ego.get_location().distance(self._episode_start_location))
        except Exception:
            traveled_dist = 0.0
        recent_window_for_log = max(
            1,
            int(self.anti_crawl_recent_window_jaywalker if is_jaywalker else self.anti_crawl_recent_window_lane),
        )
        recent_vals_for_log = list(self._recent_speed_hist)[-recent_window_for_log:]
        recent_avg_for_log = float(np.mean(recent_vals_for_log)) if recent_vals_for_log else 0.0
        nearest_obs_dist = float(info.get("nearest_obstacle_dist", -1.0))
        nearest_obs_fwd = float(info.get("nearest_obstacle_fwd", 0.0))

        if is_jaywalker:
            obs_exempt_dist = self.anti_crawl_terminate_obs_exempt_dist_jaywalker
        else:
            obs_exempt_dist = self.anti_crawl_terminate_obs_exempt_dist_lane
        near_obs_for_crawl = bool(
            nearest_obs_dist > 0.0
            and nearest_obs_fwd > -0.1
            and nearest_obs_dist < float(obs_exempt_dist)
        )

        r_crawl = 0.0
        r_bad_brake = 0.0
        if self.anti_crawl_enable and self.wrapper_crawl_penalty_enable:
            if is_jaywalker:
                v_thr = self.anti_crawl_speed_jaywalker
                k_pen = self.anti_crawl_penalty_k_jaywalker
                cap_pen = self.anti_crawl_penalty_cap_jaywalker
                near_obs_pen_scale = self.anti_crawl_penalty_near_obs_scale_jaywalker
            else:
                v_thr = self.anti_crawl_speed_lane
                k_pen = self.anti_crawl_penalty_k_lane
                cap_pen = self.anti_crawl_penalty_cap_lane
                near_obs_pen_scale = self.anti_crawl_penalty_near_obs_scale_lane

            speed_deficit = max(0.0, v_thr - speed_now)
            r_crawl = -min(cap_pen, k_pen * speed_deficit)
            if near_obs_for_crawl:
                r_crawl *= float(np.clip(near_obs_pen_scale, 0.0, 1.0))
            reward = float(reward) + float(r_crawl)

        if self.bad_brake_penalty_enable:
            if is_jaywalker:
                bad_brake_k = self.bad_brake_k_jaywalker
                v_ref = self.anti_crawl_speed_jaywalker
                bad_brake_apply_speed_min = self.bad_brake_apply_speed_min_jaywalker
                bad_brake_apply_obs_dist_min = self.bad_brake_apply_obs_dist_min_jaywalker
            else:
                bad_brake_k = self.bad_brake_k_lane
                v_ref = self.anti_crawl_speed_lane
                bad_brake_apply_speed_min = self.bad_brake_apply_speed_min_lane
                bad_brake_apply_obs_dist_min = self.bad_brake_apply_obs_dist_min_lane
            bad_brake = max(0.0, -float(a0_applied))
            far_from_obs_for_brake_pen = bool(
                (nearest_obs_dist < 0.0) or (nearest_obs_dist > float(bad_brake_apply_obs_dist_min))
            )
            if (
                (bad_brake > 0.0)
                and (not near_obs_for_crawl)
                and (speed_now >= float(bad_brake_apply_speed_min))
                and far_from_obs_for_brake_pen
            ):
                speed_factor = float(np.clip((v_ref - speed_now) / max(v_ref, 1e-6), 0.0, 1.0))
                r_bad_brake = -float(bad_brake_k) * float(bad_brake) * (0.30 + 0.70 * speed_factor)
                reward = float(reward) + float(r_bad_brake)

        crawl_fail = 0.0
        enable_crawl_terminate_now = (
            self.anti_crawl_terminate_enable
            and (self.current_episode >= int(self.anti_crawl_terminate_start_episode))
        )
        if enable_crawl_terminate_now and (not done):
            if is_jaywalker:
                warmup_steps = self.anti_crawl_warmup_steps_jaywalker
                min_avg_v = self.anti_crawl_min_avg_speed_jaywalker
                recent_window = max(1, int(self.anti_crawl_recent_window_jaywalker))
                min_recent_v = self.anti_crawl_min_recent_speed_jaywalker
                fail_penalty = self.anti_crawl_terminate_penalty_jaywalker
                min_travel_before_terminate = self.anti_crawl_min_travel_before_terminate_jaywalker
            else:
                warmup_steps = self.anti_crawl_warmup_steps_lane
                min_avg_v = self.anti_crawl_min_avg_speed_lane
                recent_window = max(1, int(self.anti_crawl_recent_window_lane))
                min_recent_v = self.anti_crawl_min_recent_speed_lane
                fail_penalty = self.anti_crawl_terminate_penalty_lane
                min_travel_before_terminate = self.anti_crawl_min_travel_before_terminate_lane

            min_v_for_terminate = float(min_avg_v) * float(np.clip(self.anti_crawl_terminate_speed_ratio, 0.1, 1.0))
            recent_vals = list(self._recent_speed_hist)[-recent_window:]
            recent_avg_speed = float(np.mean(recent_vals)) if recent_vals else 0.0
            min_recent_v_for_terminate = float(max(min_avg_v, min_recent_v)) * float(
                np.clip(self.anti_crawl_terminate_speed_ratio, 0.1, 1.0)
            )
            if (
                steps_now >= max(1, int(warmup_steps))
                and len(recent_vals) >= recent_window
                and recent_avg_speed < min_recent_v_for_terminate
                and traveled_dist >= float(min_travel_before_terminate)
                and (not near_obs_for_crawl)
            ):
                done = True
                crawl_fail = 1.0
                reward = float(reward) - float(fail_penalty)
                r_crawl = float(r_crawl) - float(fail_penalty)
                info["done_reason"] = "crawl_fail"
                info["crawl_fail"] = 1.0
                info["crawl_avg_speed"] = float(avg_speed_so_far)
                info["crawl_recent_avg_speed"] = float(recent_avg_speed)
                info["crawl_recent_window"] = int(recent_window)
                info["crawl_traveled_dist"] = float(traveled_dist)
                info["crawl_min_travel_before_terminate"] = float(min_travel_before_terminate)
                info["crawl_min_v_for_terminate"] = float(min_v_for_terminate)
                info["crawl_min_recent_v_for_terminate"] = float(min_recent_v_for_terminate)

        # ----- reward components -----
        comps = {}
        if isinstance(getattr(self.carla_env, "last_reward_components", None), dict):
            comps = self.carla_env.last_reward_components.copy()
        comps["r_yref"] = float(r_yref)
        comps["r_crawl"] = float(r_crawl)
        comps["r_bad_brake"] = float(r_bad_brake)
        self.step_reward_components = comps

        # ----- info -----
        info.update({
            "wrapped_a0_raw": float(a0),
            "wrapped_a0_applied": float(a0_applied),
            "wrapped_applied_bias": float(applied_bias),
            "wrapped_forced_throttle": int(forced_throttle),
            "wrapper_yref_penalty": float(self.yref_penalty),
            "wrapper_anti_crawl_penalty": float(r_crawl),
            "wrapper_bad_brake_penalty": float(r_bad_brake),
            "wrapper_crawl_fail": float(crawl_fail),
            "wrapper_crawl_avg_speed": float(avg_speed_so_far),
            "wrapper_crawl_recent_avg_speed": float(recent_avg_for_log),
            "wrapper_crawl_traveled_dist": float(traveled_dist),
            "wrapper_crawl_near_obs_exempt": float(1.0 if near_obs_for_crawl else 0.0),
            "wrapper_crawl_obs_dist": float(nearest_obs_dist),
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

def train_with_logging(
    agent,
    env,
    logger,
    wandb_run=None,
    episodes=1500,
    timesteps=512,
    save_every=100,
    start_episode=0,
    state_callback=None,
):
    """
    带日志记录和Wandb监控的PPO训练循环 + ✅自动早停/收敛判定
    """
    import time
    import os
    import numpy as np
    import glob
    import shutil
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
    cfg_ref = getattr(env, "config", None)
    enable_early_stop = bool(getattr(cfg_ref, "enable_early_stop", True))
    W = int(getattr(cfg_ref, "early_window", 20))  # 滑动窗口长度（建议 20~50）
    MIN_EP = int(getattr(cfg_ref, "early_min_episode", 60))  # 至少跑到这个 episode 才开始早停判断

    # ---- 收敛判据（满足就停）----
    TARGET_SUCCESS = float(getattr(cfg_ref, "early_target_success", 0.80))
    TARGET_RAN_FULL = float(getattr(cfg_ref, "early_target_ran_full", 0.80))
    TARGET_SPEED_OK = float(getattr(cfg_ref, "early_target_speed_ok", 0.80))
    TARGET_AVG_SPEED = float(getattr(cfg_ref, "early_target_avg_speed", 3.0))
    MAX_COLLISION = float(getattr(cfg_ref, "early_max_collision", 0.10))
    NEED_STABLE_WINDOWS = int(getattr(cfg_ref, "early_need_stable_windows", 3))

    # ---- 平台期判据（长时间没提升就停）----
    PLATEAU_PATIENCE = int(getattr(cfg_ref, "early_plateau_patience", 300))
    IMPROVE_EPS = float(getattr(cfg_ref, "early_improve_eps", 0.01))
    PLATEAU_MIN_EP = int(getattr(cfg_ref, "early_plateau_min_episode", 1800))
    PLATEAU_MIN_BEST_SUCCESS = float(getattr(cfg_ref, "early_plateau_min_best_success", 0.35))
    PLATEAU_MIN_PROGRESS_SCORE = float(getattr(cfg_ref, "early_plateau_min_progress_score", 0.65))
    PLATEAU_CURR_MIN_SUCCESS = float(getattr(cfg_ref, "early_plateau_curr_min_success", 0.05))
    PLATEAU_CURR_MIN_SPEED_OK = float(getattr(cfg_ref, "early_plateau_curr_min_speed_ok", 0.35))
    PLATEAU_CURR_MIN_SAFE_FAST = float(getattr(cfg_ref, "early_plateau_curr_min_safe_fast", 0.10))
    PLATEAU_CURR_MAX_COLLISION = float(getattr(cfg_ref, "early_plateau_curr_max_collision", 0.35))
    SCORE_W_SUCC = float(getattr(cfg_ref, "early_progress_w_success", 0.55))
    SCORE_W_RAN = float(getattr(cfg_ref, "early_progress_w_ran_full", 0.30))
    SCORE_W_SAFE = float(getattr(cfg_ref, "early_progress_w_no_collision", 0.15))

    # ---- 崩溃判据（明显坏掉就停）----
    BAD_MIN_EP = int(getattr(cfg_ref, "early_bad_min_episode", 2000))
    BAD_PATIENCE = int(getattr(cfg_ref, "early_bad_patience", 120))
    BAD_COLLISION = float(getattr(cfg_ref, "early_bad_collision", 0.75))
    BAD_RAN_FULL = float(getattr(cfg_ref, "early_bad_ran_full", 0.30))
    BAD_SUCCESS = float(getattr(cfg_ref, "early_bad_success", 0.20))
    BAD_AVG_SPEED = float(getattr(cfg_ref, "early_bad_avg_speed", 2.9))
    BAD_SPEED_OK = float(getattr(cfg_ref, "early_bad_speed_ok", 0.25))
    BAD_STUCK_RATE = float(getattr(cfg_ref, "early_bad_stuck_rate", 0.65))
    BAD_USE_RAN_FULL = bool(getattr(cfg_ref, "early_bad_use_ran_full", False))
    BAD_RAN_FULL_MIN_EP = int(getattr(cfg_ref, "early_bad_ran_full_min_episode", 180))
    BAD_CRAWL_FAIL = float(getattr(cfg_ref, "early_bad_crawl_fail", 0.85))
    BAD_PROGRESS_MAX = float(getattr(cfg_ref, "early_bad_progress_max", 0.12))
    BAD_SAFE_FAST = float(getattr(cfg_ref, "early_bad_safe_fast", 0.15))

    # ---- 早停重启判据：长时间几乎无有效进展时直接结束本次run ----
    RESTART_ENABLE = bool(getattr(cfg_ref, "early_restart_enable", False))
    RESTART_MIN_EP = int(getattr(cfg_ref, "early_restart_min_episode", 600))
    RESTART_PATIENCE = int(getattr(cfg_ref, "early_restart_patience", 12))
    RESTART_SUCCESS_MAX = float(getattr(cfg_ref, "early_restart_success_max", 0.03))
    RESTART_SPEED_OK_MAX = float(getattr(cfg_ref, "early_restart_speed_ok_max", 0.10))
    RESTART_SAFE_FAST_MAX = float(getattr(cfg_ref, "early_restart_safe_fast_max", 0.10))
    RESTART_AVG_SPEED_MAX = float(getattr(cfg_ref, "early_restart_avg_speed_max", 1.8))
    RESTART_COLLISION_MIN = float(getattr(cfg_ref, "early_restart_collision_min", 0.35))
    RESTART_STUCK_MIN = float(getattr(cfg_ref, "early_restart_stuck_min", 0.70))

    # ---- 硬停训：长窗口持续高碰撞 ----
    HARD_BAD_ENABLE = bool(getattr(cfg_ref, "hard_bad_enable", False))
    HARD_W = int(getattr(cfg_ref, "hard_bad_window", 120))
    HARD_BAD_COLLISION = float(getattr(cfg_ref, "hard_bad_collision", 0.85))
    HARD_BAD_MIN_EP = int(getattr(cfg_ref, "hard_bad_min_episode", 320))
    HARD_BAD_RESTORE_CONTINUE = bool(getattr(cfg_ref, "hard_bad_restore_continue", True))
    HARD_BAD_RECOVER_COOLDOWN = int(getattr(cfg_ref, "hard_bad_recover_cooldown", 40))
    HARD_BAD_MAX_RECOVERS = int(getattr(cfg_ref, "hard_bad_max_recovers", 10))
    HARD_BAD_ENT_BOOST_EP = int(getattr(cfg_ref, "hard_bad_entropy_boost_episodes", 120))
    HARD_BAD_ENT_START = float(getattr(cfg_ref, "hard_bad_entropy_boost_start", 0.045))
    HARD_BAD_ENT_END = float(getattr(cfg_ref, "hard_bad_entropy_boost_end", 0.016))

    # ---- deterministic 评估 ----
    DET_EVAL_ENABLE = bool(getattr(cfg_ref, "deterministic_eval_enable", True))
    DET_EVAL_EVERY = int(getattr(cfg_ref, "deterministic_eval_every", 50))
    DET_EVAL_EPISODES = int(getattr(cfg_ref, "deterministic_eval_episodes", 2))
    DET_EVAL_TIMESTEPS = int(getattr(cfg_ref, "deterministic_eval_timesteps", timesteps))
    DET_EVAL_USE_OLD = bool(getattr(cfg_ref, "deterministic_eval_use_old_policy", False))
    # 关键开关：PPO 应默认使用“策略采样动作”计算比率。
    # 仅当你明确要研究“执行动作拟合”时再打开。
    TRAIN_ONPOLICY_ACTION_MODE = bool(getattr(cfg_ref, "train_onpolicy_action_mode", True))
    PPO_STORE_EXECUTED_ACTION = bool(getattr(cfg_ref, "ppo_store_executed_action", True))
    # 严格 on-policy 模式下，强制使用“执行后动作”入 memory，避免策略动作与环境执行动作偏移。
    if TRAIN_ONPOLICY_ACTION_MODE and (not PPO_STORE_EXECUTED_ACTION):
        PPO_STORE_EXECUTED_ACTION = True
        print("[OnPolicy] train_onpolicy_action_mode=True -> 强制 ppo_store_executed_action=True")

    # ---- best checkpoint 自动回滚 ----
    RESTORE_BEST_ON_BAD_STOP = bool(getattr(cfg_ref, "restore_best_on_bad_stop", True))
    RESTORE_BEST_AT_END = bool(getattr(cfg_ref, "restore_best_at_end", True))
    POLICY_TARGET_KL = float(getattr(cfg_ref, "policy_target_kl", 0.022))
    POLICY_KL_STOP_MULT = float(getattr(cfg_ref, "policy_kl_stop_multiplier", 1.15))
    if hasattr(agent, "target_kl"):
        agent.target_kl = max(0.0, POLICY_TARGET_KL)
    if hasattr(agent, "kl_stop_multiplier"):
        agent.kl_stop_multiplier = max(1.0, POLICY_KL_STOP_MULT)
    print(
        "[PolicyKL] "
        f"target_kl={POLICY_TARGET_KL:.5f}, stop_mult={POLICY_KL_STOP_MULT:.3f}"
    )

    # ---- 退化回滚：连续坏窗口时恢复 best 并降低学习率，防止“中途变好后再次学坏” ----
    DEGRADE_ROLLBACK_ENABLE = bool(getattr(cfg_ref, "degrade_rollback_enable", True))
    DEGRADE_ROLLBACK_MIN_EP = int(getattr(cfg_ref, "degrade_rollback_min_episode", 320))
    DEGRADE_ROLLBACK_PATIENCE = int(getattr(cfg_ref, "degrade_rollback_patience", 8))
    DEGRADE_ROLLBACK_COOLDOWN = int(getattr(cfg_ref, "degrade_rollback_cooldown", 80))
    DEGRADE_ROLLBACK_MAX_RECOVERS = int(getattr(cfg_ref, "degrade_rollback_max_recovers", 6))
    DEGRADE_ROLLBACK_COLLISION_MIN = float(getattr(cfg_ref, "degrade_rollback_collision_min", 0.55))
    DEGRADE_ROLLBACK_SAFE_FAST_MAX = float(getattr(cfg_ref, "degrade_rollback_safe_fast_max", 0.15))
    DEGRADE_ROLLBACK_SUCCESS_MAX = float(getattr(cfg_ref, "degrade_rollback_success_max", 0.05))
    DEGRADE_POLICY_LR_SCALE = float(getattr(cfg_ref, "degrade_rollback_policy_lr_scale", 0.70))
    DEGRADE_VALUE_LR_SCALE = float(getattr(cfg_ref, "degrade_rollback_value_lr_scale", 0.80))
    DEGRADE_POLICY_LR_MIN = float(getattr(cfg_ref, "degrade_rollback_policy_lr_min", 1e-5))
    DEGRADE_VALUE_LR_MIN = float(getattr(cfg_ref, "degrade_rollback_value_lr_min", 2e-5))
    REWARD_BEST_CKPT_ENABLE = bool(getattr(cfg_ref, "reward_best_ckpt_enable", True))
    REWARD_BEST_MIN_EP = int(getattr(cfg_ref, "reward_best_ckpt_min_episode", 1))

    # 统计窗口
    dq_success = deque(maxlen=W)
    dq_ran_full = deque(maxlen=W)
    dq_no_collision = deque(maxlen=W)
    dq_avg_speed = deque(maxlen=W)
    dq_speed_ok = deque(maxlen=W)
    dq_safe_fast = deque(maxlen=W)
    dq_avg_rps = deque(maxlen=W)  # 可选：avg_reward_per_step
    dq_done_collision = deque(maxlen=W)
    dq_done_offroad = deque(maxlen=W)
    dq_done_crawl_fail = deque(maxlen=W)
    dq_done_no_progress = deque(maxlen=W)
    dq_done_time_limit = deque(maxlen=W)
    dq_steer_delta_p90 = deque(maxlen=W)
    dq_hard_collision = deque(maxlen=max(1, HARD_W))
    scenario_window = deque(maxlen=max(200, W * 10))

    stable_good_count = 0
    bad_count = 0
    restart_count = 0

    best_succ_rate = -1.0
    best_succ_episode = 0
    best_progress_score = -1.0
    best_progress_episode = 0
    best_total_reward = -float("inf")
    best_total_reward_episode = 0
    best_avg_rps_for_reward_best = -float("inf")
    plateau_count = 0
    best_ckpt_key = None
    stop_reason = None
    det_eval_last = {}
    hard_bad_recover_count = 0
    last_hard_recover_episode = -10 ** 9
    forced_entropy_until_episode = -1
    degrade_streak = 0
    degrade_rollback_count = 0
    last_degrade_rollback_episode = -10 ** 9
    policy_kl_tracker = 0.0
    policy_kl_early_stop_tracker = 0
    print(
        "[HardBadCfg] "
        f"enable={HARD_BAD_ENABLE}, window={HARD_W}, thr={HARD_BAD_COLLISION:.3f}, "
        f"restore_continue={HARD_BAD_RESTORE_CONTINUE}, cooldown={HARD_BAD_RECOVER_COOLDOWN}, "
        f"max_recovers={HARD_BAD_MAX_RECOVERS}, ent_boost_ep={HARD_BAD_ENT_BOOST_EP}"
    )
    print(
        "[RestartCfg] "
        f"enable={RESTART_ENABLE}, min_ep={RESTART_MIN_EP}, patience={RESTART_PATIENCE}, "
        f"succ_max={RESTART_SUCCESS_MAX:.3f}, speed_ok_max={RESTART_SPEED_OK_MAX:.3f}, "
        f"safe_fast_max={RESTART_SAFE_FAST_MAX:.3f}, avg_speed_max={RESTART_AVG_SPEED_MAX:.3f}, "
        f"collision_min={RESTART_COLLISION_MIN:.3f}, stuck_min={RESTART_STUCK_MIN:.3f}"
    )
    print(
        "[RewardBestCkptCfg] "
        f"enable={int(REWARD_BEST_CKPT_ENABLE)}, min_ep={REWARD_BEST_MIN_EP}"
    )

    def _save_best_checkpoint(
        ep: int,
        coll: float,
        succ: float,
        ran: float,
        avg_r: float,
        avg_speed: float = 0.0,
        speed_ok_rate: float = 0.0,
        safe_fast_rate: float = 0.0,
    ):
        try:
            best_dir = os.path.join(agent.base_path, "best_ckpt")
            os.makedirs(best_dir, exist_ok=True)
            for pattern in ("policy_net*", "value_net*", "config.json"):
                for src in glob.glob(os.path.join(agent.base_path, pattern)):
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(best_dir, os.path.basename(src)))
            meta_path = os.path.join(best_dir, "best_metrics.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(
                    f"episode={int(ep)}\n"
                    f"collision_rate_W={float(coll):.6f}\n"
                    f"success_rate_W={float(succ):.6f}\n"
                    f"ran_full_rate_W={float(ran):.6f}\n"
                    f"avg_speed_W={float(avg_speed):.6f}\n"
                    f"speed_ok_rate_W={float(speed_ok_rate):.6f}\n"
                    f"safe_fast_rate_W={float(safe_fast_rate):.6f}\n"
                    f"avg_reward_per_step_W={float(avg_r):.6f}\n"
                )
        except Exception as e:
            print(f"[BestCkpt] ⚠️ 保存best checkpoint失败: {e}")

    def _restore_best_checkpoint(reason: str) -> bool:
        best_dir = os.path.join(agent.base_path, "best_ckpt")
        lat_idx = os.path.join(best_dir, "policy_net_lat.index")
        lon_idx = os.path.join(best_dir, "policy_net_lon.index")
        val_idx = os.path.join(best_dir, "value_net.index")
        if not (os.path.exists(lat_idx) and os.path.exists(lon_idx) and os.path.exists(val_idx)):
            print(f"[BestCkpt] ⚠️ 未找到可恢复的best checkpoint，skip restore ({reason})")
            return False

        try:
            for pattern in ("policy_net*", "value_net*", "config.json"):
                for src in glob.glob(os.path.join(best_dir, pattern)):
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(agent.base_path, os.path.basename(src)))
            agent.load()
            print(f"[BestCkpt] ♻️ 已恢复最佳权重 ({reason})")
            return True
        except Exception as e:
            print(f"[BestCkpt] ⚠️ 恢复best checkpoint失败 ({reason}): {e}")
            return False

    def _save_reward_best_checkpoint(
        ep: int,
        total_reward: float,
        avg_rps: float,
        avg_speed: float,
        collision_count: int,
        done_reason: str,
        scenario_name: str,
    ):
        try:
            best_dir = os.path.join(agent.base_path, "reward_best_ckpt")
            os.makedirs(best_dir, exist_ok=True)
            for pattern in ("policy_net*", "value_net*", "config.json"):
                for src in glob.glob(os.path.join(agent.base_path, pattern)):
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(best_dir, os.path.basename(src)))
            meta_path = os.path.join(best_dir, "reward_best_metrics.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(
                    f"episode={int(ep)}\n"
                    f"total_reward={float(total_reward):.6f}\n"
                    f"avg_reward_per_step={float(avg_rps):.6f}\n"
                    f"avg_speed={float(avg_speed):.6f}\n"
                    f"collision_count={int(collision_count)}\n"
                    f"done_reason={str(done_reason)}\n"
                    f"scenario={str(scenario_name)}\n"
                )
        except Exception as e:
            print(f"[RewardBestCkpt] ⚠️ 保存reward最佳checkpoint失败: {e}")

    def _run_deterministic_eval(global_episode: int):
        if not DET_EVAL_ENABLE:
            return {}
        if DET_EVAL_EVERY <= 0 or DET_EVAL_EPISODES <= 0:
            return {}
        if global_episode % DET_EVAL_EVERY != 0:
            return {}

        old_logger = env.logger
        old_wandb = env.wandb_run
        old_episode = int(getattr(env, "current_episode", 0))
        old_obs_state = env.get_obs_norm_state() if hasattr(env, "get_obs_norm_state") else {}
        env.logger = None
        env.wandb_run = None

        coll_eps = 0
        succ_eps = 0
        speed_sum = 0.0
        step_sum = 0
        reward_sum = 0.0
        done_reasons = {}
        speed_min_lane = float(getattr(env.config, "success_min_avg_speed_lane", 3.0))
        speed_min_jaywalker = float(getattr(env.config, "success_min_avg_speed_jaywalker", 1.0))

        try:
            for _ in range(DET_EVAL_EPISODES):
                s = env.reset()
                eval_scenario = str(getattr(env.carla_env, "scenario", "")).lower()
                speed_min_eval = speed_min_jaywalker if ("jaywalker" in eval_scenario) else speed_min_lane
                ep_reward = 0.0
                ep_speed_sum = 0.0
                ep_steps = 0
                ep_coll = 0
                ep_done_reason = "unknown"
                info = {}

                for _t in range(1, DET_EVAL_TIMESTEPS + 1):
                    if isinstance(s, dict):
                        s_in = {f"state_{k}": v for k, v in s.items()}
                    else:
                        s_in = s
                    s_t = utils.to_tensor(s_in)

                    a_det = agent.network.act(
                        s_t,
                        use_old=bool(DET_EVAL_USE_OLD),
                        deterministic=True,
                    )
                    a_env = agent.convert_action(a_det)

                    s_next, r, d, info = env.step(a_env)
                    info = info or {}
                    ep_reward += float(r)
                    ep_steps += 1
                    ep_speed_sum += float(info.get("speed", 0.0))
                    if info.get("collision", 0.0):
                        ep_coll = 1
                    if d:
                        ep_done_reason = str(info.get("done_reason", "unknown"))
                        break
                    s = s_next

                coll_eps += int(ep_coll > 0)
                speed_avg_ep = ep_speed_sum / max(1, ep_steps)
                timeout_flag = bool(float((info or {}).get("timeout", 0.0)) > 0.5)
                ran_full = bool(timeout_flag or (ep_steps >= DET_EVAL_TIMESTEPS))
                bad_reasons_eval = {"collision", "offroad", "no_progress", "crawl_fail"}
                succ_eps += int(
                    (ep_coll == 0)
                    and ran_full
                    and (ep_done_reason not in bad_reasons_eval)
                    and (speed_avg_ep >= speed_min_eval)
                )
                speed_sum += speed_avg_ep
                step_sum += ep_steps
                reward_sum += ep_reward
                done_reasons[ep_done_reason] = done_reasons.get(ep_done_reason, 0) + 1
        except Exception as e:
            print(f"[DetEval] ⚠️ 评估失败: {e}")
        finally:
            env.logger = old_logger
            env.wandb_run = old_wandb
            env.current_episode = old_episode
            if old_obs_state:
                env.set_obs_norm_state(old_obs_state)

        n = max(1, DET_EVAL_EPISODES)
        return {
            "collision_rate": float(coll_eps / n),
            "success_rate": float(succ_eps / n),
            "avg_speed": float(speed_sum / n),
            "avg_steps": float(step_sum / n),
            "avg_reward": float(reward_sum / n),
            "done_reason_top": str(max(done_reasons.items(), key=lambda kv: kv[1])[0]) if done_reasons else "unknown",
        }

    def _decay_learning_rates(reason: str):
        try:
            old_p = float(getattr(agent.policy_lr, "value", agent.policy_lr()))
            old_v = float(getattr(agent.value_lr, "value", agent.value_lr()))
            new_p = max(DEGRADE_POLICY_LR_MIN, old_p * DEGRADE_POLICY_LR_SCALE)
            new_v = max(DEGRADE_VALUE_LR_MIN, old_v * DEGRADE_VALUE_LR_SCALE)

            if hasattr(agent.policy_lr, "value"):
                agent.policy_lr.value = float(new_p)
            if hasattr(agent.value_lr, "value"):
                agent.value_lr.value = float(new_v)

            print(
                f"[LRDecay] {reason}: "
                f"policy_lr {old_p:.2e}->{new_p:.2e}, value_lr {old_v:.2e}->{new_v:.2e}"
            )
        except Exception as e:
            print(f"[LRDecay] ⚠️ failed ({reason}): {e}")

    def _mean(dq):
        return float(sum(dq) / max(1, len(dq)))

    last_global_episode = int(start_episode)
    for local_episode in range(1, episodes + 1):
        episode = int(start_episode) + int(local_episode)
        last_global_episode = episode
        done_reason = "unknown"
        last_info = {}  # ✅ 保存本 episode 最后一步 info，避免作用域/未定义问题

        episode_std_mean_sum = 0.0
        episode_std_mean_count = 0
        episode_steer_delta_abs = []

        logger.start_episode(episode)

        # ============================================================
        # ✅ 课程学习：动态调整场景 + y_ref 难度（success_rate 门控）
        # ============================================================
        try:
            succ_rate_prev = _mean(dq_success) if len(dq_success) > 0 else 0.0
            window_ready = (len(dq_success) == W and local_episode >= MIN_EP)
            apply_curriculum(episode, env, succ_rate=succ_rate_prev, window_ready=window_ready)
        except Exception as e:
            print(f"[Curriculum] ⚠️ apply_curriculum failed: {e}")

        # ============================================================
        # ✅ 熵系数调度：先探索，后收敛
        # ============================================================
        entropy_rescue_active = 0
        try:
            ent_start = float(getattr(env.config, "entropy_start", 0.0200))
            ent_end = float(getattr(env.config, "entropy_end", 0.0040))
            ent_decay_ep = int(getattr(env.config, "entropy_decay_ep", 2600))
            if len(dq_done_collision) == W:
                collision_prev = _mean(dq_done_collision)
                succ_prev = _mean(dq_success) if len(dq_success) > 0 else 0.0
                rescue_coll_thr = float(getattr(env.config, "entropy_rescue_collision_rate", 0.45))
                rescue_succ_thr = float(getattr(env.config, "entropy_rescue_success_rate", 0.10))
                if collision_prev >= rescue_coll_thr and succ_prev <= rescue_succ_thr:
                    ent_start = max(ent_start, float(getattr(env.config, "entropy_rescue_start", 0.045)))
                    ent_end = max(ent_end, float(getattr(env.config, "entropy_rescue_end", 0.016)))
                    entropy_rescue_active = 1
            if episode <= forced_entropy_until_episode:
                ent_start = max(ent_start, HARD_BAD_ENT_START)
                ent_end = max(ent_end, HARD_BAD_ENT_END)
                entropy_rescue_active = 1
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
                    print(f"[EnvStep] ⚠️ simulator timeout at global_episode={episode}, t={t}: {msg}")
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

            # 记录控制后处理偏差（raw steer -> applied steer）
            raw_st = last_info.get("raw_steer", None)
            app_st = last_info.get("applied_steer", None)
            if raw_st is not None and app_st is not None:
                try:
                    raw_st = float(raw_st)
                    app_st = float(app_st)
                    if np.isfinite(raw_st) and np.isfinite(app_st):
                        episode_steer_delta_abs.append(abs(app_st - raw_st))
                except Exception:
                    pass

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

            # PPO 默认应使用“策略采样动作”的 log_prob。
            # 若改用执行后动作（applied_*），会改变优化目标，常导致比率语义失真。
            action_for_memory = action
            logp_for_memory = log_prob
            if PPO_STORE_EXECUTED_ACTION:
                exec_tb = float(last_info.get("applied_throttle_brake", a3[0]))
                exec_st = float(last_info.get("applied_steer", a3[1]))
                exec_yr = float(last_info.get("raw_y_ref", a3[2]))
                exec_action_np = np.array(
                    [np.clip(exec_tb, -1.0, 1.0), np.clip(exec_st, -1.0, 1.0), np.clip(exec_yr, -1.0, 1.0)],
                    dtype=np.float32,
                )
                action_for_memory = tf.convert_to_tensor(exec_action_np.reshape(1, 3), dtype=tf.float32)
                try:
                    exec_log_prob, _ = agent.network.log_prob_and_entropy(
                        state_t, action_for_memory, training=False, use_old=True
                    )
                    logp_for_memory = exec_log_prob
                except Exception:
                    logp_for_memory = log_prob

            agent.memory.append(state_t, action_for_memory, reward, value, logp_for_memory)

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
        if episode_steer_delta_abs:
            steer_arr = np.asarray(episode_steer_delta_abs, dtype=np.float32)
            steer_delta_mean = float(np.mean(steer_arr))
            steer_delta_p50 = float(np.percentile(steer_arr, 50))
            steer_delta_p90 = float(np.percentile(steer_arr, 90))
            steer_delta_p99 = float(np.percentile(steer_arr, 99))
        else:
            steer_delta_mean = 0.0
            steer_delta_p50 = 0.0
            steer_delta_p90 = 0.0
            steer_delta_p99 = 0.0
        logger.log_episode_reward(episode_reward)

        # ---------------------- update ----------------------
        if local_episode % agent.update_frequency == 0:
            temp_policy_loss, temp_value_loss, temp_entropy = [], [], []
            temp_policy_kl = []
            policy_kl_early_stop_count = 0

            value_batches = agent.get_value_batches()
            policy_batches = agent.get_policy_batches()

            # PPO 关键：在一次 update 开始时固定 old policy，
            # 整个 policy optimization 期间 ratio 都应基于同一份 old policy。
            if hasattr(agent, "network") and hasattr(agent.network, "update_old_policy"):
                try:
                    agent.network.update_old_policy()
                except Exception as e:
                    print(f"[PPO] ⚠️ update_old_policy(before) failed: {e}")

            stop_policy_epoch = False
            kl_stop_threshold = max(0.0, POLICY_TARGET_KL * POLICY_KL_STOP_MULT)
            for _ in range(agent.optimization_steps["policy"]):
                if stop_policy_epoch:
                    break
                for data_batch in policy_batches:
                    agent.seed_regularization()
                    total_loss, approx_kl, policy_grads = agent.get_policy_gradients(data_batch)

                    approx_kl_val = float(approx_kl.numpy()) if hasattr(approx_kl, "numpy") else float(approx_kl)
                    if np.isfinite(approx_kl_val):
                        temp_policy_kl.append(approx_kl_val)
                    if kl_stop_threshold > 0.0 and approx_kl_val > kl_stop_threshold:
                        policy_kl_early_stop_count += 1
                        stop_policy_epoch = True
                        break

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

            # policy 更新完成后，同步 old policy 供下一轮 rollout / update 使用
            if hasattr(agent, "network") and hasattr(agent.network, "update_old_policy"):
                try:
                    agent.network.update_old_policy()
                except Exception as e:
                    print(f"[PPO] ⚠️ update_old_policy(after) failed: {e}")

            # 与 PPOAgent.update() 对齐：推进 update_step（TRD/调度等依赖此计数）
            if hasattr(agent, "update_step"):
                try:
                    agent.update_step.assign_add(1)
                except Exception as e:
                    print(f"[PPO] ⚠️ update_step assign_add failed: {e}")

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
            if temp_policy_kl:
                policy_kl_tracker = float(sum(temp_policy_kl) / len(temp_policy_kl))
            policy_kl_early_stop_tracker = int(policy_kl_early_stop_count)

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

        bad_reasons = {"collision", "offroad", "no_progress", "crawl_fail"}  # ✅ 不把 time_limit 当失败
        no_collision = bool(episode_collision_count == 0)
        scenario_name = str(getattr(env.carla_env, "scenario", "")).lower()
        is_jaywalker = ("jaywalker" in scenario_name)
        success_speed_min = float(
            getattr(env.config, "success_min_avg_speed_jaywalker", 1.0)
            if is_jaywalker else
            getattr(env.config, "success_min_avg_speed_lane", 3.0)
        )
        speed_ok = bool(avg_speed >= success_speed_min)
        success = int(ran_full and no_collision and (done_reason not in bad_reasons) and speed_ok)
        safe_fast = int(no_collision and speed_ok and (done_reason not in bad_reasons))
        scenario_window.append((scenario_name, int(success), int(no_collision), int(ran_full)))

        # ============================================================
        # ✅ Early Stop 统计（每个episode都更新一次）
        # ============================================================
        dq_success.append(int(success))
        dq_ran_full.append(int(ran_full))
        dq_no_collision.append(int(no_collision))
        dq_avg_speed.append(float(avg_speed))
        dq_speed_ok.append(int(speed_ok))
        dq_safe_fast.append(int(safe_fast))
        dq_avg_rps.append(float(episode_reward / max(1, episode_steps)))
        dq_done_collision.append(int(done_reason == "collision"))
        dq_done_offroad.append(int(done_reason == "offroad"))
        dq_done_crawl_fail.append(int(done_reason == "crawl_fail"))
        dq_done_no_progress.append(int(done_reason == "no_progress"))
        dq_done_time_limit.append(int(done_reason == "time_limit"))
        dq_steer_delta_p90.append(float(steer_delta_p90))
        dq_hard_collision.append(int(episode_collision_count > 0))

        succ_rate = _mean(dq_success)
        ran_full_rate = _mean(dq_ran_full)
        no_collision_rate = _mean(dq_no_collision)
        collision_rate = 1.0 - no_collision_rate
        avg_speed_w = _mean(dq_avg_speed)
        speed_ok_rate = _mean(dq_speed_ok)
        safe_fast_rate = _mean(dq_safe_fast)
        done_collision_rate = _mean(dq_done_collision)
        done_offroad_rate = _mean(dq_done_offroad)
        done_crawl_fail_rate = _mean(dq_done_crawl_fail)
        done_no_progress_rate = _mean(dq_done_no_progress)
        done_time_limit_rate = _mean(dq_done_time_limit)
        steer_delta_p90_w = _mean(dq_steer_delta_p90)
        hard_collision_rate = _mean(dq_hard_collision)
        progress_score = (
            SCORE_W_SUCC * succ_rate
            + SCORE_W_RAN * ran_full_rate
            + SCORE_W_SAFE * no_collision_rate
        )

        # ---- 周期 deterministic 评估 ----
        det_eval = _run_deterministic_eval(episode)
        if det_eval:
            det_eval_last = det_eval.copy()
            print(
                "[DetEval] "
                f"ep={episode}, coll={det_eval.get('collision_rate', 0.0):.3f}, "
                f"succ={det_eval.get('success_rate', 0.0):.3f}, "
                f"speed={det_eval.get('avg_speed', 0.0):.3f}, "
                f"reason={det_eval.get('done_reason_top', 'unknown')}"
            )

        # ---- 收敛判据：连续满足 NEED_STABLE_WINDOWS 次 ----
        if (local_episode >= MIN_EP and len(dq_success) == W and
            succ_rate >= TARGET_SUCCESS and
            ran_full_rate >= TARGET_RAN_FULL and
            speed_ok_rate >= TARGET_SPEED_OK and
            avg_speed_w >= TARGET_AVG_SPEED and
            collision_rate <= MAX_COLLISION):
            stable_good_count += 1
        else:
            stable_good_count = 0

        # ---- 崩溃判据：连续 BAD_PATIENCE 次 ----
        bad_low_speed_stuck = (
            (avg_speed_w < BAD_AVG_SPEED)
            and (speed_ok_rate < BAD_SPEED_OK)
            and (safe_fast_rate < BAD_SAFE_FAST)
            and (progress_score < BAD_PROGRESS_MAX)
            and (
                (done_crawl_fail_rate > BAD_CRAWL_FAIL)
                or ((done_crawl_fail_rate + done_no_progress_rate) > BAD_STUCK_RATE)
            )
            and (succ_rate < BAD_SUCCESS)
        )
        if (
            local_episode >= BAD_MIN_EP and
            len(dq_success) == W and
            (
                (
                    BAD_USE_RAN_FULL and
                    local_episode >= BAD_RAN_FULL_MIN_EP and
                    ran_full_rate < BAD_RAN_FULL
                )
                or (
                    collision_rate > BAD_COLLISION and
                    (succ_rate < BAD_SUCCESS)
                )
                or bad_low_speed_stuck
            )
        ):
            bad_count += 1
        else:
            bad_count = 0

        # ---- 重启判据：长时间低速+低有效率，继续训练价值很低 ----
        restart_stuck_rate = (done_crawl_fail_rate + done_no_progress_rate)
        restart_bad = (
            RESTART_ENABLE
            and local_episode >= RESTART_MIN_EP
            and len(dq_success) == W
            and (succ_rate <= RESTART_SUCCESS_MAX)
            and (speed_ok_rate <= RESTART_SPEED_OK_MAX)
            and (safe_fast_rate <= RESTART_SAFE_FAST_MAX)
            and (avg_speed_w <= RESTART_AVG_SPEED_MAX)
            and (
                (collision_rate >= RESTART_COLLISION_MIN)
                or (restart_stuck_rate >= RESTART_STUCK_MIN)
            )
        )
        if restart_bad:
            restart_count += 1
        else:
            restart_count = 0

        # ---- 平台期判据：success + 综合进步分数长期没提升 ----
        if local_episode >= MIN_EP and len(dq_success) == W:
            improved = False
            if succ_rate > best_succ_rate + IMPROVE_EPS:
                best_succ_rate = succ_rate
                best_succ_episode = episode
                improved = True
            if progress_score > best_progress_score + IMPROVE_EPS:
                best_progress_score = progress_score
                best_progress_episode = episode
                improved = True
            if improved:
                plateau_count = 0
            else:
                plateau_count += 1

        # ---- Best checkpoint：优先成功率/安全快，其次低碰撞，再看速度 ----
        if local_episode >= MIN_EP and len(dq_success) == W:
            avg_rps_w = _mean(dq_avg_rps)
            current_key = (
                round(-float(succ_rate), 6),
                round(-float(safe_fast_rate), 6),
                round(float(collision_rate), 6),
                round(-float(speed_ok_rate), 6),
                round(-float(ran_full_rate), 6),
                round(-float(avg_speed_w), 6),
                round(-float(avg_rps_w), 6),
            )
            if (best_ckpt_key is None) or (current_key < best_ckpt_key):
                best_ckpt_key = current_key
                agent.save()
                _save_best_checkpoint(
                    ep=episode,
                    coll=collision_rate,
                    succ=succ_rate,
                    ran=ran_full_rate,
                    avg_r=avg_rps_w,
                    avg_speed=avg_speed_w,
                    speed_ok_rate=speed_ok_rate,
                    safe_fast_rate=safe_fast_rate,
                )
                print(
                    f"[BestCkpt] ✅ episode={episode}, "
                    f"collision_W={collision_rate:.3f}, success_W={succ_rate:.3f}, "
                    f"speed_ok_W={speed_ok_rate:.3f}, safe_fast_W={safe_fast_rate:.3f}"
                )

        # ---- Reward-best checkpoint：按 episode total_reward 最大保存 ----
        if REWARD_BEST_CKPT_ENABLE and local_episode >= max(1, REWARD_BEST_MIN_EP):
            avg_rps_ep = float(episode_reward / max(1, episode_steps))
            reward_better = (float(episode_reward) > float(best_total_reward) + 1e-9)
            reward_tie_better = (
                abs(float(episode_reward) - float(best_total_reward)) <= 1e-9
                and avg_rps_ep > float(best_avg_rps_for_reward_best) + 1e-9
            )
            if reward_better or reward_tie_better:
                best_total_reward = float(episode_reward)
                best_total_reward_episode = int(episode)
                best_avg_rps_for_reward_best = float(avg_rps_ep)
                agent.save()
                _save_reward_best_checkpoint(
                    ep=episode,
                    total_reward=float(episode_reward),
                    avg_rps=float(avg_rps_ep),
                    avg_speed=float(avg_speed),
                    collision_count=int(episode_collision_count),
                    done_reason=str(done_reason),
                    scenario_name=str(scenario_name),
                )
                print(
                    f"[RewardBestCkpt] ✅ episode={episode}, total_reward={episode_reward:.2f}, "
                    f"avg_rps={avg_rps_ep:.4f}, speed={avg_speed:.2f}, collision={episode_collision_count}"
                )

        # ---- 连续退化保护：回滚到 best + 降低学习率 ----
        degrade_bad_window = (
            DEGRADE_ROLLBACK_ENABLE
            and local_episode >= DEGRADE_ROLLBACK_MIN_EP
            and len(dq_success) == W
            and (best_ckpt_key is not None)
            and (collision_rate >= DEGRADE_ROLLBACK_COLLISION_MIN)
            and (safe_fast_rate <= DEGRADE_ROLLBACK_SAFE_FAST_MAX)
            and (succ_rate <= DEGRADE_ROLLBACK_SUCCESS_MAX)
        )
        if degrade_bad_window:
            degrade_streak += 1
        else:
            degrade_streak = 0

        can_rollback_now = (
            (episode - last_degrade_rollback_episode) >= max(1, DEGRADE_ROLLBACK_COOLDOWN)
            and (degrade_rollback_count < max(0, DEGRADE_ROLLBACK_MAX_RECOVERS))
        )
        if degrade_streak >= DEGRADE_ROLLBACK_PATIENCE and can_rollback_now:
            restored = False
            if RESTORE_BEST_ON_BAD_STOP:
                restored = _restore_best_checkpoint("DEGRADE_ROLLBACK")
            if restored:
                degrade_rollback_count += 1
                last_degrade_rollback_episode = episode
                degrade_streak = 0
                _decay_learning_rates("DEGRADE_ROLLBACK")
            else:
                print("[DegradeRollback] ⚠️ 恢复 best 失败，跳过本次回滚。")

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
                "episode/steer_delta_abs_mean": float(steer_delta_mean),
                "episode/steer_delta_abs_p50": float(steer_delta_p50),
                "episode/steer_delta_abs_p90": float(steer_delta_p90),
                "episode/steer_delta_abs_p99": float(steer_delta_p99),

                "training/policy_loss": float(policy_loss_tracker),
                "training/value_loss": float(value_loss_tracker),
                "training/entropy": float(entropy_tracker),
                "training/policy_approx_kl": float(policy_kl_tracker),
                "training/policy_kl_early_stop_count": int(policy_kl_early_stop_tracker),
                "training/entropy_rescue_active": int(entropy_rescue_active),
                "training/update_count": int(update_count),
                "training/std_rollout_mean": float(episode_std_mean_sum / max(1, episode_std_mean_count)),

                # ✅ early-stop 观测指标（非常建议记录，方便你看是不是在收敛）
                "early/success_rate_W": float(succ_rate),
                "early/ran_full_rate_W": float(ran_full_rate),
                "early/collision_rate_W": float(collision_rate),
                "early/avg_speed_W": float(avg_speed_w),
                "early/speed_ok_rate_W": float(speed_ok_rate),
                "early/safe_fast_rate_W": float(safe_fast_rate),
                "early/stable_good_count": int(stable_good_count),
                "early/bad_count": int(bad_count),
                "early/restart_count": int(restart_count),
                "early/plateau_count": int(plateau_count),
                "early/best_succ_rate": float(best_succ_rate),
                "early/best_succ_episode": int(best_succ_episode),
                "early/progress_score_W": float(progress_score),
                "early/best_progress_score": float(best_progress_score),
                "early/best_progress_episode": int(best_progress_episode),
                "early/best_total_reward": float(best_total_reward if np.isfinite(best_total_reward) else 0.0),
                "early/best_total_reward_episode": int(best_total_reward_episode),
                "early/best_avg_reward_per_step_for_reward_best": float(
                    best_avg_rps_for_reward_best if np.isfinite(best_avg_rps_for_reward_best) else 0.0
                ),
                "early/done_collision_rate_W": float(done_collision_rate),
                "early/done_offroad_rate_W": float(done_offroad_rate),
                "early/done_crawl_fail_rate_W": float(done_crawl_fail_rate),
                "early/done_no_progress_rate_W": float(done_no_progress_rate),
                "early/done_time_limit_rate_W": float(done_time_limit_rate),
                "early/steer_delta_abs_p90_W": float(steer_delta_p90_w),
                f"early/hard_collision_rate_W{int(HARD_W)}": float(hard_collision_rate),
                "early/hard_bad_recover_count": int(hard_bad_recover_count),
                "early/forced_entropy_until_ep": int(forced_entropy_until_episode),
                "early/degrade_streak": int(degrade_streak),
                "early/degrade_rollback_count": int(degrade_rollback_count),

                "scenario_success/cones": _scenario_metric(1, "cones"),
                "scenario_success/trimma": _scenario_metric(1, "trimma"),
                "scenario_success/construction_lane_change": _scenario_metric(1, "construction_lane_change"),
                "scenario_success/jaywalker": _scenario_metric(1, "jaywalker"),
            }
            if det_eval_last:
                wandb_metrics.update({
                    "eval_det/collision_rate": float(det_eval_last.get("collision_rate", 0.0)),
                    "eval_det/success_rate": float(det_eval_last.get("success_rate", 0.0)),
                    "eval_det/avg_speed": float(det_eval_last.get("avg_speed", 0.0)),
                    "eval_det/avg_steps": float(det_eval_last.get("avg_steps", 0.0)),
                    "eval_det/avg_reward": float(det_eval_last.get("avg_reward", 0.0)),
                })
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
                checkpoint_path = os.path.join(agent.base_path, "policy_net_lat.index")
                if os.path.exists(checkpoint_path):
                    wandb_run.save(os.path.join(agent.base_path, "policy_net*"))
                    wandb_run.save(os.path.join(agent.base_path, "value_net*"))

        if state_callback is not None:
            try:
                state_callback(int(episode), env)
            except Exception as e:
                print(f"[Resume] ⚠️ 续训状态保存失败: {e}")

        if episode % 10 == 0:
            logger.print_summary(window=10)
            print(
                "[Diag] "
                f"done_reason_W: collision={done_collision_rate:.2f}, "
                f"offroad={done_offroad_rate:.2f}, crawl_fail={done_crawl_fail_rate:.2f}, "
                f"no_progress={done_no_progress_rate:.2f}, time_limit={done_time_limit_rate:.2f} | "
                f"speed_ok_W={speed_ok_rate:.2f}, safe_fast_W={safe_fast_rate:.2f}, avg_speed_W={avg_speed_w:.2f}, "
                f"restart_count={restart_count}/{RESTART_PATIENCE} | "
                f"steer_delta_abs(ep p50/p90/p99)={steer_delta_p50:.3f}/{steer_delta_p90:.3f}/{steer_delta_p99:.3f}, "
                f"p90_W={steer_delta_p90_w:.3f}"
            )

        # ============================================================
        # ✅ Early Stop 触发：收敛 / 平台期 / 崩溃
        # ============================================================
        if enable_early_stop and local_episode >= MIN_EP and len(dq_success) == W:
            if RESTART_ENABLE and restart_count >= RESTART_PATIENCE:
                print(
                    f"\n❌ EarlyStop: RESTART_REQUIRED. "
                    f"succ_rate(W)={succ_rate:.3f}, speed_ok_rate(W)={speed_ok_rate:.3f}, "
                    f"safe_fast_rate(W)={safe_fast_rate:.3f}, avg_speed(W)={avg_speed_w:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}, "
                    f"crawl+no_progress(W)={(done_crawl_fail_rate + done_no_progress_rate):.3f}\n"
                )
                agent.save()
                stop_reason = "RESTART_REQUIRED"
                break

            if (
                HARD_BAD_ENABLE
                and local_episode >= HARD_BAD_MIN_EP
                and len(dq_hard_collision) == HARD_W
                and hard_collision_rate > HARD_BAD_COLLISION
            ):
                print(
                    f"\n❌ EarlyStop: HARD_BAD_WINDOW. "
                    f"collision_rate(W{HARD_W})={hard_collision_rate:.3f} > {HARD_BAD_COLLISION:.3f}\n"
                )
                allow_recover = (
                    HARD_BAD_RESTORE_CONTINUE
                    and ((episode - last_hard_recover_episode) >= max(1, HARD_BAD_RECOVER_COOLDOWN))
                    and (hard_bad_recover_count < max(0, HARD_BAD_MAX_RECOVERS))
                )
                cooldown_left = max(
                    0,
                    int(max(1, HARD_BAD_RECOVER_COOLDOWN) - (episode - last_hard_recover_episode)),
                )
                recovered = False
                if RESTORE_BEST_ON_BAD_STOP:
                    recovered = _restore_best_checkpoint("HARD_BAD_WINDOW")
                if allow_recover and recovered:
                    hard_bad_recover_count += 1
                    last_hard_recover_episode = episode
                    forced_entropy_until_episode = int(episode + max(0, HARD_BAD_ENT_BOOST_EP))
                    bad_count = 0
                    plateau_count = 0
                    print(
                        f"[HardRecover] ♻️ 已恢复最佳权重并继续训练 "
                        f"(recover={hard_bad_recover_count}/{HARD_BAD_MAX_RECOVERS}, "
                        f"entropy_boost_until_ep={forced_entropy_until_episode})"
                    )
                    agent.save()
                else:
                    print(
                        "[HardRecover] stop "
                        f"(allow={int(allow_recover)}, recovered={int(recovered)}, "
                        f"recover_count={hard_bad_recover_count}/{HARD_BAD_MAX_RECOVERS}, "
                        f"cooldown_left={cooldown_left}, restore_continue={int(HARD_BAD_RESTORE_CONTINUE)})"
                    )
                    agent.save()
                    stop_reason = "HARD_BAD_WINDOW"
                    break

            if stable_good_count >= NEED_STABLE_WINDOWS:
                print(
                    f"\n✅ EarlyStop: CONVERGED. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}, speed_ok_rate(W)={speed_ok_rate:.3f}, "
                    f"avg_speed(W)={avg_speed_w:.3f}\n"
                )
                agent.save()
                stop_reason = "CONVERGED"
                break

            if bad_count >= BAD_PATIENCE:
                print(
                    f"\n�� EarlyStop: DIVERGED/BAD. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}, speed_ok_rate(W)={speed_ok_rate:.3f}, "
                    f"avg_speed(W)={avg_speed_w:.3f}, crawl_fail_rate(W)={done_crawl_fail_rate:.3f}\n"
                )
                if RESTORE_BEST_ON_BAD_STOP:
                    _restore_best_checkpoint("DIVERGED_BAD")
                agent.save()
                stop_reason = "DIVERGED_BAD"
                break

            plateau_quality_ready = (
                (succ_rate >= PLATEAU_CURR_MIN_SUCCESS)
                and (speed_ok_rate >= PLATEAU_CURR_MIN_SPEED_OK)
                and (safe_fast_rate >= PLATEAU_CURR_MIN_SAFE_FAST)
                and (collision_rate <= PLATEAU_CURR_MAX_COLLISION)
            )
            plateau_guard_ready = (
                local_episode >= max(MIN_EP, PLATEAU_MIN_EP)
                and (
                    best_succ_rate >= PLATEAU_MIN_BEST_SUCCESS
                    or best_progress_score >= PLATEAU_MIN_PROGRESS_SCORE
                )
                and plateau_quality_ready
            )
            if plateau_guard_ready and plateau_count >= PLATEAU_PATIENCE:
                print(
                    f"\n�� EarlyStop: PLATEAU. "
                    f"best_succ_rate={best_succ_rate:.3f} at ep {best_succ_episode}, "
                    f"best_progress_score={best_progress_score:.3f} at ep {best_progress_episode}, "
                    f"current succ_rate(W)={succ_rate:.3f}, progress_score(W)={progress_score:.3f}\n"
                )
                if RESTORE_BEST_ON_BAD_STOP:
                    _restore_best_checkpoint("PLATEAU")
                agent.save()
                stop_reason = "PLATEAU"
                break

    if RESTORE_BEST_AT_END and best_ckpt_key is not None and stop_reason != "CONVERGED":
        _restore_best_checkpoint(f"TRAIN_END:{stop_reason or 'NATURAL_END'}")

    return int(last_global_episode)


def train_ppo(profile: str = "default"):
    profile = str(profile).strip().lower()
    if profile in ("obs30", "default", ""):
        profile = "default"
    elif profile in ("standalone", "simple", "4scenarios", "four_scenarios"):
        profile = "4scenarios" if profile in ("4scenarios", "four_scenarios") else profile
    else:
        print(f"[Profile] ⚠️ 未知profile={profile}，回退到 default")
        profile = "default"

    print("=" * 70)
    print("PPO单独训练 - 带Wandb实时监控（修正版：✅W&B step对齐 + ✅episode横轴 + ✅y_ref开关）")
    print(f"训练配置档: {profile}")
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
                    "entropy_reg": 0.0080,
                    "entropy_start": 0.0200,
                    "entropy_end": 0.0040,
                    "entropy_decay_ep": 2600,
                    "entropy_rescue_collision_rate": 0.45,
                    "entropy_rescue_success_rate": 0.10,
                    "entropy_rescue_start": 0.030,
                    "entropy_rescue_end": 0.0100,
                    "batch_size": 128,           # 修正：从256改为128
                    "update_frequency": 2,       # 新增：与agent一致
                    "optimization_steps": (4, 4),
                    "trd_loss_coef": 0.0,
                    "train_onpolicy_action_mode": True,
                    "ppo_store_executed_action": True,
                    "episodes": 1500,
                    "max_steps_per_episode": 256,
                    "wandb_step_log_interval": 50,
                    "load_existing": False,
                    "resume_quality_guard": True,
                    "resume_min_success_w": 0.03,
                    "resume_min_speed_ok_w": 0.30,
                    "resume_min_safe_fast_w": 0.15,
                    "finetune_on_resume": True,
                    "resume_policy_lr_scale": 0.5,
                    "resume_value_lr_scale": 0.5,
                    "resume_entropy_scale": 0.9,
                    "resume_opt_steps_scale": 0.75,
                    "enable_early_stop": True,
                    "early_window": 20,
                    "early_min_episode": 60,
                    "early_target_success": 0.80,
                    "early_target_ran_full": 0.80,
                    "early_target_speed_ok": 0.80,
                    "early_target_avg_speed": 3.0,
                    "early_max_collision": 0.10,
                    "early_need_stable_windows": 3,
                    "early_plateau_patience": 300,
                    "early_improve_eps": 0.01,
                    "early_plateau_min_episode": 1800,
                    "early_plateau_min_best_success": 0.35,
                    "early_plateau_min_progress_score": 0.65,
                    "early_plateau_curr_min_success": 0.05,
                    "early_plateau_curr_min_speed_ok": 0.35,
                    "early_plateau_curr_min_safe_fast": 0.10,
                    "early_plateau_curr_max_collision": 0.35,
                    "early_progress_w_success": 0.55,
                    "early_progress_w_ran_full": 0.30,
                    "early_progress_w_no_collision": 0.15,
                    "early_bad_min_episode": 2000,
                    "early_bad_patience": 120,
                    "early_bad_collision": 0.75,
                    "early_bad_ran_full": 0.30,
                    "early_bad_success": 0.20,
                    "early_bad_avg_speed": 2.9,
                    "early_bad_speed_ok": 0.25,
                    "early_bad_stuck_rate": 0.65,
                    "early_bad_use_ran_full": False,
                    "early_bad_ran_full_min_episode": 180,
                    "early_bad_crawl_fail": 0.85,
                    "early_bad_progress_max": 0.12,
                    "early_bad_safe_fast": 0.15,
                    "early_restart_enable": False,
                    "early_restart_min_episode": 600,
                    "early_restart_patience": 12,
                    "early_restart_success_max": 0.03,
                    "early_restart_speed_ok_max": 0.10,
                    "early_restart_safe_fast_max": 0.10,
                    "early_restart_avg_speed_max": 1.8,
                    "early_restart_collision_min": 0.35,
                    "early_restart_stuck_min": 0.70,
                    "hard_bad_enable": False,
                    "hard_bad_window": 120,
                    "hard_bad_collision": 0.85,
                    "hard_bad_min_episode": 320,
                    "hard_bad_restore_continue": True,
                    "hard_bad_recover_cooldown": 40,
                    "hard_bad_max_recovers": 10,
                    "hard_bad_entropy_boost_episodes": 120,
                    "hard_bad_entropy_boost_start": 0.045,
                    "hard_bad_entropy_boost_end": 0.016,
                    "restore_best_on_bad_stop": True,
                    "restore_best_at_end": True,
                    "strict_training_profile": False,
                    "deterministic_eval_enable": True,
                    "deterministic_eval_every": 50,
                    "deterministic_eval_episodes": 2,
                    "deterministic_eval_timesteps": 128,
                    "deterministic_eval_use_old_policy": False,
                    # 严格 on-policy：优先使用执行后动作重算 logp，降低动作后处理带来的偏移。
                    "ppo_store_executed_action": True,
                    "enable_forced_curriculum": True,
                    "force_stage2_episode": 140,
                    "force_stage3_episode": 360,
                    "force_stage2_min_success": 0.0,
                    "force_stage3_min_success": 0.0,
                    "stage1_success_threshold": 0.30,
                    "stage2_success_threshold": 0.60,
                    "stage1_yref_gain": 0.0015,
                    "stage2_yref_gain": 0.003,
                    "stage3_use_yref": True,
                    "stage3_yref_warmup_end": 520,
                    "stage3_yref_gain_warmup": 0.004,
                    "stage3_yref_gain_full": 0.008,

                    # 不篡改动作：保持 False（开启会破坏严格PPO假设）
                    "use_action_bias": False,
                    "use_forced_throttle": False,
                    "action_bias_strength": 0.7,
                    "action_bias_decay": 0.999,

                    # ✅ 两版本开关
                    # 兼容旧字段 + 新字段（训练中会用 curriculum 动态开关）
                    "use_yref_mapping": True,   # 兼容旧字段
                    "use_yref_in_steer": True,  # 新字段
                    "yref_gain": 0.005,
                    "yref_steer_gain": 0.005,
                    "yref_penalty": 0.0,
                    "use_aggressive_reward": False,
                    "anti_crawl_enable": True,
                    "anti_crawl_speed_lane": 3.0,
                    "anti_crawl_speed_jaywalker": 1.0,
                    "anti_crawl_penalty_k_lane": 0.32,
                    "anti_crawl_penalty_k_jaywalker": 0.08,
                    "anti_crawl_penalty_cap_lane": 1.3,
                    "anti_crawl_penalty_cap_jaywalker": 0.6,
                    "anti_crawl_terminate_enable": True,
                    "anti_crawl_warmup_steps_lane": 140,
                    "anti_crawl_warmup_steps_jaywalker": 120,
                    "anti_crawl_min_avg_speed_lane": 3.0,
                    "anti_crawl_min_avg_speed_jaywalker": 0.9,
                    "anti_crawl_recent_window_lane": 48,
                    "anti_crawl_recent_window_jaywalker": 36,
                    "anti_crawl_min_recent_speed_lane": 3.0,
                    "anti_crawl_min_recent_speed_jaywalker": 0.9,
                    "anti_crawl_terminate_penalty_lane": 45.0,
                    "anti_crawl_terminate_penalty_jaywalker": 25.0,
                    "anti_crawl_terminate_start_episode": 420,
                    "anti_crawl_terminate_speed_ratio": 0.78,
                    "anti_crawl_min_travel_before_terminate_lane": 16.0,
                    "anti_crawl_min_travel_before_terminate_jaywalker": 6.0,
                    "anti_crawl_terminate_obs_exempt_dist_lane": 12.0,
                    "anti_crawl_terminate_obs_exempt_dist_jaywalker": 18.0,
                    "anti_crawl_penalty_near_obs_scale_lane": 0.35,
                    "anti_crawl_penalty_near_obs_scale_jaywalker": 0.60,
                    "bad_brake_penalty_enable": False,
                    "bad_brake_k_lane": 0.35,
                    "bad_brake_k_jaywalker": 0.10,
                    "bad_brake_apply_speed_min_lane": 3.0,
                    "bad_brake_apply_speed_min_jaywalker": 1.2,
                    "bad_brake_apply_obs_dist_min_lane": 16.0,
                    "bad_brake_apply_obs_dist_min_jaywalker": 10.0,
                    "enable_low_speed_steer_scale": True,
                    "low_speed_steer_speed": 1.2,
                    "low_speed_steer_min_scale": 0.90,
                    "enable_steer_smoothing": False,
                    "steer_smooth_alpha": 0.82,
                    "enable_steer_rate_limit": True,
                    "steer_rate_limit_lane": 0.30,
                    "steer_rate_limit_jaywalker": 0.14,
                    "steer_rate_limit_obs_lane": 0.46,
                    "steer_rate_limit_obs_jaywalker": 0.26,
                    "policy_min_scale": 0.06,
                    "policy_min_scale_lat": 0.08,
                    "policy_min_scale_lon": 0.06,
                    "policy_raw_scale_bias_init": -1.20,
                    "policy_raw_scale_bias_lat": -1.20,
                    "policy_raw_scale_bias_lon": -1.20,
                    "policy_raw_scale_clip_low": -4.0,
                    "policy_raw_scale_clip_high": -0.35,
                    "enable_straight_stability": True,
                    "straight_stability_speed_lane": 2.8,
                    "straight_stability_speed_jaywalker": 1.4,
                    "straight_stability_lane_ratio": 0.15,
                    "straight_steer_damp": 0.60,
                    "straight_steer_deadband": 0.020,
                    "enable_steer_delta_clip": True,
                    "max_steer_postprocess_delta": 0.22,
                    "obstacle_steer_assist_enable": False,
                    "obstacle_steer_assist_dist_lane": 10.0,
                    "obstacle_steer_assist_dist_jaywalker": 10.0,
                    "obstacle_steer_assist_gain_lane": 0.14,
                    "obstacle_steer_assist_gain_jaywalker": 0.10,
                    "obstacle_steer_assist_max_lane": 0.20,
                    "obstacle_steer_assist_max_jaywalker": 0.16,
                    "obstacle_steer_assist_lat_eps": 0.15,
                    "obstacle_steer_assist_min_factor": 0.20,
                    "offroad_margin": 1.25,
                    "obs_control_lat_tol": 2.8,
                    "obs_control_lat_tol_lane": 3.5,
                    "obs_control_lat_tol_jaywalker": 3.0,
                    "obs_front_priority_fwd_min": -0.5,
                    "obs_front_priority_lat_tol": 3.2,
                    "obs_reward_lat_tol_lane": 3.2,
                    "obs_reward_lat_tol_jaywalker": 3.0,
                    "obs_reward_gate_floor_lane": 0.02,
                    "obs_reward_gate_floor_jaywalker": 0.20,
                    "avoid_fwd_lane": 30.0,
                    "avoid_fwd_jaywalker": 22.0,
                    "safe_dist_lane": 14.0,
                    "safe_dist_jaywalker": 10.0,
                    "k_avoid_lat_lane": 1.40,
                    "k_avoid_lat_jaywalker": 0.45,
                    "k_avoid_lat_cap_lane": 1.10,
                    "k_avoid_lat_cap_jaywalker": 0.18,
                    "w_obs_clear_lane": 8.5,
                    "w_obs_clear_jaywalker": 4.2,
                    "w_obs_speed_lane": 2.2,
                    "w_obs_speed_jaywalker": 1.4,
                    "w_obs_danger_lane": 4.5,
                    "w_obs_danger_jaywalker": 1.2,
                    "obs_danger_dist_lane": 16.0,
                    "obs_danger_dist_jaywalker": 9.0,
                    "w_obs_ttc_lane": 4.5,
                    "w_obs_ttc_jaywalker": 2.5,
                    "obs_ttc_crit_lane": 3.2,
                    "obs_ttc_crit_jaywalker": 2.2,
                    "speed_floor_enable_lane": True,
                    "speed_floor_target_lane": 3.0,
                    "speed_floor_k_lane": 1.2,
                    "speed_floor_obs_exempt_dist_lane": 14.0,
                    "speed_floor_lane_ratio_max": 0.85,
                    "obs_ttc_brake_crit_lane": 2.0,
                    "obs_ttc_brake_crit_jaywalker": 2.4,
                    "obs_ttc_brake_gain_lane": 0.38,
                    "obs_ttc_brake_gain_jaywalker": 0.50,
                    "k_collision_terminal": 320.0,
                    "k_offroad_terminal": 90.0,
                    "k_no_progress_terminal": 80.0,
                    "k_progress": 4.2,
                    "k_lane": 0.55,
                    "k_danger": 0.80,
                    "danger_start_ratio": 0.30,
                    "speed_reward_sigma": 1.4,
                    "safety_penalty_scale": 1.0,
                    "low_speed_brake_cut_speed": 0.8,
                    "low_speed_throttle_floor_speed": 1.3,
                    "low_speed_throttle_floor": 0.20,
                    "min_throttle_when_stuck": 0.30,

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
    config.use_yref_in_steer = True
    config.yref_steer_gain = 0.005
    # 兼容旧字段
    config.use_yref_mapping = True
    config.yref_gain = 0.005

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
    config.low_speed_throttle_floor_speed = 1.3
    config.low_speed_throttle_floor = 0.20
    config.low_speed_brake_cut_speed = 0.8
    config.min_throttle_when_stuck = 0.30
    config.obs_control_lat_tol = 2.8
    config.obs_front_priority_fwd_min = -0.5
    config.obs_front_priority_lat_tol = 3.2
    # 低速时增强转向可控性，降低 raw/applied 偏差
    config.enable_low_speed_steer_scale = True
    config.low_speed_steer_speed = 1.2
    config.low_speed_steer_min_scale = 0.90
    config.enable_steer_smoothing = False
    config.steer_smooth_alpha = 0.82
    config.enable_steer_rate_limit = True
    config.steer_rate_limit_lane = 0.30
    config.steer_rate_limit_jaywalker = 0.14
    config.steer_rate_limit_obs_lane = 0.46
    config.steer_rate_limit_obs_jaywalker = 0.26
    config.policy_min_scale = 0.06
    config.policy_min_scale_lat = 0.08
    config.policy_min_scale_lon = 0.06
    config.policy_raw_scale_bias_init = -1.20
    config.policy_raw_scale_bias_lat = -1.20
    config.policy_raw_scale_bias_lon = -1.20
    config.policy_raw_scale_clip_low = -4.0
    config.policy_raw_scale_clip_high = -0.35
    config.enable_straight_stability = True
    config.straight_stability_speed_lane = 2.8
    config.straight_stability_speed_jaywalker = 1.4
    config.straight_stability_lane_ratio = 0.15
    config.straight_steer_damp = 0.60
    config.straight_steer_deadband = 0.020
    config.enable_steer_delta_clip = True
    config.max_steer_postprocess_delta = 0.22
    config.obstacle_steer_assist_enable = False
    config.obstacle_steer_assist_dist_lane = 10.0
    config.obstacle_steer_assist_dist_jaywalker = 10.0
    config.obstacle_steer_assist_gain_lane = 0.14
    config.obstacle_steer_assist_gain_jaywalker = 0.10
    config.obstacle_steer_assist_max_lane = 0.20
    config.obstacle_steer_assist_max_jaywalker = 0.16
    config.obstacle_steer_assist_lat_eps = 0.15
    config.obstacle_steer_assist_min_factor = 0.20

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
    config.obs_throttle_cap_dist_lane = 9.0
    config.obs_throttle_cap_lane = 0.50
    config.obs_throttle_cap_dist_jaywalker = 11.0
    config.obs_throttle_cap_jaywalker = 0.24
    config.obs_control_lat_tol_lane = 3.5
    config.obs_control_lat_tol_jaywalker = 3.0
    # obstacle brake shield 分场景：lane 更少“过度刹停”，jaywalker 继续保守
    config.obs_brake_dist_lane = 5.5
    config.obs_brake_value_lane = 0.28
    config.obs_brake_hard_dist_lane = 3.0
    config.obs_brake_hard_value_lane = 0.58
    config.obs_brake_dist_jaywalker = 7.0
    config.obs_brake_value_jaywalker = 0.45
    config.obs_brake_hard_dist_jaywalker = 3.2
    config.obs_brake_hard_value_jaywalker = 0.70
    config.obs_ttc_brake_crit_lane = 2.0
    config.obs_ttc_brake_crit_jaywalker = 2.4
    config.obs_ttc_brake_gain_lane = 0.38
    config.obs_ttc_brake_gain_jaywalker = 0.50
    config.low_speed_brake_cut_speed_lane = 3.0
    config.low_speed_brake_cut_speed_jaywalker = 0.5
    config.low_speed_throttle_floor_speed_lane = 2.8
    config.low_speed_throttle_floor_speed_jaywalker = 0.7
    config.low_speed_throttle_floor_lane = 0.32
    config.low_speed_throttle_floor_jaywalker = 0.10
    config.min_throttle_when_stuck_lane = 0.45
    config.low_speed_brake_cut_scale_lane = 0.10
    config.low_speed_brake_cut_scale_jaywalker = 0.30
    config.min_throttle_when_stuck_jaywalker = 0.16

    # ✅ 场景感知速度策略（非 jaywalker 更积极，jaywalker 更保守）
    config.target_speed_lane = 4.8
    config.target_speed_jaywalker = 4.0
    config.v_max_lane = 7.0
    config.v_max_jaywalker = 5.5
    config.overspeed_start_lane = 5.2
    config.overspeed_start_jaywalker = 4.5
    config.v_cap_near_lane = 4.2
    config.v_cap_near_jaywalker = 2.2
    config.k_speed_lane = 2.10
    config.k_speed_jaywalker = 1.1
    config.k_overspeed_lane = 0.20
    config.k_overspeed_jaywalker = 0.15
    # anti-crawl: 防止“低速苟活”
    config.anti_crawl_enable = True
    # 避免与 CarlaEnv 内部低速惩罚重复叠加；训练默认关闭 wrapper 侧连续低速惩罚。
    config.wrapper_crawl_penalty_enable = False
    config.anti_crawl_speed_lane = 3.0
    config.anti_crawl_speed_jaywalker = 1.0
    config.anti_crawl_penalty_k_lane = 0.32
    config.anti_crawl_penalty_k_jaywalker = 0.08
    config.anti_crawl_penalty_cap_lane = 1.3
    config.anti_crawl_penalty_cap_jaywalker = 0.6
    config.anti_crawl_terminate_enable = True
    config.anti_crawl_warmup_steps_lane = 140
    config.anti_crawl_warmup_steps_jaywalker = 120
    config.anti_crawl_min_avg_speed_lane = 3.0
    config.anti_crawl_min_avg_speed_jaywalker = 0.9
    config.anti_crawl_recent_window_lane = 48
    config.anti_crawl_recent_window_jaywalker = 36
    config.anti_crawl_min_recent_speed_lane = 3.0
    config.anti_crawl_min_recent_speed_jaywalker = 0.9
    config.anti_crawl_terminate_penalty_lane = 45.0
    config.anti_crawl_terminate_penalty_jaywalker = 25.0
    config.anti_crawl_terminate_start_episode = 420
    config.anti_crawl_terminate_speed_ratio = 0.78
    config.anti_crawl_min_travel_before_terminate_lane = 16.0
    config.anti_crawl_min_travel_before_terminate_jaywalker = 6.0
    config.anti_crawl_terminate_obs_exempt_dist_lane = 12.0
    config.anti_crawl_terminate_obs_exempt_dist_jaywalker = 18.0
    config.anti_crawl_penalty_near_obs_scale_lane = 0.35
    config.anti_crawl_penalty_near_obs_scale_jaywalker = 0.60
    config.bad_brake_penalty_enable = False
    config.bad_brake_k_lane = 0.35
    config.bad_brake_k_jaywalker = 0.10
    config.bad_brake_apply_speed_min_lane = 3.0
    config.bad_brake_apply_speed_min_jaywalker = 1.2
    config.bad_brake_apply_obs_dist_min_lane = 16.0
    config.bad_brake_apply_obs_dist_min_jaywalker = 10.0

    config.low_speed_th_lane = 3.2
    config.low_speed_th_jaywalker = 0.6
    config.k_low_speed_lane = 0.90
    config.k_low_speed_jaywalker = 0.04
    # lane 场景最低速度地板（清晰路况且不近障碍时生效）
    config.speed_floor_enable_lane = True
    config.speed_floor_target_lane = 3.0
    config.speed_floor_k_lane = 1.2
    config.speed_floor_obs_exempt_dist_lane = 14.0
    config.speed_floor_lane_ratio_max = 0.85

    config.success_bonus_lane = 120.0
    config.success_bonus_jaywalker = 150.0
    config.success_speed_th_lane = 3.0
    config.success_speed_th_jaywalker = 1.0
    config.success_prog_th_lane = 0.05
    config.success_prog_th_jaywalker = 0.02

    # 训练侧 success 判据的最低平均速度门槛（用于课程/早停）
    config.success_min_avg_speed_lane = 3.0
    config.success_min_avg_speed_jaywalker = 1.0
    config.offroad_margin = 1.25
    config.progress_full_speed_lane = 3.8
    config.progress_min_gate_lane = 0.10
    config.clear_dist_lane = 24.0
    config.clear_speed_target_lane = 3.9
    config.k_slow_clear_lane = 0.75
    # 障碍奖励门控距离（避免过远距离就进入强避障模式）
    config.avoid_fwd_lane = 30.0
    config.avoid_fwd_jaywalker = 22.0
    config.safe_dist_lane = 14.0
    config.safe_dist_jaywalker = 10.0
    # reward 与控制的障碍横向门槛统一到同一量级，降低“奖励判危险但控制不刹”的冲突
    config.obs_reward_lat_tol_lane = 3.2
    config.obs_reward_lat_tol_jaywalker = 3.0
    config.obs_reward_gate_floor_lane = 0.02
    config.obs_reward_gate_floor_jaywalker = 0.20
    config.k_avoid_lat_lane = 1.40
    config.k_avoid_lat_jaywalker = 0.45
    config.k_avoid_lat_cap_lane = 1.10
    config.k_avoid_lat_cap_jaywalker = 0.18
    config.w_obs_clear_lane = 8.5
    config.w_obs_clear_jaywalker = 4.2
    config.w_obs_speed_lane = 2.2
    config.w_obs_speed_jaywalker = 1.4
    config.w_obs_danger_lane = 4.5
    config.w_obs_danger_jaywalker = 1.2
    config.obs_danger_dist_lane = 16.0
    config.obs_danger_dist_jaywalker = 9.0
    config.w_obs_ttc_lane = 4.5
    config.w_obs_ttc_jaywalker = 2.5
    config.obs_ttc_crit_lane = 3.2
    # 终止项与主奖励核心权重（CarlaEnv 读取这些字段）
    config.k_collision_terminal = 320.0
    config.k_offroad_terminal = 90.0
    config.k_no_progress_terminal = 80.0
    config.k_progress = 4.2
    config.k_lane = 0.55
    config.k_danger = 0.80
    config.danger_start_ratio = 0.30
    config.speed_reward_sigma = 1.4
    config.safety_penalty_scale = 1.0
    config.obs_ttc_crit_jaywalker = 2.2
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
    config.stage1_success_threshold = 0.30
    config.stage2_success_threshold = 0.60
    config.enable_forced_curriculum = True
    config.force_stage2_episode = 140
    config.force_stage3_episode = 360
    config.force_stage2_min_success = 0.0
    config.force_stage3_min_success = 0.0

    # ✅ 动作降维开关（False=完整3维动作：throttle/steer/y_ref）
    config.use_action_dim2 = False

    # ✅ Stage3 启用温和 y_ref，提升绕障意图学习
    config.stage3_use_yref = True
    config.stage1_yref_gain = 0.0015
    config.stage2_yref_gain = 0.003
    # ✅ Stage3 y_ref 渐进增益
    config.stage3_yref_warmup_end = 520
    config.stage3_yref_gain_warmup = 0.004
    config.stage3_yref_gain_full = 0.008

    # ✅ 统一episode步长：训练timesteps 与 env.max_episode_steps 对齐
    # 防止time_limit截断导致统计/收敛判断偏移
    config.max_episode_steps = 256
    config.train_episodes = 1500
    config.entropy_reg = 0.0080
    config.entropy_start = 0.0200
    config.entropy_end = 0.0040
    config.entropy_decay_ep = 2600
    config.entropy_rescue_collision_rate = 0.45
    config.entropy_rescue_success_rate = 0.10
    config.entropy_rescue_start = 0.030
    config.entropy_rescue_end = 0.010
    config.optimization_steps = (4, 4)
    config.trd_loss_coef = 0.0
    # 严格 on-policy：尽量让“训练看到的动作”与“环境实际执行动作”一致
    config.train_onpolicy_action_mode = True
    config.ppo_store_executed_action = True
    config.policy_lr = 1e-4
    config.value_lr = 2e-4
    config.gamma = 0.99
    config.lambda_ = 0.95
    config.clip_ratio = 0.2
    config.batch_size = 128
    config.update_frequency = 2

    # ✅ 早停：收敛提前停、坏训练提前止损
    config.enable_early_stop = True
    config.early_window = 20
    config.early_min_episode = 60
    config.early_target_success = 0.80
    config.early_target_ran_full = 0.80
    config.early_target_speed_ok = 0.80
    config.early_target_avg_speed = 3.0
    config.early_max_collision = 0.10
    config.early_need_stable_windows = 3
    config.early_plateau_patience = 300
    config.early_improve_eps = 0.01
    config.early_plateau_min_episode = 1800
    config.early_plateau_min_best_success = 0.35
    config.early_plateau_min_progress_score = 0.65
    config.early_plateau_curr_min_success = 0.05
    config.early_plateau_curr_min_speed_ok = 0.35
    config.early_plateau_curr_min_safe_fast = 0.10
    config.early_plateau_curr_max_collision = 0.35
    config.early_progress_w_success = 0.55
    config.early_progress_w_ran_full = 0.30
    config.early_progress_w_no_collision = 0.15
    config.early_bad_min_episode = 2000
    config.early_bad_patience = 120
    config.early_bad_collision = 0.75
    config.early_bad_ran_full = 0.30
    config.early_bad_success = 0.20
    config.early_bad_avg_speed = 2.9
    config.early_bad_speed_ok = 0.25
    config.early_bad_stuck_rate = 0.65
    config.early_bad_use_ran_full = False
    config.early_bad_ran_full_min_episode = 180
    config.early_bad_crawl_fail = 0.85
    config.early_bad_progress_max = 0.12
    config.early_bad_safe_fast = 0.15
    config.early_restart_enable = False
    config.early_restart_min_episode = 600
    config.early_restart_patience = 12
    config.early_restart_success_max = 0.03
    config.early_restart_speed_ok_max = 0.10
    config.early_restart_safe_fast_max = 0.10
    config.early_restart_avg_speed_max = 1.8
    config.early_restart_collision_min = 0.35
    config.early_restart_stuck_min = 0.70
    config.hard_bad_enable = False
    config.hard_bad_window = 120
    config.hard_bad_collision = 0.85
    config.hard_bad_min_episode = 320
    config.hard_bad_restore_continue = True
    config.hard_bad_recover_cooldown = 40
    config.hard_bad_max_recovers = 10
    config.hard_bad_entropy_boost_episodes = 120
    config.hard_bad_entropy_boost_start = 0.045
    config.hard_bad_entropy_boost_end = 0.016
    config.restore_best_on_bad_stop = True
    config.restore_best_at_end = True
    config.deterministic_eval_enable = True
    config.deterministic_eval_every = 50
    config.deterministic_eval_episodes = 2
    config.deterministic_eval_timesteps = 128
    config.deterministic_eval_use_old_policy = False

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
        config.entropy_reg = float(cfg.get("entropy_reg", 0.0080))
        config.entropy_start = float(cfg.get("entropy_start", 0.0200))
        config.entropy_end = float(cfg.get("entropy_end", 0.0040))
        config.entropy_decay_ep = int(cfg.get("entropy_decay_ep", 2600))
        config.entropy_rescue_collision_rate = float(cfg.get("entropy_rescue_collision_rate", 0.45))
        config.entropy_rescue_success_rate = float(cfg.get("entropy_rescue_success_rate", 0.10))
        config.entropy_rescue_start = float(cfg.get("entropy_rescue_start", 0.030))
        config.entropy_rescue_end = float(cfg.get("entropy_rescue_end", 0.010))
        _opt_steps = cfg.get("optimization_steps", (4, 4))
        if isinstance(_opt_steps, (list, tuple)) and len(_opt_steps) == 2:
            config.optimization_steps = (int(_opt_steps[0]), int(_opt_steps[1]))
        else:
            config.optimization_steps = (4, 4)
        config.trd_loss_coef = float(cfg.get("trd_loss_coef", 0.0))
        config.value_representation = str(cfg.get("value_representation", "scalar")).strip().lower()
        config.network_units = int(cfg.get("network_units", 128))
        config.network_layers = int(cfg.get("network_layers", 2))

        # ✅ y_ref 新旧字段统一：训练内部都用 use_yref_in_steer / yref_steer_gain
        _use_yref = bool(cfg.get("use_yref_in_steer", cfg.get("use_yref_mapping", False)))
        _yref_gain = float(cfg.get("yref_steer_gain", cfg.get("yref_gain", 0.0)))
        config.use_yref_in_steer = _use_yref
        config.yref_steer_gain = _yref_gain
        # 兼容旧字段（防止其他模块仍在读取旧名）
        config.use_yref_mapping = _use_yref
        config.yref_gain = _yref_gain
        config.yref_penalty = float(cfg.get("yref_penalty", 0.0))
        config.use_aggressive_reward = bool(cfg.get("use_aggressive_reward", False))
        config.anti_crawl_enable = bool(cfg.get("anti_crawl_enable", True))
        config.anti_crawl_speed_lane = float(cfg.get("anti_crawl_speed_lane", 3.0))
        config.anti_crawl_speed_jaywalker = float(cfg.get("anti_crawl_speed_jaywalker", 1.0))
        config.anti_crawl_penalty_k_lane = float(cfg.get("anti_crawl_penalty_k_lane", 0.32))
        config.anti_crawl_penalty_k_jaywalker = float(cfg.get("anti_crawl_penalty_k_jaywalker", 0.08))
        config.anti_crawl_penalty_cap_lane = float(cfg.get("anti_crawl_penalty_cap_lane", 1.3))
        config.anti_crawl_penalty_cap_jaywalker = float(cfg.get("anti_crawl_penalty_cap_jaywalker", 0.6))
        config.anti_crawl_terminate_enable = bool(cfg.get("anti_crawl_terminate_enable", True))
        config.anti_crawl_warmup_steps_lane = int(cfg.get("anti_crawl_warmup_steps_lane", 140))
        config.anti_crawl_warmup_steps_jaywalker = int(cfg.get("anti_crawl_warmup_steps_jaywalker", 120))
        config.anti_crawl_min_avg_speed_lane = float(cfg.get("anti_crawl_min_avg_speed_lane", 3.0))
        config.anti_crawl_min_avg_speed_jaywalker = float(cfg.get("anti_crawl_min_avg_speed_jaywalker", 0.9))
        config.anti_crawl_recent_window_lane = int(cfg.get("anti_crawl_recent_window_lane", 48))
        config.anti_crawl_recent_window_jaywalker = int(cfg.get("anti_crawl_recent_window_jaywalker", 36))
        config.anti_crawl_min_recent_speed_lane = float(cfg.get("anti_crawl_min_recent_speed_lane", 3.0))
        config.anti_crawl_min_recent_speed_jaywalker = float(cfg.get("anti_crawl_min_recent_speed_jaywalker", 0.9))
        config.anti_crawl_terminate_penalty_lane = float(cfg.get("anti_crawl_terminate_penalty_lane", 45.0))
        config.anti_crawl_terminate_penalty_jaywalker = float(cfg.get("anti_crawl_terminate_penalty_jaywalker", 25.0))
        config.anti_crawl_terminate_start_episode = int(cfg.get("anti_crawl_terminate_start_episode", 420))
        config.anti_crawl_terminate_speed_ratio = float(cfg.get("anti_crawl_terminate_speed_ratio", 0.78))
        config.anti_crawl_min_travel_before_terminate_lane = float(
            cfg.get("anti_crawl_min_travel_before_terminate_lane", 16.0)
        )
        config.anti_crawl_min_travel_before_terminate_jaywalker = float(
            cfg.get("anti_crawl_min_travel_before_terminate_jaywalker", 6.0)
        )
        config.anti_crawl_terminate_obs_exempt_dist_lane = float(
            cfg.get("anti_crawl_terminate_obs_exempt_dist_lane", 12.0)
        )
        config.anti_crawl_terminate_obs_exempt_dist_jaywalker = float(
            cfg.get("anti_crawl_terminate_obs_exempt_dist_jaywalker", 18.0)
        )
        config.anti_crawl_penalty_near_obs_scale_lane = float(
            cfg.get("anti_crawl_penalty_near_obs_scale_lane", 0.35)
        )
        config.anti_crawl_penalty_near_obs_scale_jaywalker = float(
            cfg.get("anti_crawl_penalty_near_obs_scale_jaywalker", 0.60)
        )
        config.bad_brake_penalty_enable = bool(cfg.get("bad_brake_penalty_enable", False))
        config.bad_brake_k_lane = float(cfg.get("bad_brake_k_lane", 0.35))
        config.bad_brake_k_jaywalker = float(cfg.get("bad_brake_k_jaywalker", 0.10))
        config.bad_brake_apply_speed_min_lane = float(cfg.get("bad_brake_apply_speed_min_lane", 3.0))
        config.bad_brake_apply_speed_min_jaywalker = float(cfg.get("bad_brake_apply_speed_min_jaywalker", 1.2))
        config.bad_brake_apply_obs_dist_min_lane = float(cfg.get("bad_brake_apply_obs_dist_min_lane", 16.0))
        config.bad_brake_apply_obs_dist_min_jaywalker = float(cfg.get("bad_brake_apply_obs_dist_min_jaywalker", 10.0))
        config.enable_low_speed_steer_scale = bool(cfg.get("enable_low_speed_steer_scale", True))
        config.low_speed_steer_speed = float(cfg.get("low_speed_steer_speed", 1.2))
        config.low_speed_steer_min_scale = float(cfg.get("low_speed_steer_min_scale", 0.90))
        config.enable_steer_smoothing = bool(cfg.get("enable_steer_smoothing", False))
        config.steer_smooth_alpha = float(cfg.get("steer_smooth_alpha", 0.82))
        config.enable_steer_rate_limit = bool(cfg.get("enable_steer_rate_limit", True))
        config.steer_rate_limit_lane = float(cfg.get("steer_rate_limit_lane", 0.30))
        config.steer_rate_limit_jaywalker = float(cfg.get("steer_rate_limit_jaywalker", 0.14))
        config.steer_rate_limit_obs_lane = float(cfg.get("steer_rate_limit_obs_lane", 0.46))
        config.steer_rate_limit_obs_jaywalker = float(cfg.get("steer_rate_limit_obs_jaywalker", 0.26))
        config.policy_min_scale = float(cfg.get("policy_min_scale", 0.06))
        config.policy_min_scale_lat = float(cfg.get("policy_min_scale_lat", 0.08))
        config.policy_min_scale_lon = float(cfg.get("policy_min_scale_lon", 0.06))
        config.policy_raw_scale_bias_init = float(cfg.get("policy_raw_scale_bias_init", -1.20))
        config.policy_raw_scale_bias_lat = float(cfg.get("policy_raw_scale_bias_lat", -1.20))
        config.policy_raw_scale_bias_lon = float(cfg.get("policy_raw_scale_bias_lon", -1.20))
        config.policy_raw_scale_clip_low = float(cfg.get("policy_raw_scale_clip_low", -4.0))
        config.policy_raw_scale_clip_high = float(cfg.get("policy_raw_scale_clip_high", -0.35))
        config.enable_straight_stability = bool(cfg.get("enable_straight_stability", True))
        config.straight_stability_speed_lane = float(cfg.get("straight_stability_speed_lane", 2.8))
        config.straight_stability_speed_jaywalker = float(cfg.get("straight_stability_speed_jaywalker", 1.4))
        config.straight_stability_lane_ratio = float(cfg.get("straight_stability_lane_ratio", 0.15))
        config.straight_steer_damp = float(cfg.get("straight_steer_damp", 0.60))
        config.straight_steer_deadband = float(cfg.get("straight_steer_deadband", 0.020))
        config.enable_steer_delta_clip = bool(cfg.get("enable_steer_delta_clip", True))
        config.max_steer_postprocess_delta = float(cfg.get("max_steer_postprocess_delta", 0.22))
        config.obstacle_steer_assist_enable = bool(cfg.get("obstacle_steer_assist_enable", False))
        config.obstacle_steer_assist_dist_lane = float(cfg.get("obstacle_steer_assist_dist_lane", 10.0))
        config.obstacle_steer_assist_dist_jaywalker = float(cfg.get("obstacle_steer_assist_dist_jaywalker", 10.0))
        config.obstacle_steer_assist_gain_lane = float(cfg.get("obstacle_steer_assist_gain_lane", 0.14))
        config.obstacle_steer_assist_gain_jaywalker = float(cfg.get("obstacle_steer_assist_gain_jaywalker", 0.10))
        config.obstacle_steer_assist_max_lane = float(cfg.get("obstacle_steer_assist_max_lane", 0.20))
        config.obstacle_steer_assist_max_jaywalker = float(cfg.get("obstacle_steer_assist_max_jaywalker", 0.16))
        config.obstacle_steer_assist_lat_eps = float(cfg.get("obstacle_steer_assist_lat_eps", 0.15))
        config.obstacle_steer_assist_min_factor = float(cfg.get("obstacle_steer_assist_min_factor", 0.20))
        config.obs_control_lat_tol = float(cfg.get("obs_control_lat_tol", 2.8))
        config.obs_control_lat_tol_lane = float(cfg.get("obs_control_lat_tol_lane", 3.5))
        config.obs_control_lat_tol_jaywalker = float(cfg.get("obs_control_lat_tol_jaywalker", 3.0))
        config.obs_front_priority_fwd_min = float(cfg.get("obs_front_priority_fwd_min", -0.5))
        config.obs_front_priority_lat_tol = float(cfg.get("obs_front_priority_lat_tol", 3.2))
        config.obs_reward_lat_tol_lane = float(cfg.get("obs_reward_lat_tol_lane", 3.2))
        config.obs_reward_lat_tol_jaywalker = float(cfg.get("obs_reward_lat_tol_jaywalker", 3.0))
        config.obs_reward_gate_floor_lane = float(cfg.get("obs_reward_gate_floor_lane", 0.02))
        config.obs_reward_gate_floor_jaywalker = float(cfg.get("obs_reward_gate_floor_jaywalker", 0.20))
        config.avoid_fwd_lane = float(cfg.get("avoid_fwd_lane", 30.0))
        config.avoid_fwd_jaywalker = float(cfg.get("avoid_fwd_jaywalker", 22.0))
        config.safe_dist_lane = float(cfg.get("safe_dist_lane", 14.0))
        config.safe_dist_jaywalker = float(cfg.get("safe_dist_jaywalker", 10.0))
        config.k_avoid_lat_lane = float(cfg.get("k_avoid_lat_lane", 1.40))
        config.k_avoid_lat_jaywalker = float(cfg.get("k_avoid_lat_jaywalker", 0.45))
        config.k_avoid_lat_cap_lane = float(cfg.get("k_avoid_lat_cap_lane", 1.10))
        config.k_avoid_lat_cap_jaywalker = float(cfg.get("k_avoid_lat_cap_jaywalker", 0.18))
        config.w_obs_clear_lane = float(cfg.get("w_obs_clear_lane", 8.5))
        config.w_obs_clear_jaywalker = float(cfg.get("w_obs_clear_jaywalker", 4.2))
        config.w_obs_speed_lane = float(cfg.get("w_obs_speed_lane", 2.2))
        config.w_obs_speed_jaywalker = float(cfg.get("w_obs_speed_jaywalker", 1.4))
        config.w_obs_danger_lane = float(cfg.get("w_obs_danger_lane", 4.5))
        config.w_obs_danger_jaywalker = float(cfg.get("w_obs_danger_jaywalker", 1.2))
        config.obs_danger_dist_lane = float(cfg.get("obs_danger_dist_lane", 16.0))
        config.obs_danger_dist_jaywalker = float(cfg.get("obs_danger_dist_jaywalker", 9.0))
        config.w_obs_ttc_lane = float(cfg.get("w_obs_ttc_lane", 4.5))
        config.w_obs_ttc_jaywalker = float(cfg.get("w_obs_ttc_jaywalker", 2.5))
        config.obs_ttc_crit_lane = float(cfg.get("obs_ttc_crit_lane", 3.2))
        config.obs_ttc_crit_jaywalker = float(cfg.get("obs_ttc_crit_jaywalker", 2.2))
        config.speed_floor_enable_lane = bool(cfg.get("speed_floor_enable_lane", True))
        config.speed_floor_target_lane = float(cfg.get("speed_floor_target_lane", 3.0))
        config.speed_floor_k_lane = float(cfg.get("speed_floor_k_lane", 1.2))
        config.speed_floor_obs_exempt_dist_lane = float(cfg.get("speed_floor_obs_exempt_dist_lane", 14.0))
        config.speed_floor_lane_ratio_max = float(cfg.get("speed_floor_lane_ratio_max", 0.85))
        config.obs_ttc_brake_crit_lane = float(cfg.get("obs_ttc_brake_crit_lane", 2.0))
        config.obs_ttc_brake_crit_jaywalker = float(cfg.get("obs_ttc_brake_crit_jaywalker", 2.4))
        config.obs_ttc_brake_gain_lane = float(cfg.get("obs_ttc_brake_gain_lane", 0.38))
        config.obs_ttc_brake_gain_jaywalker = float(cfg.get("obs_ttc_brake_gain_jaywalker", 0.50))
        config.k_collision_terminal = float(cfg.get("k_collision_terminal", 320.0))
        config.k_offroad_terminal = float(cfg.get("k_offroad_terminal", 90.0))
        config.k_no_progress_terminal = float(cfg.get("k_no_progress_terminal", 80.0))
        config.k_progress = float(cfg.get("k_progress", 4.2))
        config.k_lane = float(cfg.get("k_lane", 0.55))
        config.k_danger = float(cfg.get("k_danger", 0.80))
        config.danger_start_ratio = float(cfg.get("danger_start_ratio", 0.30))
        config.speed_reward_sigma = float(cfg.get("speed_reward_sigma", 1.4))
        config.safety_penalty_scale = float(cfg.get("safety_penalty_scale", 1.0))
        config.offroad_margin = float(cfg.get("offroad_margin", 1.25))
        config.log_steer_saturation = bool(cfg.get("log_steer_saturation", True))
        config.steer_sat_threshold = float(cfg.get("steer_sat_threshold", 0.999))
        config.wandb_step_log_interval = int(cfg.get("wandb_step_log_interval", 50))
        config.load_existing = bool(cfg.get("load_existing", False))
        config.resume_quality_guard = bool(cfg.get("resume_quality_guard", True))
        config.resume_min_success_w = float(cfg.get("resume_min_success_w", 0.03))
        config.resume_min_speed_ok_w = float(cfg.get("resume_min_speed_ok_w", 0.30))
        config.resume_min_safe_fast_w = float(cfg.get("resume_min_safe_fast_w", 0.15))
        config.finetune_on_resume = bool(cfg.get("finetune_on_resume", True))
        config.resume_policy_lr_scale = float(cfg.get("resume_policy_lr_scale", 0.5))
        config.resume_value_lr_scale = float(cfg.get("resume_value_lr_scale", 0.5))
        config.resume_entropy_scale = float(cfg.get("resume_entropy_scale", 0.9))
        config.resume_opt_steps_scale = float(cfg.get("resume_opt_steps_scale", 0.75))
        config.enable_early_stop = bool(cfg.get("enable_early_stop", True))
        config.early_window = int(cfg.get("early_window", 20))
        config.early_min_episode = int(cfg.get("early_min_episode", 60))
        config.early_target_success = float(cfg.get("early_target_success", 0.80))
        config.early_target_ran_full = float(cfg.get("early_target_ran_full", 0.80))
        config.early_target_speed_ok = float(cfg.get("early_target_speed_ok", 0.80))
        config.early_target_avg_speed = float(cfg.get("early_target_avg_speed", 3.0))
        config.early_max_collision = float(cfg.get("early_max_collision", 0.10))
        config.early_need_stable_windows = int(cfg.get("early_need_stable_windows", 3))
        config.early_plateau_patience = int(cfg.get("early_plateau_patience", 300))
        config.early_improve_eps = float(cfg.get("early_improve_eps", 0.01))
        config.early_plateau_min_episode = int(cfg.get("early_plateau_min_episode", 1800))
        config.early_plateau_min_best_success = float(cfg.get("early_plateau_min_best_success", 0.35))
        config.early_plateau_min_progress_score = float(cfg.get("early_plateau_min_progress_score", 0.65))
        config.early_plateau_curr_min_success = float(cfg.get("early_plateau_curr_min_success", 0.05))
        config.early_plateau_curr_min_speed_ok = float(cfg.get("early_plateau_curr_min_speed_ok", 0.35))
        config.early_plateau_curr_min_safe_fast = float(cfg.get("early_plateau_curr_min_safe_fast", 0.10))
        config.early_plateau_curr_max_collision = float(cfg.get("early_plateau_curr_max_collision", 0.35))
        config.early_progress_w_success = float(cfg.get("early_progress_w_success", 0.55))
        config.early_progress_w_ran_full = float(cfg.get("early_progress_w_ran_full", 0.30))
        config.early_progress_w_no_collision = float(cfg.get("early_progress_w_no_collision", 0.15))
        config.early_bad_min_episode = int(cfg.get("early_bad_min_episode", 2000))
        config.early_bad_patience = int(cfg.get("early_bad_patience", 120))
        config.early_bad_collision = float(cfg.get("early_bad_collision", 0.75))
        config.early_bad_ran_full = float(cfg.get("early_bad_ran_full", 0.30))
        config.early_bad_success = float(cfg.get("early_bad_success", 0.20))
        config.early_bad_avg_speed = float(cfg.get("early_bad_avg_speed", 2.9))
        config.early_bad_speed_ok = float(cfg.get("early_bad_speed_ok", 0.25))
        config.early_bad_stuck_rate = float(cfg.get("early_bad_stuck_rate", 0.65))
        config.early_bad_use_ran_full = bool(cfg.get("early_bad_use_ran_full", False))
        config.early_bad_ran_full_min_episode = int(cfg.get("early_bad_ran_full_min_episode", 180))
        config.early_bad_crawl_fail = float(cfg.get("early_bad_crawl_fail", 0.85))
        config.early_bad_progress_max = float(cfg.get("early_bad_progress_max", 0.12))
        config.early_bad_safe_fast = float(cfg.get("early_bad_safe_fast", 0.15))
        config.early_restart_enable = bool(cfg.get("early_restart_enable", False))
        config.early_restart_min_episode = int(cfg.get("early_restart_min_episode", 600))
        config.early_restart_patience = int(cfg.get("early_restart_patience", 12))
        config.early_restart_success_max = float(cfg.get("early_restart_success_max", 0.03))
        config.early_restart_speed_ok_max = float(cfg.get("early_restart_speed_ok_max", 0.10))
        config.early_restart_safe_fast_max = float(cfg.get("early_restart_safe_fast_max", 0.10))
        config.early_restart_avg_speed_max = float(cfg.get("early_restart_avg_speed_max", 1.8))
        config.early_restart_collision_min = float(cfg.get("early_restart_collision_min", 0.35))
        config.early_restart_stuck_min = float(cfg.get("early_restart_stuck_min", 0.70))
        config.hard_bad_enable = bool(cfg.get("hard_bad_enable", False))
        config.hard_bad_window = int(cfg.get("hard_bad_window", 120))
        config.hard_bad_collision = float(cfg.get("hard_bad_collision", 0.85))
        config.hard_bad_min_episode = int(cfg.get("hard_bad_min_episode", 320))
        config.hard_bad_restore_continue = bool(cfg.get("hard_bad_restore_continue", True))
        config.hard_bad_recover_cooldown = int(cfg.get("hard_bad_recover_cooldown", 40))
        config.hard_bad_max_recovers = int(cfg.get("hard_bad_max_recovers", 10))
        config.hard_bad_entropy_boost_episodes = int(cfg.get("hard_bad_entropy_boost_episodes", 120))
        config.hard_bad_entropy_boost_start = float(cfg.get("hard_bad_entropy_boost_start", 0.045))
        config.hard_bad_entropy_boost_end = float(cfg.get("hard_bad_entropy_boost_end", 0.016))
        config.restore_best_on_bad_stop = bool(cfg.get("restore_best_on_bad_stop", True))
        config.restore_best_at_end = bool(cfg.get("restore_best_at_end", True))
        config.deterministic_eval_enable = bool(cfg.get("deterministic_eval_enable", True))
        config.deterministic_eval_every = int(cfg.get("deterministic_eval_every", 50))
        config.deterministic_eval_episodes = int(cfg.get("deterministic_eval_episodes", 2))
        config.deterministic_eval_timesteps = int(cfg.get("deterministic_eval_timesteps", 128))
        config.deterministic_eval_use_old_policy = bool(cfg.get("deterministic_eval_use_old_policy", False))
        config.train_onpolicy_action_mode = bool(cfg.get("train_onpolicy_action_mode", True))
        config.ppo_store_executed_action = bool(cfg.get("ppo_store_executed_action", True))
        config.stage1_success_threshold = float(cfg.get("stage1_success_threshold", 0.30))
        config.stage2_success_threshold = float(cfg.get("stage2_success_threshold", 0.60))
        config.stage1_yref_gain = float(cfg.get("stage1_yref_gain", 0.0015))
        config.stage2_yref_gain = float(cfg.get("stage2_yref_gain", 0.003))
        config.stage3_use_yref = bool(cfg.get("stage3_use_yref", True))
        config.stage3_yref_warmup_end = int(cfg.get("stage3_yref_warmup_end", 520))
        config.stage3_yref_gain_warmup = float(cfg.get("stage3_yref_gain_warmup", 0.004))
        config.stage3_yref_gain_full = float(cfg.get("stage3_yref_gain_full", 0.008))
        config.enable_forced_curriculum = bool(cfg.get("enable_forced_curriculum", True))
        config.force_stage2_episode = int(cfg.get("force_stage2_episode", 140))
        config.force_stage3_episode = int(cfg.get("force_stage3_episode", 360))
        config.force_stage2_min_success = float(cfg.get("force_stage2_min_success", 0.0))
        config.force_stage3_min_success = float(cfg.get("force_stage3_min_success", 0.0))
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
        config.entropy_reg = float(getattr(config, "entropy_reg", 0.0080))
        config.entropy_start = float(getattr(config, "entropy_start", 0.0200))
        config.entropy_end = float(getattr(config, "entropy_end", 0.0040))
        config.entropy_decay_ep = int(getattr(config, "entropy_decay_ep", 2600))
        config.entropy_rescue_collision_rate = float(getattr(config, "entropy_rescue_collision_rate", 0.45))
        config.entropy_rescue_success_rate = float(getattr(config, "entropy_rescue_success_rate", 0.10))
        config.entropy_rescue_start = float(getattr(config, "entropy_rescue_start", 0.030))
        config.entropy_rescue_end = float(getattr(config, "entropy_rescue_end", 0.010))
        config.optimization_steps = tuple(getattr(config, "optimization_steps", (4, 4)))
        config.trd_loss_coef = float(getattr(config, "trd_loss_coef", 0.0))
        config.value_representation = str(getattr(config, "value_representation", "scalar")).strip().lower()
        config.network_units = int(getattr(config, "network_units", 128))
        config.network_layers = int(getattr(config, "network_layers", 2))

        # ✅ y_ref 新旧字段统一
        config.use_yref_in_steer = bool(getattr(config, "use_yref_in_steer", False))
        config.yref_steer_gain = float(getattr(config, "yref_steer_gain", 0.0))
        config.use_yref_mapping = bool(config.use_yref_in_steer)
        config.yref_gain = float(config.yref_steer_gain)
        config.yref_penalty = float(getattr(config, "yref_penalty", 0.0))
        config.use_aggressive_reward = False
        config.anti_crawl_enable = True
        config.anti_crawl_speed_lane = 3.0
        config.anti_crawl_speed_jaywalker = 1.0
        config.anti_crawl_penalty_k_lane = 0.32
        config.anti_crawl_penalty_k_jaywalker = 0.08
        config.anti_crawl_penalty_cap_lane = 1.3
        config.anti_crawl_penalty_cap_jaywalker = 0.6
        config.anti_crawl_terminate_enable = True
        config.anti_crawl_warmup_steps_lane = 140
        config.anti_crawl_warmup_steps_jaywalker = 120
        config.anti_crawl_min_avg_speed_lane = 3.0
        config.anti_crawl_min_avg_speed_jaywalker = 0.9
        config.anti_crawl_recent_window_lane = int(getattr(config, "anti_crawl_recent_window_lane", 48))
        config.anti_crawl_recent_window_jaywalker = int(getattr(config, "anti_crawl_recent_window_jaywalker", 36))
        config.anti_crawl_min_recent_speed_lane = float(getattr(config, "anti_crawl_min_recent_speed_lane", 3.0))
        config.anti_crawl_min_recent_speed_jaywalker = float(getattr(config, "anti_crawl_min_recent_speed_jaywalker", 0.9))
        config.anti_crawl_terminate_penalty_lane = 45.0
        config.anti_crawl_terminate_penalty_jaywalker = 25.0
        config.anti_crawl_terminate_start_episode = 420
        config.anti_crawl_terminate_speed_ratio = 0.78
        config.anti_crawl_min_travel_before_terminate_lane = float(
            getattr(config, "anti_crawl_min_travel_before_terminate_lane", 16.0)
        )
        config.anti_crawl_min_travel_before_terminate_jaywalker = float(
            getattr(config, "anti_crawl_min_travel_before_terminate_jaywalker", 6.0)
        )
        config.anti_crawl_terminate_obs_exempt_dist_lane = 12.0
        config.anti_crawl_terminate_obs_exempt_dist_jaywalker = 18.0
        config.anti_crawl_penalty_near_obs_scale_lane = 0.35
        config.anti_crawl_penalty_near_obs_scale_jaywalker = 0.60
        config.bad_brake_penalty_enable = bool(getattr(config, "bad_brake_penalty_enable", False))
        config.bad_brake_k_lane = float(getattr(config, "bad_brake_k_lane", 0.35))
        config.bad_brake_k_jaywalker = float(getattr(config, "bad_brake_k_jaywalker", 0.10))
        config.enable_low_speed_steer_scale = bool(getattr(config, "enable_low_speed_steer_scale", True))
        config.low_speed_steer_speed = float(getattr(config, "low_speed_steer_speed", 1.2))
        config.low_speed_steer_min_scale = float(getattr(config, "low_speed_steer_min_scale", 0.90))
        config.enable_steer_smoothing = bool(getattr(config, "enable_steer_smoothing", False))
        config.steer_smooth_alpha = float(getattr(config, "steer_smooth_alpha", 0.82))
        config.enable_steer_rate_limit = bool(getattr(config, "enable_steer_rate_limit", True))
        config.steer_rate_limit_lane = float(getattr(config, "steer_rate_limit_lane", 0.30))
        config.steer_rate_limit_jaywalker = float(getattr(config, "steer_rate_limit_jaywalker", 0.14))
        config.steer_rate_limit_obs_lane = float(getattr(config, "steer_rate_limit_obs_lane", 0.46))
        config.steer_rate_limit_obs_jaywalker = float(getattr(config, "steer_rate_limit_obs_jaywalker", 0.26))
        config.policy_min_scale = float(getattr(config, "policy_min_scale", 0.06))
        config.policy_min_scale_lat = float(getattr(config, "policy_min_scale_lat", 0.08))
        config.policy_min_scale_lon = float(getattr(config, "policy_min_scale_lon", 0.06))
        config.policy_raw_scale_bias_init = float(getattr(config, "policy_raw_scale_bias_init", -1.20))
        config.policy_raw_scale_bias_lat = float(getattr(config, "policy_raw_scale_bias_lat", -1.20))
        config.policy_raw_scale_bias_lon = float(getattr(config, "policy_raw_scale_bias_lon", -1.20))
        config.policy_raw_scale_clip_low = float(getattr(config, "policy_raw_scale_clip_low", -4.0))
        config.policy_raw_scale_clip_high = float(getattr(config, "policy_raw_scale_clip_high", -0.35))
        config.enable_straight_stability = bool(getattr(config, "enable_straight_stability", True))
        config.straight_stability_speed_lane = float(getattr(config, "straight_stability_speed_lane", 2.8))
        config.straight_stability_speed_jaywalker = float(getattr(config, "straight_stability_speed_jaywalker", 1.4))
        config.straight_stability_lane_ratio = float(getattr(config, "straight_stability_lane_ratio", 0.15))
        config.straight_steer_damp = float(getattr(config, "straight_steer_damp", 0.60))
        config.straight_steer_deadband = float(getattr(config, "straight_steer_deadband", 0.020))
        config.enable_steer_delta_clip = bool(getattr(config, "enable_steer_delta_clip", True))
        config.max_steer_postprocess_delta = float(getattr(config, "max_steer_postprocess_delta", 0.22))
        config.obstacle_steer_assist_enable = bool(getattr(config, "obstacle_steer_assist_enable", False))
        config.obstacle_steer_assist_dist_lane = float(getattr(config, "obstacle_steer_assist_dist_lane", 10.0))
        config.obstacle_steer_assist_dist_jaywalker = float(getattr(config, "obstacle_steer_assist_dist_jaywalker", 10.0))
        config.obstacle_steer_assist_gain_lane = float(getattr(config, "obstacle_steer_assist_gain_lane", 0.14))
        config.obstacle_steer_assist_gain_jaywalker = float(getattr(config, "obstacle_steer_assist_gain_jaywalker", 0.10))
        config.obstacle_steer_assist_max_lane = float(getattr(config, "obstacle_steer_assist_max_lane", 0.20))
        config.obstacle_steer_assist_max_jaywalker = float(getattr(config, "obstacle_steer_assist_max_jaywalker", 0.16))
        config.obstacle_steer_assist_lat_eps = float(getattr(config, "obstacle_steer_assist_lat_eps", 0.15))
        config.obstacle_steer_assist_min_factor = float(getattr(config, "obstacle_steer_assist_min_factor", 0.20))
        config.obs_control_lat_tol = float(getattr(config, "obs_control_lat_tol", 2.8))
        config.obs_control_lat_tol_lane = float(getattr(config, "obs_control_lat_tol_lane", 3.5))
        config.obs_control_lat_tol_jaywalker = float(getattr(config, "obs_control_lat_tol_jaywalker", 3.0))
        config.obs_front_priority_fwd_min = float(getattr(config, "obs_front_priority_fwd_min", -0.5))
        config.obs_front_priority_lat_tol = float(getattr(config, "obs_front_priority_lat_tol", 3.2))
        config.obs_reward_lat_tol_lane = float(getattr(config, "obs_reward_lat_tol_lane", 3.2))
        config.obs_reward_lat_tol_jaywalker = float(getattr(config, "obs_reward_lat_tol_jaywalker", 3.0))
        config.obs_reward_gate_floor_lane = float(getattr(config, "obs_reward_gate_floor_lane", 0.02))
        config.obs_reward_gate_floor_jaywalker = float(getattr(config, "obs_reward_gate_floor_jaywalker", 0.20))
        config.k_avoid_lat_lane = float(getattr(config, "k_avoid_lat_lane", 1.40))
        config.k_avoid_lat_jaywalker = float(getattr(config, "k_avoid_lat_jaywalker", 0.45))
        config.k_avoid_lat_cap_lane = float(getattr(config, "k_avoid_lat_cap_lane", 1.10))
        config.k_avoid_lat_cap_jaywalker = float(getattr(config, "k_avoid_lat_cap_jaywalker", 0.18))
        config.w_obs_clear_lane = float(getattr(config, "w_obs_clear_lane", 8.5))
        config.w_obs_clear_jaywalker = float(getattr(config, "w_obs_clear_jaywalker", 4.2))
        config.w_obs_speed_lane = float(getattr(config, "w_obs_speed_lane", 2.2))
        config.w_obs_speed_jaywalker = float(getattr(config, "w_obs_speed_jaywalker", 1.4))
        config.w_obs_danger_lane = float(getattr(config, "w_obs_danger_lane", 4.5))
        config.w_obs_danger_jaywalker = float(getattr(config, "w_obs_danger_jaywalker", 1.2))
        config.obs_danger_dist_lane = float(getattr(config, "obs_danger_dist_lane", 16.0))
        config.obs_danger_dist_jaywalker = float(getattr(config, "obs_danger_dist_jaywalker", 9.0))
        config.w_obs_ttc_lane = float(getattr(config, "w_obs_ttc_lane", 4.5))
        config.w_obs_ttc_jaywalker = float(getattr(config, "w_obs_ttc_jaywalker", 2.5))
        config.obs_ttc_crit_lane = float(getattr(config, "obs_ttc_crit_lane", 3.2))
        config.obs_ttc_crit_jaywalker = float(getattr(config, "obs_ttc_crit_jaywalker", 2.2))
        config.obs_ttc_brake_crit_lane = float(getattr(config, "obs_ttc_brake_crit_lane", 2.0))
        config.obs_ttc_brake_crit_jaywalker = float(getattr(config, "obs_ttc_brake_crit_jaywalker", 2.4))
        config.obs_ttc_brake_gain_lane = float(getattr(config, "obs_ttc_brake_gain_lane", 0.38))
        config.obs_ttc_brake_gain_jaywalker = float(getattr(config, "obs_ttc_brake_gain_jaywalker", 0.50))
        config.k_collision_terminal = float(getattr(config, "k_collision_terminal", 320.0))
        config.k_offroad_terminal = float(getattr(config, "k_offroad_terminal", 90.0))
        config.k_no_progress_terminal = float(getattr(config, "k_no_progress_terminal", 80.0))
        config.k_progress = float(getattr(config, "k_progress", 4.2))
        config.k_lane = float(getattr(config, "k_lane", 0.55))
        config.k_danger = float(getattr(config, "k_danger", 0.80))
        config.danger_start_ratio = float(getattr(config, "danger_start_ratio", 0.30))
        config.speed_reward_sigma = float(getattr(config, "speed_reward_sigma", 1.4))
        config.safety_penalty_scale = float(getattr(config, "safety_penalty_scale", 1.0))
        config.offroad_margin = float(getattr(config, "offroad_margin", 1.25))
        config.log_steer_saturation = True
        config.steer_sat_threshold = 0.999
        config.wandb_step_log_interval = int(getattr(config, "wandb_step_log_interval", 50))
        config.load_existing = bool(getattr(config, "load_existing", False))
        config.resume_quality_guard = bool(getattr(config, "resume_quality_guard", True))
        config.resume_min_success_w = float(getattr(config, "resume_min_success_w", 0.03))
        config.resume_min_speed_ok_w = float(getattr(config, "resume_min_speed_ok_w", 0.30))
        config.resume_min_safe_fast_w = float(getattr(config, "resume_min_safe_fast_w", 0.15))
        config.finetune_on_resume = bool(getattr(config, "finetune_on_resume", True))
        config.resume_policy_lr_scale = float(getattr(config, "resume_policy_lr_scale", 0.5))
        config.resume_value_lr_scale = float(getattr(config, "resume_value_lr_scale", 0.5))
        config.resume_entropy_scale = float(getattr(config, "resume_entropy_scale", 0.9))
        config.resume_opt_steps_scale = float(getattr(config, "resume_opt_steps_scale", 0.75))
        config.enable_early_stop = bool(getattr(config, "enable_early_stop", True))
        config.early_window = int(getattr(config, "early_window", 20))
        config.early_min_episode = int(getattr(config, "early_min_episode", 60))
        config.early_target_success = float(getattr(config, "early_target_success", 0.80))
        config.early_target_ran_full = float(getattr(config, "early_target_ran_full", 0.80))
        config.early_target_speed_ok = float(getattr(config, "early_target_speed_ok", 0.80))
        config.early_target_avg_speed = float(getattr(config, "early_target_avg_speed", 3.0))
        config.early_max_collision = float(getattr(config, "early_max_collision", 0.10))
        config.early_need_stable_windows = int(getattr(config, "early_need_stable_windows", 3))
        config.early_plateau_patience = int(getattr(config, "early_plateau_patience", 300))
        config.early_improve_eps = float(getattr(config, "early_improve_eps", 0.01))
        config.early_plateau_min_episode = int(getattr(config, "early_plateau_min_episode", 1800))
        config.early_plateau_min_best_success = float(getattr(config, "early_plateau_min_best_success", 0.35))
        config.early_plateau_min_progress_score = float(getattr(config, "early_plateau_min_progress_score", 0.65))
        config.early_plateau_curr_min_success = float(getattr(config, "early_plateau_curr_min_success", 0.05))
        config.early_plateau_curr_min_speed_ok = float(getattr(config, "early_plateau_curr_min_speed_ok", 0.35))
        config.early_plateau_curr_min_safe_fast = float(getattr(config, "early_plateau_curr_min_safe_fast", 0.10))
        config.early_plateau_curr_max_collision = float(getattr(config, "early_plateau_curr_max_collision", 0.35))
        config.early_progress_w_success = float(getattr(config, "early_progress_w_success", 0.55))
        config.early_progress_w_ran_full = float(getattr(config, "early_progress_w_ran_full", 0.30))
        config.early_progress_w_no_collision = float(getattr(config, "early_progress_w_no_collision", 0.15))
        config.early_bad_min_episode = int(getattr(config, "early_bad_min_episode", 2000))
        config.early_bad_patience = int(getattr(config, "early_bad_patience", 120))
        config.early_bad_collision = float(getattr(config, "early_bad_collision", 0.75))
        config.early_bad_ran_full = float(getattr(config, "early_bad_ran_full", 0.30))
        config.early_bad_success = float(getattr(config, "early_bad_success", 0.20))
        config.early_bad_avg_speed = float(getattr(config, "early_bad_avg_speed", 2.9))
        config.early_bad_speed_ok = float(getattr(config, "early_bad_speed_ok", 0.25))
        config.early_bad_stuck_rate = float(getattr(config, "early_bad_stuck_rate", 0.65))
        config.early_bad_use_ran_full = bool(getattr(config, "early_bad_use_ran_full", False))
        config.early_bad_ran_full_min_episode = int(getattr(config, "early_bad_ran_full_min_episode", 180))
        config.early_bad_crawl_fail = float(getattr(config, "early_bad_crawl_fail", 0.85))
        config.early_bad_progress_max = float(getattr(config, "early_bad_progress_max", 0.12))
        config.early_bad_safe_fast = float(getattr(config, "early_bad_safe_fast", 0.15))
        config.early_restart_enable = bool(getattr(config, "early_restart_enable", False))
        config.early_restart_min_episode = int(getattr(config, "early_restart_min_episode", 600))
        config.early_restart_patience = int(getattr(config, "early_restart_patience", 12))
        config.early_restart_success_max = float(getattr(config, "early_restart_success_max", 0.03))
        config.early_restart_speed_ok_max = float(getattr(config, "early_restart_speed_ok_max", 0.10))
        config.early_restart_safe_fast_max = float(getattr(config, "early_restart_safe_fast_max", 0.10))
        config.early_restart_avg_speed_max = float(getattr(config, "early_restart_avg_speed_max", 1.8))
        config.early_restart_collision_min = float(getattr(config, "early_restart_collision_min", 0.35))
        config.early_restart_stuck_min = float(getattr(config, "early_restart_stuck_min", 0.70))
        config.hard_bad_enable = bool(getattr(config, "hard_bad_enable", False))
        config.hard_bad_window = int(getattr(config, "hard_bad_window", 120))
        config.hard_bad_collision = float(getattr(config, "hard_bad_collision", 0.85))
        config.hard_bad_min_episode = int(getattr(config, "hard_bad_min_episode", 320))
        config.hard_bad_restore_continue = bool(getattr(config, "hard_bad_restore_continue", True))
        config.hard_bad_recover_cooldown = int(getattr(config, "hard_bad_recover_cooldown", 40))
        config.hard_bad_max_recovers = int(getattr(config, "hard_bad_max_recovers", 10))
        config.hard_bad_entropy_boost_episodes = int(getattr(config, "hard_bad_entropy_boost_episodes", 120))
        config.hard_bad_entropy_boost_start = float(getattr(config, "hard_bad_entropy_boost_start", 0.045))
        config.hard_bad_entropy_boost_end = float(getattr(config, "hard_bad_entropy_boost_end", 0.016))
        config.restore_best_on_bad_stop = bool(getattr(config, "restore_best_on_bad_stop", True))
        config.restore_best_at_end = bool(getattr(config, "restore_best_at_end", True))
        config.deterministic_eval_enable = bool(getattr(config, "deterministic_eval_enable", True))
        config.deterministic_eval_every = int(getattr(config, "deterministic_eval_every", 50))
        config.deterministic_eval_episodes = int(getattr(config, "deterministic_eval_episodes", 2))
        config.deterministic_eval_timesteps = int(getattr(config, "deterministic_eval_timesteps", 128))
        config.deterministic_eval_use_old_policy = bool(getattr(config, "deterministic_eval_use_old_policy", False))
        config.train_onpolicy_action_mode = bool(getattr(config, "train_onpolicy_action_mode", True))
        config.ppo_store_executed_action = bool(getattr(config, "ppo_store_executed_action", True))
        config.stage1_success_threshold = float(getattr(config, "stage1_success_threshold", 0.30))
        config.stage2_success_threshold = float(getattr(config, "stage2_success_threshold", 0.60))
        config.stage1_yref_gain = float(getattr(config, "stage1_yref_gain", 0.0015))
        config.stage2_yref_gain = float(getattr(config, "stage2_yref_gain", 0.003))
        config.stage3_use_yref = bool(getattr(config, "stage3_use_yref", True))
        config.stage3_yref_warmup_end = int(getattr(config, "stage3_yref_warmup_end", 520))
        config.stage3_yref_gain_warmup = float(getattr(config, "stage3_yref_gain_warmup", 0.004))
        config.stage3_yref_gain_full = float(getattr(config, "stage3_yref_gain_full", 0.008))
        config.enable_forced_curriculum = bool(getattr(config, "enable_forced_curriculum", True))
        config.force_stage2_episode = int(getattr(config, "force_stage2_episode", 140))
        config.force_stage3_episode = int(getattr(config, "force_stage3_episode", 360))
        config.force_stage2_min_success = float(getattr(config, "force_stage2_min_success", 0.0))
        config.force_stage3_min_success = float(getattr(config, "force_stage3_min_success", 0.0))

    # ===== 根据版本自动设置 yref_penalty =====
    # 不映射版：给一个小惩罚把 y_ref 压到 0
    # 映射版：不惩罚（或很小）
    if bool(getattr(config, "use_yref_in_steer", config.use_yref_mapping)):
        config.yref_penalty = 0.0  # 或 0.01（看你是否希望更平滑）
    else:
        config.yref_penalty = 0.05  # 推荐从 0.05 开始（范围 0.02~0.1）

    # ===== 可选：强制训练档覆盖（默认关闭，避免“改参不生效”） =====
    # 当 strict_training_profile=True 时，才会进行大幅参数覆盖与裁剪。
    strict_training_profile = bool(getattr(config, "strict_training_profile", False))
    if strict_training_profile:
        print("[Config] strict_training_profile=True，启用强制训练档覆盖。")

        config.enable_forced_curriculum = False
        config.ppo_store_executed_action = bool(getattr(config, "ppo_store_executed_action", True))
        config.stage1_yref_gain = max(float(getattr(config, "stage1_yref_gain", 0.0015)), 0.0015)
        config.stage2_yref_gain = max(float(getattr(config, "stage2_yref_gain", 0.003)), 0.003)
        config.stage3_use_yref = bool(getattr(config, "stage3_use_yref", True))
        config.stage1_min_episode = max(int(getattr(config, "stage1_min_episode", 800)), 800)
        config.stage2_min_episode = max(int(getattr(config, "stage2_min_episode", 950)), 1000)
        config.stage2_min_episode = max(config.stage2_min_episode, config.stage1_min_episode + 150)
        config.stage1_success_threshold = max(float(getattr(config, "stage1_success_threshold", 0.30)), 0.40)
        config.stage2_success_threshold = max(float(getattr(config, "stage2_success_threshold", 0.60)), 0.70)

        config.early_restart_enable = False
        config.hard_bad_enable = False
        config.early_plateau_min_episode = max(int(getattr(config, "early_plateau_min_episode", 220)), 1800)
        config.early_plateau_patience = max(int(getattr(config, "early_plateau_patience", 60)), 400)
        config.early_bad_min_episode = max(int(getattr(config, "early_bad_min_episode", 180)), 1800)
        config.early_bad_patience = max(int(getattr(config, "early_bad_patience", 16)), 120)

        config.success_min_avg_speed_lane = max(float(getattr(config, "success_min_avg_speed_lane", 3.0)), 3.0)
        config.success_speed_th_lane = max(float(getattr(config, "success_speed_th_lane", 3.0)), 3.0)
        config.anti_crawl_speed_lane = max(float(getattr(config, "anti_crawl_speed_lane", 3.0)), 3.0)
        config.anti_crawl_min_avg_speed_lane = max(float(getattr(config, "anti_crawl_min_avg_speed_lane", 3.0)), 3.0)
        config.anti_crawl_min_recent_speed_lane = max(float(getattr(config, "anti_crawl_min_recent_speed_lane", 3.0)), 3.0)

        config.policy_min_scale = min(float(getattr(config, "policy_min_scale", 0.14)), 0.08)
        config.policy_min_scale_lat = min(float(getattr(config, "policy_min_scale_lat", 0.22)), 0.08)
        config.policy_min_scale_lon = min(float(getattr(config, "policy_min_scale_lon", 0.18)), 0.07)
        config.policy_raw_scale_bias_init = min(float(getattr(config, "policy_raw_scale_bias_init", -0.35)), -1.9)
        config.policy_raw_scale_bias_lat = min(float(getattr(config, "policy_raw_scale_bias_lat", -0.05)), -2.1)
        config.policy_raw_scale_bias_lon = min(float(getattr(config, "policy_raw_scale_bias_lon", -0.18)), -2.2)

        config.enable_low_speed_steer_scale = False
        config.enable_anti_stall = False
        config.enable_straight_stability = False
        config.enable_steer_rate_limit = False
        config.enable_steer_delta_clip = False

        config.safe_dist_lane = float(np.clip(float(getattr(config, "safe_dist_lane", 16.0)), 12.0, 18.0))
        config.safe_dist_jaywalker = float(np.clip(float(getattr(config, "safe_dist_jaywalker", 12.0)), 8.0, 14.0))
        config.avoid_fwd_lane = float(np.clip(float(getattr(config, "avoid_fwd_lane", 32.0)), 24.0, 36.0))
        config.avoid_fwd_jaywalker = float(np.clip(float(getattr(config, "avoid_fwd_jaywalker", 24.0)), 16.0, 30.0))
        config.w_obs_clear_lane = min(float(getattr(config, "w_obs_clear_lane", 6.8)), 4.0)
        config.w_obs_clear_jaywalker = min(float(getattr(config, "w_obs_clear_jaywalker", 4.2)), 3.2)
        config.obs_danger_dist_lane = min(float(getattr(config, "obs_danger_dist_lane", 13.5)), 9.5)
    else:
        print("[Config] strict_training_profile=False，保留用户/W&B原始参数（不再强制覆盖）。")

    # ===== 运行时防呆：避免历史配置把训练推到不可学区间 =====
    try:
        # 0) on-policy 行为一致性：严格模式下强制用执行后动作训练
        config.train_onpolicy_action_mode = bool(getattr(config, "train_onpolicy_action_mode", True))
        if config.train_onpolicy_action_mode:
            config.ppo_store_executed_action = True

        # 1) policy std 参数必须自洽
        ps_clip_low = float(getattr(config, "policy_raw_scale_clip_low", -4.0))
        ps_clip_high = float(getattr(config, "policy_raw_scale_clip_high", -0.35))
        if ps_clip_low > ps_clip_high:
            ps_clip_low, ps_clip_high = ps_clip_high, ps_clip_low
        config.policy_raw_scale_clip_low = ps_clip_low
        config.policy_raw_scale_clip_high = ps_clip_high

        config.policy_min_scale = float(np.clip(float(getattr(config, "policy_min_scale", 0.06)), 0.02, 0.20))
        config.policy_min_scale_lat = float(np.clip(float(getattr(config, "policy_min_scale_lat", 0.08)), 0.02, 0.22))
        config.policy_min_scale_lon = float(np.clip(float(getattr(config, "policy_min_scale_lon", 0.06)), 0.02, 0.20))
        config.policy_raw_scale_bias_init = float(
            np.clip(float(getattr(config, "policy_raw_scale_bias_init", -1.20)), ps_clip_low, ps_clip_high)
        )
        config.policy_raw_scale_bias_lat = float(
            np.clip(float(getattr(config, "policy_raw_scale_bias_lat", -1.20)), ps_clip_low, ps_clip_high)
        )
        config.policy_raw_scale_bias_lon = float(
            np.clip(float(getattr(config, "policy_raw_scale_bias_lon", -1.20)), ps_clip_low, ps_clip_high)
        )

        # 2) 障碍物 shaping 距离参数：避免“过远就强惩罚”
        config.avoid_fwd_lane = float(np.clip(float(getattr(config, "avoid_fwd_lane", 30.0)), 18.0, 40.0))
        config.avoid_fwd_jaywalker = float(np.clip(float(getattr(config, "avoid_fwd_jaywalker", 22.0)), 12.0, 32.0))
        config.safe_dist_lane = float(np.clip(float(getattr(config, "safe_dist_lane", 14.0)), 8.0, 20.0))
        config.safe_dist_jaywalker = float(np.clip(float(getattr(config, "safe_dist_jaywalker", 10.0)), 6.0, 16.0))
        config.obs_reward_gate_floor_lane = float(
            np.clip(float(getattr(config, "obs_reward_gate_floor_lane", 0.02)), 0.0, 0.20)
        )
        config.obs_reward_gate_floor_jaywalker = float(
            np.clip(float(getattr(config, "obs_reward_gate_floor_jaywalker", 0.20)), 0.0, 0.30)
        )

        # 3) 若 y_ref 未参与控制，则显式屏蔽第三维对环境的影响，减少噪声通道
        if not bool(getattr(config, "use_yref_in_steer", True)):
            config.use_action_dim2 = True

        # 4) 早停防呆：避免在课程未跑完前被“坏窗口/平台期”过早截断
        if bool(getattr(config, "enable_early_stop", True)):
            train_eps = max(1, int(getattr(config, "train_episodes", 1500)))
            bad_min_floor = int(max(320, min(900, int(0.35 * train_eps))))
            plateau_min_floor = int(max(480, min(1200, int(0.65 * train_eps))))
            config.early_bad_min_episode = max(
                int(getattr(config, "early_bad_min_episode", bad_min_floor)),
                bad_min_floor,
            )
            config.early_plateau_min_episode = max(
                int(getattr(config, "early_plateau_min_episode", plateau_min_floor)),
                plateau_min_floor,
            )
            config.early_bad_patience = max(
                int(getattr(config, "early_bad_patience", 120)),
                30,
            )
            config.early_plateau_patience = max(
                int(getattr(config, "early_plateau_patience", 300)),
                120,
            )

            # 默认关闭“激进止损”触发器，避免把可恢复训练误停。
            aggressive_early_stop = bool(getattr(config, "aggressive_early_stop", False))
            if not aggressive_early_stop:
                config.early_restart_enable = False
                config.hard_bad_enable = False
    except Exception as e:
        print(f"[Config] ⚠️ runtime sanitize failed: {e}")

    # =========================
    # ✅ Profile 覆盖：统一旧脚本入口到一套稳定训练链
    # =========================
    def _apply_profile_overrides(cfg, profile_name: str):
        # 所有 profile 的公共稳态配置
        cfg.normalize_obs = bool(getattr(cfg, "normalize_obs", True))
        cfg.value_representation = str(getattr(cfg, "value_representation", "scalar")).strip().lower()
        if cfg.value_representation not in ("scalar", "decomposed"):
            cfg.value_representation = "scalar"
        cfg.network_units = int(getattr(cfg, "network_units", 128))
        cfg.network_layers = int(getattr(cfg, "network_layers", 2))
        cfg.trd_loss_coef = float(getattr(cfg, "trd_loss_coef", 0.0))
        cfg.use_action_bias = bool(getattr(cfg, "use_action_bias", False))
        cfg.use_forced_throttle = bool(getattr(cfg, "use_forced_throttle", False))
        cfg.train_onpolicy_action_mode = bool(getattr(cfg, "train_onpolicy_action_mode", True))
        if cfg.train_onpolicy_action_mode:
            cfg.ppo_store_executed_action = True

        if profile_name == "standalone":
            cfg.agent_name = "ppo-carla-standalone"
            cfg.random_scenario = False
            cfg.scenario = "parked_obstacles"
            cfg.scenario_pool = ["parked_obstacles"]
            cfg.observations_type = "state"
            cfg.obs_obstacle_k = 0
            cfg.use_yref_in_steer = False
            cfg.use_yref_mapping = False
            cfg.yref_steer_gain = 0.0
            cfg.yref_gain = 0.0
            cfg.yref_penalty = 0.0
            cfg.max_episode_steps = int(getattr(cfg, "max_episode_steps", 512))
            cfg.train_episodes = int(getattr(cfg, "train_episodes", 1000))
            cfg.batch_size = int(getattr(cfg, "batch_size", 256))
            cfg.update_frequency = int(getattr(cfg, "update_frequency", 1))
            cfg.optimization_steps = tuple(getattr(cfg, "optimization_steps", (6, 6)))
            cfg.policy_lr = float(getattr(cfg, "policy_lr", 5e-5))
            cfg.value_lr = float(getattr(cfg, "value_lr", 1e-4))
        elif profile_name == "simple":
            cfg.agent_name = "ppo-carla-simple"
            cfg.random_scenario = False
            cfg.scenario = "parked_obstacles"
            cfg.scenario_pool = ["parked_obstacles"]
            cfg.observations_type = "state"
            cfg.obs_obstacle_k = 0
            cfg.use_yref_in_steer = False
            cfg.use_yref_mapping = False
            cfg.yref_steer_gain = 0.0
            cfg.yref_gain = 0.0
            cfg.yref_penalty = 0.0
            cfg.max_episode_steps = int(getattr(cfg, "max_episode_steps", 512))
            cfg.train_episodes = int(getattr(cfg, "train_episodes", 600))
            cfg.batch_size = int(getattr(cfg, "batch_size", 256))
            cfg.update_frequency = int(getattr(cfg, "update_frequency", 1))
            cfg.optimization_steps = tuple(getattr(cfg, "optimization_steps", (6, 6)))
            cfg.policy_lr = float(getattr(cfg, "policy_lr", 5e-5))
            cfg.value_lr = float(getattr(cfg, "value_lr", 1e-4))
        elif profile_name == "4scenarios":
            cfg.agent_name = "ppo-carla-4scenarios"
            cfg.random_scenario = True
            cfg.scenario_pool = ["cones", "jaywalker", "trimma", "construction_lane_change"]
            cfg.observations_type = "state_lane_obstacles"
            cfg.max_episode_steps = int(getattr(cfg, "max_episode_steps", 512))
            cfg.train_episodes = int(getattr(cfg, "train_episodes", 300))
            cfg.update_frequency = int(getattr(cfg, "update_frequency", 1))
            cfg.optimization_steps = tuple(getattr(cfg, "optimization_steps", (5, 5)))
            cfg.policy_lr = float(getattr(cfg, "policy_lr", 1e-4))
            cfg.value_lr = float(getattr(cfg, "value_lr", 2e-4))
            cfg.use_yref_in_steer = bool(getattr(cfg, "use_yref_in_steer", True))
            cfg.use_yref_mapping = bool(getattr(cfg, "use_yref_in_steer", True))
        else:
            cfg.agent_name = str(getattr(cfg, "agent_name", "ppo-carla-obs30"))
            cfg.value_representation = str(getattr(cfg, "value_representation", "scalar")).strip().lower()

    _apply_profile_overrides(config, profile)
    print(
        "[Profile] "
        f"name={profile}, agent_name={getattr(config, 'agent_name', 'ppo-carla-obs30')}, "
        f"obs={getattr(config, 'observations_type', None)}, value_repr={getattr(config, 'value_representation', None)}, "
        f"net=({getattr(config, 'network_units', None)}x{getattr(config, 'network_layers', None)}), "
        f"trd_coef={getattr(config, 'trd_loss_coef', None)}"
    )

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

    # reward config：默认使用环境原生reward；可开关激进版
    use_aggressive_reward = bool(getattr(config, "use_aggressive_reward", False))
    if use_aggressive_reward:
        print("\n[0] 加载激进版Reward配置...")
        try:
            from reward_config_aggressive import get_aggressive_config
            config.reward_config = get_aggressive_config()
            print("✅ 激进版Reward已加载")
        except Exception as e:
            print(f"⚠️  无法加载激进版reward，使用默认配置: {e}")
    else:
        print("\n[0] 使用默认Reward配置（未启用激进版）")

    print("\n[1] 创建训练日志记录器...")
    logger = TrainingLogger(log_file="training_log.json", auto_save_interval=5)
    print("✅ 日志记录器创建成功")

    print("\n[DEBUG] Config y_ref switch:")
    print("  use_yref_in_steer =", getattr(config, "use_yref_in_steer", None))
    print("  yref_steer_gain   =", getattr(config, "yref_steer_gain", None))
    print("  use_yref_mapping  =", getattr(config, "use_yref_mapping", None))
    print("  yref_gain         =", getattr(config, "yref_gain", None))
    print("  yref_penalty     =", getattr(config, "yref_penalty", None))
    print("  train_onpolicy_action_mode =", getattr(config, "train_onpolicy_action_mode", None))
    print("  ppo_store_executed_action  =", getattr(config, "ppo_store_executed_action", None))
    print("\n[DEBUG] Config signature (safety_v3):")
    print("  stage1_success_threshold =", getattr(config, "stage1_success_threshold", None))
    print("  stage2_success_threshold =", getattr(config, "stage2_success_threshold", None))
    print("  stage3_use_yref         =", getattr(config, "stage3_use_yref", None))
    print("  entropy(start/end/decay)=", (
        getattr(config, "entropy_start", None),
        getattr(config, "entropy_end", None),
        getattr(config, "entropy_decay_ep", None),
    ))
    print("  obs_reward_lat_tol_lane =", getattr(config, "obs_reward_lat_tol_lane", None))
    print("  k_avoid_lat_lane        =", getattr(config, "k_avoid_lat_lane", None))
    print("  w_obs_clear_lane        =", getattr(config, "w_obs_clear_lane", None))
    print("  w_obs_speed_lane        =", getattr(config, "w_obs_speed_lane", None))
    print("  w_obs_danger_lane       =", getattr(config, "w_obs_danger_lane", None))
    print("  w_obs_ttc_lane          =", getattr(config, "w_obs_ttc_lane", None))
    print("  obs_ttc_crit_lane       =", getattr(config, "obs_ttc_crit_lane", None))
    print("  obs_danger_dist_lane    =", getattr(config, "obs_danger_dist_lane", None))
    print("  steer_postprocess flags =", (
        getattr(config, "enable_steer_smoothing", None),
        getattr(config, "enable_steer_rate_limit", None),
        getattr(config, "enable_straight_stability", None),
        getattr(config, "obstacle_steer_assist_enable", None),
    ))
    print("  policy_min_scale(lat/lon) =", (
        getattr(config, "policy_min_scale_lat", None),
        getattr(config, "policy_min_scale_lon", None),
    ))
    print("  policy_raw_scale_clip   =", (
        getattr(config, "policy_raw_scale_clip_low", None),
        getattr(config, "policy_raw_scale_clip_high", None),
    ))

    agent_name = str(getattr(config, "agent_name", "ppo-carla-obs30"))
    weights_root = "./weights"
    agent_base_path = os.path.join(weights_root, agent_name)
    dump_effective_config(config, os.path.join(agent_base_path, "effective_config.json"))

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

    request_load_existing = bool(getattr(config, "load_existing", False))
    resume_policy_only = bool(getattr(config, "resume_policy_only", False))
    ckpt_ready = has_compatible_checkpoint(agent_base_path)
    load_existing = bool(request_load_existing and ckpt_ready)
    resume_quality_guard = bool(getattr(config, "resume_quality_guard", True))
    resume_min_success_w = float(getattr(config, "resume_min_success_w", 0.03))
    resume_min_speed_ok_w = float(getattr(config, "resume_min_speed_ok_w", 0.30))
    resume_min_safe_fast_w = float(getattr(config, "resume_min_safe_fast_w", 0.15))

    if request_load_existing and ckpt_ready:
        if resume_quality_guard:
            m = read_best_ckpt_metrics(agent_base_path)
            succ_w = float(m.get("success_rate_W", m.get("success_rate_w", -1.0))) if m else -1.0
            speed_ok_w = float(m.get("speed_ok_rate_W", m.get("speed_ok_rate_w", -1.0))) if m else -1.0
            safe_fast_w = float(m.get("safe_fast_rate_W", m.get("safe_fast_rate_w", -1.0))) if m else -1.0
            if (
                m
                and (succ_w >= 0.0 and succ_w < resume_min_success_w)
                and (speed_ok_w >= 0.0 and speed_ok_w < resume_min_speed_ok_w)
                and (safe_fast_w >= 0.0 and safe_fast_w < resume_min_safe_fast_w)
            ):
                load_existing = False
                print(
                    f"\n⚠️ 检测到checkpoint质量过低，自动改为从头训练: {agent_base_path}\n"
                    f"   success_W={succ_w:.3f}, speed_ok_W={speed_ok_w:.3f}, safe_fast_W={safe_fast_w:.3f}"
                )
            else:
                print(f"\n✅ 检测到可用checkpoint，启用续训: {agent_base_path}")
        else:
            print(f"\n✅ 检测到可用checkpoint，启用续训: {agent_base_path}")
    elif request_load_existing and (not ckpt_ready):
        print(f"\n⚠️ 未检测到完整checkpoint，自动从头训练: {agent_base_path}")
    else:
        print("\n✅ 配置为从头训练（load_existing=False）")

    base_policy_lr = float(config.policy_lr)
    base_value_lr = float(config.value_lr)
    base_entropy_start = float(config.entropy_start)
    base_entropy_end = float(config.entropy_end)
    base_entropy_reg = float(config.entropy_reg)
    base_opt_steps = tuple(config.optimization_steps)

    if load_existing and bool(getattr(config, "finetune_on_resume", True)):
        lr_p_scale = max(0.05, float(getattr(config, "resume_policy_lr_scale", 0.5)))
        lr_v_scale = max(0.05, float(getattr(config, "resume_value_lr_scale", 0.5)))
        ent_scale = max(0.1, float(getattr(config, "resume_entropy_scale", 0.9)))
        opt_scale = max(0.25, float(getattr(config, "resume_opt_steps_scale", 0.75)))

        config.policy_lr = max(1e-6, float(config.policy_lr) * lr_p_scale)
        config.value_lr = max(1e-6, float(config.value_lr) * lr_v_scale)
        config.entropy_start = max(1e-4, float(config.entropy_start) * ent_scale)
        config.entropy_end = max(1e-5, float(config.entropy_end) * ent_scale)
        if config.entropy_start < config.entropy_end:
            config.entropy_start, config.entropy_end = config.entropy_end, config.entropy_start
        config.entropy_reg = min(float(config.entropy_reg), float(config.entropy_start))

        _p_opt, _v_opt = tuple(config.optimization_steps)
        config.optimization_steps = (
            max(2, int(round(_p_opt * opt_scale))),
            max(2, int(round(_v_opt * opt_scale))),
        )
        print(
            "[Resume] finetune_on_resume=True -> "
            f"policy_lr={config.policy_lr:.2e}, value_lr={config.value_lr:.2e}, "
            f"entropy=({config.entropy_start:.4f}->{config.entropy_end:.4f}), "
            f"opt_steps={config.optimization_steps}"
        )

    # ✅ 修复P1级和P2级问题：优化超参数
    agent_kwargs = dict(
        env=env,
        seed=int(getattr(config, "seed", 10)),
        policy_lr=float(config.policy_lr),
        value_lr=float(config.value_lr),
        gamma=float(config.gamma),
        lambda_=float(config.lambda_),
        clip_ratio=float(config.clip_ratio),
        entropy_regularization=float(config.entropy_reg),
        target_kl=float(getattr(config, "policy_target_kl", 0.022)),
        kl_stop_multiplier=float(getattr(config, "policy_kl_stop_multiplier", 1.15)),
        optimization_steps=tuple(config.optimization_steps),
        batch_size=int(config.batch_size),
        update_frequency=int(config.update_frequency),
        value_representation=str(getattr(config, "value_representation", "scalar")),
        network=dict(
            units=int(getattr(config, "network_units", 128)),
            num_layers=int(getattr(config, "network_layers", 2)),
            policy=dict(
                min_scale=float(getattr(config, "policy_min_scale", 0.06)),
                min_scale_lat=float(getattr(config, "policy_min_scale_lat", 0.08)),
                min_scale_lon=float(getattr(config, "policy_min_scale_lon", 0.06)),
                raw_scale_bias=float(getattr(config, "policy_raw_scale_bias_init", -1.20)),
                raw_scale_bias_lat=float(getattr(config, "policy_raw_scale_bias_lat", -1.20)),
                raw_scale_bias_lon=float(getattr(config, "policy_raw_scale_bias_lon", -1.20)),
                raw_scale_clip_min=float(getattr(config, "policy_raw_scale_clip_low", -4.0)),
                raw_scale_clip_max=float(getattr(config, "policy_raw_scale_clip_high", -0.35)),
            ),
            value=dict(),
        ),
        name=agent_name,
    )
    try:
        agent = PPOAgent(load=load_existing, **agent_kwargs)
    except Exception as e:
        if not load_existing:
            raise
        print(f"[Resume] ⚠️ checkpoint加载失败，回退从头训练: {e}")
        load_existing = False
        config.policy_lr = base_policy_lr
        config.value_lr = base_value_lr
        config.entropy_start = base_entropy_start
        config.entropy_end = base_entropy_end
        config.entropy_reg = base_entropy_reg
        config.optimization_steps = base_opt_steps
        env.global_env_step = 0
        agent = PPOAgent(load=False, **agent_kwargs)

    # 兼容不同 PPOAgent 版本：构造后再设置 TRD 系数，避免 __init__ 参数不兼容
    trd_coef = float(getattr(config, "trd_loss_coef", 0.0))
    if hasattr(agent, "trd_loss_coef"):
        agent.trd_loss_coef = trd_coef
    if hasattr(agent, "trd_coef_max"):
        agent.trd_coef_max = trd_coef

    print("✅ PPO Agent创建成功")
    print(f"  - 模型保存路径: {agent.base_path}")

    start_episode = 0
    if load_existing:
        resume_state = load_resume_state(agent.base_path)
        if resume_state.get("loaded", False):
            start_episode = int(resume_state.get("global_episode", 0))
            env.global_env_step = int(resume_state.get("global_env_step", 0))
            restored_rms = env.set_obs_norm_state({
                "mean": resume_state.get("obs_norm_mean"),
                "var": resume_state.get("obs_norm_var"),
                "count": resume_state.get("obs_norm_count"),
            })
            print(
                f"[Resume] ✅ 已恢复训练状态: start_episode={start_episode}, "
                f"global_env_step={env.global_env_step}, obs_norm_restored={int(restored_rms)}"
            )
        else:
            print("[Resume] ⚠️ 未找到resume_state.npz，将从 episode=0 开始计数续训。")

    # ✅ 仅恢复 policy，避免 reward 变化导致 value 失配
    if resume_policy_only:
        lat_path = os.path.join(agent.base_path, "policy_net_lat")
        lon_path = os.path.join(agent.base_path, "policy_net_lon")
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

    last_saved_episode = int(start_episode)

    def _state_callback(global_episode: int, env_obj: CarlaGymEnv):
        nonlocal last_saved_episode
        last_saved_episode = int(global_episode)
        obs_state = env_obj.get_obs_norm_state() if hasattr(env_obj, "get_obs_norm_state") else {}
        save_resume_state(
            base_path=agent.base_path,
            global_episode=last_saved_episode,
            global_env_step=int(getattr(env_obj, "global_env_step", 0)),
            obs_norm_state=obs_state,
        )

    try:
        last_saved_episode = train_with_logging(
            agent, env, logger,
            wandb_run=wandb_run,
            episodes=(debug_vis_episodes if debug_vis else int(getattr(config, "train_episodes", 1500))),
            # ✅ 和 env.max_episode_steps 保持一致，避免time_limit截断影响统计
            timesteps=int(getattr(config, "max_episode_steps", 512)),
            save_every=100,
            start_episode=int(start_episode),
            state_callback=_state_callback,
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练异常中止: {e}")
    finally:
        try:
            final_obs_state = env.get_obs_norm_state() if hasattr(env, "get_obs_norm_state") else {}
            save_resume_state(
                base_path=agent.base_path,
                global_episode=int(last_saved_episode),
                global_env_step=int(getattr(env, "global_env_step", 0)),
                obs_norm_state=final_obs_state,
            )
            print(
                f"[Resume] 已保存续训状态: episode={int(last_saved_episode)}, "
                f"env_step={int(getattr(env, 'global_env_step', 0))}"
            )
        except Exception as e:
            print(f"[Resume] ⚠️ finally保存续训状态失败: {e}")
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
