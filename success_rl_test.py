from __future__ import annotations

"""
Rule-Based Agent Success Test:
- 四个场景各跑有效 episode 50 次（仅计场景初始化 + ego 生成成功的 episode）
- 记录：场景名、episode编号、是否碰撞
- 若无碰撞，记录 EVA 的 safety / comfort / efficiency（按 episode 均值）

输出：
  /home/ajifang/il_data_collect/il_data/success_rl_report.csv
"""

import sys
import time
import csv
import random
import math
import os
import contextlib
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Any

sys.path.append('/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
import carla
sys.path.append("/home/ajifang/RL_selector")

from env.scenarios import (
    ConesScenario,
    JaywalkerScenario,
    TrimmaScenario,
    ConstructionLaneChangeScenario,
)

# 引入你的 rl planner (PPO)
sys.path.append("/home/ajifang/SAC_carla/planners")
from rl_agent_only.agents.ppo import PPOAgent

# ================================
# 复制版：carla_utils（不依赖 il_data_collector）
# ================================
@contextlib.contextmanager
def carla_sync_mode(client: carla.Client, world: carla.World, enabled: bool, fixed_dt: float = 0.05):
    if not enabled:
        yield
        return

    original_settings = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        yield
    finally:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)


EGO_BP_CANDIDATES = [
    "vehicle.tesla.model3",
]


def get_ego_blueprint(world: carla.World) -> carla.ActorBlueprint:
    lib = world.get_blueprint_library()
    for name in EGO_BP_CANDIDATES:
        try:
            bp = lib.find(name)
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                bp.set_attribute("color", colors[0] if colors else "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp
        except Exception:
            continue

    for bp in lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and bp.get_attribute("number_of_wheels").as_int() == 4:
            if bp.has_attribute("color"):
                bp.set_attribute("color", "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp

    raise RuntimeError("未找到可用车辆蓝图。")


def set_spectator_follow_ego(
    world: carla.World,
    ego: carla.Actor,
    mode: str = "chase",
    distance: float = 22.0,
    height: float = 7.0,
    pitch_deg: float = -12.0,
    side_offset: float = 1.5,
):
    spec = world.get_spectator()
    tf = ego.get_transform()
    yaw = tf.rotation.yaw
    rad = math.radians(yaw)
    fx, fy = math.cos(rad), math.sin(rad)
    rx, ry = math.sin(rad), -math.cos(rad)

    if mode == "top":
        cam_loc = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + height)
        cam_rot = carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0)
    else:
        cam_loc = carla.Location(
            x=tf.location.x - fx * distance + rx * side_offset,
            y=tf.location.y - fy * distance + ry * side_offset,
            z=tf.location.z + height,
        )
        cam_rot = carla.Rotation(pitch=pitch_deg, yaw=yaw, roll=0.0)

    spec.set_transform(carla.Transform(cam_loc, cam_rot))


# ================================
# 复制版：EVA Monitor（不依赖 il_data_collector）
# ================================
try:
    import pygame
except Exception as e:
    raise RuntimeError("需要 pygame 才能显示 EVA HUD，请先安装 pygame") from e


@dataclass
class EVAConfig:
    w_saf: float = 1.0
    w_com: float = 0.2
    w_eff: float = 0.1

    w_col: float = 1.0
    w_dev: float = 0.3
    w_head: float = 0.2
    w_off: float = 0.6
    w_obs: float = 0.6
    w_spd_pen: float = 0.2

    w_steer: float = 0.3
    w_jerk: float = 0.4
    w_brake: float = 0.3

    w_spd: float = 0.6
    w_prog: float = 0.4

    max_dev: float = 2.0
    max_head: float = 0.35
    speed_limit: float = 12.0
    min_speed: float = 0.5
    ttc_crit: float = 3.0
    obs_dist_crit: float = 8.0
    brake_thr: float = 0.5


class EvaMonitor:
    def __init__(self, cfg: EVAConfig = EVAConfig(), width: int = 520, height: int = 320):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.world: Optional[carla.World] = None
        self.ego: Optional[carla.Actor] = None

        self.prev_speed: Optional[float] = None
        self.prev_acc: Optional[float] = None
        self.prev_steer: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.progress_s: float = 0.0

        self.hist_len = 120
        self.hist_total = deque(maxlen=self.hist_len)
        self.hist_speed = deque(maxlen=self.hist_len)
        self.hist_saf = deque(maxlen=self.hist_len)
        self.hist_com = deque(maxlen=self.hist_len)
        self.hist_eff = deque(maxlen=self.hist_len)

        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "20,20")
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("EVA Monitor")
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.font_micro = pygame.font.SysFont("Arial", 12)

        self.bg = (16, 18, 22)
        self.card = (36, 38, 44)
        self.card_2 = (30, 32, 38)
        self.card_3 = (26, 28, 34)
        self.text = (228, 232, 236)
        self.text_dim = (160, 165, 175)
        self.accent = (255, 156, 66)
        self.accent2 = (96, 192, 255)
        self.good = (112, 212, 140)
        self.warn = (255, 178, 77)
        self.bad = (219, 88, 96)

    def attach(self, world: carla.World, ego: carla.Actor):
        self.world = world
        self.ego = ego
        self.prev_speed = None
        self.prev_acc = None
        self.prev_steer = None
        self.prev_time = None
        self.progress_s = 0.0
        self.hist_total.clear()
        self.hist_speed.clear()
        self.hist_saf.clear()
        self.hist_com.clear()
        self.hist_eff.clear()

    def _get_lateral_and_heading_error(self) -> (float, float):
        if self.world is None or self.ego is None:
            return 0.0, 0.0
        amap = self.world.get_map()
        ego_tf = self.ego.get_transform()
        wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0, 0.0

        dx = ego_tf.location.x - wp.transform.location.x
        dy = ego_tf.location.y - wp.transform.location.y
        yaw_wp = math.radians(wp.transform.rotation.yaw)
        nx = -math.sin(yaw_wp)
        ny = math.cos(yaw_wp)
        dev = dx * nx + dy * ny

        yaw_ego = math.radians(ego_tf.rotation.yaw)
        head_err = (yaw_ego - yaw_wp + math.pi) % (2 * math.pi) - math.pi
        return float(dev), float(head_err)

    def _is_offroad(self) -> bool:
        if self.world is None or self.ego is None:
            return False
        amap = self.world.get_map()
        wp = amap.get_waypoint(self.ego.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving)
        return wp is None

    def _nearest_obstacle_info(self) -> (float, float):
        if self.world is None or self.ego is None:
            return 999.0, 999.0

        ego_tf = self.ego.get_transform()
        ego_vel = self.ego.get_velocity()
        ego_speed = math.hypot(ego_vel.x, ego_vel.y)
        fwd = ego_tf.get_forward_vector()

        nearest_dist = 999.0
        ttc = 999.0

        actors = self.world.get_actors().filter("vehicle.*|walker.pedestrian.*")
        for a in actors:
            if a.id == self.ego.id:
                continue
            loc = a.get_location()
            dx = loc.x - ego_tf.location.x
            dy = loc.y - ego_tf.location.y
            ahead = dx * fwd.x + dy * fwd.y
            if ahead <= 0:
                continue
            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                a_vel = a.get_velocity()
                rel_v = ego_speed - math.hypot(a_vel.x, a_vel.y)
                if rel_v > 0.1:
                    ttc = dist / rel_v
                else:
                    ttc = 999.0

        return nearest_dist, ttc

    def _speed_reward(self, v: float) -> float:
        v_des = min(self.cfg.speed_limit, max(self.cfg.min_speed, self.cfg.speed_limit))
        sigma = 2.0
        return math.exp(-((v - v_des) ** 2) / (2 * sigma ** 2))

    def tick(self) -> Dict[str, float]:
        if self.world is None or self.ego is None:
            return {}

        snap = self.world.get_snapshot()
        now = snap.timestamp.elapsed_seconds
        ego_vel = self.ego.get_velocity()
        speed = math.hypot(ego_vel.x, ego_vel.y)

        if self.prev_speed is None or self.prev_time is None:
            acc = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            acc = (speed - self.prev_speed) / dt

        if self.prev_acc is None or self.prev_time is None:
            jerk = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            jerk = (acc - self.prev_acc) / dt

        ctrl = self.ego.get_control()
        steer = float(ctrl.steer)

        dev, head_err = self._get_lateral_and_heading_error()
        offroad = 1.0 if self._is_offroad() else 0.0
        obs_dist, ttc = self._nearest_obstacle_info()

        r_col = 0.0
        r_dev = -min(abs(dev) / self.cfg.max_dev, 1.0)
        r_head = -min(abs(head_err) / self.cfg.max_head, 1.0)
        r_off = -offroad
        r_obs = -1.0 if (obs_dist < self.cfg.obs_dist_crit or ttc < self.cfg.ttc_crit) else 0.0
        r_spd_pen = -max(0.0, (speed - self.cfg.speed_limit) / max(1e-3, self.cfg.speed_limit))

        r_saf = (self.cfg.w_col * r_col +
                 self.cfg.w_dev * r_dev +
                 self.cfg.w_head * r_head +
                 self.cfg.w_off * r_off +
                 self.cfg.w_obs * r_obs +
                 self.cfg.w_spd_pen * r_spd_pen)

        r_steer = -(steer ** 2)
        r_jerk = -min(1.0, abs(jerk) / 5.0)
        r_brake = -max(0.0, (ctrl.brake - self.cfg.brake_thr))

        r_com = (self.cfg.w_steer * r_steer +
                 self.cfg.w_jerk * r_jerk +
                 self.cfg.w_brake * r_brake)

        r_spd = self._speed_reward(speed)
        r_prog = speed / max(1e-3, self.cfg.speed_limit)
        r_eff = self.cfg.w_spd * r_spd + self.cfg.w_prog * r_prog

        r_total = self.cfg.w_saf * r_saf + self.cfg.w_com * r_com + self.cfg.w_eff * r_eff

        self.prev_speed = speed
        self.prev_acc = acc
        self.prev_steer = steer
        self.prev_time = now

        data = {
            "r_total": r_total,
            "r_saf": r_saf,
            "r_com": r_com,
            "r_eff": r_eff,
            "r_dev": r_dev,
            "r_head": r_head,
            "r_off": r_off,
            "r_obs": r_obs,
            "r_spd_pen": r_spd_pen,
            "r_steer": r_steer,
            "r_jerk": r_jerk,
            "r_brake": r_brake,
            "r_spd": r_spd,
            "r_prog": r_prog,
            "speed": speed,
            "dev": dev,
            "head_err": head_err,
            "ttc": ttc,
            "obs_dist": obs_dist,
        }

        self.hist_total.append(r_total)
        self.hist_speed.append(speed)
        self.hist_saf.append(r_saf)
        self.hist_com.append(r_com)
        self.hist_eff.append(r_eff)

        return data

    def render(self, data: Optional[Dict[str, float]] = None):
        if data is None:
            data = {}

        self.screen.fill(self.bg)
        panel = pygame.Rect(10, 10, self.width - 20, self.height - 20)
        pygame.draw.rect(self.screen, self.card, panel, border_radius=14)

        title = self.font_title.render("EVA Monitor", True, self.text)
        self.screen.blit(title, (24, 18))
        sub = self.font_small.render("Safety / Comfort / Efficiency Dashboard", True, self.text_dim)
        self.screen.blit(sub, (24, 44))

        left = pygame.Rect(14, 64, 230, 230)
        pygame.draw.rect(self.screen, self.card_2, left, border_radius=12)

        right = pygame.Rect(252, 64, 250, 230)
        pygame.draw.rect(self.screen, self.card_2, right, border_radius=12)

        def draw_bar(label, value, y, color):
            v = max(-1.0, min(1.0, float(value)))
            pygame.draw.rect(self.screen, self.card_3, (28, y + 18, 185, 10), border_radius=6)
            w = int(92 + 92 * v)
            x0 = 28
            pygame.draw.rect(self.screen, color, (x0, y + 18, w, 10), border_radius=6)
            text = self.font_small.render(f"{label}: {value: .3f}", True, self.text)
            self.screen.blit(text, (28, y))
            txt_r = self.font_small.render(f"{value: .3f}", True, self.text_dim)
            self.screen.blit(txt_r, (200, y))

        draw_bar("Total", data.get("r_total", 0.0), 86, self.accent)
        draw_bar("Safety", data.get("r_saf", 0.0), 124, self.good)
        draw_bar("Comfort", data.get("r_com", 0.0), 162, self.accent)
        draw_bar("Efficiency", data.get("r_eff", 0.0), 200, self.accent2)

        def draw_gauge_card(x, y, w, h, title, value, unit, color, vmin, vmax):
            pygame.draw.rect(self.screen, self.card_3, (x, y, w, h), border_radius=10)
            cx, cy = x + 28, y + 26
            radius = 18
            arc_rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            pygame.draw.arc(self.screen, (70, 74, 82), arc_rect, math.radians(200), math.radians(340), 3)
            v = max(vmin, min(vmax, float(value)))
            ang = (v - vmin) / (vmax - vmin + 1e-6) * 140 + 200
            rad = math.radians(ang)
            px = cx + int((radius - 2) * math.cos(rad))
            py = cy + int((radius - 2) * math.sin(rad))
            pygame.draw.line(self.screen, color, (cx, cy), (px, py), 2)
            pygame.draw.circle(self.screen, color, (cx, cy), 3)
            t = self.font_micro.render(title, True, self.text_dim)
            self.screen.blit(t, (x + 54, y + 6))
            vtxt = self.font.render(f"{value:.2f} {unit}", True, self.text)
            self.screen.blit(vtxt, (x + 54, y + 24))

        draw_gauge_card(20, 238, 108, 40, "Speed", data.get("speed", 0.0), "m/s", self.accent2, 0.0, 20.0)
        draw_gauge_card(132, 238, 108, 40, "Deviation", data.get("dev", 0.0), "m", self.good, 0.0, self.cfg.max_dev)
        draw_gauge_card(20, 282, 108, 40, "Head Error", data.get("head_err", 0.0), "rad", self.warn, 0.0, self.cfg.max_head)
        draw_gauge_card(132, 282, 108, 40, "TTC", data.get("ttc", 0.0), "s", self.accent, 0.0, 10.0)

        def draw_line_chart(rect, series, color, label, y_min=-1.0, y_max=1.0, scale: float = 1.0):
            x, y, w, h = rect
            pygame.draw.rect(self.screen, self.card_3, rect, border_radius=10)
            if len(series) < 2:
                return
            pts = []
            for i, v in enumerate(series):
                vx = x + int(i * (w - 8) / max(1, len(series) - 1)) + 4
                vv = max(y_min, min(y_max, float(v) * scale))
                vy = y + h - int((vv - y_min) / (y_max - y_min + 1e-6) * (h - 8)) - 4
                pts.append((vx, vy))
            pygame.draw.lines(self.screen, color, False, pts, 2)
            if scale != 1.0:
                label = f"{label} x{scale:.0f}"
            txt = self.font_small.render(label, True, self.text)
            self.screen.blit(txt, (x + 8, y + 6))

        draw_line_chart((264, 84, 230, 52), self.hist_total, self.accent, "Total (History)", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 142, 230, 46), self.hist_saf, self.good, "Safety", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 194, 230, 46), self.hist_com, self.warn, "Comfort", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 246, 230, 46), self.hist_eff, self.accent2, "Efficiency", -1.0, 1.0, scale=10.0)

        pygame.display.flip()


# ================================
# PPO Planner（与 train_ppo_with_wandb 对齐）
# ================================
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("需要 numpy 才能运行 PPO planner") from e

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("需要 tensorflow 才能运行 PPO planner") from e


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
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
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class DummyGymEnv:
    def __init__(self, obs_dim: int = 30, action_dim: int = 3):
        import gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )


class RLPPOPlanner:
    _shared_agent: Optional[PPOAgent] = None
    _shared_normalizer: Optional[RunningMeanStd] = None
    _shared_obs_dim: Optional[int] = None

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # --- obs config (与 train_ppo_with_wandb 对齐) ---
        self.obs_use_lane = True
        self.obs_use_obstacles = True
        self.obs_obstacle_k = 5
        self.obs_obstacle_range = 50.0

        self.base_state_dim = 9
        self.lane_dim = 6 if self.obs_use_lane else 0
        self.obstacle_dim = (self.obs_obstacle_k * 3) if self.obs_use_obstacles else 0
        self.obs_dim = self.base_state_dim + self.lane_dim + self.obstacle_dim

        # --- action config ---
        self.action_dim = 3
        self.use_yref_in_steer = True
        self.yref_steer_gain = 0.03

        # --- obs normalization ---
        self.normalize_obs = True
        self.obs_clip = 10.0

        # --- control shaping (与 CarlaEnv.step 对齐) ---
        self.brake_deadzone = 0.05
        self.low_speed_brake_cut_speed = 0.6
        self.low_speed_throttle_floor_speed = 0.6
        self.low_speed_throttle_floor = 0.05
        self.enable_low_speed_steer_scale = True
        self.low_speed_steer_speed = 2.0
        self.low_speed_steer_min_scale = 0.25
        self.enable_steer_smoothing = True
        self.steer_smooth_alpha = 0.7
        self.enable_anti_stall = True
        self.stuck_speed_thresh = 0.30
        self.min_throttle_when_stuck = 0.20
        self.low_speed_steer_scale = 0.60

        # --- runtime ---
        self.world: Optional[carla.World] = None
        self.ego: Optional[carla.Actor] = None
        self.map: Optional[carla.Map] = None
        self.target_wp = None
        self.episode_steps = 0
        self.prev_control_for_smooth: Optional[carla.VehicleControl] = None

        if RLPPOPlanner._shared_agent is None:
            self._init_agent()
            RLPPOPlanner._shared_agent = self.agent
            RLPPOPlanner._shared_obs_dim = self.obs_dim
        else:
            if RLPPOPlanner._shared_obs_dim != self.obs_dim:
                raise RuntimeError("PPO obs_dim 不匹配，请检查模型与观测维度")
            self.agent = RLPPOPlanner._shared_agent

        if self.normalize_obs:
            if RLPPOPlanner._shared_normalizer is None:
                RLPPOPlanner._shared_normalizer = RunningMeanStd(shape=(self.obs_dim,))
            self.obs_normalizer = RLPPOPlanner._shared_normalizer
        else:
            self.obs_normalizer = None

    def _init_agent(self):
        weights_dir = os.path.dirname(self.model_dir.rstrip("/"))
        name = os.path.basename(self.model_dir.rstrip("/"))
        dummy_env = DummyGymEnv(obs_dim=self.obs_dim, action_dim=self.action_dim)

        self.agent = PPOAgent(
            env=dummy_env,
            policy_lr=1e-4,
            value_lr=2e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_ratio=0.2,
            entropy_regularization=0.12,
            optimization_steps=(10, 10),
            batch_size=128,
            update_frequency=2,
            name=name,
            weights_dir=weights_dir,
            load=False
        )

        self.agent.load_weights()

    def _wrap_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _world_to_ego_frame(self, dx: float, dy: float, ego_yaw_rad: float):
        c = math.cos(ego_yaw_rad)
        s = math.sin(ego_yaw_rad)
        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy
        return rel_x, rel_y

    def _compute_lane_obs(self) -> np.ndarray:
        loc = self.ego.get_location()
        tfm = self.ego.get_transform()
        ego_yaw = math.radians(float(tfm.rotation.yaw))

        wp = self.map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            return np.zeros((6,), dtype=np.float32)

        lane_dev = float(math.hypot(loc.x - wp.transform.location.x, loc.y - wp.transform.location.y))
        lane_w = float(getattr(wp, "lane_width", 3.5))

        fwd = wp.transform.get_forward_vector()
        lane_yaw = math.atan2(float(fwd.y), float(fwd.x))
        heading_err = self._wrap_angle(lane_yaw - ego_yaw)

        tgt = self.target_wp
        if tgt is None:
            nxt = wp.next(5.0)
            tgt = nxt[0] if nxt else wp

        tgt_loc = tgt.transform.location
        dx = float(tgt_loc.x - loc.x)
        dy = float(tgt_loc.y - loc.y)
        wp_rel_x, wp_rel_y = self._world_to_ego_frame(dx, dy, ego_yaw)

        lane_dev_n = float(np.clip(lane_dev / max(1e-3, lane_w), 0.0, 3.0))
        lane_w_n = float(np.clip(lane_w / 4.0, 0.0, 2.0))
        wp_rel_x_n = float(np.clip(wp_rel_x / 20.0, -2.0, 2.0))
        wp_rel_y_n = float(np.clip(wp_rel_y / 10.0, -2.0, 2.0))

        return np.array([
            lane_dev_n,
            math.sin(heading_err),
            math.cos(heading_err),
            lane_w_n,
            wp_rel_x_n,
            wp_rel_y_n,
        ], dtype=np.float32)

    def _collect_obstacle_candidates(self):
        if hasattr(self, "obstacle_actors") and self.obstacle_actors:
            out = []
            for a in self.obstacle_actors:
                if a is None:
                    continue
                try:
                    _ = a.get_location()
                    out.append(a)
                except Exception:
                    continue
            return out

        try:
            actors = self.world.get_actors()
            out = []
            for v in actors.filter("vehicle.*"):
                if self.ego is not None and v.id == self.ego.id:
                    continue
                out.append(v)

            for p in actors.filter("static.prop.*"):
                tid = getattr(p, "type_id", "")
                if ("trafficcone" in tid) or ("cone" in tid) or ("barrier" in tid) or ("construction" in tid):
                    out.append(p)
            return out
        except Exception:
            return []

    def _compute_obstacle_obs(self) -> np.ndarray:
        K = int(self.obs_obstacle_k)
        R = float(self.obs_obstacle_range)

        candidates = self._collect_obstacle_candidates()
        if not candidates or (self.ego is None):
            return np.zeros((K * 3,), dtype=np.float32)

        ego_loc = self.ego.get_location()
        ego_tf = self.ego.get_transform()
        ego_fwd = ego_tf.get_forward_vector()
        ego_right = ego_tf.get_right_vector()

        items = []
        for a in candidates:
            try:
                a_loc = a.get_location()
            except Exception:
                continue

            dx = float(a_loc.x - ego_loc.x)
            dy = float(a_loc.y - ego_loc.y)
            dist = float(math.hypot(dx, dy))
            if dist > R:
                continue

            rel_x = dx * float(ego_fwd.x) + dy * float(ego_fwd.y)
            ego_left = carla.Vector3D(x=-ego_right.x, y=-ego_right.y, z=-ego_right.z)
            rel_y = dx * ego_left.x + dy * ego_left.y

            items.append((dist, rel_x, rel_y))

        if not items:
            return np.zeros((K * 3,), dtype=np.float32)

        items.sort(key=lambda x: x[0])
        items = items[:K]

        feats = []
        for dist, rel_x, rel_y in items:
            rel_x_n = float(np.clip(rel_x / R, -1.0, 1.0))
            rel_y_n = float(np.clip(rel_y / R, -1.0, 1.0))
            dist_n = float(np.clip(dist / R, 0.0, 1.0))
            feats.extend([rel_x_n, rel_y_n, dist_n])

        while len(feats) < K * 3:
            feats.extend([0.0, 0.0, 0.0])

        return np.array(feats, dtype=np.float32)

    def _get_state_obs(self) -> np.ndarray:
        tfm = self.ego.get_transform()
        loc, rot = tfm.location, tfm.rotation
        acc = self._vector_to_scalar(self.ego.get_acceleration())
        ang = self._vector_to_scalar(self.ego.get_angular_velocity())
        vel = self._vector_to_scalar(self.ego.get_velocity())

        base = np.array(
            [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll, acc, ang, vel],
            dtype=np.float32,
        )

        parts = [base]
        if self.obs_use_lane:
            parts.append(self._compute_lane_obs())
        if self.obs_use_obstacles:
            parts.append(self._compute_obstacle_obs())

        obs = np.concatenate(parts, axis=0).astype(np.float32)
        if obs.shape[0] != int(self.obs_dim):
            raise RuntimeError(f"PPO obs_dim mismatch: got {obs.shape[0]} expected {self.obs_dim}")
        return obs

    def _vector_to_scalar(self, vector):
        return float(math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2))

    def _action_to_control(self, action: np.ndarray) -> (float, float, float):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        a0 = float(action[0]) if action.shape[0] > 0 else 0.0
        a1 = float(action[1]) if action.shape[0] > 1 else 0.0
        a2 = float(action[2]) if action.shape[0] > 2 else 0.0

        a0 = float(np.clip(a0, -1.0, 1.0))
        a1 = float(np.clip(a1, -1.0, 1.0))
        y_ref = float(np.clip(a2, -1.0, 1.0))

        if a0 >= 0.0:
            throttle = float(np.clip(a0, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            b = float(np.clip(-a0, 0.0, 1.0))
            b = 0.0 if b < self.brake_deadzone else b
            brake = b

        steer_raw = float(np.clip(a1, -1.0, 1.0))
        steer = steer_raw
        if self.use_yref_in_steer:
            steer = float(np.clip(steer_raw + self.yref_steer_gain * y_ref, -1, 1))

        speed = None
        if self.ego is not None:
            v = self.ego.get_velocity()
            speed = float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

        if self.enable_low_speed_steer_scale and (speed is not None):
            if speed < self.low_speed_steer_speed:
                frac = speed / max(self.low_speed_steer_speed, 1e-6)
                scale = self.low_speed_steer_min_scale + (1.0 - self.low_speed_steer_min_scale) * frac
                steer = float(np.clip(steer * scale, -1.0, 1.0))

        if speed is not None:
            if speed < self.low_speed_brake_cut_speed:
                brake = brake * 0.3
            if speed < self.low_speed_throttle_floor_speed:
                throttle = max(throttle, self.low_speed_throttle_floor)

        if self.enable_steer_smoothing and (self.prev_control_for_smooth is not None):
            alpha = float(np.clip(self.steer_smooth_alpha, 0.0, 1.0))
            steer = alpha * steer + (1.0 - alpha) * float(self.prev_control_for_smooth.steer)
            steer = float(np.clip(steer, -1.0, 1.0))

        if self.enable_anti_stall and (self.ego is not None):
            if speed is None:
                v = self.ego.get_velocity()
                speed = float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
            if speed < self.stuck_speed_thresh:
                brake = 0.0
                throttle = max(throttle, self.min_throttle_when_stuck)
                steer = float(np.clip(steer * self.low_speed_steer_scale, -1.0, 1.0))

        self.prev_control_for_smooth = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        return throttle, steer, brake

    def update_corridor(self, world: carla.World, ego: carla.Actor):
        if self.world is None or self.ego is None or (self.ego.id != ego.id):
            self.world = world
            self.ego = ego
            self.map = world.get_map()
            self.target_wp = None
            self.episode_steps = 0
            self.prev_control_for_smooth = None

        self.episode_steps += 1
        obs = self._get_state_obs()
        if self.normalize_obs and self.obs_normalizer is not None:
            self.obs_normalizer.update(obs.reshape(1, -1))
            obs = self.obs_normalizer.normalize(obs)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        state = {"state": tf.expand_dims(tf.constant(obs, dtype=tf.float32), axis=0)}
        action, _, _, _, _ = self.agent.predict(state)
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        throttle, steer, brake = self._action_to_control(action)
        return throttle, steer, brake, {"raw_action": action}


SCENARIOS = ["cones", "jaywalker", "trimma", "construction"]
TARGET_EPISODES_PER_SCENE = 50
MAX_STEPS = 600

OUTPUT_CSV = "/home/ajifang/il_data_collect/il_data/success_rl_report.csv"

PPO_MODEL_DIR = "/home/ajifang/SAC_carla/weights/ppo-carla-obs30"


def make_scenario(name: str, world: carla.World, amap: carla.Map, tm_port: int):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.tm_port = tm_port
    cfg.enable_traffic_flow = True
    if name == "cones":
        return ConesScenario(world, amap, cfg)
    if name == "jaywalker":
        return JaywalkerScenario(world, amap, cfg)
    if name == "trimma":
        return TrimmaScenario(world, amap, cfg)
    if name == "construction":
        return ConstructionLaneChangeScenario(world, amap, cfg)
    raise ValueError(name)


def _snapshot_actor_ids(world: carla.World) -> set:
    try:
        return set([a.id for a in world.get_actors()])
    except Exception:
        return set()


def _cleanup_new_actors(world: carla.World, before_ids: set):
    try:
        for a in world.get_actors():
            if a.id in before_ids:
                continue
            if (
                a.type_id.startswith("vehicle.")
                or a.type_id.startswith("walker.")
                or a.type_id.startswith("sensor.")
                or a.type_id.startswith("static.prop")
            ):
                try:
                    a.destroy()
                except Exception:
                    pass
    except Exception:
        pass


def _load_progress(path: str) -> (Dict[str, int], Dict[str, int]):
    counts = {s: 0 for s in SCENARIOS}
    max_seed = {s: 0 for s in SCENARIOS}
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = row.get("scenario")
                if scene not in counts:
                    continue
                counts[scene] += 1
                try:
                    seed_val = int(row.get("seed", "0"))
                    if seed_val > max_seed[scene]:
                        max_seed[scene] = seed_val
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return counts, max_seed


class CollisionWatcher:
    def __init__(self, world: carla.World, ego: carla.Actor):
        self.world = world
        self.ego = ego
        self.has_collided = False
        self.sensor = None
        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent):
        self.has_collided = True

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


def run_one_episode(client: carla.Client, scenario_name: str, tm_port: int) -> Dict:
    world = client.get_world()
    amap = world.get_map()

    # 场景创建
    scenario = make_scenario(scenario_name, world, amap, tm_port)
    if not scenario.setup():
        scenario.cleanup()
        return {"valid": False}

    # 生成 ego
    ego_bp = get_ego_blueprint(world)
    spawn_tf = scenario.get_spawn_transform()
    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        spawn_tf.location.z += 0.5
        ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        scenario.cleanup()
        return {"valid": False}

    # 预热
    try:
        for _ in range(3):
            world.tick()
    except Exception:
        pass

    # spectator
    try:
        set_spectator_follow_ego(world, ego, mode="chase")
    except Exception:
        pass

    # rl planner (PPO)
    planner = RLPPOPlanner(PPO_MODEL_DIR)

    # eva
    eva = EvaMonitor()
    eva.attach(world, ego)

    # collision
    collision = CollisionWatcher(world, ego)

    # 统计 EVA
    saf_list: List[float] = []
    com_list: List[float] = []
    eff_list: List[float] = []

    for _ in range(MAX_STEPS):
        world.tick()

        # 场景特殊逻辑
        if isinstance(scenario, JaywalkerScenario):
            scenario.check_and_trigger(ego.get_location())
            scenario.tick_update()

        # --- 修改部分：适配 PPO planner ---
        try:
            ctrl_result = planner.update_corridor(world, ego)

            if ctrl_result and len(ctrl_result) >= 3:
                throttle, steer, brake = ctrl_result[0], ctrl_result[1], ctrl_result[2]
                ego.apply_control(carla.VehicleControl(throttle=float(throttle),
                                                       steer=float(steer),
                                                       brake=float(brake)))
        except Exception as e:
            print(f"Planner Error: {e}")
            break
        # ------------------------------------------

        data = eva.tick()
        eva.render(data)

        if not collision.has_collided:
            saf_list.append(float(data.get("r_saf", 0.0)))
            com_list.append(float(data.get("r_com", 0.0)))
            eff_list.append(float(data.get("r_eff", 0.0)))

        if collision.has_collided:
            break

    # 清理
    collision.destroy()
    if ego and ego.is_alive:
        ego.destroy()
    scenario.cleanup()

    if collision.has_collided:
        return {"valid": True, "collision": True, "r_saf": None, "r_com": None, "r_eff": None}

    def _avg(arr):
        return sum(arr) / max(1, len(arr))

    return {
        "valid": True,
        "collision": False,
        "r_saf": _avg(saf_list),
        "r_com": _avg(com_list),
        "r_eff": _avg(eff_list),
    }


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()

    # ✅ 确保输出目录存在
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(OUTPUT_CSV)
    file_nonempty = file_exists and os.path.getsize(OUTPUT_CSV) > 0
    resume_counts, resume_max_seed = _load_progress(OUTPUT_CSV) if file_nonempty else ({}, {})
    if file_nonempty:
        print(f"检测到已有报告，将从已有进度继续: {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "a" if file_nonempty else "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "seed", "trial", "collision",
                "r_saf", "r_com", "r_eff",
                "seed_bits"
            ],
        )
        if not file_nonempty:
            writer.writeheader()

        with carla_sync_mode(client, world, enabled=True, fixed_dt=0.05):
            for scene in SCENARIOS:
                success_count = resume_counts.get(scene, 0)
                attempt = resume_max_seed.get(scene, 0)
                if success_count >= TARGET_EPISODES_PER_SCENE:
                    print(
                        f"\n=== {scene} | 已有有效次数 {success_count}/{TARGET_EPISODES_PER_SCENE}，跳过 ==="
                    )
                    continue
                while success_count < TARGET_EPISODES_PER_SCENE:
                    attempt += 1
                    print(
                        f"\n=== {scene} | Attempt {attempt} | "
                        f"Success {success_count}/{TARGET_EPISODES_PER_SCENE} ==="
                    )

                    # 固定随机种子（用 attempt，失败会重试）
                    random.seed(attempt)
                    np.random.seed(attempt)

                    before_ids = _snapshot_actor_ids(world)
                    result = run_one_episode(client, scene, tm_port=8000)

                    if not result.get("valid", True):
                        _cleanup_new_actors(world, before_ids)
                        time.sleep(0.5)
                        continue

                    success_count += 1
                    collided = int(result["collision"])

                    writer.writerow({
                        "scenario": scene,
                        "seed": attempt,
                        "trial": success_count,
                        "collision": collided,
                        "r_saf": "" if result["r_saf"] is None else f"{result['r_saf']:.6f}",
                        "r_com": "" if result["r_com"] is None else f"{result['r_com']:.6f}",
                        "r_eff": "" if result["r_eff"] is None else f"{result['r_eff']:.6f}",
                        "seed_bits": "1" if collided else "0",
                    })
                    f.flush()

                    _cleanup_new_actors(world, before_ids)
                    time.sleep(0.5)

    print(f"\n✅ 成功写入报告: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
