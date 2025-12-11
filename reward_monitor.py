# reward_monitor.py

from dataclasses import dataclass, asdict
import math
import os
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

import carla


# ==========================
# 配置与数据结构
# ==========================

@dataclass
class RewardWeights:
    # 顶层权重
    w_safety: float = 1.0
    w_comfort: float = 0.2
    w_efficiency: float = 0.1

    # safety 内部
    w_collision: float = 5.0
    w_offroad: float = 3.0
    w_deviation: float = 1.0
    w_heading: float = 1.0
    w_obs_dist: float = 2.0
    w_speed_pen: float = 1.0

    # comfort 内部
    w_steer: float = 0.5
    w_jerk: float = 1.0
    w_brake: float = 0.5

    # efficiency 内部
    w_speed: float = 0.7
    w_progress: float = 0.3


@dataclass
class RewardThresholds:
    # 跟车 /距离相关
    time_headway_safe: float = 1.8  # 安全时距（秒）
    lane_dev_max: float = 1.5       # 允许的最大横向偏移 (m)
    heading_diff_max_deg: float = 30.0  # 允许的最大航向差 (deg)

    # 舒适阈值
    jerk_thr: float = 0.5       # m/s^3
    brake_thr: float = 3.0      # m/s^2，超过认为太暴力
    steer_rate_max: float = 0.5 # 每秒最大转向变化（归一化 steer 值）

    # 速度
    min_speed_ratio: float = 0.3  # 低于限速 30% 认为太慢


@dataclass
class RewardComponents:
    # safety
    collision: float = 0.0
    offroad: float = 0.0
    deviation: float = 0.0
    heading: float = 0.0
    obs_dist: float = 0.0
    speed_pen: float = 0.0

    # comfort
    steer_cost: float = 0.0
    jerk_cost: float = 0.0
    brake_cost: float = 0.0

    # efficiency
    speed_reward: float = 0.0
    progress: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RewardMonitorConfig:
    weights: RewardWeights = RewardWeights()
    thresholds: RewardThresholds = RewardThresholds()
    default_dt: float = 0.05  # 若 timestamp 里没有 delta_seconds，用这个
    enable_vis: bool = True
    vis_update_interval: int = 2  # 每多少 step 更新一次 plot
    save_dir: str = "./reward_logs"


# ==========================
# 主类：RewardMonitor
# ==========================

class RewardMonitor:
    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        route_waypoints: Optional[List[carla.Waypoint]] = None,
        config: RewardMonitorConfig = RewardMonitorConfig(),
    ):
        self.world = world
        self.map = world.get_map()
        self.ego = ego_vehicle
        self.route = route_waypoints or []
        self.cfg = config

        # 历史量，用来计算 jerk / progress / steer rate
        self.prev_location: Optional[carla.Location] = None
        self.prev_velocity: Optional[carla.Vector3D] = None
        self.prev_acc: Optional[carla.Vector3D] = None
        self.prev_control: Optional[carla.VehicleControl] = None
        self.prev_s: float = 0.0  # 近似沿路程
        self.step_count: int = 0
        self.episode_id: int = 0

        # 记录时间序列
        self.history_total: List[float] = []
        self.history_components: Dict[str, List[float]] = {
            k: [] for k in RewardComponents().__dict__.keys()
        }
        self.history_planner_id: List[int] = []
        # 顶层三大项的历史（方便单独画曲线）
        self.history_safety: List[float] = []
        self.history_comfort: List[float] = []
        self.history_efficiency: List[float] = []

        # 可视化相关
        self.fig = None
        self.ax_safety = None
        self.ax_comfort = None
        self.ax_eff = None
        self.ax_total = None
        self.bars_safety = None
        self.bars_comfort = None
        self.bars_eff = None
        self.line_total = None
        self.line_safe = None
        self.line_comf = None
        self.line_eff = None

        if self.cfg.enable_vis:
            self._init_fig()

        os.makedirs(self.cfg.save_dir, exist_ok=True)

    # --------------------------
    # 可视化初始化
    # --------------------------
    def _init_fig(self):
        """初始化一个简洁的 2x3 仪表盘: 上面一行曲线，下面三块柱状图."""
        if not self.cfg.enable_vis:
            self.fig = None
            return

        plt.ion()

        # 全局样式：白底、细线、浅灰网格
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "axes.edgecolor":   "#d4d4d4",
            "axes.labelcolor":  "#111827",
            "axes.titlesize":   11,
            "axes.titleweight": "semibold",
            "xtick.color":      "#6b7280",
            "ytick.color":      "#6b7280",
            "grid.color":       "#e5e7eb",
            "grid.linestyle":   "-",
            "grid.linewidth":   0.6,
        })

        # 总布局：2 行 3 列
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(
            2, 3,
            height_ratios=[2.0, 1.6],
            wspace=0.45,
            hspace=0.45,
        )

        # 上面整行：Total + 三大项曲线
        self.ax_total = self.fig.add_subplot(gs[0, :])

        # 下面三块：Safety / Comfort / Efficiency 的子项柱状图
        self.ax_safety = self.fig.add_subplot(gs[1, 0])
        self.ax_comfort = self.fig.add_subplot(gs[1, 1])
        self.ax_eff = self.fig.add_subplot(gs[1, 2])

        # -------- 上方曲线区：Total + 三大项 --------
        self.ax_total.set_title("Reward Decomposition over Time")
        self.ax_total.set_xlabel("Step")
        self.ax_total.set_ylabel("Reward")
        self.ax_total.grid(True, alpha=0.6)

        # 四条线：Total / Safety / Comfort / Efficiency
        self.line_total, = self.ax_total.plot([], [], label="Total",      linewidth=1.6, color="#2563eb")
        self.line_safe,  = self.ax_total.plot([], [], label="Safety",     linewidth=1.2, color="#ef4444")
        self.line_comf,  = self.ax_total.plot([], [], label="Comfort",    linewidth=1.2, color="#f97316")
        self.line_eff,   = self.ax_total.plot([], [], label="Efficiency", linewidth=1.2, color="#16a34a")

        self.ax_total.legend(loc="upper right", frameon=False, fontsize=8)

        # -------- 下方三个柱状卡片 --------

        # 1) Safety 子项
        safety_labels = ["coll", "off", "dev", "head", "obs", "spd"]
        self.ax_safety.set_title("Safety components", pad=6)
        self.ax_safety.grid(True, axis="y", alpha=0.5)
        self.ax_safety.axhline(0.0, color="#9ca3af", linewidth=0.8)
        self.bars_safety = self.ax_safety.bar(
            range(len(safety_labels)),
            [0.0] * len(safety_labels),
            width=0.6,
            color="#f97373",
            alpha=0.85,
        )
        self.ax_safety.set_xticks(range(len(safety_labels)))
        self.ax_safety.set_xticklabels(safety_labels, fontsize=8)
        for spine in ["top", "right"]:
            self.ax_safety.spines[spine].set_visible(False)

        # 2) Comfort 子项
        comfort_labels = ["steer", "jerk", "brake"]
        self.ax_comfort.set_title("Comfort components", pad=6)
        self.ax_comfort.grid(True, axis="y", alpha=0.5)
        self.ax_comfort.axhline(0.0, color="#9ca3af", linewidth=0.8)
        self.bars_comfort = self.ax_comfort.bar(
            range(len(comfort_labels)),
            [0.0] * len(comfort_labels),
            width=0.6,
            color="#a5b4fc",
            alpha=0.9,
        )
        self.ax_comfort.set_xticks(range(len(comfort_labels)))
        self.ax_comfort.set_xticklabels(comfort_labels, fontsize=8)
        for spine in ["top", "right"]:
            self.ax_comfort.spines[spine].set_visible(False)

        # 3) Efficiency 子项
        eff_labels = ["speed", "prog"]
        self.ax_eff.set_title("Efficiency components", pad=6)
        self.ax_eff.grid(True, axis="y", alpha=0.5)
        self.ax_eff.axhline(0.0, color="#9ca3af", linewidth=0.8)
        self.bars_eff = self.ax_eff.bar(
            range(len(eff_labels)),
            [0.0] * len(eff_labels),
            width=0.6,
            color="#34d399",
            alpha=0.9,
        )
        self.ax_eff.set_xticks(range(len(eff_labels)))
        self.ax_eff.set_xticklabels(eff_labels, fontsize=8)
        for spine in ["top", "right"]:
            self.ax_eff.spines[spine].set_visible(False)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # --------------------------
    # 每个 episode 开始时重置
    # --------------------------
    def reset(self, episode_id: int = 0):
        self.episode_id = episode_id
        self.prev_location = None
        self.prev_velocity = None
        self.prev_acc = None
        self.prev_control = None
        self.prev_s = 0.0
        self.step_count = 0

        # 清空历史记录
        self.history_total.clear()
        for k in self.history_components:
            self.history_components[k].clear()
        self.history_planner_id.clear()

        self.history_safety.clear()
        self.history_comfort.clear()
        self.history_efficiency.clear()

        # 不再 cla()，只重置线和柱子的数值
        if self.cfg.enable_vis and self.fig is not None and self.line_total is not None:
            # 清空四条曲线
            self.line_total.set_data([], [])
            self.line_safe.set_data([], [])
            self.line_comf.set_data([], [])
            self.line_eff.set_data([], [])

            # 柱状图清零
            for bars in (self.bars_safety, self.bars_comfort, self.bars_eff):
                if bars is not None:
                    for b in bars:
                        b.set_height(0.0)

            # 给一个默认的轴范围，避免空列表时报错
            self.ax_total.set_xlim(0, 5)
            self.ax_total.set_ylim(-1.0, 1.0)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    # --------------------------
    # 工具函数
    # --------------------------
    @staticmethod
    def _vec3d_to_np(v: carla.Vector3D) -> np.ndarray:
        return np.array([v.x, v.y, v.z], dtype=np.float32)

    @staticmethod
    def _location_to_np(loc: carla.Location) -> np.ndarray:
        return np.array([loc.x, loc.y, loc.z], dtype=np.float32)

    # 最近前车（简化：同方向且在前方）
    def _get_lead_vehicle(self, ego_wp: carla.Waypoint) -> Optional[carla.Actor]:
        actors = self.world.get_actors().filter('vehicle.*')
        ego_loc = ego_wp.transform.location
        ego_forward = ego_wp.transform.get_forward_vector()
        ego_forward_np = self._vec3d_to_np(ego_forward)[:2]
        ego_forward_np /= (np.linalg.norm(ego_forward_np) + 1e-6)

        min_dist = 1e9
        lead_vehicle = None
        for v in actors:
            if v.id == self.ego.id:
                continue
            loc = v.get_location()
            vec = self._location_to_np(loc - ego_loc)[:2]
            proj = np.dot(vec, ego_forward_np)
            if proj <= 0:
                continue  # 不在前方
            lateral = np.linalg.norm(vec - proj * ego_forward_np)
            if lateral > 4.0:  # 侧向太远，认为不是本车道
                continue
            dist = np.linalg.norm(vec)
            if dist < min_dist:
                min_dist = dist
                lead_vehicle = v
        return lead_vehicle

    # --------------------------
    # 计算所有 reward 子项
    # --------------------------
    def _compute_components(
        self,
        control: carla.VehicleControl,
        dt: float,
        collision_flag: bool = False,
    ) -> RewardComponents:
        cfg_w = self.cfg.weights
        cfg_t = self.cfg.thresholds

        transform = self.ego.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = self.ego.get_velocity()
        acc = self.ego.get_acceleration()

        # 速度 (m/s)
        v_vec = self._vec3d_to_np(velocity)
        speed = float(np.linalg.norm(v_vec))  # m/s

        # ego 的 waypoint
        ego_wp = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk | carla.LaneType.Shoulder
        )

        comps = RewardComponents()

        # ---------- Safety 1: Collision ----------
        if collision_flag:
            comps.collision = -1.0
        else:
            comps.collision = 0.0

        # ---------- Safety 2: Offroad ----------
        # 简化：如果 lane_type 不是 Driving/Shoulder，则认为 offroad
        if ego_wp.lane_type != carla.LaneType.Driving and ego_wp.lane_type != carla.LaneType.Shoulder:
            comps.offroad = -1.0
        else:
            comps.offroad = 0.0

        # ---------- Safety 3: Lane Deviation ----------
        lane_center = ego_wp.transform.location
        d_lat = location.distance(lane_center)  # 这里是3D距离，横向为主，近似当作横向偏移
        d_lat_norm = min(d_lat / cfg_t.lane_dev_max, 1.0)
        comps.deviation = - (d_lat_norm ** 2)

        # ---------- Safety 4: Heading Alignment ----------
        lane_yaw = ego_wp.transform.rotation.yaw  # degrees
        ego_yaw = rotation.yaw
        d_yaw = abs((ego_yaw - lane_yaw + 180.0) % 360.0 - 180.0)  # wrap 到 [-180,180]
        d_yaw_norm = min(d_yaw / cfg_t.heading_diff_max_deg, 1.0)
        comps.heading = - (d_yaw_norm ** 2)

        # ---------- Safety 5: Close-Obstacle Distance ----------
        lead = self._get_lead_vehicle(ego_wp)
        if lead is not None:
            lead_loc = lead.get_location()
            d_min = location.distance(lead_loc)
            # 安全距离 ~ v * T
            d_safe = max(speed * cfg_t.time_headway_safe, 0.1)
            if d_min < d_safe:
                comps.obs_dist = - (1.0 - d_min / d_safe)
            else:
                comps.obs_dist = 0.0
        else:
            comps.obs_dist = 0.0

        # ---------- Safety 6: Speed Penalty (超速/过慢) ----------
        # 获取限速（CARLA 每个 lane_segment 可以设置 speed limit）
        speed_limit = ego_wp.lane_width  # NOTE: 这里只是占位，你可以改为 ego_wp.get_speed_limit()
        if hasattr(ego_wp, "get_speed_limit"):
            speed_limit = ego_wp.get_speed_limit() / 3.6  # 转成 m/s
        else:
            speed_limit = 15.0  # 默认 54 km/h

        # 过慢惩罚
        v_min = cfg_t.min_speed_ratio * speed_limit
        slow_pen = 0.0
        if speed < v_min:
            slow_pen = ((v_min - speed) / max(v_min, 1e-3)) ** 2

        # 超速惩罚
        over_pen = 0.0
        if speed > speed_limit:
            over_pen = ((speed - speed_limit) / speed_limit) ** 2

        comps.speed_pen = - (slow_pen + over_pen)

        # ---------- Comfort 1: Steering Cost ----------
        if self.prev_control is not None:
            steer_rate = abs(control.steer - self.prev_control.steer) / max(dt, 1e-3)
            steer_norm = min(steer_rate / cfg_t.steer_rate_max, 1.0)
            comps.steer_cost = - (steer_norm ** 2)
        else:
            comps.steer_cost = 0.0

        # ---------- Comfort 2: Jerk ----------
        if self.prev_acc is not None:
            a_vec = self._vec3d_to_np(acc)
            prev_a_vec = self._vec3d_to_np(self.prev_acc)
            jerk_vec = (a_vec - prev_a_vec) / max(dt, 1e-3)
            jerk_mag = float(np.linalg.norm(jerk_vec))
            jerk_norm = min(jerk_mag / cfg_t.jerk_thr, 1.0)
            comps.jerk_cost = - (jerk_norm ** 2)
        else:
            comps.jerk_cost = 0.0

        # ---------- Comfort 3: Brake Cost ----------
        # 这里用纵向减速度近似制动强度
        if self.prev_velocity is not None:
            v_prev = self._vec3d_to_np(self.prev_velocity)
            dv = (v_vec - v_prev) / max(dt, 1e-3)
            a_long = float(np.dot(
                dv[:2],
                self._vec3d_to_np(ego_wp.transform.get_forward_vector())[:2]
            ))
            # a_long < 0 表示减速
            brake_pen = 0.0
            if a_long < 0:
                decel = abs(a_long)
                if decel > cfg_t.brake_thr:
                    brake_pen = ((decel - cfg_t.brake_thr) / cfg_t.brake_thr) ** 2
            comps.brake_cost = - brake_pen
        else:
            comps.brake_cost = 0.0

        # ---------- Efficiency 1: Speed Reward ----------
        # 理想速度接近 speed_limit
        v_des = speed_limit * 0.9
        sigma_v = 0.3 * speed_limit
        if sigma_v < 1e-3:
            comps.speed_reward = 0.0
        else:
            comps.speed_reward = math.exp(- ((speed - v_des) ** 2) / (2 * sigma_v ** 2))

        # ---------- Efficiency 2: Progress ----------
        if self.prev_location is not None:
            # 直接用位移近似 progress
            prog = location.distance(self.prev_location) / max(dt, 1e-3)  # m/s
        else:
            prog = 0.0
        # 归一化，大概除以一个期望速度
        comps.progress = prog / max(speed_limit, 1e-3)
        comps.progress = float(max(min(comps.progress, 1.0), -1.0))

        # 更新历史状态
        self.prev_location = location
        self.prev_velocity = velocity
        self.prev_acc = acc
        self.prev_control = control

        return comps

    # --------------------------
    # 将子项组合成总 reward
    # --------------------------
    def _combine_components(self, comps: RewardComponents):
        w = self.cfg.weights

        # safety
        r_safety = (
            w.w_collision * comps.collision +
            w.w_offroad * comps.offroad +
            w.w_deviation * comps.deviation +
            w.w_heading * comps.heading +
            w.w_obs_dist * comps.obs_dist +
            w.w_speed_pen * comps.speed_pen
        )

        # comfort
        r_comfort = (
            w.w_steer * comps.steer_cost +
            w.w_jerk * comps.jerk_cost +
            w.w_brake * comps.brake_cost
        )

        # efficiency
        r_eff = (
            w.w_speed * comps.speed_reward +
            w.w_progress * comps.progress
        )

        total = (
            w.w_safety * r_safety +
            w.w_comfort * r_comfort +
            w.w_efficiency * r_eff
        )
        return float(total), r_safety, r_comfort, r_eff

    # --------------------------
    # 更新可视化
    # --------------------------
    def _update_vis(
        self,
        comps: RewardComponents,
        total_reward: float,
        r_safety: float,
        r_comfort: float,
        r_eff: float,
    ):
        if not self.cfg.enable_vis or self.fig is None:
            return

        # -------- 上方曲线：Total / Safety / Comfort / Efficiency --------
        steps = np.arange(len(self.history_total), dtype=np.int32)
        y_total = np.array(self.history_total, dtype=np.float32)
        y_safe = np.array(self.history_safety, dtype=np.float32)
        y_comf = np.array(self.history_comfort, dtype=np.float32)
        y_eff = np.array(self.history_efficiency, dtype=np.float32)

        self.line_total.set_data(steps, y_total)
        self.line_safe.set_data(steps, y_safe)
        self.line_comf.set_data(steps, y_comf)
        self.line_eff.set_data(steps, y_eff)

        if len(y_total) > 1:
            ymin = float(min(y_total.min(), y_safe.min(), y_comf.min(), y_eff.min()))
            ymax = float(max(y_total.max(), y_safe.max(), y_comf.max(), y_eff.max()))
        else:
            ymin, ymax = -1.0, 1.0

        # 🔧 防止NaN/Inf导致matplotlib报错
        if np.isnan(ymin) or np.isinf(ymin):
            ymin = -1.0
        if np.isnan(ymax) or np.isinf(ymax):
            ymax = 1.0

        pad = 0.15 * max(1.0, ymax - ymin)
        self.ax_total.set_xlim(0, max(5, len(y_total)))
        self.ax_total.set_ylim(ymin - pad, ymax + pad)

        # -------- 下方三块：每类子项柱状图 --------
        # Safety
        safety_vals = [
            comps.collision,
            comps.offroad,
            comps.deviation,
            comps.heading,
            comps.obs_dist,
            comps.speed_pen,
        ]
        for bar, val in zip(self.bars_safety, safety_vals):
            bar.set_height(val)
        s_min = min(safety_vals + [-1.0])
        s_max = max(safety_vals + [0.0])
        # 🔧 防止NaN/Inf
        if np.isnan(s_min) or np.isinf(s_min):
            s_min = -1.0
        if np.isnan(s_max) or np.isinf(s_max):
            s_max = 0.0
        self.ax_safety.set_ylim(s_min - 0.1, s_max + 0.1)

        # Comfort
        comfort_vals = [
            comps.steer_cost,
            comps.jerk_cost,
            comps.brake_cost,
        ]
        for bar, val in zip(self.bars_comfort, comfort_vals):
            bar.set_height(val)
        c_min = min(comfort_vals + [-1.0])
        c_max = max(comfort_vals + [0.0])
        # 🔧 防止NaN/Inf
        if np.isnan(c_min) or np.isinf(c_min):
            c_min = -1.0
        if np.isnan(c_max) or np.isinf(c_max):
            c_max = 0.0
        self.ax_comfort.set_ylim(c_min - 0.1, c_max + 0.1)

        # Efficiency
        eff_vals = [
            comps.speed_reward,
            comps.progress,
        ]
        for bar, val in zip(self.bars_eff, eff_vals):
            bar.set_height(val)
        e_min = min(eff_vals + [-1.0])
        e_max = max(eff_vals + [1.0])
        # 🔧 防止NaN/Inf
        if np.isnan(e_min) or np.isinf(e_min):
            e_min = -1.0
        if np.isnan(e_max) or np.isinf(e_max):
            e_max = 1.0
        self.ax_eff.set_ylim(e_min - 0.1, e_max + 0.1)

        # -------- 刷新画布 --------
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # --------------------------
    # 对外主接口：一步更新 + 返回 reward
    # --------------------------
    def update(
        self,
        control: carla.VehicleControl,
        planner_id: Optional[int] = None,
        collision_flag: bool = False,
        done: bool = False,
    ):
        snapshot = self.world.get_snapshot()
        dt = getattr(snapshot.timestamp, "delta_seconds", self.cfg.default_dt)

        comps = self._compute_components(control, dt, collision_flag=collision_flag)
        total_reward, r_safety, r_comfort, r_eff = self._combine_components(comps)

        # ====== 记录历史 ======
        self.history_total.append(total_reward)
        self.history_safety.append(r_safety)
        self.history_comfort.append(r_comfort)
        self.history_efficiency.append(r_eff)

        for k, v in comps.to_dict().items():
            self.history_components[k].append(float(v))
        self.history_planner_id.append(-1 if planner_id is None else int(planner_id))
        self.step_count += 1

        # 可视化（低频更新）
        if self.cfg.enable_vis and (self.step_count % self.cfg.vis_update_interval == 0):
            self._update_vis(comps, total_reward, r_safety, r_comfort, r_eff)

        # episode 结束时保存图
        if done:
            self.save_curves()

        return total_reward, comps

    # --------------------------
    # 保存曲线图与数据
    # --------------------------
    def save_curves(self):
        ep_dir = os.path.join(self.cfg.save_dir, f"ep_{self.episode_id:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # 保存 total reward 曲线
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.history_total, linewidth=1.0)
        ax.set_title("Total Reward over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        fig.tight_layout()
        fig.savefig(os.path.join(ep_dir, "total_reward.png"))
        plt.close(fig)

        # 保存所有子项的时间序列
        np.save(os.path.join(ep_dir, "total_reward.npy"), np.array(self.history_total))

        # 三大项
        np.save(os.path.join(ep_dir, "safety.npy"), np.array(self.history_safety))
        np.save(os.path.join(ep_dir, "comfort.npy"), np.array(self.history_comfort))
        np.save(os.path.join(ep_dir, "efficiency.npy"), np.array(self.history_efficiency))

        for k, v in self.history_components.items():
            np.save(os.path.join(ep_dir, f"{k}.npy"), np.array(v))
        np.save(os.path.join(ep_dir, "planner_ids.npy"), np.array(self.history_planner_id))
