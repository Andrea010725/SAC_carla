# carla_base/carla_env.py
import os
import sys
import math
import random
import time
from typing import Optional, Tuple, List
import xml.etree.ElementTree as ET

import gym
import numpy as np
from gym import spaces

from .carla_sync_mode import CarlaSyncMode
from .carla_weather import Weather
from .scenario_manager import ScenarioFactory, ScenarioBase

CARLA_ROOT = "/home/ajifang/carla"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.insert(
    0,
    os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-0.9.15-py3.7-linux-x86_64.egg"),
)

import carla  # noqa: E402
import pygame  # noqa: E402

# matplotlib for reward visualization (使用 Agg 后端，避免与 pygame 冲突)
try:
    import matplotlib
    matplotlib.use('Agg')  # ✅ 使用 Agg 后端（不创建窗口，只保存图片）
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# 尝试导入 RewardMonitor，如果失败则使用占位类
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reward_monitor import RewardMonitor, RewardMonitorConfig
except ImportError:
    RewardMonitor = None
    RewardMonitorConfig = None


class CarlaEnv(gym.Env):
    """
    ✅ 3维动作环境（写死）：
      action = [throttle_brake, steer, y_ref] in [-1, 1]^3
        - throttle_brake: >=0 -> throttle, <0 -> brake
        - steer: 原始方向盘
        - y_ref: “参考/偏置”信号（可选参与控制）

    观测：9维 state
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, carla_port: int, tm_port: int):
        super().__init__()
        self.config = config
        self.carla_port = int(carla_port)
        self.tm_port = int(tm_port)

        # ==== 训练/观测参数 ====
        # ==== 训练/观测参数 ====
        self.max_episode_steps = int(getattr(config, "max_episode_steps", 2000))

        # ===== 观测类型：支持扩展 =====
        # 可选：
        #   "state"                   -> 9
        #   "state_lane"              -> 9 + 6
        #   "state_lane_obstacles"    -> 9 + 6 + 3*K
        self.observations_type = str(getattr(config, "observations_type", "state_lane_obstacles")).lower()

        # obstacles 配置
        self.obs_obstacle_k = int(getattr(config, "obs_obstacle_k", 5))
        self.obs_obstacle_range = float(getattr(config, "obs_obstacle_range", 50.0))  # meters

        # lane 配置
        self.obs_use_lane = bool(getattr(config, "obs_use_lane", True))
        self.obs_use_obstacles = bool(getattr(config, "obs_use_obstacles", True))

        # 根据 observations_type 强制开关
        if self.observations_type == "state":
            self.obs_use_lane = False
            self.obs_use_obstacles = False
        elif self.observations_type in ["state_lane", "lane"]:
            self.obs_use_lane = True
            self.obs_use_obstacles = False
        elif self.observations_type in ["state_lane_obstacles", "lane_obstacles", "full"]:
            self.obs_use_lane = True
            self.obs_use_obstacles = True
        else:
            raise ValueError(f"[CarlaEnv] Unknown observations_type={self.observations_type}")

        self.base_state_dim = 9
        self.lane_dim = 6 if self.obs_use_lane else 0
        self.obstacle_dim = (self.obs_obstacle_k * 3) if self.obs_use_obstacles else 0

        # ✅ 最终 obs_dim：自动计算，别再写死 9
        self.obs_dim = self.base_state_dim + self.lane_dim + self.obstacle_dim

        # ✅ 3维动作保持不变
        self.action_dim = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # ✅ 可选：未来如果你希望场景生成时“注册障碍物”，这里先留个列表（现在不用也行）
        self.obstacle_actors: List[carla.Actor] = []

        # 渲染/同步
        self.render_display = bool(getattr(config, "render", True))
        self.fixed_dt = float(getattr(config, "fixed_dt", 0.05))  # 20 FPS
        self.map_name = str(getattr(config, "map_name", "Town05"))

        # 车辆与交通
        self.vehicle_name = str(getattr(config, "vehicle_name", "tesla.cybertruck"))
        self.traffic = bool(getattr(config, "traffic", False))

        # HUD 显示的 Planner 名
        self.planner_mode = str(getattr(config, "planner_mode", "RL")).upper()

        # =========================
        # 动作映射：防站桩参数
        # =========================
        self.brake_deadzone = float(getattr(config, "brake_deadzone", 0.05))
        self.stuck_speed_thresh = float(getattr(config, "stuck_speed_thresh", 0.30))  # m/s
        self.min_throttle_when_stuck = float(getattr(config, "min_throttle_when_stuck", 0.20))
        self.low_speed_steer_scale = float(getattr(config, "low_speed_steer_scale", 0.60))
        self.enable_anti_stall = bool(getattr(config, "enable_anti_stall", True))

        # =========================
        # ✅ y_ref 是否参与控制（建议显式开关）
        # =========================
        self.use_yref_in_steer = bool(getattr(config, "use_yref_in_steer", True))
        # 原来 0.30 太猛，训练早期很容易推到边界；先用 0.10 更稳
        self.yref_steer_gain = float(getattr(config, "yref_steer_gain", 0.03))

        # =========================
        # Reward 参数
        # =========================
        self.use_forward_progress = bool(getattr(config, "use_forward_progress", True))
        self.k_progress = float(getattr(config, "k_progress", 1.2))
        self.progress_clip = float(getattr(config, "progress_clip", 1.0))

        self.speed_gate_by_lane = bool(getattr(config, "speed_gate_by_lane", True))
        self.speed_gate_width = float(getattr(config, "speed_gate_width", 2.2))  # 建议 >=2.0

        self.smooth_only_above_speed = float(getattr(config, "smooth_only_above_speed", 1.0))
        self.smooth_clip_min = float(getattr(config, "smooth_clip_min", -0.08))
        self.k_smooth = float(getattr(config, "k_smooth", 0.03))

        self.no_progress_speed_thresh = float(getattr(config, "no_progress_speed_thresh", 0.25))
        self.no_progress_fwd_thresh = float(getattr(config, "no_progress_fwd_thresh", 0.01))
        self.k_idle = float(getattr(config, "k_idle", 0.003))
        self.no_progress_limit = int(getattr(config, "no_progress_limit", 250))

        # ✅ offroad 自适应 margin（固定3.0太容易误杀）
        self.offroad_margin = float(getattr(config, "offroad_margin", 1.0))

        # ==== CARLA 连接 ====
        self.client: carla.Client = carla.Client("127.0.0.1", self.carla_port)
        self.client.set_timeout(180.0)

        self.world: carla.World = self.client.get_world()
        self._original_settings = self.world.get_settings()

        current_map_name = self.world.get_map().name
        if self.map_name not in current_map_name:
            print(f"[CarlaEnv] 当前地图 {current_map_name} != 期望地图 {self.map_name}，正在加载...")
            self.world = self.client.load_world(self.map_name)
        self.map: carla.Map = self.world.get_map()

        # 同步设置（强制）
        self._apply_sync_settings(self.fixed_dt)

        # Traffic Manager - 带重试
        self.tm = None
        tm_success = False
        tm_ports_to_try = [self.tm_port, self.tm_port + 1, self.tm_port + 500, 8500, 8501, 8502]
        for try_port in tm_ports_to_try:
            try:
                self.tm = self.client.get_trafficmanager(try_port)
                self.tm.set_synchronous_mode(True)
                self.tm_port = try_port
                tm_success = True
                break
            except RuntimeError as e:
                if "bind error" in str(e):
                    continue
                raise e
        if not tm_success:
            raise RuntimeError(f"无法连接Traffic Manager，尝试的端口: {tm_ports_to_try}")

        # 运行时句柄
        self.ego: Optional[carla.Vehicle] = None
        self.camera_display: Optional[carla.Sensor] = None
        self.collision_sensor: Optional[carla.Sensor] = None
        self.sync_mode: Optional[CarlaSyncMode] = None
        self.weather: Optional[Weather] = Weather(self.world, float(getattr(config, "changing_weather_speed", 0.0)))

        # 场景相关
        self.scenario = str(getattr(config, "scenario", "plain")).lower()
        self.scenario_instance: Optional[ScenarioBase] = None  # 场景实例
        self._first_cone_tf: Optional[carla.Transform] = None
        self._last_cone_tf: Optional[carla.Transform] = None
        self.initial_spawn_tf = getattr(config, "initial_spawn_tf", None)
        self.spectator_mode = str(getattr(config, "spectator_mode", "none")).lower()

        # Pygame
        self.screen = None
        self.font_small = None
        self.font_big = None
        self.clock = None
        if self.render_display:
            # ✅ 尝试使用真实显示，如果失败则回退到虚拟模式
            try:
                # 首先尝试使用 x11 显示（真实窗口）
                if 'SDL_VIDEODRIVER' in os.environ:
                    del os.environ['SDL_VIDEODRIVER']

                # 先初始化 pygame
                pygame.init()

                # 尝试创建窗口（使用更安全的标志）
                self.screen = pygame.display.set_mode((800, 600), pygame.SWSURFACE)
                self.font_big = get_font(size=24)
                self.font_small = get_font(size=14)
                self.clock = pygame.time.Clock()
                print("[CarlaEnv] ✅ Pygame显示已启用（真实窗口模式）")
            except Exception as e:
                # 如果真实显示失败，回退到虚拟模式
                print(f"[CarlaEnv] ⚠️ 无法创建真实窗口: {e}")
                print("[CarlaEnv] 回退到虚拟显示模式...")

                # 清理之前的 pygame 初始化
                try:
                    pygame.quit()
                except:
                    pass

                # 使用虚拟显示驱动
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
                self.screen = pygame.display.set_mode((400, 300), pygame.SWSURFACE)
                self.font_big = get_font(size=24)
                self.font_small = get_font(size=14)
                self.clock = pygame.time.Clock()
                print("[CarlaEnv] ✅ Pygame显示已启用（虚拟模式）")

        # Matplotlib for reward visualization
        self.reward_fig = None
        self.reward_ax = None
        self.reward_bars = None
        self.reward_history = []  # 存储历史数据
        self.reward_update_interval = 10  # 每10步更新一次
        self.reward_save_dir = "./reward_plots"  # 保存图片的目录
        if self.render_display and MATPLOTLIB_AVAILABLE:
            # 创建保存目录
            os.makedirs(self.reward_save_dir, exist_ok=True)
            self._init_reward_plot()

        # XML/地图来源
        self.xml_file = getattr(config, "xml_file", "/home/ajifang/czw/RL_selector/env/waypoints.xml")
        self.xml_dir = getattr(config, "xml_dir", None)
        self.randomize_town = bool(getattr(config, "randomize_town", False))
        self.town_pool = list(getattr(config, "town_pool", ["Town01", "Town03", "Town05"]))

        # cones 参数
        self.cone_num = int(getattr(config, "cone_num", 10))
        self.cone_step_behind = float(getattr(config, "cone_step_behind", 3.0))
        self.cone_step_lateral = float(getattr(config, "cone_step_lateral", 0.35))
        self.cone_z_offset = float(getattr(config, "cone_z_offset", 0.0))
        self.cone_min_gap_from_junction = float(getattr(config, "cone_min_gap_from_junction", 15.0))
        self.cone_grid = float(getattr(config, "cone_grid", 5.0))
        self.cone_lane_margin = float(getattr(config, "cone_lane_margin", 0.25))

        # spawn 控制
        self.spawn_wp_step = float(getattr(config, "spawn_wp_step", 2.0))
        self.spawn_min_gap_from_cone = float(getattr(config, "spawn_min_gap_from_cone", 20.0))

        # parked_obstacles 参数
        self.num_parked_cars = int(getattr(config, "num_parked_cars", 4))
        self.parked_car_spacing = float(getattr(config, "parked_car_spacing", 50.0))
        self.parked_car_offset = float(getattr(config, "parked_car_offset", 1.8))
        self.parked_car_start_distance = float(getattr(config, "parked_car_start_distance", 30.0))

        # 统计 & 管理
        self.episode_steps = 0
        self.collision = False
        self._actors: List[carla.Actor] = []

        # Reward monitor
        self.reward_monitor: Optional[RewardMonitor] = None
        self.episode_id: int = 0
        self.last_control: Optional[carla.VehicleControl] = None

        # ✅ 可视化开关
        self.enable_debug_drawing = bool(getattr(config, "enable_debug_drawing", True))
        self.debug_draw_interval = int(getattr(config, "debug_draw_interval", 5))  # 每5步绘制一次

        # ✅ 可视化细节开关
        self.draw_detection_range = bool(getattr(config, "draw_detection_range", True))  # 青色圆圈
        self.draw_ego_direction = bool(getattr(config, "draw_ego_direction", True))      # 绿色箭头
        self.draw_obstacle_boxes = bool(getattr(config, "draw_obstacle_boxes", True))    # 障碍物边界框
        self.draw_lane_center = bool(getattr(config, "draw_lane_center", True))          # 白色车道线

        # ===== reward 内部状态 =====
        self.target_wp = None
        self.prev_wp_dist = None
        self.wp_step_dist = float(getattr(config, "wp_step_dist", 5.0))
        self.wp_reach_thresh = float(getattr(config, "wp_reach_thresh", 2.0))
        self.no_progress_steps = 0
        self.prev_control_for_smooth = None
        self.prev_loc: Optional[carla.Location] = None  # forward progress

        self.progress_ema = 0.0
        self.last_obs = None  # 用于 reward 读取障碍信息（在 step 里更新）

    # ----------------- 同步设置 & 切图 -----------------
    def _apply_sync_settings(self, fixed_dt: float):
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = fixed_dt
        # ✅ 训练阶段关闭渲染：降低内存占用，减少CARLA断连
        s.no_rendering_mode = (not self.render_display)
        self.world.apply_settings(s)

    def _load_map_if_needed(self, town: str):
        town = str(town).strip()
        cur = self.world.get_map().name if self.world and self.world.get_map() else ""
        if town and town != cur:
            print(f"[CarlaEnv] Loading map: {town} (was {cur})")
            self._cleanup_actors()
            if self.sync_mode is not None:
                try:
                    if hasattr(self.sync_mode, "_settings") and self.sync_mode._settings is not None:
                        self.world.apply_settings(self.sync_mode._settings)
                except Exception:
                    pass
                self.sync_mode = None

            self.world = self.client.load_world(town)
            self.map = self.world.get_map()
            self._apply_sync_settings(self.fixed_dt)

            self.tm = self.client.get_trafficmanager(self.tm_port)
            self.tm.set_synchronous_mode(True)
            self.weather = Weather(self.world, float(getattr(self.config, "changing_weather_speed", 0.0)))

    def _get_required_town_for_scenario(self, scenario_name: str) -> Optional[str]:
        """
        获取场景需要的地图名称（从 XML 配置中获取）

        Args:
            scenario_name: 场景名称

        Returns:
            str: 地图名称，如果场景不需要特定地图则返回 None
        """
        try:
            from carla_base.scenario_xml_parser import get_xml_parser, SCENARIO_TYPE_MAPPING

            # 获取场景对应的 ScenarioRunner 类型
            scenario_type = SCENARIO_TYPE_MAPPING.get(scenario_name)
            if not scenario_type:
                return None

            # 获取 XML 解析器
            parser = get_xml_parser()

            # 获取该场景类型的所有配置
            scenarios = parser.get_scenarios_for_type(scenario_type, town=None)

            if scenarios:
                # 随机选择一个场景配置，获取其 town
                selected = random.choice(scenarios)
                town = selected.get('town')
                if town:
                    print(f"[CarlaEnv] 从 XML 获取场景 {scenario_name} 的地图: {town}")
                    return town
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ 获取场景地图失败: {e}")

        return None

    def set_planner_mode(self, name: str):
        self.planner_mode = str(name).upper()

    def _sync_runtime_config(self):
        """
        ✅ 运行时同步 config（支持训练过程动态改参）
        说明：
        - 训练脚本可能会做 curriculum / y_ref 开关 / anti-stall 参数调节
        - 如果不在 reset 前同步，这些改动不会生效
        """
        # y_ref 控制开关 + 增益（用于降低训练早期难度）
        self.use_yref_in_steer = bool(getattr(self.config, "use_yref_in_steer", self.use_yref_in_steer))
        self.yref_steer_gain = float(getattr(self.config, "yref_steer_gain", self.yref_steer_gain))

        # anti-stall 相关（训练中可动态收紧/放松）
        self.enable_anti_stall = bool(getattr(self.config, "enable_anti_stall", self.enable_anti_stall))
        self.min_throttle_when_stuck = float(
            getattr(self.config, "min_throttle_when_stuck", self.min_throttle_when_stuck)
        )
        self.low_speed_steer_scale = float(
            getattr(self.config, "low_speed_steer_scale", self.low_speed_steer_scale)
        )

    # ----------------- reset/step -----------------
    def reset(self):
        # ✅ 训练中可能会动态修改 config，这里每个 episode 同步一次
        self._sync_runtime_config()

        # ✅ 随机场景选择（如果启用）
        if getattr(self.config, "random_scenario", False):
            scenario_pool = getattr(self.config, "scenario_pool", ["parked_obstacles", "cones"])
            self.scenario = random.choice(scenario_pool)
            print(f"\n[RandomScenario] 本次Episode场景: {self.scenario}")

        # ✅ 温和的清理方案：先清空队列，但不立即关闭 sync_mode
        if self.sync_mode is not None:
            try:
                # 清空队列中的残留数据
                for q in self.sync_mode._queues:
                    try:
                        while not q.empty():
                            q.get_nowait()
                    except:
                        pass
            except Exception as e:
                print(f"[CarlaEnv] ⚠️ 清空队列失败: {e}")

        # 清理 actors（包括传感器）
        self._cleanup_actors()

        # ✅ 现在关闭 sync_mode（传感器已经被销毁）
        if self.sync_mode is not None:
            try:
                # 只恢复设置，不再尝试停止传感器（已经销毁了）
                if self.sync_mode._settings is not None:
                    self.world.apply_settings(self.sync_mode._settings)
            except Exception as e:
                print(f"[CarlaEnv] ⚠️ 恢复设置失败: {e}")
            finally:
                self.sync_mode = None

        # ✅ 给 CARLA 更多时间来稳定（特别是在清理大量 actors 后）
        time.sleep(0.5)

        # 强制同步设置，避免某次异常导致 world settings 漂掉
        self._apply_sync_settings(self.fixed_dt)

        # 切图逻辑
        if self.scenario == "cones_xml":
            xml_path = self._pick_random_xml_file()
            if xml_path:
                town, _ = self._parse_xml_waypoints(xml_path)
                if town:
                    self._load_map_if_needed(town)
            else:
                if self.randomize_town and self.town_pool:
                    self._load_map_if_needed(random.choice(self.town_pool))
        else:
            # ✅ 检查场景是否需要特定的 town（从 XML 获取）
            required_town = self._get_required_town_for_scenario(self.scenario)
            if required_town:
                print(f"[CarlaEnv] 场景 {self.scenario} 需要地图 {required_town}，正在切换...")
                self._load_map_if_needed(required_town)
            elif self.randomize_town and self.town_pool:
                self._load_map_if_needed(random.choice(self.town_pool))

        # 搭场景 + spawn
        spawn_tf = self._maybe_setup_scene_and_pick_spawn()

        self.ego = self._spawn_ego_with_transform(spawn_tf)
        if self.ego is None:
            raise RuntimeError("spawn ego failed；请确认地图有可用 spawn 点。")

        bp_lib = self.world.get_blueprint_library()
        if self.render_display:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", "400")
            cam_bp.set_attribute("image_size_y", "300")
            cam_bp.set_attribute("fov", "90")
            cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
            self.camera_display = self.world.try_spawn_actor(cam_bp, cam_tf, attach_to=self.ego)
            if self.camera_display:
                self._actors.append(self.camera_display)

        col_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.try_spawn_actor(col_bp, carla.Transform(), attach_to=self.ego)
        if self.collision_sensor is not None:
            self.collision_sensor.listen(lambda e: self._on_collision(e))
            self._actors.append(self.collision_sensor)

        fps = int(round(1.0 / self.fixed_dt))
        # ✅ 同步模式：训练时可关闭渲染，减少内存占用与断连风险
        if self.render_display and self.camera_display is not None:
            self.sync_mode = CarlaSyncMode(
                self.world,
                self.camera_display,
                fps=fps,
                no_rendering_mode=(not self.render_display),
            )
        else:
            self.sync_mode = CarlaSyncMode(
                self.world,
                fps=fps,
                no_rendering_mode=(not self.render_display),
            )

        # 初始控制：刹停一帧，确保稳定
        # ✅ 增加重试机制，防止 RPC 超时
        max_autopilot_retries = 3
        for retry in range(max_autopilot_retries):
            try:
                self.ego.set_autopilot(False)
                break
            except RuntimeError as e:
                if retry < max_autopilot_retries - 1:
                    print(f"⚠️  set_autopilot 失败 (尝试 {retry+1}/{max_autopilot_retries}): {e}")
                    time.sleep(0.5)
                else:
                    print(f"⚠️  set_autopilot 最终失败，继续执行: {e}")
                    # 不抛出异常，继续执行

        init_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        init_control.gear = 1
        init_control.manual_gear_shift = True
        self.ego.apply_control(init_control)
        self.last_control = init_control

        # 第一次 tick 给更长 timeout
        # ✅ 传感器在第一次 tick 时可能还没准备好，CarlaSyncMode 会自动处理
        try:
            self._tick_once(timeout=5.0)
        except Exception as e:
            print(f"⚠️  第一次tick失败: {e}")
            # 重试一次
            time.sleep(0.5)
            self._tick_once(timeout=5.0)

        # reset reward state
        self._reset_reward_state()

        self.episode_steps = 0
        self.collision = False

        # RewardMonitor reset
        self.episode_id += 1

        # ✅ 打印场景初始化信息
        self._print_scene_info()

        save_root = getattr(self.config, "record_path", "./logs")
        if RewardMonitor is not None and RewardMonitorConfig is not None:
            if hasattr(self.config, "reward_config") and self.config.reward_config is not None:
                monitor_cfg = self.config.reward_config
                monitor_cfg.save_dir = os.path.join(save_root, "reward_logs")
                monitor_cfg.enable_vis = self.render_display
            else:
                monitor_cfg = RewardMonitorConfig(
                    enable_vis=self.render_display,
                    save_dir=os.path.join(save_root, "reward_logs")
                )

            if self.reward_monitor is None:
                self.reward_monitor = RewardMonitor(world=self.world, ego_vehicle=self.ego, config=monitor_cfg)
            else:
                self.reward_monitor.world = self.world
                self.reward_monitor.ego = self.ego
                self.reward_monitor.cfg = monitor_cfg

            self.reward_monitor.reset(episode_id=self.episode_id)

        return self._get_state_obs()

    def step(self, action):
        """
        action = [throttle_brake, steer, y_ref] in [-1,1]^3
        """
        # --------- 0) 兼容输入类型，并确保长度 >= 3 ---------
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        a0 = float(action[0]) if action.shape[0] > 0 else 0.0
        a1 = float(action[1]) if action.shape[0] > 1 else 0.0
        a2 = float(action[2]) if action.shape[0] > 2 else 0.0

        a0 = float(np.clip(a0, -1.0, 1.0))
        a1 = float(np.clip(a1, -1.0, 1.0))
        y_ref = float(np.clip(a2, -1.0, 1.0))

        # --------- 1) throttle/brake ---------
        if a0 >= 0.0:
            throttle = float(np.clip(a0, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            b = float(np.clip(-a0, 0.0, 1.0))
            b = 0.0 if b < self.brake_deadzone else b
            brake = b

        # --------- 2) steer：env 不再做 y_ref 映射，只执行最终 steer ---------
        steer_raw = float(np.clip(a1, -1.0, 1.0))  # 这里 steer_raw 就是 applied steer
        steer = steer_raw
        if self.use_yref_in_steer:
            steer = float(np.clip(steer_raw + self.yref_steer_gain * y_ref, -1, 1))

        # y_ref 仍然读出来，但只用于日志（不用于控制）
        y_ref = float(np.clip(a2, -1.0, 1.0))

        # --------- 3) anti-stall ---------
        if self.enable_anti_stall and (self.ego is not None):
            v = self.ego.get_velocity()
            speed = float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
            if speed < self.stuck_speed_thresh:
                brake = 0.0
                throttle = max(throttle, self.min_throttle_when_stuck)
                steer = float(np.clip(steer * self.low_speed_steer_scale, -1.0, 1.0))

        # --------- 4) apply control ---------
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        control.gear = 1
        control.manual_gear_shift = True
        self.ego.apply_control(control)
        self.last_control = control

        snapshot, display_image = self._tick_once(timeout=15.0)  # ✅ 增加到 15 秒

        # ✅ 绘制调试信息（每N步绘制一次，避免性能影响）
        if self.episode_steps % self.debug_draw_interval == 0:
            self._draw_debug_info()

        if self.render_display and display_image is not None:
            self._draw_display(display_image, throttle, steer, brake)

        next_obs = self._get_state_obs()

        self.last_obs = next_obs
        # 在调用 _get_reward() 之前就预告本步是否将触发 time limit
        will_timeout = (self.episode_steps + 1 >= self.max_episode_steps)
        self.timeout_flag = bool(will_timeout)

        # env 内部 reward / done_reason（collision/offroad/no_progress 等）都由 _get_reward 给
        reward_env, done_env, info = self._get_reward()

        # ✅ 更新 matplotlib reward 可视化（每 N 步更新一次）
        if self.render_display and MATPLOTLIB_AVAILABLE and (self.episode_steps % self.reward_update_interval == 0):
            self._update_reward_plot()
        if info is None:
            info = {}

        # ===== DEBUG: obstacle obs check =====
        if getattr(self, "obs_use_obstacles", False) and hasattr(self, "last_obs") and (self.last_obs is not None):
            K = int(getattr(self, "obs_obstacle_k", 5))
            obs_obs = self.last_obs[-K * 3:]
            info["dbg_obs_obstacles_sumabs"] = float(np.sum(np.abs(obs_obs)))
            # 可选：只存前一个障碍物的3个数，避免 wandb 太长
            info["dbg_obs_obstacles_first3"] = [float(x) for x in obs_obs[:3]]

        # --------- 5) time limit（仅截断语义，不惩罚）---------
        self.episode_steps += 1
        timeout = (self.episode_steps >= self.max_episode_steps)  # 字段名 timeout 保留给日志
        self.timeout_flag = bool(timeout)  # ✅ 新增

        # reset() 里：初始化时
        self.timeout_flag = False  # ✅ 新增
        done = bool(done_env or timeout)

        # --------- 6) done_reason 语义修正 ----------
        # done_env=True 时：保持 _get_reward 的 done_reason（collision/offroad/no_progress）
        # done_env=False 且 timeout=True 时：这是自然截断 -> time_limit
        if timeout and (not done_env):
            dr = info.get("done_reason", "running")
            if dr in ["running", "unknown", None, ""]:
                info["done_reason"] = "time_limit"

        # --------- 7) reward components 闭合 ----------
        comps = {}
        if hasattr(self, "last_reward_components") and isinstance(self.last_reward_components, dict):
            comps = self.last_reward_components.copy()

        # time_limit 不给惩罚：0.0
        comps["r_timeout"] = 0.0

        # 最终 reward：严格由 components 求和，保证和分解一致
        reward = float(sum(comps.values()))

        # 给 train_with_logging 用（你那边用的是 env.step_reward_components）
        self.last_reward_components = comps.copy()
        self.step_reward_components = comps.copy()

        # --------- 8) info/debug ----------
        info["timeout"] = float(timeout)  # 仍然保留给面板/日志

        info["raw_throttle_brake"] = float(a0)
        info["raw_steer"] = float(steer_raw)  # 这就是 applied steer
        info["raw_y_ref"] = float(y_ref)

        info["applied_steer"] = float(steer)

        # env 不再使用 y_ref 映射
        info["yref_steer_gain"] = float(self.yref_steer_gain)
        info["yref_used"] = 1.0 if self.use_yref_in_steer else 0.0
        info["steer_delta_from_yref"] = float(steer - steer_raw)

        if self.use_yref_in_steer:
            info["steer_delta_from_yref"] = float(steer - steer_raw)
        else:
            info["steer_delta_from_yref"] = 0.0

        return next_obs, float(reward), bool(done), info

    def render(self, mode="human"):
        return None

    def close(self):
        self._cleanup_actors()
        try:
            # 恢复世界设置，避免退出后 server 被同步模式卡住
            if self.world is not None and self._original_settings is not None:
                self.world.apply_settings(self._original_settings)
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    # ----------------- reward state reset -----------------
    def _reset_reward_state(self):
        # 彻底重置 episode 内部 reward 状态
        self.no_progress_steps = 0
        self.prev_control_for_smooth = None

        self.target_wp = None
        self.prev_wp_dist = None

        ego_loc = self.ego.get_location()
        self.prev_loc = ego_loc

        ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True)
        step_dist = float(getattr(self, "wp_step_dist", 5.0))
        reach_th = float(getattr(self, "wp_reach_thresh", 2.0))
        self.wp_step_dist = step_dist
        self.wp_reach_thresh = reach_th

        next_wps = ego_wp.next(step_dist)
        self.target_wp = next_wps[0] if next_wps else ego_wp

        target_loc = self.target_wp.transform.location
        self.prev_wp_dist = float(math.hypot(ego_loc.x - target_loc.x, ego_loc.y - target_loc.y))

    # ----------------- Tick -----------------
    def _tick_once(self, timeout=2.0):
        if self.sync_mode is None:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
            return None, None

        ret = self.sync_mode.tick(timeout=timeout)
        if isinstance(ret, (list, tuple)) and len(ret) > 0:
            snapshot = ret[0]
            image = ret[1] if len(ret) > 1 else None
        else:
            snapshot, image = None, None

        if self.weather is not None:
            self.weather.tick()
        return snapshot,  image

    # ----------------- spawn & spectator -----------------
    def _spawn_ego_with_transform(self, tf: carla.Transform) -> Optional[carla.Vehicle]:
        bp_lib = self.world.get_blueprint_library()
        model_id = f"vehicle.{self.vehicle_name}"
        bp = bp_lib.find(model_id) if bp_lib.find(model_id) else bp_lib.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")

        ego = self.world.try_spawn_actor(bp, tf)
        if ego:
            self._maybe_place_spectator(ego)
            self._actors.append(ego)
            return ego

        spawns = self.map.get_spawn_points()
        random.shuffle(spawns)
        for alt in spawns:
            ego = self.world.try_spawn_actor(bp, alt)
            if ego:
                self._maybe_place_spectator(ego)
                self._actors.append(ego)
                return ego
        return None

    def _maybe_place_spectator(self, ego: carla.Actor):
        mode = (self.spectator_mode or "none").lower()
        spec = self.world.get_spectator()
        if mode == "none":
            return
        tf = ego.get_transform()
        yaw = tf.rotation.yaw
        rad = math.radians(yaw)
        if mode == "fixed":
            back, height, side = 28.0, 7.0, 0.0
            fx, fy = math.cos(rad), math.sin(rad)
            rx, ry = math.sin(rad), -math.cos(rad)
            cam_loc = carla.Location(
                x=tf.location.x - fx * back + rx * side,
                y=tf.location.y - fy * back + ry * side,
                z=tf.location.z + height
            )
            tgt = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 1.5)
            vx, vy, vz = (tgt.x - cam_loc.x), (tgt.y - cam_loc.y), (tgt.z - cam_loc.z)
            yaw_cam = math.degrees(math.atan2(vy, vx))
            dist_xy = max(1e-6, math.hypot(vx, vy))
            pitch_cam = -math.degrees(math.atan2(vz, dist_xy))
            spec.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=pitch_cam, yaw=yaw_cam, roll=0.0)))
        elif mode == "chase":
            back, height, side, pitch = 23.0, 9.0, 2.0, -12.0
            fx, fy = math.cos(rad), math.sin(rad)
            rx, ry = math.sin(rad), -math.cos(rad)
            cam_loc = carla.Location(
                x=tf.location.x - fx * back + rx * side,
                y=tf.location.y - fy * back + ry * side,
                z=tf.location.z + height
            )
            spec.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)))

    # ✅ 新增：绘制调试信息
    def _draw_debug_info(self):
        """在CARLA世界中绘制调试信息（障碍物、检测范围等）"""
        if not self.enable_debug_drawing:
            return

        if self.ego is None:
            return

        debug = self.world.debug
        ego_loc = self.ego.get_location()
        ego_tf = self.ego.get_transform()

        # 1. 绘制自车检测范围（用多边形近似圆形）
        if self.draw_detection_range:
            R = float(self.obs_obstacle_range)
            num_points = 32  # 圆形近似点数
            for i in range(num_points):
                angle1 = 2 * math.pi * i / num_points
                angle2 = 2 * math.pi * (i + 1) / num_points
                p1 = carla.Location(
                    x=ego_loc.x + R * math.cos(angle1),
                    y=ego_loc.y + R * math.sin(angle1),
                    z=ego_loc.z + 0.5
                )
                p2 = carla.Location(
                    x=ego_loc.x + R * math.cos(angle2),
                    y=ego_loc.y + R * math.sin(angle2),
                    z=ego_loc.z + 0.5
                )
                debug.draw_line(
                    begin=p1,
                    end=p2,
                    thickness=0.05,
                    color=carla.Color(0, 255, 255),  # 青色
                    life_time=0.1
                )

        # 2. 绘制自车前向方向（箭头）
        if self.draw_ego_direction:
            fwd = ego_tf.get_forward_vector()
            end_loc = carla.Location(
                x=ego_loc.x + fwd.x * 10.0,
                y=ego_loc.y + fwd.y * 10.0,
                z=ego_loc.z + 1.0
            )
            debug.draw_arrow(
                begin=carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 1.0),
                end=end_loc,
                thickness=0.2,
                arrow_size=0.5,
                color=carla.Color(0, 255, 0),  # 绿色
                life_time=0.1
            )

        # 3. 绘制障碍物边界框
        if self.draw_obstacle_boxes:
            for obs in self.obstacle_actors:
                if obs is None:
                    continue
                try:
                    obs_loc = obs.get_location()
                    bbox = obs.bounding_box

                    # 计算距离
                    dist = math.hypot(obs_loc.x - ego_loc.x, obs_loc.y - ego_loc.y)

                    # 根据距离选择颜色
                    if dist < 10.0:
                        color = carla.Color(255, 0, 0)  # 红色：很近
                    elif dist < 20.0:
                        color = carla.Color(255, 165, 0)  # 橙色：中等
                    else:
                        color = carla.Color(255, 255, 0)  # 黄色：较远

                    # 绘制边界框
                    debug.draw_box(
                        box=bbox,
                        rotation=obs.get_transform().rotation,
                        thickness=0.1,
                        color=color,
                        life_time=0.1
                    )

                    # 绘制距离文本
                    debug.draw_string(
                        location=carla.Location(x=obs_loc.x, y=obs_loc.y, z=obs_loc.z + 2.0),
                        text=f"{dist:.1f}m",
                        draw_shadow=True,
                        color=color,
                        life_time=0.1
                    )
                except Exception:
                    continue

        # 4. 绘制车道中心线
        if self.draw_lane_center:
            try:
                wp = self.map.get_waypoint(ego_loc, project_to_road=True)
                if wp:
                    # 绘制前方车道中心线
                    cur_wp = wp
                    for _ in range(20):
                        nxt = cur_wp.next(2.0)
                        if not nxt:
                            break
                        next_wp = nxt[0]
                        debug.draw_line(
                            begin=carla.Location(
                                x=cur_wp.transform.location.x,
                                y=cur_wp.transform.location.y,
                                z=cur_wp.transform.location.z + 0.5
                            ),
                            end=carla.Location(
                                x=next_wp.transform.location.x,
                                y=next_wp.transform.location.y,
                                z=next_wp.transform.location.z + 0.5
                            ),
                            thickness=0.1,
                            color=carla.Color(255, 255, 255),  # 白色
                            life_time=0.1
                        )
                        cur_wp = next_wp
            except Exception:
                pass

    # ✅ 新增：打印场景信息
    def _print_scene_info(self):
        """打印当前场景的详细信息"""
        print("\n" + "="*70)
        print(f"🎬 Episode {self.episode_id} - 场景初始化")
        print("="*70)

        # 地图信息
        map_name = self.map.name if self.map else "Unknown"
        print(f"📍 地图: {map_name}")
        print(f"🎭 场景类型: {self.scenario}")

        # 自车信息
        if self.ego:
            ego_loc = self.ego.get_location()
            ego_rot = self.ego.get_transform().rotation
            print(f"\n🚗 自车信息:")
            print(f"   - 位置: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
            print(f"   - 朝向: Yaw={ego_rot.yaw:.1f}°")
            print(f"   - 车型: {self.vehicle_name}")

        # 障碍物信息
        if self.obstacle_actors:
            print(f"\n🚧 障碍物信息 (共{len(self.obstacle_actors)}个):")
            for i, obs in enumerate(self.obstacle_actors, 1):
                if obs is None:
                    continue
                try:
                    obs_loc = obs.get_location()
                    obs_type = obs.type_id
                    if self.ego:
                        ego_loc = self.ego.get_location()
                        dist = math.hypot(obs_loc.x - ego_loc.x, obs_loc.y - ego_loc.y)
                        print(f"   [{i}] {obs_type}")
                        print(f"       位置: ({obs_loc.x:.1f}, {obs_loc.y:.1f}, {obs_loc.z:.1f})")
                        print(f"       距离自车: {dist:.1f}m")
                except Exception:
                    continue
        else:
            print(f"\n🚧 障碍物: 无")

        # 观测配置
        print(f"\n👁️  观测配置:")
        print(f"   - 类型: {self.observations_type}")
        print(f"   - 维度: {self.obs_dim}")
        print(f"   - 障碍物检测数量: {self.obs_obstacle_k}")
        print(f"   - 检测范围: {self.obs_obstacle_range}m")

        # 可视化配置
        print(f"\n🎨 可视化:")
        print(f"   - Pygame渲染: {'✅' if self.render_display else '❌'}")
        print(f"   - Spectator模式: {self.spectator_mode}")
        print(f"   - 调试绘制: {'✅' if self.enable_debug_drawing else '❌'}")

        print("="*70 + "\n")

    # ----------------- XML & scene -----------------
    def _pick_random_xml_file(self) -> Optional[str]:
        if self.xml_file and os.path.isfile(self.xml_file):
            return self.xml_file
        if self.xml_dir and os.path.isdir(self.xml_dir):
            cands = [os.path.join(self.xml_dir, f) for f in os.listdir(self.xml_dir)
                     if f.lower().endswith(".xml")]
            if cands:
                return random.choice(cands)
        return None

    @staticmethod
    def _parse_xml_waypoints(xml_path: str) -> Tuple[str, List[dict]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        town = ""
        route_elem = root.find(".//route")
        if route_elem is not None and "town" in route_elem.attrib:
            town = route_elem.attrib.get("town", "").strip()

        wps = []
        for wp in root.findall(".//route/waypoints/position"):
            try:
                wps.append({
                    "x": float(wp.get("x")),
                    "y": float(wp.get("y")),
                    "z": float(wp.get("z")),
                    "yaw": float(wp.get("yaw", 0.0)),
                    "pitch": float(wp.get("pitch", 0.0)),
                    "roll": float(wp.get("roll", 0.0)),
                })
            except Exception:
                continue
        return town, wps

    def _maybe_setup_scene_and_pick_spawn(self) -> carla.Transform:
        """
        场景初始化和自车生成位置选择

        Returns:
            carla.Transform: 自车生成位置
        """
        # 1) 如果有initial_spawn_tf，直接使用
        if isinstance(self.initial_spawn_tf, dict):
            try:
                loc = carla.Location(
                    x=float(self.initial_spawn_tf["x"]),
                    y=float(self.initial_spawn_tf["y"]),
                    z=float(self.initial_spawn_tf.get("z", 0.3)),
                )
                yaw = float(self.initial_spawn_tf.get("yaw", 0.0))
                return carla.Transform(loc, carla.Rotation(yaw=yaw))
            except Exception as e:
                print(f"[CarlaEnv] invalid initial_spawn_tf, fallback: {e}")

        # 2) 使用新的场景管理系统
        if self.scenario != "plain":
            # 创建场景实例
            self.scenario_instance = ScenarioFactory.create_scenario(
                scenario_name=self.scenario,
                world=self.world,
                carla_map=self.map,
                config=self.config
            )

            if self.scenario_instance is not None:
                # 初始化场景
                success = self.scenario_instance.setup()

                if success:
                    # 获取障碍物actors（用于观测）
                    self.obstacle_actors = self.scenario_instance.get_obstacle_actors()

                    # 获取自车生成位置
                    spawn_tf = self.scenario_instance.get_spawn_transform()
                    if spawn_tf is not None:
                        return spawn_tf
                    else:
                        print(f"[CarlaEnv] ⚠️ 场景 {self.scenario} 未返回spawn位置，使用默认")
                else:
                    print(f"[CarlaEnv] ⚠️ 场景 {self.scenario} 初始化失败，使用默认spawn")
            else:
                print(f"[CarlaEnv] ⚠️ 未知场景 {self.scenario}，使用默认spawn")

        # 3) 兼容旧的cones_xml场景（如果需要保留）
        if self.scenario == "cones_xml":
            xml_path = self._pick_random_xml_file()
            if xml_path:
                _, wps = self._parse_xml_waypoints(xml_path)
                if wps:
                    pick = random.choice(wps)
                    start_loc = carla.Location(x=pick["x"], y=pick["y"], z=pick["z"])
                    start_wp = self.map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if start_wp:
                        self._place_cones_conditionally_behind(
                            start_wp=start_wp,
                            num_cones=self.cone_num,
                            step_behind=self.cone_step_behind,
                            step_lateral_per_cone=self.cone_step_lateral,
                            z_offset=self.cone_z_offset,
                            lane_margin=self.cone_lane_margin,
                        )
                        if self.world.get_settings().synchronous_mode:
                            for _ in range(3):
                                self.world.tick()
                        tf = self._spawn_tf_from_first_cone()
                        if tf is not None:
                            return tf

        # 4) 兼容旧的cones场景（如果需要保留）
        if self.scenario == "cones_old":
            start_wp = self._pick_random_start_waypoint(
                min_gap_from_junction=self.cone_min_gap_from_junction,
                grid=self.cone_grid
            )
            if start_wp:
                self._place_cones_conditionally_behind(
                    start_wp=start_wp,
                    num_cones=self.cone_num,
                    step_behind=self.cone_step_behind,
                    step_lateral_per_cone=self.cone_step_lateral,
                    z_offset=self.cone_z_offset,
                    lane_margin=self.cone_lane_margin
                )
                if self.world.get_settings().synchronous_mode:
                    for _ in range(3):
                        self.world.tick()
                tf = self._spawn_tf_from_first_cone()
                if tf is not None:
                    return tf

        # 5) fallback: 使用地图默认spawn点
        spawns = self.map.get_spawn_points()
        if not spawns:
            return carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
        return random.choice(spawns)

    # ============== 旧的cones函数（保留用于兼容） ==============
    def _spawn_tf_from_first_cone(self) -> Optional[carla.Transform]:
        if self._first_cone_tf is None:
            return None
        amap = self.world.get_map()
        cone_wp = amap.get_waypoint(
            self._first_cone_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if cone_wp is None:
            return None

        traveled = 0.0
        step = max(0.5, float(self.spawn_wp_step))
        wp = cone_wp
        while traveled < self.spawn_min_gap_from_cone:
            prevs = wp.previous(step)
            if not prevs:
                break
            wp = prevs[0]
            traveled += step
        return wp.transform

    def _pick_random_start_waypoint(self, min_gap_from_junction: float = 15.0, grid: float = 5.0, max_tries: int = 300):
        amap = self.world.get_map()
        cands = [wp for wp in amap.generate_waypoints(grid) if wp.lane_type == carla.LaneType.Driving]
        if not cands:
            return None
        random.shuffle(cands)

        def _is_near_junction(wp: carla.Waypoint, dist: float = 15.0, step: float = 1.0):
            cur = wp
            traveled = 0.0
            while traveled < dist:
                nxt = cur.next(step)
                if not nxt:
                    break
                cur = nxt[0]
                traveled += step
                if cur.is_junction:
                    return True
            cur = wp
            traveled = 0.0
            while traveled < dist:
                prv = cur.previous(step)
                if not prv:
                    break
                cur = prv[0]
                traveled += step
                if cur.is_junction:
                    return True
            return False

        tries = 0
        for wp in cands:
            tries += 1
            if (not wp.is_junction) and (not _is_near_junction(wp, dist=min_gap_from_junction, step=1.0)):
                return wp
            if tries >= max_tries:
                break
        for wp in cands:
            if not wp.is_junction:
                return wp
        return cands[0]

    def _place_cones_conditionally_behind(
        self,
        start_wp: carla.Waypoint,
        num_cones: int = 10,
        step_behind: float = 3.0,
        step_lateral_per_cone: float = 0.35,
        z_offset: float = 0.0,
        lane_margin: float = 0.25,
    ):
        """
        ⚠️ 已废弃 - 此函数已移至 scenario_manager.ConesScenario
        保留此函数仅用于向后兼容
        """
        print("[DEPRECATED] _place_cones_conditionally_behind 已废弃，请使用 scenario_manager.ConesScenario")
        world = self.world
        lib = world.get_blueprint_library()
        try:
            cone_bp = lib.find("static.prop.trafficcone01")
        except Exception:
            cone_bp = lib.find("static.prop.trafficcone")

        left_lane_wp = start_wp.get_left_lane()
        right_lane_wp = start_wp.get_right_lane()
        is_left_driving = left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving
        is_right_driving = right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving

        lateral_multiplier = 1.0
        if is_left_driving and not is_right_driving:
            lateral_multiplier = 1.0
        elif is_right_driving and not is_left_driving:
            lateral_multiplier = -1.0
        elif is_left_driving and is_right_driving:
            lateral_multiplier = random.choice([-1.0, 1.0])

        cones_spawned: List[carla.Actor] = []
        first_tf = None
        last_tf = None
        cur_wp = start_wp

        for i in range(num_cones):
            if not cur_wp:
                break
            wp_tf = cur_wp.transform
            right_vec = wp_tf.get_right_vector()
            half_w = cur_wp.lane_width * 0.5

            start_offset_signed = (half_w - lane_margin) * -lateral_multiplier
            progression_offset = (i * step_lateral_per_cone) * lateral_multiplier
            desired = start_offset_signed + progression_offset
            max_pos = half_w - lane_margin
            max_neg = -(half_w - lane_margin)
            actual = max(max_neg, min(max_pos, desired))

            cone_loc = wp_tf.location + right_vec * actual
            cone_loc.z += z_offset
            cone_tf = carla.Transform(cone_loc, wp_tf.rotation)

            cone_actor = world.try_spawn_actor(cone_bp, cone_tf)
            if cone_actor:
                cones_spawned.append(cone_actor)
                if first_tf is None:
                    first_tf = cone_tf
                last_tf = cone_tf

            prv = cur_wp.previous(step_behind)
            cur_wp = prv[0] if prv else None

        self._first_cone_tf = first_tf
        self._last_cone_tf = last_tf
        self._actors.extend(cones_spawned)
        return cones_spawned, first_tf, last_tf

    def _place_parked_vehicles(self, start_wp: carla.Waypoint):
        """
        ⚠️ 已废弃 - 此函数已移至 scenario_manager.ParkedObstaclesScenario
        保留此函数仅用于向后兼容

        双车场景：在ego车前方12-20米范围内生成2辆障碍车
        第一辆: 12-20米随机
        第二辆: 第一辆后8米
        """
        print("[DEPRECATED] _place_parked_vehicles 已废弃，请使用 scenario_manager.ParkedObstaclesScenario")
        world = self.world
        lib = world.get_blueprint_library()

        # 调试输出
        start_loc = start_wp.transform.location
        num_cars = int(getattr(self.config, "num_parked_cars", 1))
        print(f"\n[OBSTACLE] 双车场景 - 开始生成{num_cars}辆障碍车:")
        print(f"  - 起始waypoint: ({start_loc.x:.1f}, {start_loc.y:.1f}, {start_loc.z:.1f})")
        print(f"  - 车道类型: {start_wp.lane_type}")
        print(f"  - 车道宽度: {start_wp.lane_width:.1f}m")

        # 获取车辆blueprints
        vehicle_bps = [
            lib.filter("vehicle.tesla.model3"),
            lib.filter("vehicle.audi.a2"),
            lib.filter("vehicle.bmw.grandtourer"),
            lib.filter("vehicle.toyota.prius"),
        ]
        available_bps = [bp for bps in vehicle_bps for bp in bps if bps]
        if not available_bps:
            available_bps = lib.filter("vehicle.*")

        parked_vehicles = []

        # ✅ 第一辆车：随机距离（12-20米）
        min_dist = float(getattr(self.config, "parked_car_start_distance_min", 12.0))
        max_dist = float(getattr(self.config, "parked_car_start_distance_max", 20.0))
        first_car_distance = random.uniform(min_dist, max_dist)

        print(f"  - 第一辆车目标距离: {first_car_distance:.1f}m (范围: {min_dist}-{max_dist}m)")

        # 生成第一辆车
        cur_wp = start_wp
        traveled = 0.0
        step_size = 2.0

        while traveled < first_car_distance:
            nxt = cur_wp.next(step_size)
            if not nxt:
                print(f"    ⚠️ 无法继续前进（路径尽头），已前进{traveled:.1f}m")
                break
            cur_wp = nxt[0]
            traveled += step_size

        if cur_wp:
            vehicle = self._spawn_single_vehicle(world, cur_wp, available_bps, 1, traveled)
            if vehicle:
                parked_vehicles.append(vehicle)

        # ✅ 第二辆车：在第一辆车后spacing米
        if num_cars >= 2 and cur_wp:
            spacing = float(getattr(self.config, "parked_car_spacing", 8.0))
            print(f"  - 第二辆车间隔: {spacing:.1f}m")

            # 继续前进spacing米
            second_car_traveled = 0.0
            while second_car_traveled < spacing:
                nxt = cur_wp.next(step_size)
                if not nxt:
                    print(f"    ⚠️ 无法继续前进（路径尽头），已前进{second_car_traveled:.1f}m")
                    break
                cur_wp = nxt[0]
                second_car_traveled += step_size

            if cur_wp:
                total_distance = traveled + second_car_traveled
                vehicle = self._spawn_single_vehicle(world, cur_wp, available_bps, 2, total_distance)
                if vehicle:
                    parked_vehicles.append(vehicle)

        # 汇总输出
        print(f"\n[OBSTACLE DEBUG] 双车场景生成完成:")
        print(f"  - 尝试生成: {num_cars} 辆")
        print(f"  - 成功生成: {len(parked_vehicles)} 辆")
        if len(parked_vehicles) >= 1:
            v1_loc = parked_vehicles[0].get_location()
            print(f"  - 车辆1位置: ({v1_loc.x:.1f}, {v1_loc.y:.1f}, {v1_loc.z:.1f})")
        if len(parked_vehicles) >= 2:
            v2_loc = parked_vehicles[1].get_location()
            print(f"  - 车辆2位置: ({v2_loc.x:.1f}, {v2_loc.y:.1f}, {v2_loc.z:.1f})")
            # 计算两车距离
            if len(parked_vehicles) >= 2:
                v1_loc = parked_vehicles[0].get_location()
                v2_loc = parked_vehicles[1].get_location()
                dist = math.sqrt((v2_loc.x - v1_loc.x)**2 + (v2_loc.y - v1_loc.y)**2)
                print(f"  - 两车间距: {dist:.1f}m")
        print(f"  - 横向偏移: 0.0m (车道中心)")
        print(f"  - 注册到obstacle_actors: {len(self.obstacle_actors)} 个\n")

        return parked_vehicles

    def _spawn_single_vehicle(self, world, waypoint, available_bps, car_index, distance_from_ego):
        """
        在指定waypoint生成单个车辆
        """
        wp_loc = waypoint.transform.location
        wp_rot = waypoint.transform.rotation

        print(f"  - 车辆{car_index}: 目标距离={distance_from_ego:.1f}m, 位置=({wp_loc.x:.1f}, {wp_loc.y:.1f})")

        # 策略1: 车道中心（最优）
        spawn_loc = carla.Location(
            x=wp_loc.x,
            y=wp_loc.y,
            z=wp_loc.z + 0.5  # 抬高0.5米防止穿地
        )
        spawn_tf = carla.Transform(spawn_loc, wp_rot)

        vehicle_bp = random.choice(available_bps)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

        # 策略2: 如果失败，尝试更高位置
        if not vehicle:
            print(f"    ⚠️ 车道中心生成失败，尝试更高位置...")
            spawn_loc.z = wp_loc.z + 1.0
            spawn_tf = carla.Transform(spawn_loc, wp_rot)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

        # 策略3: 如果还失败，尝试稍微偏移
        if not vehicle:
            print(f"    ⚠️ 高位置生成失败，尝试轻微偏移...")
            right_vec = wp_rot.get_right_vector()
            spawn_loc = carla.Location(
                x=wp_loc.x + right_vec.x * 0.5,
                y=wp_loc.y + right_vec.y * 0.5,
                z=wp_loc.z + 0.5
            )
            spawn_tf = carla.Transform(spawn_loc, wp_rot)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

        if vehicle:
            vehicle.set_simulate_physics(False)
            self._actors.append(vehicle)
            self.obstacle_actors.append(vehicle)

            v_loc = vehicle.get_location()
            print(f"    ✅ 成功生成！ID={vehicle.id}, 实际位置=({v_loc.x:.1f}, {v_loc.y:.1f}, {v_loc.z:.1f})")
            return vehicle
        else:
            print(f"    ❌ 所有策略都失败！")
            return None

    # ----------------- obs / collision -----------------
    def _on_collision(self, event):
        self.collision = True

    def _wrap_angle(self, a: float) -> float:
        # wrap to [-pi, pi]
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _world_to_ego_frame(self, dx: float, dy: float, ego_yaw_rad: float):
        c = math.cos(ego_yaw_rad)
        s = math.sin(ego_yaw_rad)
        # ego x+: forward, ego y+: left
        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy
        return rel_x, rel_y

    def _compute_lane_obs(self) -> np.ndarray:
        """
        lane obs (6):
          [lane_dev_norm, sin(heading_err), cos(heading_err), lane_width_norm, wp_rel_x_norm, wp_rel_y_norm]
        """
        loc = self.ego.get_location()
        tf = self.ego.get_transform()
        ego_yaw = math.radians(float(tf.rotation.yaw))

        wp = self.map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            return np.zeros((6,), dtype=np.float32)

        # lane deviation (meters)
        lane_dev = float(math.hypot(loc.x - wp.transform.location.x, loc.y - wp.transform.location.y))
        lane_w = float(getattr(wp, "lane_width", 3.5))

        # heading error
        fwd = wp.transform.get_forward_vector()
        lane_yaw = math.atan2(float(fwd.y), float(fwd.x))
        heading_err = self._wrap_angle(lane_yaw - ego_yaw)

        # target waypoint direction (use self.target_wp if exists)
        tgt = self.target_wp
        if tgt is None:
            nxt = wp.next(float(getattr(self, "wp_step_dist", 5.0)))
            tgt = nxt[0] if nxt else wp

        tgt_loc = tgt.transform.location
        dx = float(tgt_loc.x - loc.x)
        dy = float(tgt_loc.y - loc.y)
        wp_rel_x, wp_rel_y = self._world_to_ego_frame(dx, dy, ego_yaw)

        # normalization (very important)
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
        """
        优先：self.obstacle_actors（如果你在 spawn 障碍物时注册了）
        否则：fallback 扫 world 里的 vehicle.* + static.prop.*（cone/barrier 等）
        """
        # 1) 优先使用注册过的 obstacles
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

        # 2) fallback：扫全世界 actors
        try:
            actors = self.world.get_actors()
            out = []

            # vehicles
            for v in actors.filter("vehicle.*"):
                if self.ego is not None and v.id == self.ego.id:
                    continue
                out.append(v)

            # static props（cones / barriers / 等）
            for p in actors.filter("static.prop.*"):
                # 你可以按需扩展关键词
                tid = getattr(p, "type_id", "")
                if ("trafficcone" in tid) or ("cone" in tid) or ("barrier" in tid) or ("construction" in tid):
                    out.append(p)

            return out
        except Exception:
            return []

    def _compute_obstacle_obs(self) -> np.ndarray:
        """
        obstacles obs: K * 3
          per obstacle: [rel_x_norm, rel_y_norm, dist_norm]
        其中：
          rel_x = 前向投影（自车 forward 点乘）
          rel_y = 横向投影（自车 right 点乘，右为正）
        """
        K = int(self.obs_obstacle_k)
        R = float(self.obs_obstacle_range)

        candidates = self._collect_obstacle_candidates()

        # ✅ 添加调试输出（每50步输出一次）
        if self.episode_steps % 50 == 0:
            ego_loc = self.ego.get_location() if self.ego else None
            print(f"\n[OBSTACLE DETECTION] Step {self.episode_steps}:")
            print(f"  - 检测范围: {R}m")
            print(f"  - 候选障碍物数量: {len(candidates)}")
            if ego_loc:
                print(f"  - Ego位置: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
                if candidates:
                    for i, obs in enumerate(candidates[:5]):
                        try:
                            obs_loc = obs.get_location()
                            dist = math.hypot(obs_loc.x - ego_loc.x, obs_loc.y - ego_loc.y)
                            in_range = "✅" if dist <= R else "❌"
                            print(f"  - 障碍物{i+1}: 距离={dist:.1f}m {in_range}, 位置=({obs_loc.x:.1f}, {obs_loc.y:.1f})")
                        except:
                            print(f"  - 障碍物{i+1}: 无效")
            else:
                print(f"  - Ego: None")

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

            # ✅ 用点乘得到 ego-frame 的前向/横向
            rel_x = dx * float(ego_fwd.x) + dy * float(ego_fwd.y)

            # left vector = -right vector
            ego_left = carla.Vector3D(x=-ego_right.x, y=-ego_right.y, z=-ego_right.z)
            rel_y = dx * ego_left.x + dy * ego_left.y  # 左为正

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

        result = np.array(feats, dtype=np.float32)

        # ✅ 添加调试输出（每50步输出一次）
        if self.episode_steps % 50 == 0:
            non_zero = np.count_nonzero(result)
            print(f"  - 观测值非零元素: {non_zero}/{K*3}")
            if non_zero > 0:
                print(f"  - 前3个障碍物观测: {result[:9]}")

        return result

    def _get_state_obs(self):
        tf = self.ego.get_transform()
        loc, rot = tf.location, tf.rotation
        acc = vector_to_scalar(self.ego.get_acceleration())
        ang = vector_to_scalar(self.ego.get_angular_velocity())
        vel = vector_to_scalar(self.ego.get_velocity())

        base = np.array(
            [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll, acc, ang, vel],
            dtype=np.float32,
        )

        parts = [base]

        if getattr(self, "obs_use_lane", False):
            parts.append(self._compute_lane_obs())

        if getattr(self, "obs_use_obstacles", False):
            parts.append(self._compute_obstacle_obs())

        obs = np.concatenate(parts, axis=0).astype(np.float32)

        # 防御：维度必须一致
        if obs.shape[0] != int(self.obs_dim):
            raise RuntimeError(f"[CarlaEnv] obs_dim mismatch: got {obs.shape[0]} expected {self.obs_dim}")

        return obs

    def _get_reward(self):
        import math
        import numpy as np

        # ============================================================
        # ✅ Safety-first Reward (更稳定/更少组件版)
        # 目标：先学会不碰撞、不出界，低速稳定穿过四个场景
        # ============================================================

        # ----------------- terminal penalties -----------------
        # ✅ 终止惩罚：碰撞/出界更痛一点，推动安全学习
        K_COLLISION_TERMINAL = 180.0
        K_OFFROAD_TERMINAL = 120.0
        K_NO_PROGRESS_TERM = 50.0
        # ✅ 允许更长时间尝试起步，避免刚学走就被判“无进展”
        NO_PROGRESS_LIMIT = 260

        # ----------------- progress (门控后才给) -----------------
        # ✅ 提高前向进度奖励，避免“保守趴地上”
        # ✅ 让进度回报更“看得见”，避免长期负回报卡平台
        K_PROGRESS = 1.20
        PROG_CLIP = 0.60
        PROG_EMA_A = 0.08

        # ----------------- speed (鼓励更合理的低速通过，而不是爬行) -----------------
        # ✅ 速度目标再上调一点，配合低速惩罚，让策略别“爬行”
        TARGET_SPEED = 4.0
        V_MAX = 7.0
        OVERSPEED_START = 5.5
        K_SPEED = 0.35
        K_OVERSPEED = 0.15

        # ----------------- lane keeping -----------------
        # ✅ 轻微降低车道惩罚，减少“怕偏一点就停车”的行为
        # ✅ 轻微降低车道惩罚，避免“怕偏一点就停车”
        K_LANE = 0.35
        SOFT_START_RATIO = 0.60
        K_OFFROAD_SOFT = 1.6

        # ----------------- danger (仍然保守，但不要过于强惩罚) -----------------
        # ✅ 危险惩罚仍保守，但稍微缓和强度
        DANGER_START = 0.40
        K_DANGER = 0.50
        DANGER_CLIP = 0.80

        # ----------------- obstacle shaping (更早更强) -----------------
        HAVE_OBS = bool(getattr(self, "obs_use_obstacles", False))
        K_OBS = int(getattr(self, "obs_obstacle_k", 5))
        R_OBS = float(getattr(self, "obs_obstacle_range", 50.0))
        USE_LANE_FEAT = bool(getattr(self, "obs_use_lane", True))
        LANE_DIM = 6 if USE_LANE_FEAT else 0

        # ✅ 观察更远一点，提前对障碍做引导惩罚
        AVOID_FWD = 55.0
        SAFE_DIST = 22.0
        LAT_TOL = 4.5
        W_OBS_CLEAR = 2.5  # 近距惩罚稍强，减少碰撞率
        W_OBS_SPEED = 1.0  # 近障碍限速惩罚
        V_CAP_NEAR = 2.2   # 近障碍速度上限略收紧

        # ----------------- alive / success -----------------
        # ✅ 略微提高 alive，降低纯“苟活”收益（配合低速惩罚）
        # ✅ 生存奖励小幅增加，成功奖励更明确
        R_ALIVE = 0.03
        SUCCESS_BONUS = 80.0

        # ----------------- terminal flags -----------------
        collision_flag = bool(getattr(self, "collision", False))
        done = False
        done_reason = "running"

        ego = getattr(self, "ego", None)
        world = getattr(self, "world", None)
        m = getattr(self, "map", None)

        if ego is None or world is None or m is None:
            return 0.0, False, {"done": 0.0, "done_reason": "no_ego"}

        # ----------------- ego states -----------------
        vel = ego.get_velocity()
        speed = float(math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

        loc = ego.get_location()
        wp = m.get_waypoint(loc, project_to_road=True)

        lane_width = float(getattr(wp, "lane_width", 3.5))
        lane_center = wp.transform.location
        lane_dev = float(math.hypot(loc.x - lane_center.x, loc.y - lane_center.y))

        # ----------------- progress (ego-forward projection) -----------------
        if getattr(self, "prev_loc", None) is None:
            self.prev_loc = loc

        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()

        dx = float(loc.x - self.prev_loc.x)
        dy = float(loc.y - self.prev_loc.y)
        prog_fwd = dx * float(fwd.x) + dy * float(fwd.y)
        prog_fwd = float(np.clip(prog_fwd, -PROG_CLIP, PROG_CLIP))
        self.prev_loc = loc

        # EMA for idle
        if not hasattr(self, "progress_ema"):
            self.progress_ema = 0.0
        self.progress_ema = (1.0 - PROG_EMA_A) * float(self.progress_ema) + PROG_EMA_A * float(prog_fwd)

        if not hasattr(self, "no_progress_steps"):
            self.no_progress_steps = 0
        # ✅ 低速阈值上调，鼓励更快进入“可控前进”
        is_idle = (abs(self.progress_ema) < 0.01) and (speed < 0.60)
        self.no_progress_steps = self.no_progress_steps + 1 if is_idle else 0

        r_no_progress_term = 0.0
        if self.no_progress_steps > int(NO_PROGRESS_LIMIT):
            done = True
            done_reason = "no_progress"
            r_no_progress_term = -K_NO_PROGRESS_TERM

        # ----------------- lane/offroad -----------------
        offroad_thresh = 0.5 * lane_width + 1.0
        lane_ratio = lane_dev / max(offroad_thresh, 1e-6)
        offroad = bool(lane_dev > offroad_thresh)

        # soft offroad penalty (提前拉回)
        r_offroad_soft = 0.0
        soft_start = SOFT_START_RATIO * offroad_thresh
        if lane_dev > soft_start:
            x = (lane_dev - soft_start) / max(offroad_thresh - soft_start, 1e-6)
            x = float(np.clip(x, 0.0, 2.0))
            r_offroad_soft = -K_OFFROAD_SOFT * float(x * x)

        # lane dense penalty
        r_lane = -K_LANE * float(np.clip(lane_ratio, 0.0, 2.0) ** 2)

        # danger: 提前触发 + 与速度耦合（更强）
        danger_excess = max(0.0, lane_ratio - DANGER_START)
        speed_ratio = float(np.clip(speed / max(TARGET_SPEED, 1e-6), 0.0, 2.0))
        r_danger = -K_DANGER * float(danger_excess ** 2) * speed_ratio
        r_danger = float(np.clip(r_danger, -DANGER_CLIP, 0.0))

        # ----------------- nearest obstacle (obs优先 + fallback) -----------------
        nearest_dist = None
        nearest_fwd = None
        nearest_lat = None
        obstacle_gate = 0.0

        obs_vec = getattr(self, "last_obs", None)

        def _fallback_nearest_world():
            try:
                tf2 = ego.get_transform()
                e_loc = tf2.location
                e_fwd = tf2.get_forward_vector()
                e_right = tf2.get_right_vector()
            except Exception:
                return None, None, None

            best = None
            for a in getattr(self, "obstacle_actors", []):
                if a is None:
                    continue
                try:
                    a_loc = a.get_location()
                except Exception:
                    continue
                dx_ = float(a_loc.x - e_loc.x)
                dy_ = float(a_loc.y - e_loc.y)
                dist_ = float(math.hypot(dx_, dy_))
                if dist_ < 1e-3 or dist_ > R_OBS:
                    continue
                fwdp = dx_ * float(e_fwd.x) + dy_ * float(e_fwd.y)
                if fwdp <= 0.0:
                    continue
                latp = dx_ * float(e_right.x) + dy_ * float(e_right.y)
                if (best is None) or (dist_ < best[0]):
                    best = (dist_, fwdp, latp)

            if best is None:
                return None, None, None
            return best[0], best[1], best[2]

        # 从 obs 解析（保持兼容）
        if HAVE_OBS and (obs_vec is not None) and (len(obs_vec) >= 9 + LANE_DIM + K_OBS * 3):
            x = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
            start = 9 + LANE_DIM
            block = x[start:start + K_OBS * 3].reshape(K_OBS, 3)

            valid = block[:, 2] > 1e-6
            if np.any(valid):
                relx_n = block[valid, 0]
                rely_n = block[valid, 1]
                dist_n = block[valid, 2]

                relx = relx_n * R_OBS
                rely = rely_n * R_OBS
                dist = dist_n * R_OBS

                yaw = math.radians(float(ego_tf.rotation.yaw))
                cy, sy = math.cos(yaw), math.sin(yaw)
                dx_w = relx * cy - rely * sy
                dy_w = relx * sy + rely * cy

                right = ego_tf.get_right_vector()
                fwd_proj = dx_w * float(fwd.x) + dy_w * float(fwd.y)
                lat_proj = dx_w * float(right.x) + dy_w * float(right.y)

                front = fwd_proj > 0.0
                if np.any(front):
                    dist2 = dist[front]
                    fwd2 = fwd_proj[front]
                    lat2 = lat_proj[front]
                    j = int(np.argmin(dist2))
                    nearest_dist = float(dist2[j])
                    nearest_fwd = float(fwd2[j])
                    nearest_lat = float(lat2[j])
                else:
                    nearest_dist, nearest_fwd, nearest_lat = _fallback_nearest_world()
            else:
                nearest_dist, nearest_fwd, nearest_lat = _fallback_nearest_world()
        else:
            nearest_dist, nearest_fwd, nearest_lat = _fallback_nearest_world()

        # ----------------- obstacle penalties (删掉 r_obs_sep，避免刷奖励抖动) -----------------
        r_obs_clear = 0.0
        r_obs_speed = 0.0

        if (nearest_dist is not None) and (nearest_fwd is not None):
            # 前向门控：越近越强
            g = (AVOID_FWD - float(nearest_fwd)) / max(AVOID_FWD, 1e-6)
            g = float(np.clip(g, 0.0, 1.0))

            # 横向门控：别太严
            lat_gate = 1.0
            if nearest_lat is not None:
                lat_gate = float(np.clip(1.0 - abs(float(nearest_lat)) / max(LAT_TOL, 1e-6), 0.0, 1.0))

            obstacle_gate = g * lat_gate

            # 进入 SAFE_DIST：平方强惩罚（主导安全）
            if nearest_dist < SAFE_DIST:
                x = (SAFE_DIST - float(nearest_dist)) / max(SAFE_DIST, 1e-6)
                r_obs_clear = -W_OBS_CLEAR * obstacle_gate * float(x * x)

            # 近障碍限速：超过 V_CAP_NEAR 就罚
            if obstacle_gate > 0.05:
                over = max(0.0, speed - V_CAP_NEAR)
                r_obs_speed = -W_OBS_SPEED * obstacle_gate * float((over / max(V_CAP_NEAR, 1e-6)) ** 2)

        # ----------------- speed reward (强门控：危险时不给“快”) -----------------
        # 只鼓励低速接近 TARGET_SPEED
        err = abs(speed - TARGET_SPEED) / max(TARGET_SPEED, 1e-3)
        speed_score = float(np.clip(1.0 - err, 0.0, 1.0))
        r_speed = K_SPEED * speed_score

        # 门控：偏离车道/近障碍 -> 速度奖励变小
        # ✅ 给一个最低门槛，避免奖励被完全“熄火”
        # ✅ 门控别太狠：避免奖励完全“熄火”，导致收敛到龟速
        safety_gate = float(np.clip(1.0 - 0.6 * lane_ratio, 0.0, 1.0))
        safety_gate *= float(np.clip(1.0 - 0.8 * obstacle_gate, 0.0, 1.0))
        safety_gate = max(0.35, safety_gate)
        r_speed *= safety_gate

        # overspeed penalty：只保留一条，更干净
        r_overspeed = 0.0
        if speed > OVERSPEED_START:
            over = (speed - OVERSPEED_START) / max(OVERSPEED_START, 1e-6)
            r_overspeed = -K_OVERSPEED * float(over * over)
        if speed > V_MAX:
            # 超过硬上限再加一层（防止失控）
            r_overspeed -= 0.5 * float(((speed - V_MAX) / max(V_MAX, 1e-6)) ** 2)

        # ----------------- progress (强门控：危险时不奖励“冲”) -----------------
        # 关键：progress 只有在“比较安全”的时候才给，避免为了进度硬撞
        r_progress = K_PROGRESS * prog_fwd
        r_progress *= safety_gate

        # ----------------- 低速惩罚（防止“龟速苟活”） -----------------
        # ✅ 低速惩罚加重：逼迫策略走出“慢速保命”局部最优
        LOW_SPEED_TH = 1.20
        K_LOW_SPEED = 0.12
        r_low_speed = 0.0
        if speed < LOW_SPEED_TH:
            # 低速越接近 0，惩罚越大
            x = (LOW_SPEED_TH - speed) / max(LOW_SPEED_TH, 1e-6)
            r_low_speed = -K_LOW_SPEED * float(x * x)

        # ----------------- terminal checks -----------------
        r_collision = 0.0
        if collision_flag:
            done = True
            done_reason = "collision"
            r_collision = -K_COLLISION_TERMINAL

        r_offroad = 0.0
        if (not done) and offroad:
            done = True
            done_reason = "offroad"
            r_offroad = -K_OFFROAD_TERMINAL

        # ----------------- alive + success -----------------
        r_alive = R_ALIVE if not done else 0.0

        r_success = 0.0
        timeout_flag = bool(getattr(self, "timeout_flag", False))
        if timeout_flag and (not collision_flag) and (not offroad):
            r_success = SUCCESS_BONUS

        # ----------------- total reward -----------------
        components = {
            "r_progress": float(r_progress),
            "r_speed": float(r_speed),
            "r_lane": float(r_lane),
            "r_danger": float(r_danger),
            "r_offroad_soft": float(r_offroad_soft),
            "r_obs_clear": float(r_obs_clear),
            "r_obs_speed": float(r_obs_speed),
            "r_overspeed": float(r_overspeed),
            "r_low_speed": float(r_low_speed),
            "r_no_progress_terminal": float(r_no_progress_term),
            "r_offroad": float(r_offroad),
            "r_collision": float(r_collision),
            "r_alive": float(r_alive),
            "r_success": float(r_success),
        }

        total_reward = float(sum(components.values()))
        self.last_reward_components = components.copy()

        # ----------------- info -----------------
        info = {
            "done": float(done),
            "done_reason": done_reason,
            "collision": float(collision_flag),
            "speed": float(speed),
            "lane_deviation": float(lane_dev),
            "lane_width": float(lane_width),
            "offroad_thresh": float(offroad_thresh),
            "lane_ratio": float(lane_ratio),
            "offroad": float(offroad),
            "progress_fwd": float(prog_fwd),
            "progress_ema": float(getattr(self, "progress_ema", 0.0)),
            "no_progress_steps": float(getattr(self, "no_progress_steps", 0)),
            "nearest_obstacle_dist": float(nearest_dist) if nearest_dist is not None else -1.0,
            "nearest_obstacle_fwd": float(nearest_fwd) if nearest_fwd is not None else 0.0,
            "nearest_obstacle_lat": float(nearest_lat) if nearest_lat is not None else 0.0,
            "obstacle_gate": float(obstacle_gate),
            "safety_gate": float(safety_gate),
        }

        # reset collision latch
        self.collision = False
        return float(total_reward), bool(done), info

    # ----------------- render -----------------
    def _draw_display(self, image, throttle, steer, brake):
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            draw_image(self.screen, image)

            panel_w, panel_h = 280, 120
            panel_x, panel_y = 12, 12
            border_radius = 14
            panel_bg = (15, 18, 22)
            panel_alpha = 150
            bar_w = 6
            planner_color = get_planner_color(self.planner_mode)

            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 0))
            try:
                pygame.draw.rect(
                    panel,
                    (*panel_bg, panel_alpha),
                    pygame.Rect(0, 0, panel_w, panel_h),
                    border_radius=border_radius,
                )
            except TypeError:
                pygame.draw.rect(panel, (*panel_bg, panel_alpha), pygame.Rect(0, 0, panel_w, panel_h))
            pygame.draw.rect(panel, (*planner_color, 220), pygame.Rect(0, 0, bar_w, panel_h),
                             border_radius=border_radius)

            pad_left = 14
            cur_x = bar_w + pad_left
            cur_y = 10
            line_gap = 22

            speed_ms = vector_to_scalar(self.ego.get_velocity())
            speed_kmh = speed_ms * 3.6

            if self.font_big is not None:
                draw_text_with_shadow(
                    panel,
                    self.font_big,
                    f"Planner: {self.planner_mode}",
                    (cur_x, cur_y),
                    color=(240, 240, 240),
                )
            cur_y += 36

            if self.font_small is not None:
                draw_text_with_shadow(panel, self.font_small, f"Throttle: {throttle:.2f}", (cur_x, cur_y))
                cur_y += line_gap
                draw_text_with_shadow(panel, self.font_small, f"Steer:    {steer:.2f}", (cur_x, cur_y))
                cur_y += line_gap
                draw_text_with_shadow(panel, self.font_small, f"Brake:    {brake:.2f}", (cur_x, cur_y))

                sp_text = f"{speed_kmh:5.1f} km/h"
                sp_surf = self.font_small.render(sp_text, True, (220, 220, 220))
                panel.blit(sp_surf, (panel_w - sp_surf.get_width() - 10, panel_h - sp_surf.get_height() - 8))

            self.screen.blit(panel, (panel_x, panel_y))
            pygame.display.flip()

            if self.clock is not None:
                fps = int(round(1.0 / self.fixed_dt))
                self.clock.tick(fps)
        except Exception:
            pass

    def _init_reward_plot(self):
        """初始化 matplotlib reward 可视化（Agg 后端，保存图片）"""
        try:
            self.reward_fig, self.reward_ax = plt.subplots(figsize=(12, 6))

            # 设置样式
            self.reward_ax.set_facecolor('#1e1e1e')
            self.reward_fig.patch.set_facecolor('#2d2d2d')
            self.reward_ax.grid(True, alpha=0.3, linestyle='--')
            self.reward_ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)

            # 设置标签
            self.reward_ax.set_xlabel('Reward Component', color='white', fontsize=10)
            self.reward_ax.set_ylabel('Value', color='white', fontsize=10)
            self.reward_ax.set_title('Reward Components', color='white', fontsize=12, fontweight='bold')

            # 设置刻度颜色
            self.reward_ax.tick_params(colors='white', labelsize=8)

            plt.tight_layout()
            print(f"[CarlaEnv] ✅ Matplotlib reward 可视化已初始化（保存到 {self.reward_save_dir}）")
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ Matplotlib 初始化失败: {e}")
            self.reward_fig = None

    def _update_reward_plot(self):
        """更新 matplotlib reward 可视化并保存图片"""
        if not hasattr(self, 'last_reward_components') or not self.last_reward_components:
            return

        if self.reward_fig is None:
            return

        try:
        # 定义要显示的 reward 组成部分（按重要性排序）
            important_keys = [
            ("base_reward", "Base"),
            ("r_wp", "WP"),
            ("r_speed", "Speed"),
            ("r_low_speed", "Low\nSpeed"),
            ("r_lane", "Lane"),
            ("r_collision", "Collision"),
            ("r_offroad", "Offroad"),
            ("r_alive", "Alive"),
                ("r_danger", "Danger"),
                ("r_obstacle_clear", "Obs\nClear"),
                ("r_obstacle_sep", "Obs\nSep"),
                ("r_obstacle_speed", "Obs\nSpeed"),
                ("r_obstacle_stuck", "Obs\nStuck"),
                ("r_smooth", "Smooth"),
                ("r_mag", "Mag"),
                ("r_steer_speed", "Steer\nSpeed"),
                ("r_idle", "Idle"),
            ]

            # 提取数据
            labels = []
            values = []
            colors = []

            for key, label in important_keys:
                if key in self.last_reward_components:
                    value = self.last_reward_components[key]
                    labels.append(label)
                    values.append(value)

                    # 根据数值选择颜色
                    if abs(value) < 0.0001:
                        colors.append('#666666')  # 灰色
                    elif value > 0:
                        colors.append('#00ff00')  # 绿色
                    else:
                        colors.append('#ff4444')  # 红色

            # 清空并重绘
            self.reward_ax.clear()

            # 绘制柱状图
            bars = self.reward_ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

            # 设置 x 轴标签
            self.reward_ax.set_xticks(range(len(labels)))
            self.reward_ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

            # 在柱子上显示数值
            for i, (bar, value) in enumerate(zip(bars, values)):
                if abs(value) > 0.001:  # 只显示非零值
                    height = bar.get_height()
                    self.reward_ax.text(bar.get_x() + bar.get_width()/2., height,
                                      f'{value:.3f}',
                                      ha='center', va='bottom' if height > 0 else 'top',
                                      fontsize=7, color='white', fontweight='bold')

            # 计算总 reward
            total_reward = sum(values)

            # 重新设置样式
            self.reward_ax.set_facecolor('#1e1e1e')
            self.reward_ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            self.reward_ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)

            # 设置标签和标题
            self.reward_ax.set_xlabel('Reward Component', color='white', fontsize=10)
            self.reward_ax.set_ylabel('Value', color='white', fontsize=10)

            # 标题显示总 reward
            title_color = '#00ff00' if total_reward >= 0 else '#ff4444'
            self.reward_ax.set_title(f'Reward Components | Total: {total_reward:+.4f} | Episode: {self.episode_id} | Step: {self.episode_steps}',
                                    color=title_color, fontsize=12, fontweight='bold')

            # 设置刻度颜色
            self.reward_ax.tick_params(colors='white', labelsize=8)

            # 调整 y 轴范围
            if values:
                y_max = max(max(values), 0.1)
                y_min = min(min(values), -0.1)
                margin = (y_max - y_min) * 0.2
                self.reward_ax.set_ylim(y_min - margin, y_max + margin)

            plt.tight_layout()

            # 保存图片（覆盖同一个文件，实时更新）
            save_path = os.path.join(self.reward_save_dir, "reward_components_latest.png")
            self.reward_fig.savefig(save_path, dpi=100, facecolor='#2d2d2d')

            # 每100步保存一个带时间戳的版本
            if self.episode_steps % 100 == 0:
                timestamped_path = os.path.join(self.reward_save_dir,
                                               f"reward_ep{self.episode_id}_step{self.episode_steps}.png")
                self.reward_fig.savefig(timestamped_path, dpi=100, facecolor='#2d2d2d')

        except Exception as e:
            print(f"[CarlaEnv] ⚠️ Reward plot 更新失败: {e}")

    # ----------------- cleanup -----------------
    def _cleanup_actors(self):
        def _safe_destroy(actor):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        # ✅ 清理场景实例中的 actors
        if self.scenario_instance is not None:
            try:
                self.scenario_instance.cleanup()
                # 给 CARLA 更多时间来处理 actor 销毁（增加 tick 次数和等待时间）
                if self.world.get_settings().synchronous_mode:
                    try:
                        for _ in range(10):  # ✅ 从 5 次增加到 10 次
                            self.world.tick()
                            time.sleep(0.1)  # ✅ 从 0.05 增加到 0.1
                    except Exception:
                        pass
                print(f"[CarlaEnv] ✅ 场景 {self.scenario} 已清理")
            except Exception as e:
                print(f"[CarlaEnv] ⚠️ 场景清理失败: {e}")
            self.scenario_instance = None

        for a in self._actors:
            _safe_destroy(a)
        self._actors = []

        # ✅ 先停止 sensor 监听，再销毁
        if self.camera_display is not None:
            try:
                self.camera_display.stop()
            except Exception:
                pass
        _safe_destroy(self.camera_display)
        self.camera_display = None

        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass
        _safe_destroy(self.collision_sensor)
        self.collision_sensor = None

        _safe_destroy(self.ego)
        self.ego = None
        self.obstacle_actors = []


# ----------------- utils -----------------
def vector_to_scalar(vector):
    return float(np.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2))


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font(size=14):
    try:
        fonts = [x for x in pygame.font.get_fonts()]
        if len(fonts) == 0:
            return pygame.font.Font(None, size)
        default_font = "ubuntumono"
        font_name = default_font if default_font in fonts else fonts[0]
        font_path = pygame.font.match_font(font_name)
        if font_path is None:
            return pygame.font.Font(None, size)
        return pygame.font.Font(font_path, size)
    except Exception:
        return pygame.font.Font(None, size)


def draw_text_with_shadow(surface, font, text, pos, color=(235, 235, 235), shadow=(0, 0, 0), offset=(1, 1)):
    shadow_surf = font.render(text, True, shadow)
    surface.blit(shadow_surf, (pos[0] + offset[0], pos[1] + offset[1]))
    text_surf = font.render(text, True, color)
    surface.blit(text_surf, pos)


def get_planner_color(planner_mode: str):
    mode = (planner_mode or "").upper()
    if mode == "RULE":
        return (64, 220, 190)
    elif mode == "IL":
        return (255, 170, 60)
    elif mode == "RL":
        return (165, 110, 245)
    return (120, 150, 200)
