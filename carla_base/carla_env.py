# carla_base/carla_env.py
import os
import sys
import math
import random
from typing import Optional, Tuple, List
import xml.etree.ElementTree as ET

import gym
import numpy as np
from gym import spaces

from .carla_sync_mode import CarlaSyncMode
from .carla_weather import Weather
CARLA_ROOT = "/home/ajifang/carla"
# ====== 指定 CARLA 安装路径（按你的实际路径）======
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist",
                                "carla-0.9.15-py3.7-linux-x86_64.egg"))

import carla   # noqa: E402
import pygame  # noqa: E402

# 尝试导入 RewardMonitor，如果失败则使用占位类
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reward_monitor import RewardMonitor, RewardMonitorConfig
except ImportError:
    RewardMonitor = None
    RewardMonitorConfig = None


class CarlaEnv(gym.Env):
    """
    训练友好的 Env：
    - 观测：9 维 state（x,y,z,pitch,yaw,roll,acc,ang_vel,vel）
    - 动作：2 维连续 [-1,1] -> (throttle/brake, steer)
    - 渲染：Pygame 显示前视相机 + HUD（Planner 名、控制量、速度）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config, carla_port: int, tm_port: int):
        super().__init__()

        self.config = config
        self.carla_port = int(carla_port)
        self.tm_port = int(tm_port)

        # ==== 训练/观测参数（先于 spaces）====
        self.max_episode_steps = int(getattr(config, "max_episode_steps", 2000))
        self.obs_dim = int(getattr(config, "obs_dim", 9))
        self.action_dim = int(getattr(config, "action_dim", 2))

        # ==== Gym 空间 ====
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # ==== 其他配置 ====
        self.observations_type = str(getattr(config, "observations_type", "state"))
        assert self.observations_type == "state", "当前实现仅返回 9 维 state。"

        # 渲染/同步
        self.render_display = bool(getattr(config, "render", True))
        self.fixed_dt = float(getattr(config, "fixed_dt", 0.05))  # 20 FPS
        self.map_name = str(getattr(config, "map_name", "Town05"))

        # 车辆与交通
        self.vehicle_name = str(getattr(config, "vehicle_name", "tesla.cybertruck"))
        self.traffic = bool(getattr(config, "traffic", False))

        # HUD 显示的 Planner 名
        self.planner_mode = str(getattr(config, "planner_mode", "RL")).upper()

        # ==== CARLA 连接 ====
        self.client: carla.Client = carla.Client("127.0.0.1", self.carla_port)
        self.client.set_timeout(60.0)  # 增加超时时间到60秒（切换大地图需要更长时间）

        # 初始世界 - 先获取当前世界，如果地图不对再加载
        self.world: carla.World = self.client.get_world()
        current_map_name = self.world.get_map().name
        if self.map_name not in current_map_name:
            print(f"[CarlaEnv] 当前地图 {current_map_name} != 期望地图 {self.map_name}，正在加载...")
            self.world = self.client.load_world(self.map_name)
        self.map: carla.Map = self.world.get_map()

        # 同步设置
        self._apply_sync_settings(self.fixed_dt)

        # Traffic Manager - 带重试机制
        self.tm = None
        tm_success = False
        tm_ports_to_try = [self.tm_port, self.tm_port + 1, self.tm_port + 500, 8500, 8501, 8502]

        for try_port in tm_ports_to_try:
            try:
                self.tm = self.client.get_trafficmanager(try_port)
                self.tm.set_synchronous_mode(True)
                self.tm_port = try_port  # 更新实际使用的端口
                tm_success = True
                break
            except RuntimeError as e:
                if "bind error" in str(e):
                    continue
                else:
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
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font_big = get_font(size=24)
            self.font_small = get_font(size=14)
            self.clock = pygame.time.Clock()

        # ====== XML/地图来源 ======
        self.xml_file = getattr(config, "xml_file", "/home/ajifang/czw/RL_selector/env/waypoints.xml")
        self.xml_dir = getattr(config, "xml_dir", None)
        self.randomize_town = bool(getattr(config, "randomize_town", False))
        self.town_pool = list(getattr(config, "town_pool", ["Town01", "Town03", "Town05"]))

        # ====== cones 参数 ======
        self.cone_num = int(getattr(config, "cone_num", 10))
        self.cone_step_behind = float(getattr(config, "cone_step_behind", 3.0))
        self.cone_step_lateral = float(getattr(config, "cone_step_lateral", 0.35))
        self.cone_z_offset = float(getattr(config, "cone_z_offset", 0.0))
        self.cone_min_gap_from_junction = float(getattr(config, "cone_min_gap_from_junction", 15.0))
        self.cone_grid = float(getattr(config, "cone_grid", 5.0))
        self.cone_lane_margin = float(getattr(config, "cone_lane_margin", 0.25))

        # ====== 自车 spawn 控制 ======
        self.spawn_wp_step = float(getattr(config, "spawn_wp_step", 2.0))  # m
        self.spawn_min_gap_from_cone = float(getattr(config, "spawn_min_gap_from_cone", 20.0))  # m

        # 🔧 新增：停车障碍场景参数（模拟 Overtaking）
        self.num_parked_cars = int(getattr(config, "num_parked_cars", 4))
        self.parked_car_spacing = float(getattr(config, "parked_car_spacing", 50.0))
        self.parked_car_offset = float(getattr(config, "parked_car_offset", 1.8))
        self.parked_car_start_distance = float(getattr(config, "parked_car_start_distance", 30.0))

        # ==== 统计 & 管理 ====
        self.episode_steps = 0
        self.collision = False
        self._actors: List[carla.Actor] = []  # 场景临时 actor

        # ====== Reward 监控器相关 ======
        self.reward_monitor: Optional[RewardMonitor] = None
        self.episode_id: int = 0
        self.last_control: Optional[carla.VehicleControl] = None


    # ----------------- 公共：同步设置 & 切图 -----------------
    def _apply_sync_settings(self, fixed_dt: float):
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = fixed_dt
        self.world.apply_settings(s)

    def _load_map_if_needed(self, town: str):
        """若切换地图就先切，再进行场景搭建"""
        town = str(town).strip()
        cur = self.world.get_map().name if self.world and self.world.get_map() else ""
        if town and town != cur:
            print(f"[CarlaEnv] Loading map: {town} (was {cur})")
            # 清理现有 actor / 传感器 / sync_mode
            self._cleanup_actors()
            if self.sync_mode is not None:
                try:
                    if hasattr(self.sync_mode, '_settings') and self.sync_mode._settings is not None:
                        self.world.apply_settings(self.sync_mode._settings)
                except Exception:
                    pass
                self.sync_mode = None

            # 切图
            self.world = self.client.load_world(town)
            self.map = self.world.get_map()

            # 恢复同步/TM/天气
            self._apply_sync_settings(self.fixed_dt)
            self.tm = self.client.get_trafficmanager(self.tm_port)
            self.tm.set_synchronous_mode(True)
            self.weather = Weather(self.world, float(getattr(self.config, "changing_weather_speed", 0.0)))

    # 允许外部动态切换 HUD 上显示的 planner 名
    def set_planner_mode(self, name: str):
        self.planner_mode = str(name).upper()

    # ----------------- 核心接口 -----------------
    def reset(self):
        self._cleanup_actors()

        # 清理旧的 sync_mode
        if self.sync_mode is not None:
            try:
                if hasattr(self.sync_mode, '_settings') and self.sync_mode._settings is not None:
                    self.world.apply_settings(self.sync_mode._settings)
            except Exception:
                pass
            self.sync_mode = None

        # 先决定是否切图（基于 XML 或随机）
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
            if self.randomize_town and self.town_pool:
                self._load_map_if_needed(random.choice(self.town_pool))

        # 搭场景并选择自车 spawn
        spawn_tf = self._maybe_setup_scene_and_pick_spawn()

        # 生成自车
        self.ego = self._spawn_ego_with_transform(spawn_tf)
        if self.ego is None:
            raise RuntimeError("spawn ego failed；请确认地图有可用 spawn 点。")

        # 传感器：相机 + 碰撞
        bp_lib = self.world.get_blueprint_library()
        if self.render_display:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", "800")
            cam_bp.set_attribute("image_size_y", "600")
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

        # CarlaSyncMode
        fps = int(round(1.0 / self.fixed_dt))
        if self.render_display and self.camera_display is not None:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, fps=fps)
        else:
            self.sync_mode = CarlaSyncMode(self.world, fps=fps)

        # 🔧 先设置档位和初始控制，再tick
        self.ego.set_autopilot(False)
        init_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        init_control.gear = 1  # 🔧 设置为前进档
        init_control.manual_gear_shift = True  # 🔧 启用手动档位
        self.ego.apply_control(init_control)

        # 🔧 重要：让车辆落到地面（如果spawn在空中）
        # 检查是否在空中（z > 0.2米）
        ego_loc = self.ego.get_location()
        if ego_loc.z > 0.2:
            # 释放刹车，让车辆自由落体
            fall_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)
            fall_control.gear = 1
            fall_control.manual_gear_shift = True
            self.ego.apply_control(fall_control)

            # Tick直到着地（最多10次）
            for _ in range(10):
                self._tick_once()
                new_loc = self.ego.get_location()
                # 如果z坐标稳定（变化<0.01米），说明已着地
                if abs(new_loc.z - ego_loc.z) < 0.01:
                    break
                ego_loc = new_loc

            # 着地后重新应用刹车
            self.ego.apply_control(init_control)

        # 初始一帧（让档位设置生效）
        self._tick_once()

        # ========== ✅ 在这里开始加入 waypoint 相关初始化 ==========
        # 当前自车位置 & 所在 waypoint
        ego_loc = self.ego.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True)
        # 每次沿着道路向前看的距离（单位：米），比如 5 米一段
        self.wp_step_dist = 5.0

        # 判定“到达当前目标 waypoint”的阈值（距离 < 2m 就认为到了）
        self.wp_reach_thresh = 2.0
        # 使用 CARLA API 获取“前方一段距离”的 waypoint
        next_wps = ego_wp.next(self.wp_step_dist)
        if len(next_wps) > 0:
            self.target_wp = next_wps[0]
        else:
            # 如果前方没有 next（少见），就用当前 wp 顶上，防止 None
            self.target_wp = ego_wp

            # 计算当前到目标 waypoint 的距离，并存为“上一步”的距离
            target_loc = self.target_wp.transform.location
            self.prev_wp_dist = math.hypot(
                ego_loc.x - target_loc.x,
                ego_loc.y - target_loc.y,
            )
        # 初始化“连续低速步数”计数器（防止站桩）
        self.idle_steps = 0

        # ========== ✅ waypoint 初始化结束 ==========

        self.episode_steps = 0
        self.collision = False
        self.last_control = init_control

        # ====== 初始化 / 重置 RewardMonitor ======
        self.episode_id += 1

        # 保存路径：record_path/reward_logs/ep_xxxx
        save_root = getattr(self.config, "record_path", "./logs")

        if RewardMonitor is not None and RewardMonitorConfig is not None:
            # 🏎️ 检查是否有自定义reward配置（激进版）
            if hasattr(self.config, "reward_config") and self.config.reward_config is not None:
                # 使用自定义配置（激进版）
                monitor_cfg = self.config.reward_config
                # 更新save_dir和enable_vis（可能被自定义配置覆盖）
                monitor_cfg.save_dir = os.path.join(save_root, "reward_logs")
                monitor_cfg.enable_vis = self.render_display
            else:
                # 使用默认配置
                monitor_cfg = RewardMonitorConfig(
                    enable_vis=self.render_display,
                    save_dir=os.path.join(save_root, "reward_logs")
                )

            if self.reward_monitor is None:
                # 第一次创建
                self.reward_monitor = RewardMonitor(
                world=self.world,
                ego_vehicle=self.ego,
                config=monitor_cfg,
            )
            else:
                # 地图可能切换过，需要更新 world/ego & config
                self.reward_monitor.world = self.world
                self.reward_monitor.ego = self.ego
                self.reward_monitor.cfg = monitor_cfg

            # 重置当前 episode 的历史记录
            self.reward_monitor.reset(episode_id=self.episode_id)

        return self._get_state_obs()


    def step(self, action):
        a0 = float(action[0]); a1 = float(action[1])
        if a0 >= 0.0:
            throttle, brake = np.clip(a0, 0.0, 1.0), 0.0
        else:
            throttle, brake = 0.0, np.clip(-a0, 0.0, 1.0)
        steer = float(np.clip(a1, -1.0, 1.0))

        # 创建control对象并应用
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        control.gear = 1  # 🔧 确保始终在前进档
        control.manual_gear_shift = True  # 🔧 启用手动档位
        self.ego.apply_control(control)

        # ✅ 更新last_control，用于reward计算
        self.last_control = control

        snapshot, display_image = self._tick_once()

        if self.render_display and display_image is not None:
            self._draw_display(display_image, throttle, steer, brake)

        next_obs = self._get_state_obs()
        reward, done, info = self._get_reward()

        self.episode_steps += 1

        timeout = (self.episode_steps >= self.max_episode_steps)
        # success = self._check_overtake_success()
        # offroad = self._check_offroad()
        done = done or timeout  #or offroad  #or success
        info["collision"] = float(self.collision)
        info["timeout"] = float(timeout)
        # info["success"] = float(success)
        # info["offroad"] = float(offroad)

        return next_obs, reward, done, info

    def render(self, mode="human"):
        return None

    def close(self):
        self._cleanup_actors()
        try:
            pygame.quit()
        except Exception:
            pass

    # ----------------- Tick/Spawn/Scene -----------------
    def _tick_once(self):
        if self.sync_mode is None:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
            return None, None

        ret = self.sync_mode.tick(timeout=2.0)
        if isinstance(ret, (list, tuple)) and len(ret) > 0:
            snapshot = ret[0]
            image = ret[1] if len(ret) > 1 else None
        else:
            snapshot, image = None, None

        if self.weather is not None:
            self.weather.tick()
        return snapshot, image

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

        # 如果这个点失败了，退回随机点兜底
        spawns = self.map.get_spawn_points()
        random.shuffle(spawns)
        for alt in spawns:
            # 🔧 不修改z坐标，使用spawn point原始高度
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
                x=tf.location.x - fx*back + rx*side,
                y=tf.location.y - fy*back + ry*side,
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
                x=tf.location.x - fx*back + rx*side,
                y=tf.location.y - fy*back + ry*side,
                z=tf.location.z + height
            )
            spec.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)))

    # ----------------- XML & 场景搭建 -----------------
    def _pick_random_xml_file(self) -> Optional[str]:
        # 优先用 xml_file；否则在 xml_dir 随机挑
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
        """
        返回 (town, waypoints列表)。优先从 <route town="TownXX"> 取 town；
        若没有，则返回空字符串，沿用当前地图。
        waypoints: [{'x','y','z','yaw','pitch','roll'}]
        """
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

    def _spawn_tf_from_first_cone(self) -> Optional[carla.Transform]:
        """
        基于 first_cone_tf，沿车道中心向“上游(previous)”回溯，
        直到满足最小纵向间距，再用该路点作为自车 spawn。
        """
        if self._first_cone_tf is None:
            return None
        amap = self.world.get_map()
        cone_wp = amap.get_waypoint(self._first_cone_tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
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

        tf = wp.transform
        # 🔧 不修改z坐标，使用waypoint原始高度
        return tf

    def _maybe_setup_scene_and_pick_spawn(self) -> carla.Transform:
        """
        选择自车初始位姿：
        1) 若配置给了 initial_spawn_tf -> 直接用；
        2) cones_xml：从 XML 随机 waypoint 搭锥桶；基于锥桶回溯选 spawn；
        3) cones：随机起点搭锥桶；基于锥桶回溯选 spawn；
        4) 否则随机 spawn 点。
        注意：切图已在 reset() 最前面处理，这里不做 load_world。
        """
        # 1) initial_spawn_tf
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

        # 2) cones_xml：已在当前地图上
        if self.scenario == "cones_xml":
            xml_path = self._pick_random_xml_file()
            if xml_path:
                _, wps = self._parse_xml_waypoints(xml_path)
                if wps:
                    pick = random.choice(wps)
                    start_loc = carla.Location(x=pick["x"], y=pick["y"], z=pick["z"])
                    start_wp = self.map.get_waypoint(start_loc, project_to_road=True,
                                                     lane_type=carla.LaneType.Driving)
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
                            try:
                                d = self._first_cone_tf.location.distance(tf.location)
                                print(f"[CarlaEnv] spawn-from-xml-cone: gap ~ {d:.1f} m (>= {self.spawn_min_gap_from_cone:.1f})")
                            except Exception:
                                pass
                            return tf

        # 3) cones：老逻辑随机起点
        if self.scenario == "cones":
            start_wp = self._pick_random_start_waypoint(
                min_gap_from_junction=self.cone_min_gap_from_junction,
                grid=self.cone_grid
            )
            if start_wp:
                cones_spawned, first_tf, last_tf = self._place_cones_conditionally_behind(
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

        # 🔧 新增：4) parked_obstacles - 停车障碍场景（模拟 Overtaking）
        if self.scenario == "parked_obstacles":
            start_wp = self._pick_random_start_waypoint(
                min_gap_from_junction=15.0,
                grid=5.0
            )
            if start_wp:
                # 放置停车障碍
                parked_vehicles = self._place_parked_vehicles(start_wp)

                # 同步
                if self.world.get_settings().synchronous_mode:
                    for _ in range(3):
                        self.world.tick()

                # 自车spawn在起始点
                ego_spawn = start_wp.transform
                # 🔧 不修改z坐标，使用waypoint原始高度
                return ego_spawn

        # 4) 兜底：随机 spawn
        spawns = self.map.get_spawn_points()
        if not spawns:
            return carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
        tf = random.choice(spawns)
        # 🔧 不修改z坐标，使用spawn point原始高度
        return tf

    # --- 随机起点（非 XML） ---
    def _pick_random_start_waypoint(self, min_gap_from_junction: float = 15.0, grid: float = 5.0, max_tries: int = 300):
        amap = self.world.get_map()
        cands = [wp for wp in amap.generate_waypoints(grid) if wp.lane_type == carla.LaneType.Driving]
        if not cands:
            return None
        random.shuffle(cands)

        def _is_near_junction(wp: carla.Waypoint, dist: float = 15.0, step: float = 1.0):
            cur = wp; traveled = 0.0
            while traveled < dist:
                nxt = cur.next(step)
                if not nxt: break
                cur = nxt[0]; traveled += step
                if cur.is_junction: return True
            cur = wp; traveled = 0.0
            while traveled < dist:
                prv = cur.previous(step)
                if not prv: break
                cur = prv[0]; traveled += step
                if cur.is_junction: return True
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

    def _place_cones_conditionally_behind(self,
                                          start_wp: carla.Waypoint,
                                          num_cones: int = 10,
                                          step_behind: float = 3.0,
                                          step_lateral_per_cone: float = 0.35,
                                          z_offset: float = 0.0,
                                          lane_margin: float = 0.25):
        world = self.world
        lib = world.get_blueprint_library()
        try:
            cone_bp = lib.find("static.prop.trafficcone01")
        except Exception:
            cone_bp = lib.find("static.prop.trafficcone")

        # 方向：优先把锥桶从“靠近路肩的一侧”往另一侧推进
        left_lane_wp = start_wp.get_left_lane()
        right_lane_wp = start_wp.get_right_lane()
        is_left_driving = left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving
        is_right_driving = right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving

        lateral_multiplier = 1.0  # +右 -左
        if is_left_driving and not is_right_driving:
            lateral_multiplier = 1.0  #-1.0
        elif is_right_driving and not is_left_driving:
            lateral_multiplier = -1.0  #1.0
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
            if prv:
                cur_wp = prv[0]
            else:
                cur_wp = None

        self._first_cone_tf = first_tf
        self._last_cone_tf = last_tf
        self._actors.extend(cones_spawned)

        return cones_spawned, first_tf, last_tf

    # 🔧 新增：放置停车障碍车辆（模拟 Overtaking ParkedObstacle 场景）
    def _place_parked_vehicles(self, start_wp: carla.Waypoint):
        """
        沿着路线放置停车障碍车辆

        Args:
            start_wp: 起始waypoint

        Returns:
            List[carla.Actor]: 生成的停车车辆列表
        """
        world = self.world
        lib = world.get_blueprint_library()

        # 获取车辆蓝图（优先使用常见车型）
        vehicle_bps = [
            lib.filter('vehicle.tesla.model3'),
            lib.filter('vehicle.audi.a2'),
            lib.filter('vehicle.bmw.grandtourer'),
            lib.filter('vehicle.toyota.prius'),
        ]
        available_bps = [bp for bps in vehicle_bps for bp in bps if bps]
        if not available_bps:
            available_bps = lib.filter('vehicle.*')

        parked_vehicles = []
        cur_wp = start_wp

        # 前进到第一辆车的起始位置
        for _ in range(int(self.parked_car_start_distance / 2.0)):
            next_wps = cur_wp.next(2.0)
            if next_wps:
                cur_wp = next_wps[0]
            else:
                break

        # 放置每辆停车
        for i in range(self.num_parked_cars):
            if not cur_wp:
                break

            # 计算停车位置（右侧路边）
            wp_tf = cur_wp.transform
            right_vec = wp_tf.get_right_vector()

            # 停车横向偏移（正值=右侧）
            parked_loc = wp_tf.location + right_vec * self.parked_car_offset
            parked_loc.z += 0.1  # 稍微抬高避免卡地面

            parked_tf = carla.Transform(parked_loc, wp_tf.rotation)

            # 随机选择车型
            vehicle_bp = random.choice(available_bps)

            # 尝试spawn
            vehicle = world.try_spawn_actor(vehicle_bp, parked_tf)
            if vehicle:
                # 设置为静止（不模拟物理）
                vehicle.set_simulate_physics(False)
                parked_vehicles.append(vehicle)
                self._actors.append(vehicle)  # 记录以便清理

            # 前进到下一个停车位置
            distance_covered = 0.0
            while distance_covered < self.parked_car_spacing:
                next_wps = cur_wp.next(5.0)
                if next_wps:
                    cur_wp = next_wps[0]
                    distance_covered += 5.0
                else:
                    break

        return parked_vehicles

    # ----------------- 观测 / 奖励 -----------------
    def _on_collision(self, event):
        self.collision = True

    def _get_state_obs(self):
        tf = self.ego.get_transform()
        loc, rot = tf.location, tf.rotation
        acc = vector_to_scalar(self.ego.get_acceleration())
        ang = vector_to_scalar(self.ego.get_angular_velocity())
        vel = vector_to_scalar(self.ego.get_velocity())
        return np.array([loc.x, loc.y, loc.z,
                         rot.pitch, rot.yaw, rot.roll,
                         acc, ang, vel], dtype=np.float64)

    # def _get_reward(self):
    #     loc = self.ego.get_location()
    #     wp = self.map.get_waypoint(loc, project_to_road=True)
    #     d = math.hypot(loc.x - wp.transform.location.x, loc.y - wp.transform.location.y)
    #     follow_waypoint_reward = -d
    #
    #     done = bool(self.collision)
    #     collision_reward = -1 if self.collision else 0
    #     total_reward = 100 * follow_waypoint_reward + 100 * collision_reward
    #
    #     info = {
    #         "follow_waypoint_reward": follow_waypoint_reward,
    #         "collision_reward": collision_reward,
    #         "cost": 0.0
    #     }
    #
    #     self.collision = False
    #     return total_reward, done, info

    def _get_reward(self):
        """
        使用 RewardMonitor 计算多目标 reward，并在外面叠加
        overtaking 场景需要的 shaping：
        - 朝着下一 waypoint 前进（核心：r_wp）
        - 鼓励保持合理车速
        - 惩罚长时间龟速（防止站桩 / 原地打转）
        - 碰撞给适度终止惩罚
        """
        # ==== 基础信息 ====
        done = bool(self.collision)

        # 当前车速（m/s）
        vel = self.ego.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        # 当前位置 + 车道中心偏移
        loc = self.ego.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True)
        lane_deviation = math.hypot(
            loc.x - wp.transform.location.x,
            loc.y - wp.transform.location.y
        )

        # 记录“连续低速/静止”的步数（防止站桩）
        if not hasattr(self, "idle_steps"):
            self.idle_steps = 0
        if speed < 0.2:  # 小于 0.2m/s 视为几乎不动
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        # ==== 1. 朝下一 waypoint 的进度奖励 r_wp ====
        r_wp = 0.0
        if hasattr(self, "target_wp") and self.target_wp is not None:
            target_loc = self.target_wp.transform.location
            dist_to_wp = math.hypot(
                loc.x - target_loc.x,
                loc.y - target_loc.y
            )

            # 如果没有 prev_wp_dist，先用当前距离初始化
            if not hasattr(self, "prev_wp_dist"):
                self.prev_wp_dist = dist_to_wp

            # 本帧相对上一帧，距离缩短了多少（>0 表示朝 waypoint 走近）
            delta_dist = self.prev_wp_dist - dist_to_wp
            self.prev_wp_dist = dist_to_wp

            # 奖励系数：1.0 可以先试，太小就放大
            r_wp = 1.0 * delta_dist

            # 如果已经很接近当前 waypoint，则切换到下一个 waypoint
            if dist_to_wp < getattr(self, "wp_reach_thresh", 2.0):
                step_dist = getattr(self, "wp_step_dist", 5.0)
                next_wps = self.target_wp.next(step_dist)
                if len(next_wps) > 0:
                    self.target_wp = next_wps[0]
                    new_target_loc = self.target_wp.transform.location
                    self.prev_wp_dist = math.hypot(
                        loc.x - new_target_loc.x,
                        loc.y - new_target_loc.y
                    )

        # ==== 2. 基于 RewardMonitor 的基础 reward（弱化权重） ====
        base_reward = 0.0
        rm_info = {}

        if self.reward_monitor is not None and self.last_control is not None:
            planner_id_map = {"RULE": 0, "IL": 1, "RL": 2}
            planner_id = planner_id_map.get(self.planner_mode, 2)  # 默认为 RL

            rm_total, comps = self.reward_monitor.update(
                control=self.last_control,
                planner_id=planner_id,
                collision_flag=self.collision,
                done=done,
            )

            # 只当作轻量 shaping，避免动辄 -100 这种尺度
            base_reward = 0.02 * rm_total
            rm_info = {
                "reward_components": comps.to_dict(),
                "planner_id": planner_id,
            }

        # ==== 3. overtaking 专用的其他 shaping ====

        # (a) 车速奖励：鼓励保持中等速度，不要一直 0.x m/s
        target_speed = getattr(self, "target_speed", 10.0)  # 10m/s ≈ 36km/h
        norm_speed = min(speed, target_speed) / max(target_speed, 1e-3)
        r_speed = 0.4 * norm_speed  # 不再 *50，而是 0~0.4 左右

        # (b) 车道偏移惩罚：越偏离中心线惩罚越大
        r_lane = -0.5 * lane_deviation

        # (c) 碰撞惩罚
        r_collision = -5.0 if self.collision else 0.0

        # (d) 长时间龟速惩罚：防止站桩拖到 timeout（包括原地打转但前进很少）
        r_idle = 0.0
        idle_limit = 50
        if self.idle_steps > idle_limit:
            r_idle = -10.0
            done = True  # 直接终止这一局（idle fail）

        # ==== 4. 汇总 reward ====
        total_reward = (
                base_reward +
                r_wp +
                r_speed +
                r_lane +
                r_collision +
                r_idle
        )

        # ==== 5. 组织 info，方便 monitor 统计 ====
        info = {
            **rm_info,
            "collision": float(self.collision),
            "speed": float(speed),
            "lane_deviation": float(lane_deviation),
            "idle_steps": float(self.idle_steps),
            "wp_delta": float(r_wp),  # 本帧朝 waypoint 的进度
            # "wp_dist": float(dist_to_wp)  # 如果上面定义了 dist_to_wp，可以加上
        }

        # 本帧的碰撞已经计入 reward，下一帧清零
        self.collision = False

        return total_reward, done, info






    # ----------------- 渲染 -----------------
    def _draw_display(self, image, throttle, steer, brake):
        try:
            # ✅ 处理pygame事件，防止窗口"无响应"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass  # 忽略关闭事件，由训练循环控制

            draw_image(self.screen, image)

            # HUD 面板
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
                pygame.draw.rect(panel, (*panel_bg, panel_alpha), pygame.Rect(0, 0, panel_w, panel_h),
                                 border_radius=border_radius)
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
                draw_text_with_shadow(panel, self.font_big, f"Planner: {self.planner_mode}",
                                      (cur_x, cur_y), color=(240, 240, 240))
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

    # ----------------- 清理 -----------------
    def _cleanup_actors(self):
        def _safe_destroy(actor):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        # 清理注册的临时 actor（包含锥桶、相机、碰撞等）
        for a in self._actors:
            _safe_destroy(a)
        self._actors = []

        _safe_destroy(self.camera_display); self.camera_display = None
        _safe_destroy(self.collision_sensor); self.collision_sensor = None
        _safe_destroy(self.ego); self.ego = None


# ----------------- 工具函数 -----------------
def vector_to_scalar(vector):
    return float(np.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2))


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # BGRA -> BGR -> RGB
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font(size=14):
    """获取字体,如果系统字体不可用则使用pygame默认字体"""
    try:
        fonts = [x for x in pygame.font.get_fonts()]
        if len(fonts) == 0:
            # 系统字体不可用,使用pygame默认字体
            return pygame.font.Font(None, size)

        default_font = "ubuntumono"
        font_name = default_font if default_font in fonts else fonts[0]
        font_path = pygame.font.match_font(font_name)

        if font_path is None:
            # 如果找不到字体文件,使用默认字体
            return pygame.font.Font(None, size)

        return pygame.font.Font(font_path, size)
    except Exception:
        # 任何错误都使用pygame默认字体
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
