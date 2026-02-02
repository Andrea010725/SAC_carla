"""
场景生成器 - Corner Cases场景管理
包含5个场景类型：
1. parked_obstacles - 停放车辆避让（已实现）
2. cones - 锥桶场景（待实现）
3. jaywalker - 鬼探头（待实现）
4. trimma - Trimma场景（待实现）
5. construction_lane_change - 施工+变道高交通流（待实现）
"""

import random
import math
from typing import Optional, List, Tuple, Dict, Any
import carla
import sys
from tiny_scenarios_obstacle import ahead_obstacle_scenario
from carla_data_provider import CarlaDataProvider


class ScenarioBase:
    """
    场景基类 - 所有场景必须继承此类

    子类需要实现：
    1. setup() - 场景初始化和障碍物生成
    2. get_spawn_transform() - 返回自车生成位置
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        """
        初始化场景

        Args:
            world: CARLA世界对象
            carla_map: CARLA地图对象
            config: 配置对象
        """
        self.world = world
        self.map = carla_map
        self.config = config

        # 场景生成的actors（需要在reset时清理）
        self.scenario_actors: List[carla.Actor] = []

        # 场景元数据
        self.scenario_name = "base"
        self.scenario_description = "Base scenario"

    def setup(self) -> bool:
        """
        场景初始化 - 生成障碍物、设置环境等

        Returns:
            bool: 是否成功初始化
        """
        raise NotImplementedError("子类必须实现 setup() 方法")

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """
        获取自车生成位置

        Returns:
            carla.Transform: 自车生成的Transform，如果失败返回None
        """
        raise NotImplementedError("子类必须实现 get_spawn_transform() 方法")

    def cleanup(self):
        """清理场景中生成的所有actors"""
        for actor in self.scenario_actors:
            if actor is not None:
                try:
                    actor.destroy()
                except RuntimeError as e:
                    # Actor 可能已经被销毁（例如碰撞后）
                    if "not found" not in str(e):
                        print(f"[Scenario] ⚠️ 清理 actor 失败: {e}")
                except Exception as e:
                    print(f"[Scenario] ⚠️ 清理 actor 失败: {e}")
        self.scenario_actors.clear()

    def get_obstacle_actors(self) -> List[carla.Actor]:
        """
        获取场景中的障碍物actors（用于观测）

        Returns:
            List[carla.Actor]: 障碍物列表
        """
        return self.scenario_actors.copy()

    def get_scenario_info(self) -> Dict[str, Any]:
        """
        获取场景信息（用于日志和调试）

        Returns:
            Dict: 场景信息字典
        """
        return {
            "name": self.scenario_name,
            "description": self.scenario_description,
            "num_actors": len(self.scenario_actors),
        }

    def cleanup(self):
        """通用清理：销毁本场景创建的所有actors"""
        # 1) 先把仍在运动的 actor 停住（尤其是walker）
        for actor in self.scenario_actors:
            if actor is None:
                continue
            try:
                if actor.type_id.startswith("walker.pedestrian"):
                    stop_ctrl = carla.WalkerControl()
                    stop_ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
                    stop_ctrl.speed = 0.0
                    actor.apply_control(stop_ctrl)

            except:
                pass

        # 2) destroy（推荐倒序销毁更安全）
        for actor in reversed(self.scenario_actors):
            if actor is None:
                continue
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass

        # 3) 清空列表
        self.scenario_actors.clear()


# ============================================================================
# 场景1: 停放车辆避让（已实现）
# ============================================================================

class ParkedObstaclesScenario(ScenarioBase):
    """
    停放车辆避让场景

    场景描述：
    - 在自车前方12-20米范围内生成多辆静止车辆
    - 车辆停放在车道中心或侧方
    - 自车需要避让这些车辆

    配置参数：
    - num_parked_cars: 停放车辆数量
    - parked_car_spacing: 车辆间距（米）
    - parked_car_start_distance_min: 第一辆车最小距离（米）
    - parked_car_start_distance_max: 第一辆车最大距离（米）
    - parked_car_offset: 横向偏移（米，0=车道中心）
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "parked_obstacles"
        self.scenario_description = "停放车辆避让场景"

        # 读取配置
        self.num_parked_cars = int(getattr(config, "num_parked_cars", 4))
        self.parked_car_spacing = float(getattr(config, "parked_car_spacing", 8.0))
        self.parked_car_start_distance_min = float(getattr(config, "parked_car_start_distance_min", 12.0))
        self.parked_car_start_distance_max = float(getattr(config, "parked_car_start_distance_max", 20.0))
        self.parked_car_offset = float(getattr(config, "parked_car_offset", 0.0))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None

    def setup(self) -> bool:
        """生成停放车辆场景"""
        # 1. 选择自车生成点
        spawns = self.map.get_spawn_points()
        if not spawns:
            print("[ParkedObstacles] ❌ 地图没有预定义spawn点")
            return False

        self.ego_spawn_transform = random.choice(spawns)
        ego_loc = self.ego_spawn_transform.location

        print(f"\n[ParkedObstacles] 使用spawn点: ({ego_loc.x:.1f}, {ego_loc.y:.1f})")

        # 2. 获取对应的waypoint
        start_wp = self.map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not start_wp:
            print("[ParkedObstacles] ❌ 无法获取waypoint")
            return False

        # 3. 生成停放车辆
        success = self._place_parked_vehicles(start_wp)

        # 4. 等待物理稳定
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        return success

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

    def _place_parked_vehicles(self, start_wp: carla.Waypoint) -> bool:
        """
        生成停放车辆

        Args:
            start_wp: 起始waypoint

        Returns:
            bool: 是否成功生成
        """
        lib = self.world.get_blueprint_library()

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

        print(f"\n[ParkedObstacles] 开始生成{self.num_parked_cars}辆障碍车:")

        # 第一辆车：随机距离
        first_car_distance = random.uniform(
            self.parked_car_start_distance_min,
            self.parked_car_start_distance_max
        )
        print(f"  - 第一辆车目标距离: {first_car_distance:.1f}m")

        # 前进到第一辆车位置
        cur_wp = start_wp
        traveled = 0.0
        step_size = 2.0

        while traveled < first_car_distance:
            nxt = cur_wp.next(step_size)
            if not nxt:
                print(f"    ⚠️ 路径尽头，已前进{traveled:.1f}m")
                break
            cur_wp = nxt[0]
            traveled += step_size

        # 生成第一辆车
        if cur_wp:
            vehicle = self._spawn_single_vehicle(cur_wp, available_bps, 1, traveled)
            if vehicle:
                self.scenario_actors.append(vehicle)

        # 生成后续车辆
        for i in range(2, self.num_parked_cars + 1):
            if not cur_wp:
                break

            # 前进spacing距离
            spacing_traveled = 0.0
            while spacing_traveled < self.parked_car_spacing:
                nxt = cur_wp.next(step_size)
                if not nxt:
                    print(f"    ⚠️ 路径尽头，已前进{spacing_traveled:.1f}m")
                    break
                cur_wp = nxt[0]
                spacing_traveled += step_size

            if cur_wp:
                total_distance = traveled + spacing_traveled
                vehicle = self._spawn_single_vehicle(cur_wp, available_bps, i, total_distance)
                if vehicle:
                    self.scenario_actors.append(vehicle)
                traveled = total_distance

        # 汇总输出
        print(f"\n[ParkedObstacles] 生成完成:")
        print(f"  - 尝试生成: {self.num_parked_cars} 辆")
        print(f"  - 成功生成: {len(self.scenario_actors)} 辆")

        return len(self.scenario_actors) > 0

    def _spawn_single_vehicle(
        self,
        waypoint: carla.Waypoint,
        available_bps: List[carla.ActorBlueprint],
        car_index: int,
        distance_from_ego: float
    ) -> Optional[carla.Actor]:
        """
        在指定waypoint生成单个车辆

        Args:
            waypoint: 生成位置
            available_bps: 可用的车辆blueprints
            car_index: 车辆编号
            distance_from_ego: 距离自车的距离

        Returns:
            carla.Actor: 生成的车辆，失败返回None
        """
        wp_loc = waypoint.transform.location
        wp_rot = waypoint.transform.rotation

        print(f"  - 车辆{car_index}: 距离={distance_from_ego:.1f}m, 位置=({wp_loc.x:.1f}, {wp_loc.y:.1f})")

        # 计算生成位置（考虑横向偏移）
        right_vec = wp_rot.get_right_vector()
        spawn_loc = carla.Location(
            x=wp_loc.x + right_vec.x * self.parked_car_offset,
            y=wp_loc.y + right_vec.y * self.parked_car_offset,
            z=wp_loc.z + 0.5
        )
        spawn_tf = carla.Transform(spawn_loc, wp_rot)

        # 尝试生成
        vehicle_bp = random.choice(available_bps)
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

        # 如果失败，尝试更高位置
        if not vehicle:
            spawn_loc.z = wp_loc.z + 1.0
            spawn_tf = carla.Transform(spawn_loc, wp_rot)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

        if vehicle:
            vehicle.set_simulate_physics(False)  # 静止车辆
            v_loc = vehicle.get_location()
            print(f"    ✅ 成功！ID={vehicle.id}, 位置=({v_loc.x:.1f}, {v_loc.y:.1f})")
            return vehicle
        else:
            print(f"    ❌ 生成失败")
            return None


# ============================================================================
# 场景2: 锥桶场景（待实现）
# ============================================================================

class ConesScenario(ScenarioBase):
    """
    锥桶场景

    场景描述：
    - 在自车车道内放置一系列锥桶
    - 锥桶从车道一侧边缘开始，逐渐向另一侧移动
    - 形成"收窄"或"S形"效果
    - 自车需要横向调整避让锥桶

    配置参数：
    - cone_num: 锥桶数量（默认15）
    - cone_step_behind: 锥桶纵向间距（米，默认3.0）
    - cone_step_lateral: 锥桶横向递进距离（米，默认0.4）
    - cone_z_offset: 锥桶高度偏移（米，默认0.0）
    - cone_lane_margin: 锥桶距离车道边缘的最小距离（米，默认0.25）
    - cone_min_gap_from_junction: 距离路口的最小距离（米，默认15.0）
    - cone_grid: waypoint网格间距（米，默认5.0）
    - spawn_min_gap_from_cone: 自车距离第一个锥桶的距离（米，默认20.0）
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "cones"
        self.scenario_description = "锥桶避让场景"

        # 读取配置参数
        self.cone_num = int(getattr(config, "cone_num", 8))
        self.cone_step_behind = float(getattr(config, "cone_step_behind", 3.0))
        self.cone_step_lateral = float(getattr(config, "cone_step_lateral", 0.4))
        self.cone_z_offset = float(getattr(config, "cone_z_offset", 0.5))
        self.cone_lane_margin = float(getattr(config, "cone_lane_margin", 0.25))
        self.cone_min_gap_from_junction = float(getattr(config, "cone_min_gap_from_junction", 15.0))
        self.cone_grid = float(getattr(config, "cone_grid", 5.0))
        self.spawn_min_gap_from_cone = float(getattr(config, "spawn_min_gap_from_cone", 20.0))

        # ✅ 交通流参数
        self.tm_port = int(getattr(config, "tm_port", 8000))
        self.enable_traffic_flow = bool(getattr(config, "enable_traffic_flow", True))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.first_cone_transform: Optional[carla.Transform] = None
        self.traffic_flow_spawner = None

    def setup(self) -> bool:
        """实现锥桶场景生成"""
        print(f"\n[Cones] 开始生成锥桶场景...")
        print(f"  - 锥桶数量: {self.cone_num}")
        print(f"  - 纵向间距: {self.cone_step_behind}m")
        print(f"  - 横向递进: {self.cone_step_lateral}m")

        # 1. 选择起始waypoint（远离路口）
        start_wp = self._pick_random_start_waypoint()
        if not start_wp:
            print("[Cones] ❌ 无法找到合适的起始位置")
            return False

        print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, "
              f"{start_wp.transform.location.y:.1f})")

        # 2. 放置锥桶
        cones = self._place_cones_conditionally_behind(start_wp)
        if not cones:
            print("[Cones] ❌ 锥桶生成失败")
            return False

        self.scenario_actors.extend(cones)

        # 3. 设置自车spawn位置
        if self.first_cone_transform:
            self.ego_spawn_transform = self._calculate_ego_spawn()
        else:
            print("[Cones] ⚠️ 无法确定自车spawn位置，使用起始waypoint")
            self.ego_spawn_transform = start_wp.transform

        # 4. ✅ 生成交通流
        if self.enable_traffic_flow:
            print(f"[Cones] 开始生成交通流...")
            try:
                # 导入 TrafficFlowSpawner
                import sys
                sys.path.insert(0, '/home/ajifang/DriveAdapter/tools')
                from custom_eval import TrafficFlowSpawner

                # 创建 client
                client = carla.Client("localhost", 2000)
                client.set_timeout(5.0)

                # 创建 TrafficFlowSpawner
                self.traffic_flow_spawner = TrafficFlowSpawner(client, self.world, self.tm_port)

                # 收集锥桶位置作为避让区域
                cone_locs = [c.get_location() for c in cones if c is not None]

                # 生成交通流
                traffic_vehicles = self.traffic_flow_spawner.spawn_high_density_surrounding_flow(
                    base_wp=start_wp,
                    lanes_num=1,
                    opposite_lanes_num=2,
                    enable_same_lane=False,  # 不在当前车道生成（有锥桶）
                    enable_left=True,
                    enable_right=True,
                    enable_opposite=True,
                    density_per_100m=8.0,
                    range_ahead=100.0,
                    range_behind=80.0,
                    speed_diff_pct=20.0,
                    disable_lane_change=True,
                    follow_dist=3.0,
                    ego_loc=self.ego_spawn_transform.location,
                    min_gap_to_ego=20.0,
                    avoid_centers=cone_locs,
                    avoid_radius=15.0,
                    total_spawn_cap=60,
                )

                self.scenario_actors.extend(traffic_vehicles)
                print(f"[Cones] ✅ 交通流生成完成，车辆数量: {len(traffic_vehicles)}")

            except Exception as e:
                print(f"[Cones] ⚠️ 交通流生成失败: {e}")
                import traceback
                traceback.print_exc()

        # 5. 等待物理稳定
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        print(f"[Cones] ✅ 成功生成 {len(cones)} 个锥桶")
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

    def _pick_random_start_waypoint(self) -> Optional[carla.Waypoint]:
        """选择远离路口的随机waypoint"""
        # 生成候选waypoints
        candidates = [
            wp for wp in self.map.generate_waypoints(self.cone_grid)
            if wp.lane_type == carla.LaneType.Driving
        ]

        if not candidates:
            return None

        random.shuffle(candidates)

        # 查找远离路口的waypoint
        max_tries = 300
        for i, wp in enumerate(candidates):
            if i >= max_tries:
                break

            # 检查是否在路口
            if wp.is_junction:
                continue

            # 检查前后是否靠近路口
            if self._is_near_junction(wp, self.cone_min_gap_from_junction):
                continue

            return wp

        # 如果找不到理想位置，返回第一个非路口waypoint
        for wp in candidates:
            if not wp.is_junction:
                return wp

        return candidates[0] if candidates else None

    def _is_near_junction(self, wp: carla.Waypoint, dist: float = 15.0) -> bool:
        """检查waypoint是否靠近路口"""
        step = 1.0

        # 检查前方
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

        # 检查后方
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

    def _place_cones_conditionally_behind(
        self,
        start_wp: carla.Waypoint
    ) -> List[carla.Actor]:
        """
        沿着车道放置锥桶，从一侧边缘逐渐向另一侧移动

        Args:
            start_wp: 起始waypoint

        Returns:
            List[carla.Actor]: 生成的锥桶列表
        """
        lib = self.world.get_blueprint_library()

        # 获取锥桶blueprint
        try:
            cone_bp = lib.find("static.prop.trafficcone01")
        except Exception:
            cone_bp = lib.find("static.prop.trafficcone")

        # 检测左右车道类型
        left_lane_wp = start_wp.get_left_lane()
        right_lane_wp = start_wp.get_right_lane()
        is_left_driving = left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving
        is_right_driving = right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving

        # 决定放置侧
        lateral_multiplier = 1.0
        if is_left_driving and not is_right_driving:
            lateral_multiplier = 1.0  # 从左向右
            print(f"  - 放置策略: 从左侧向右侧移动（右侧是非行车道）")
        elif is_right_driving and not is_left_driving:
            lateral_multiplier = -1.0  # 从右向左
            print(f"  - 放置策略: 从右侧向左侧移动（左侧是非行车道）")
        elif is_left_driving and is_right_driving:
            lateral_multiplier = random.choice([-1.0, 1.0])
            direction = "从左向右" if lateral_multiplier == 1.0 else "从右向左"
            print(f"  - 放置策略: {direction}（两侧都是行车道，随机选择）")

        cones_spawned: List[carla.Actor] = []
        cur_wp = start_wp

        # 放置锥桶
        for i in range(self.cone_num):
            if not cur_wp:
                break

            wp_tf = cur_wp.transform
            right_vec = wp_tf.get_right_vector()
            half_w = cur_wp.lane_width * 0.5

            # 计算横向偏移
            start_offset_signed = (half_w - self.cone_lane_margin) * -lateral_multiplier
            progression_offset = (i * self.cone_step_lateral) * lateral_multiplier
            desired = start_offset_signed + progression_offset

            # 限制在车道范围内
            max_pos = half_w - self.cone_lane_margin
            min_offset = -(half_w - self.cone_lane_margin)
            actual = max(min_offset, min(max_pos, desired))

            # 计算锥桶位置
            cone_loc = carla.Location(
                x=wp_tf.location.x + right_vec.x * actual,
                y=wp_tf.location.y + right_vec.y * actual,
                z=wp_tf.location.z + self.cone_z_offset
            )
            cone_tf = carla.Transform(cone_loc, wp_tf.rotation)

            # 生成锥桶
            cone_actor = self.world.try_spawn_actor(cone_bp, cone_tf)
            if cone_actor:
                cones_spawned.append(cone_actor)
                if self.first_cone_transform is None:
                    self.first_cone_transform = cone_tf

            # ✅ 修改：向前移动到下一个位置（使用next），让锥桶在自车前方
            nxt = cur_wp.next(self.cone_step_behind)
            cur_wp = nxt[0] if nxt else None

        return cones_spawned

    def _calculate_ego_spawn(self) -> Optional[carla.Transform]:
        """计算自车spawn位置（第一个锥桶前方20米）"""
        if not self.first_cone_transform:
            return None

        # 获取第一个锥桶位置对应的waypoint
        cone_wp = self.map.get_waypoint(
            self.first_cone_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not cone_wp:
            return None

        # ✅ 修改：自车spawn在第一个锥桶后方25米，锥桶在自车前方
        traveled = 0.0
        step = 2.0
        wp = cone_wp
        target_distance = 25.0  # 后退25米

        while traveled < target_distance:
            prevs = wp.previous(step)  # 向后移动
            if not prevs:
                break
            wp = prevs[0]
            traveled += step

        tf = wp.transform

        # 关键：车辆 spawn 给一个安全高度，避免底盘/地面穿插
        safe_loc = carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.5)

        # 可选但推荐：清掉 pitch/roll（道路接缝/坡度会让 vehicle spawn 更容易失败）
        safe_rot = carla.Rotation(pitch=0.0, yaw=tf.rotation.yaw, roll=0.0)

        print(f"[Cones] 自车spawn位置: ({safe_loc.x:.1f}, {safe_loc.y:.1f})")
        print(f"[Cones] 第一个锥桶位置: ({self.first_cone_transform.location.x:.1f}, {self.first_cone_transform.location.y:.1f})")

        return carla.Transform(safe_loc, safe_rot)



# ============================================================================
# 场景3: 鬼探头（待实现）
# ============================================================================

class JaywalkerScenario(ScenarioBase):
    """
    鬼探头场景（行人突然横穿马路）

    场景描述：
    - 行人在自车前方一定距离处，从道路一侧横向穿行到另一侧
    - 当自车接近到触发距离时，行人开始移动
    - 考验自车的紧急制动能力和反应速度
    - 可选：添加遮挡物（停放车辆）增加难度

    场景布局：
    ```
    [路边]  |  [车道]  |  [路边]
            |          |
          |          |
       ↓    |          |
       →→→→→|→→→→→→→→→|  (行人横穿)
            |          |
            |        |  (自车接近)
            |    ↑     |
    ```

    配置参数：
    - jaywalker_distance: 行人位置距离自车spawn点（米，默认20.0）
    - jaywalker_speed: 行人移动速度（m/s，默认2.0-3.0）
    - jaywalker_trigger_distance: 触发距离（自车距离多远时行人开始移动，默认15.0）
    - jaywalker_start_side: 行人起始侧（"left"/"right"/"random"，默认"random"）
    - use_occlusion_vehicle: 是否使用遮挡车辆（默认False）
    - occlusion_vehicle_distance: 遮挡车辆距离（米，默认18.0）

    实现要点：
    1. 行人生成：使用 walker.pedestrian.* blueprint
    2. 行人控制：使用 controller.ai.walker 控制器
    3. 触发机制：需要在step中检测自车距离，触发行人移动
    4. 横向移动：计算道路宽度，让行人从一侧移动到另一侧
    5. 遮挡物：可选在行人前方放置停放车辆

    训练价值：
    - 测试紧急制动能力
    - 测试障碍物检测灵敏度
    - 测试反应速度
    - 真实场景常见（城市道路）

    难度：⭐⭐⭐⭐⭐ 非常困难
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "jaywalker"
        self.scenario_description = "鬼探头场景（行人突然横穿）"

        self.jaywalker_distance = float(getattr(config, "jaywalker_distance", 20.0))
        self.jaywalker_speed = float(getattr(config, "jaywalker_speed", 2.0))
        self.jaywalker_trigger_distance = float(getattr(config, "jaywalker_trigger_distance", 15.0))
        self.jaywalker_start_side = str(getattr(config, "jaywalker_start_side", "random"))
        self.use_occlusion_vehicle = bool(getattr(config, "use_occlusion_vehicle", False))
        self.occlusion_vehicle_distance = float(getattr(config, "occlusion_vehicle_distance", 18.0))

        # ✅ 交通流参数
        self.tm_port = int(getattr(config, "tm_port", 8000))
        self.enable_traffic_flow = bool(getattr(config, "enable_traffic_flow", True))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.pedestrian: Optional[carla.Actor] = None
        self.pedestrian_start_location: Optional[carla.Location] = None
        self.pedestrian_target_location: Optional[carla.Location] = None
        self.triggered: bool = False

        self._ped_manual_velocity: bool = True  # ✅ 开启手动速度控制
        self._ped_reach_eps: float = 0.6  # ✅ 到目标点多少米算到达
        self._ped_stop_after_reach: bool = True  # ✅ 到达后停止
        self._ped_vel_vector = None  # ✅ 保存当前速度向量（可选）

        # 记录"碰撞触发点"（用于更精确触发）
        self.trigger_location: Optional[carla.Location] = None
        self.traffic_flow_spawner = None

    # --------------------------
    # 1) 选直道/远离路口 waypoint
    # --------------------------
    def _pick_random_straight_road(self) -> Optional[carla.Waypoint]:
        candidates = [
            wp for wp in self.map.generate_waypoints(5.0)
            if wp.lane_type == carla.LaneType.Driving and (not wp.is_junction)
        ]
        if not candidates:
            return None
        random.shuffle(candidates)

        # 过滤：前后都离路口远一点（避免你 cones 那种前推进 junction）
        safe_gap = 25.0  # 你也可以用 config 控制
        for wp in candidates[:400]:
            if self._is_near_junction(wp, safe_gap):
                continue
            # 直道过滤（yaw变化不大）
            if not self._is_straight_enough(wp, lookahead=15.0, yaw_thresh=15.0):
                continue
            return wp

        return candidates[0]

    def _is_straight_enough(self, wp: carla.Waypoint, lookahead=15.0, yaw_thresh=15.0) -> bool:
        """判断一段路是否近似直道：前方若干米 yaw 变化不超过阈值"""
        base_yaw = wp.transform.rotation.yaw
        cur = wp
        traveled = 0.0
        step = 2.0
        while traveled < lookahead:
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
            traveled += step
            dyaw = abs((cur.transform.rotation.yaw - base_yaw + 180) % 360 - 180)
            if dyaw > yaw_thresh:
                return False
        return True

    def _is_near_junction(self, wp: carla.Waypoint, dist: float = 20.0) -> bool:
        step = 1.0
        # 前方
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
        # 后方
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

    # --------------------------
    # 2) 沿车道前进到指定距离
    # --------------------------
    def _advance_waypoint(self, start_wp: carla.Waypoint, distance: float) -> Optional[carla.Waypoint]:
        traveled = 0.0
        cur = start_wp
        step = 1.0
        while traveled < distance and (not cur.is_junction):
            nxt = cur.next(step)
            if not nxt:
                break
            nxt_wp = nxt[-1]
            traveled += nxt_wp.transform.location.distance(cur.transform.location)
            cur = nxt_wp
        return cur

    # --------------------------
    # 3) 找 sidewalk waypoint（向左或向右找）
    # --------------------------
    def _find_sidewalk_waypoint(self, road_wp: carla.Waypoint, side: str) -> Optional[carla.Waypoint]:
        """
        side: 'left' or 'right'
        """
        cur = road_wp
        for _ in range(10):
            if cur.lane_type == carla.LaneType.Sidewalk:
                return cur
            nxt = cur.get_left_lane() if side == "left" else cur.get_right_lane()
            if nxt is None:
                break
            cur = nxt
        return None

    # --------------------------
    # 4) sidewalk 上生成行人 transform（复用 GhostA 的 get_sidewalk_transform 思路）
    # --------------------------
    def _get_sidewalk_transform(self, sidewalk_wp: carla.Waypoint, face_to_road=True, z_offset=0.5) -> carla.Transform:
        tf = sidewalk_wp.transform
        rot = tf.rotation
        # 让行人朝向道路（鬼探头更自然）
        if face_to_road:
            rot = carla.Rotation(pitch=0.0, yaw=rot.yaw + 270.0, roll=0.0)

        loc = tf.location
        loc = carla.Location(loc.x, loc.y, loc.z + z_offset)
        return carla.Transform(loc, rot)

    def _spawn_pedestrian(self, spawn_tf: carla.Transform) -> Optional[carla.Actor]:
        lib = self.world.get_blueprint_library()

        walker_bps = lib.filter("walker.pedestrian.*")
        if not walker_bps:
            return None
        walker_bp = random.choice(walker_bps)

        ped = self.world.try_spawn_actor(walker_bp, spawn_tf)
        if ped is None:
            return None

        # ✅ 开启物理（可选，建议开）
        try:
            ped.set_simulate_physics(True)
        except:
            pass

        return ped

    # --------------------------
    # 6) （可选）生成遮挡车辆
    # --------------------------
    def _spawn_occlusion_vehicle(self, base_wp: carla.Waypoint, distance_ahead: float = 0.0) -> Optional[carla.Actor]:
        lib = self.world.get_blueprint_library()
        veh_bp = lib.find("vehicle.audi.tt") if lib.find("vehicle.audi.tt") else lib.filter("vehicle.*")[0]

        # 放在“自车车道右侧车道”或“路边”，这里给一个简单实现：放在右侧车道上
        right_wp = base_wp.get_right_lane()
        if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
            right_wp = base_wp

        occ_wp = self._advance_waypoint(right_wp, distance_ahead) if distance_ahead > 0 else right_wp
        tf = occ_wp.transform

        # 稍微抬高，避免 spawn fail
        tf = carla.Transform(tf.location + carla.Location(z=0.5), tf.rotation)

        v = self.world.try_spawn_actor(veh_bp, tf)
        if v:
            v.set_autopilot(False)
        return v

    # --------------------------
    # setup 主逻辑（实现）
    # --------------------------
    def setup(self) -> bool:
        print(f"\n[Jaywalker] 开始生成鬼探头场景...")
        print(f"  - 行人距离: {self.jaywalker_distance}m")
        print(f"  - 行人速度: {self.jaywalker_speed}m/s")
        print(f"  - 触发距离: {self.jaywalker_trigger_distance}m")
        print(f"  - 遮挡车辆: {'是' if self.use_occlusion_vehicle else '否'}")

        # 1) 选直道/远离路口起点
        start_wp = self._pick_random_straight_road()
        if not start_wp:
            print("[Jaywalker] ❌ 找不到合适直道路段")
            return False

        # 2) 自车 spawn 点（就用 start_wp，建议给 z + 0.5，避免你 cones 那种失败）
        ego_tf = start_wp.transform
        ego_tf = carla.Transform(ego_tf.location + carla.Location(z=0.5),
                                 carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0))
        self.ego_spawn_transform = ego_tf

        # ======== 行人从车道线附近横穿（核心修复）========
        ped_road_wp = self._advance_waypoint(start_wp, self.jaywalker_distance)
        if not ped_road_wp:
            print("[Jaywalker] ❌ 无法前进到行人触发位置")
            return False
        wp_tf = ped_road_wp.transform
        right_vec = wp_tf.get_right_vector()
        half_w = ped_road_wp.lane_width * 0.5

        # 行人离车道线的安全距离（别刚好压在线上）
        edge_margin = 0.2  # 你可调：0.2~0.5都行

        # 决定从哪侧出现：left / right
        if self.jaywalker_start_side == "random":
            side = random.choice(["left", "right"])
        else:
            side = self.jaywalker_start_side.lower()
            side = side if side in ["left", "right"] else "right"

        # left: 在左车道线附近 => offset 为负
        # right: 在右车道线附近 => offset 为正
        side_sign = 1.0 if side == "left" else -1.0

        # 起点：在当前车道边界（车道线附近）
        start_offset = side_sign * (half_w + 0.5)

        start_loc = carla.Location(
            x=wp_tf.location.x + right_vec.x * start_offset,
            y=wp_tf.location.y + right_vec.y * start_offset,
            z=wp_tf.location.z + 0.5,  # 关键：+0.5 避免贴地spawn失败
        )

        # 终点：对侧车道线附近
        target_offset = -start_offset
        target_loc = carla.Location(
            x=wp_tf.location.x + right_vec.x * target_offset,
            y=wp_tf.location.y + right_vec.y * target_offset,
            z=wp_tf.location.z + 0.5,
        )

        self.pedestrian_start_location = start_loc
        self.pedestrian_target_location = target_loc

        # 行人初始朝向：面对横穿方向（可选但推荐）
        dx = target_loc.x - start_loc.x
        dy = target_loc.y - start_loc.y
        yaw = math.degrees(math.atan2(dy, dx))  # 指向目标点

        ped_spawn_tf = carla.Transform(
            start_loc,
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        )

        ped = self._spawn_pedestrian(ped_spawn_tf)
        if ped is None:
            print("[Jaywalker] ❌ 行人生成失败")
            return False

        self.pedestrian = ped
        self.pedestrian_controller = None  # ✅ 明确不用controller
        self.triggered = False

        # ✅ 注册 actor（只注册行人）
        self.scenario_actors.append(ped)

        # 同步模式下稳定一下
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        # ✅ 生成交通流
        if self.enable_traffic_flow:
            print(f"[Jaywalker] 开始生成交通流...")
            try:
                import sys
                sys.path.insert(0, '/home/ajifang/DriveAdapter/tools')
                from custom_eval import TrafficFlowSpawner

                client = carla.Client("localhost", 2000)
                client.set_timeout(5.0)
                self.traffic_flow_spawner = TrafficFlowSpawner(client, self.world, self.tm_port)

                # 避让行人位置
                avoid_locs = [self.pedestrian_start_location, self.pedestrian_target_location]

                traffic_vehicles = self.traffic_flow_spawner.spawn_high_density_surrounding_flow(
                    base_wp=start_wp,
                    lanes_num=1,
                    opposite_lanes_num=2,
                    enable_same_lane=True,
                    enable_left=True,
                    enable_right=True,
                    enable_opposite=True,
                    density_per_100m=8.0,
                    range_ahead=100.0,
                    range_behind=80.0,
                    speed_diff_pct=20.0,
                    disable_lane_change=True,
                    follow_dist=3.0,
                    ego_loc=self.ego_spawn_transform.location,
                    min_gap_to_ego=20.0,
                    avoid_centers=avoid_locs,
                    avoid_radius=15.0,
                    total_spawn_cap=60,
                )

                self.scenario_actors.extend(traffic_vehicles)
                print(f"[Jaywalker] ✅ 交通流生成完成，车辆数量: {len(traffic_vehicles)}")

            except Exception as e:
                print(f"[Jaywalker] ⚠️ 交通流生成失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"[Jaywalker] ✅ 场景生成成功")
        print(f"  - ego spawn: ({self.ego_spawn_transform.location.x:.1f}, {self.ego_spawn_transform.location.y:.1f})")
        print(
            f"  - ped spawn: ({self.pedestrian_start_location.x:.1f}, {self.pedestrian_start_location.y:.1f}) side={side}")
        print(f"  - ped target: ({self.pedestrian_target_location.x:.1f}, {self.pedestrian_target_location.y:.1f})")

        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        return self.ego_spawn_transform

    # --------------------------
    def trigger_pedestrian(self):
        if self.triggered:
            return

        if not self.pedestrian or not self.pedestrian_target_location:
            return

        self.triggered = True
        print(f"[Jaywalker] ✅ 行人开始横穿（manual velocity） speed={self.jaywalker_speed:.2f}")

    def check_and_trigger(self, ego_location: carla.Location):
        if self.triggered or not self.pedestrian:
            return

        ped_loc = self.pedestrian.get_location()

        # 这里推荐用“到触发点的距离”而不是到行人距离（更像 GhostA 的 collision_location）
        if self.trigger_location is not None:
            d = math.hypot(ego_location.x - self.trigger_location.x, ego_location.y - self.trigger_location.y)
        else:
            d = math.hypot(ego_location.x - ped_loc.x, ego_location.y - ped_loc.y)

        if d < self.jaywalker_trigger_distance:
            self.trigger_pedestrian()

    def tick_update(self):
        """
        ✅ 每一帧调用一次：用 WalkerControl 推动行人移动（不依赖 AI controller）
        """
        if (not self._ped_manual_velocity) or (not self.triggered):
            return
        if (self.pedestrian is None) or (self.pedestrian_target_location is None):
            return

        try:
            ped_loc = self.pedestrian.get_location()
            tgt = self.pedestrian_target_location

            dx = tgt.x - ped_loc.x
            dy = tgt.y - ped_loc.y
            dist = math.hypot(dx, dy)

            # ✅ 到达目标点：停下
            if dist < self._ped_reach_eps:
                stop_ctrl = carla.WalkerControl()
                stop_ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
                stop_ctrl.speed = 0.0
                self.pedestrian.apply_control(stop_ctrl)
                return

            # ✅ 方向单位化
            ux = dx / (dist + 1e-6)
            uy = dy / (dist + 1e-6)

            # ✅ 速度下限（避免过慢引发不稳定）
            speed = max(0.8, float(self.jaywalker_speed))

            ctrl = carla.WalkerControl()
            ctrl.direction = carla.Vector3D(ux, uy, 0.0)
            ctrl.speed = speed
            ctrl.jump = False

            self.pedestrian.apply_control(ctrl)

        except RuntimeError:
            # actor 无效/被销毁
            return
        except Exception as e:
            print(f"[Jaywalker] ⚠️ tick_update异常: {e}")


# ============================================================================
# 场景4: Trimma场景
# ============================================================================

class TrimmaScenario(ScenarioBase):
    """
    Trimma场景（包围突围）

    场景描述：
    - 自车被其他车辆包围（前左右都有车）
    - 周围车辆以不同速度行驶（有的快有的慢）
    - 自车需要找到合适的gap，借道超车或变道
    - 考验自车的变道决策、超车能力和安全性

    场景布局（俯视图）：
    ```
    [左车道]      [自车车道]      [右车道]
                                
       ↑             ↑              ↑
      慢速          前车            快速
                   (中速)

                    
                    ↑
                  自车
                 (需要超车)

    配置参数：
    - front_vehicle_distance: 前车距离（米，默认 18.0）
    - side_vehicle_offset: 左右车相对自车的纵向偏移（米，默认 +3.0）
        - 3.0 表示左右车在自车前方3米左右（更像夹击）
        - 0.0 表示左右车和自车并排
        - -3.0 表示左右车在自车后方3米
    - min_lane_count: 最少车道数（默认 3，要求左右都存在 Driving Lane）
    - tm_port: Traffic Manager 端口（默认 8000）
    - tm_global_distance: TM 安全车距（默认 2.5m）
    - front_speed_diff_pct: 前车速度差百分比（默认 -20，负数=比限速快）
    - side_speed_diff_pct: 左右车速度差百分比（默认 +30，正数=比限速慢）
    - disable_lane_change: 是否禁止周围车辆变道（默认 True）
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "trimma"
        self.scenario_description = "Trimma场景（左右夹击 + 前车更快）"

        # 读取配置参数
        self.front_vehicle_distance = float(getattr(config, "front_vehicle_distance", 18.0))
        self.side_vehicle_offset = float(getattr(config, "side_vehicle_offset", 3.0))
        self.min_lane_count = int(getattr(config, "min_lane_count", 3))

        self.tm_port = int(getattr(config, "tm_port", 8000))
        self.tm_global_distance = float(getattr(config, "tm_global_distance", 2.5))

        # ✅ 速度差：前车比左右车快一点点，左右车很慢
        self.front_speed_diff_pct = float(getattr(config, "front_speed_diff_pct", +70.0))  # 前车：慢70%
        self.side_speed_diff_pct = float(getattr(config, "side_speed_diff_pct", +80.0))   # 左右车：慢80%

        self.disable_lane_change = bool(getattr(config, "disable_lane_change", True))

        # ✅ 交通流参数
        self.enable_traffic_flow = bool(getattr(config, "enable_traffic_flow", True))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.traffic_manager = None

        # 记录三辆关键车（便于 debug）
        self.front_vehicle: Optional[carla.Actor] = None
        self.left_vehicle: Optional[carla.Actor] = None
        self.right_vehicle: Optional[carla.Actor] = None
        self.traffic_flow_spawner = None

    # --------------------------
    # 1) 选择一个“左右都存在 Driving lane”的中心车道 waypoint（保证 >= 3 lanes）
    # --------------------------
    def _pick_center_lane_waypoint(self) -> Optional[carla.Waypoint]:
        candidates = [
            wp for wp in self.map.generate_waypoints(5.0)
            if wp.lane_type == carla.LaneType.Driving and (not wp.is_junction)
        ]
        if not candidates:
            return None

        random.shuffle(candidates)

        for wp in candidates[:600]:
            left_wp = wp.get_left_lane()
            right_wp = wp.get_right_lane()

            # ✅ 必须左右都存在 Driving lane（至少三车道结构）
            if left_wp is None or right_wp is None:
                continue
            if left_wp.lane_type != carla.LaneType.Driving:
                continue
            if right_wp.lane_type != carla.LaneType.Driving:
                continue

            # 可选：过滤过窄车道
            if wp.lane_width < 3.0:
                continue

            # 可选：前后离路口远一点，避免车刚生成就进 junction
            if self._is_near_junction(wp, dist=25.0):
                continue

            return wp

        return candidates[0]

    def _is_near_junction(self, wp: carla.Waypoint, dist: float = 20.0) -> bool:
        step = 1.0

        # 前方
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

        # 后方
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

    # --------------------------
    # 2) 沿道路前/后移动 waypoint
    # --------------------------
    def _move_along_lane(self, start_wp: carla.Waypoint, dist: float) -> Optional[carla.Waypoint]:
        """
        dist > 0 前进；dist < 0 后退
        """
        if start_wp is None:
            return None

        step = 1.0
        traveled = 0.0
        cur = start_wp

        target = abs(dist)
        forward = dist >= 0

        while traveled < target and (not cur.is_junction):
            nxts = cur.next(step) if forward else cur.previous(step)
            if not nxts:
                break
            nxt_wp = nxts[-1]
            traveled += nxt_wp.transform.location.distance(cur.transform.location)
            cur = nxt_wp

        return cur

    # --------------------------
    # 3) 在某个 waypoint 生成 vehicle（安全 z offset + try 多次）
    # --------------------------
    def _spawn_vehicle_at_waypoint(self, wp: carla.Waypoint) -> Optional[carla.Actor]:
        lib = self.world.get_blueprint_library()

        # 选一个常见车（更容易生成）
        preferred = ["vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2020"]
        bp = None
        for name in preferred:
            try:
                bp = lib.find(name)
                if bp:
                    break
            except:
                pass
        if bp is None:
            bps = lib.filter("vehicle.*")
            if not bps:
                return None
            bp = random.choice(bps)

        tf = wp.transform
        tf = carla.Transform(tf.location + carla.Location(z=0.5), tf.rotation)

        # try_spawn_actor 可能失败，稍微抬高再试几次
        for z_try in [0.5, 0.8, 1.0]:
            tf_try = carla.Transform(wp.transform.location + carla.Location(z=z_try), wp.transform.rotation)
            actor = self.world.try_spawn_actor(bp, tf_try)
            if actor is not None:
                return actor

        return None

    # --------------------------
    # 4) Traffic Manager 速度/行为设置
    # --------------------------
    def _apply_tm_settings(self, veh: carla.Actor, speed_diff_pct: float):
        """
        speed_diff_pct:
        - >0 慢于限速
        - <0 快于限速
        """
        if veh is None or self.traffic_manager is None:
            return

        try:
            veh.set_autopilot(True, self.tm_port)
        except Exception:
            # 某些版本不需要传 port
            try:
                veh.set_autopilot(True)
            except:
                return

        # 禁止变道（防止乱跑）
        if self.disable_lane_change:
            try:
                self.traffic_manager.auto_lane_change(veh, False)
            except:
                pass

        # 设置跟车距离
        try:
            self.traffic_manager.distance_to_leading_vehicle(veh, self.tm_global_distance)
        except:
            pass

        # 设置速度差
        try:
            self.traffic_manager.vehicle_percentage_speed_difference(veh, speed_diff_pct)
        except:
            pass

    # --------------------------
    # setup 主逻辑（完整实现）
    # --------------------------
    def setup(self) -> bool:
        print(f"\n[Trimma] 开始生成 Trimma 场景（左右慢 + 前车快）...")
        print(f"  - 前车距离: {self.front_vehicle_distance}m")
        print(f"  - 左右车偏移: {self.side_vehicle_offset}m (相对自车纵向)")
        print(f"  - 前车速度差: {self.front_speed_diff_pct}% (负=更快)")
        print(f"  - 左右车速度差: {self.side_speed_diff_pct}% (正=更慢)")
        print(f"  - 最少车道数: {self.min_lane_count}")

        # 1) 选中心车道 waypoint（确保左右都有 Driving）
        center_wp = self._pick_center_lane_waypoint()
        if not center_wp:
            print("[Trimma] ❌ 找不到满足条件的多车道中心 waypoint")
            return False

        left_wp = center_wp.get_left_lane()
        right_wp = center_wp.get_right_lane()
        if left_wp is None or right_wp is None:
            print("[Trimma] ❌ 中心 waypoint 左右车道不存在")
            return False
        if left_wp.lane_type != carla.LaneType.Driving or right_wp.lane_type != carla.LaneType.Driving:
            print("[Trimma] ❌ 左右车道不是 Driving lane")
            return False

        # 2) ego spawn（中间车道）
        ego_tf = center_wp.transform
        ego_tf = carla.Transform(
            ego_tf.location + carla.Location(z=0.5),
            carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0)
        )
        self.ego_spawn_transform = ego_tf

        # 3) 获取 Traffic Manager（稳健写法）
        self.traffic_manager = None
        try:
            # 有的版本是 client.get_trafficmanager，这里只能尽量兼容
            self.traffic_manager = self.world.get_traffic_manager()  # 如果你环境支持
        except:
            pass

        if self.traffic_manager is None:
            # 兜底：自己连一个 client 拿 tm
            try:
                client = carla.Client("localhost", 2000)
                client.set_timeout(5.0)
                self.traffic_manager = client.get_trafficmanager(self.tm_port)
            except Exception as e:
                print(f"[Trimma] ⚠️ 获取 Traffic Manager 失败：{e}")
                self.traffic_manager = None

        if self.traffic_manager is None:
            print("[Trimma] ❌ 没有 Traffic Manager，无法设置速度与 autopilot")
            return False

        # 确保 tm 端口一致
        try:
            self.traffic_manager.set_synchronous_mode(self.world.get_settings().synchronous_mode)
        except:
            pass

        # 4) 生成前车（同车道，前方 dist）
        front_wp = self._move_along_lane(center_wp, self.front_vehicle_distance)
        if not front_wp:
            print("[Trimma] ❌ 找不到前车 waypoint")
            return False

        front_vehicle = self._spawn_vehicle_at_waypoint(front_wp)
        if not front_vehicle:
            print("[Trimma] ❌ 前车生成失败")
            return False

        self.front_vehicle = front_vehicle
        self.scenario_actors.append(front_vehicle)

        # 5) 生成左车（左车道，略微前方/并排）
        left_base_wp = self._move_along_lane(left_wp, self.side_vehicle_offset)
        if not left_base_wp:
            print("[Trimma] ❌ 找不到左车 waypoint")
            return False

        left_vehicle = self._spawn_vehicle_at_waypoint(left_base_wp)
        if not left_vehicle:
            print("[Trimma] ❌ 左车生成失败")
            return False

        self.left_vehicle = left_vehicle
        self.scenario_actors.append(left_vehicle)

        # 6) 生成右车（右车道，略微前方/并排）
        right_base_wp = self._move_along_lane(right_wp, self.side_vehicle_offset)
        if not right_base_wp:
            print("[Trimma] ❌ 找不到右车 waypoint")
            return False

        right_vehicle = self._spawn_vehicle_at_waypoint(right_base_wp)
        if not right_vehicle:
            print("[Trimma] ❌ 右车生成失败")
            return False

        self.right_vehicle = right_vehicle
        self.scenario_actors.append(right_vehicle)

        # 7) 设置 TM 行为与速度：前车快，左右慢
        self._apply_tm_settings(front_vehicle, self.front_speed_diff_pct)
        self._apply_tm_settings(left_vehicle, self.side_speed_diff_pct)
        self._apply_tm_settings(right_vehicle, self.side_speed_diff_pct)

        # 同步模式下 tick 稳定一下
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        # ✅ 生成额外的交通流（前后方向）
        if self.enable_traffic_flow:
            print(f"[Trimma] 开始生成额外交通流...")
            try:
                import sys
                sys.path.insert(0, '/home/ajifang/DriveAdapter/tools')
                from custom_eval import TrafficFlowSpawner

                client = carla.Client("localhost", 2000)
                client.set_timeout(5.0)
                self.traffic_flow_spawner = TrafficFlowSpawner(client, self.world, self.tm_port)

                # 避让三辆关键车的位置
                avoid_locs = [
                    front_vehicle.get_location(),
                    left_vehicle.get_location(),
                    right_vehicle.get_location(),
                ]

                traffic_vehicles = self.traffic_flow_spawner.spawn_high_density_surrounding_flow(
                    base_wp=center_wp,
                    lanes_num=1,
                    opposite_lanes_num=2,
                    enable_same_lane=True,
                    enable_left=True,
                    enable_right=True,
                    enable_opposite=True,
                    density_per_100m=8.0,
                    range_ahead=100.0,
                    range_behind=80.0,
                    speed_diff_pct=20.0,
                    disable_lane_change=True,
                    follow_dist=3.0,
                    ego_loc=self.ego_spawn_transform.location,
                    min_gap_to_ego=20.0,
                    avoid_centers=avoid_locs,
                    avoid_radius=20.0,
                    total_spawn_cap=60,
                )

                self.scenario_actors.extend(traffic_vehicles)
                print(f"[Trimma] ✅ 交通流生成完成，车辆数量: {len(traffic_vehicles)}")

            except Exception as e:
                print(f"[Trimma] ⚠️ 交通流生成失败: {e}")
                import traceback
                traceback.print_exc()

        # 8) 打印信息
        ego_loc = self.ego_spawn_transform.location
        f_loc = front_vehicle.get_location()
        l_loc = left_vehicle.get_location()
        r_loc = right_vehicle.get_location()

        print("[Trimma] ✅ 场景生成成功")
        print(f"  - Ego  : ({ego_loc.x:.1f}, {ego_loc.y:.1f})")
        print(f"  - Front: ({f_loc.x:.1f}, {f_loc.y:.1f}) speed_diff={self.front_speed_diff_pct}%")
        print(f"  - Left : ({l_loc.x:.1f}, {l_loc.y:.1f}) speed_diff={self.side_speed_diff_pct}%")
        print(f"  - Right: ({r_loc.x:.1f}, {r_loc.y:.1f}) speed_diff={self.side_speed_diff_pct}%")

        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

# ============================================================================
# 场景5: 施工
# ============================================================================

class ConstructionLaneChangeScenario(ScenarioBase):
    """
        施工封道 + 高密度交通流变道场景

        场景设计：
        - 自车所在车道前方生成施工封道区域（锥桶/水马/杂物/施工人员）
        - 当前车道被迫不可通行 => 自车必须向相邻车道变道绕行
        - 相邻车道存在高密度交通流（gap 小，不容易插入）
        - 训练自车的“找 gap + 安全变道 + 避让施工区”的综合能力

        配置参数：
        - construction_distance: 施工区域距离自车多远开始（米，默认30）
        - construction_length: 施工区域长度（米，默认20）   # 这里主要用于交通流生成范围
        - traffic_density: 相邻车道交通密度（辆/100m，默认3）
        - traffic_speed: 交通流速度（m/s，默认8.0）
        - min_gap_for_lane_change: 最小变道 gap（米，默认12.0）  # 这里只做记录/调试，实际是否变道由你的planner完成
        - construction_type: 施工类型（construction1 / construction2，默认construction1）
        - flow_range: 在施工区前后各生成多少米的交通流（默认80m）
        """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "construction_lane_change"
        self.scenario_description = "施工封道 + 高密度交通流变道场景"

        self.construction_distance = float(getattr(config, "construction_distance", 30.0))
        self.construction_length = float(getattr(config, "construction_length", 20.0))
        self.traffic_density = float(getattr(config, "traffic_density", 3.0))  # 车/100m
        self.traffic_speed = float(getattr(config, "traffic_speed", 8.0))  # m/s
        self.min_gap_for_lane_change = float(getattr(config, "min_gap_for_lane_change", 12.0))
        self.flow_range = float(getattr(config, "flow_range", 80.0))

        # 施工生成器配置
        self.construction_type = str(getattr(config, "construction_type", "construction1"))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.traffic_manager = None
        self.tm_port = int(getattr(config, "tm_port", 8000))

        # ✅ 交通流参数
        self.enable_traffic_flow = bool(getattr(config, "enable_traffic_flow", True))
        self.traffic_flow_spawner = None

        # 记录关键点（可用于 debug / trigger）
        self.construction_location: Optional[carla.Location] = None
        self.adjacent_lane_id: Optional[int] = None

    # ---------------------------------------------------------
    # 选一条“直道 + 至少2车道 + 远离路口”的起点 waypoint
    # ---------------------------------------------------------
    def _pick_multi_lane_straight_road(self) -> Optional[carla.Waypoint]:
        candidates = [
            wp for wp in self.map.generate_waypoints(5.0)
            if wp.lane_type == carla.LaneType.Driving and (not wp.is_junction)
        ]
        if not candidates:
            return None

        random.shuffle(candidates)

        def is_near_junction(wp: carla.Waypoint, dist=30.0) -> bool:
            step = 1.0
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

        def has_adjacent_lane(wp: carla.Waypoint) -> bool:
            l = wp.get_left_lane()
            r = wp.get_right_lane()
            ok_l = l is not None and l.lane_type == carla.LaneType.Driving
            ok_r = r is not None and r.lane_type == carla.LaneType.Driving
            return ok_l or ok_r

        # 找一条：有相邻车道 + 不靠路口
        for wp in candidates[:600]:
            if is_near_junction(wp, 35.0):
                continue
            if not has_adjacent_lane(wp):
                continue
            return wp

        # 实在找不到就退化
        for wp in candidates:
            if has_adjacent_lane(wp):
                return wp
        return candidates[0]

    # ---------------------------------------------------------
    # 沿当前车道前进一定距离（用于找施工位置）
    # ---------------------------------------------------------
    def _advance_waypoint(self, start_wp: carla.Waypoint, distance: float) -> Optional[carla.Waypoint]:
        traveled = 0.0
        cur = start_wp
        step = 1.0
        while traveled < distance and (not cur.is_junction):
            nxt = cur.next(step)
            if not nxt:
                break
            nxt_wp = nxt[-1]
            traveled += nxt_wp.transform.location.distance(cur.transform.location)
            cur = nxt_wp
        return cur

    # ---------------------------------------------------------
    # 在相邻车道生成高密度交通流
    # ---------------------------------------------------------
    def _spawn_dense_traffic_flow(self, lane_wp: carla.Waypoint, center_wp: carla.Waypoint):
        """
        在 lane_wp 这条车道上，围绕 center_wp 位置前后刷车：
        - 密度：traffic_density (辆/100m)
        - 范围：flow_range（前后各 flow_range 米）
        """
        # 间距 = 100 / density
        spacing = max(6.0, 100.0 / max(0.5, self.traffic_density))  # 最小不要太小，避免 spawn 失败
        num_each_side = int(self.flow_range / spacing)

        lib = self.world.get_blueprint_library()
        vehicle_bps = lib.filter("vehicle.*")

        def try_spawn_at_wp(wp: carla.Waypoint):
            bp = random.choice(vehicle_bps)
            tf = wp.transform
            tf = carla.Transform(tf.location + carla.Location(z=0.5), tf.rotation)  # 防止贴地 spawn fail
            v = self.world.try_spawn_actor(bp, tf)
            return v

        spawned: List[carla.Actor] = []

        # 中心点先来一辆（可选）
        center_vehicle = try_spawn_at_wp(center_wp)
        if center_vehicle:
            spawned.append(center_vehicle)

        # 前方刷车
        cur = center_wp
        for _ in range(num_each_side):
            nxt = cur.next(spacing)
            if not nxt:
                break
            cur = nxt[0]
            v = try_spawn_at_wp(cur)
            if v:
                spawned.append(v)

        # 后方刷车
        cur = center_wp
        for _ in range(num_each_side):
            prv = cur.previous(spacing)
            if not prv:
                break
            cur = prv[0]
            v = try_spawn_at_wp(cur)
            if v:
                spawned.append(v)

        # 设置 TM 控制（速度固定、禁止变道）
        if self.traffic_manager:
            for v in spawned:
                try:
                    v.set_autopilot(True, self.tm_port)
                    self.traffic_manager.auto_lane_change(v, False)

                    # TM 的 set_desired_speed 单位是 km/h
                    speed_kmh = float(self.traffic_speed) * 3.6
                    self.traffic_manager.set_desired_speed(v, speed_kmh)

                    # 保持车距（稍微小一点更“难插入”）
                    self.traffic_manager.distance_to_leading_vehicle(v, 4.0)
                except Exception as e:
                    print(f"[ConstructionLaneChange] ⚠️ traffic flow TM设置失败: {e}")

        return spawned

    # ---------------------------------------------------------
    # setup 主逻辑
    # ---------------------------------------------------------
    def setup(self) -> bool:
        print(f"\n[ConstructionLaneChange] 开始生成施工变道场景...")
        print(f"  - 施工距离: {self.construction_distance}m")
        print(f"  - 施工长度: {self.construction_length}m")
        print(f"  - 交通密度: {self.traffic_density} 辆/100m")
        print(f"  - 交通速度: {self.traffic_speed} m/s")
        print(f"  - 最小可插入gap(参考): {self.min_gap_for_lane_change}m")
        print(f"  - 施工类型: {self.construction_type}")

        # 1) 选择合适道路
        start_wp = self._pick_multi_lane_straight_road()
        if not start_wp:
            print("[ConstructionLaneChange] ❌ 找不到合适道路")
            return False

        # 2) 自车 spawn
        ego_tf = start_wp.transform
        ego_tf = carla.Transform(ego_tf.location + carla.Location(z=0.5), ego_tf.rotation)
        self.ego_spawn_transform = ego_tf

        # 3) 获取 traffic manager（用 client 强制一致）
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(5.0)
            self.traffic_manager = client.get_trafficmanager(self.tm_port)
            try:
                self.traffic_manager.set_synchronous_mode(self.world.get_settings().synchronous_mode)
            except:
                pass
        except Exception as e:
            print(f"[ConstructionLaneChange] ⚠️ 获取TrafficManager失败: {e}")
            self.traffic_manager = None

        # 4) 找施工位置 waypoint（在自车前方 construction_distance）
        construction_wp = self._advance_waypoint(start_wp, self.construction_distance)
        if not construction_wp:
            print("[ConstructionLaneChange] ❌ 无法定位施工位置 waypoint")
            return False

        # 5) 施工区生成（复用你的 ahead_obstacle_scenario）
        #    注意：你的施工生成器不返回 actor 列表，所以我们用“前后 actor diff”自动收集
        before_ids = set([a.id for a in self.world.get_actors()])

        scene_cfg = {
            "num_cones": int(max(5, self.construction_length / 3.0)),  # 粗略：长度越长 cones 越多
            "cone_interval": 3,
            "num_garbage": 30,
            "num_workers": 3,
        }
        gen_cfg = {"gen_cfg": self.construction_type}

        try:
            self.construction_location = ahead_obstacle_scenario(
                self.world,
                construction_wp,
                actor_list=[],
                actor_desc=[],
                scene_cfg=scene_cfg,
                gen_cfg=gen_cfg
            )
        except Exception as e:
            print(f"[ConstructionLaneChange] ❌ 施工区生成失败: {e}")
            import traceback
            traceback.print_exc()  # <<< 就加这一行

            return False

        # 施工生成后，检查 static.prop 和 walker 是否存在
        all_actors = self.world.get_actors()
        props = [a for a in all_actors if a.type_id.startswith("static.prop")]
        walkers = [a for a in all_actors if a.type_id.startswith("walker.pedestrian")]

        print("[DEBUG] static.prop count =", len(props))
        print("[DEBUG] walkers count =", len(walkers))

        # 打印离施工点最近的 10 个 static.prop
        if self.construction_location:
            props_sorted = sorted(
                props,
                key=lambda a: a.get_location().distance(self.construction_location)
            )
            for a in props_sorted[:10]:
                d = a.get_location().distance(self.construction_location)
                print(f"[DEBUG] prop near construction: {a.type_id} id={a.id} dist={d:.1f}")

        after_actors = self.world.get_actors()
        new_actors = [a for a in after_actors if a.id not in before_ids]

        # 记录这些新 actor，便于 cleanup
        self.scenario_actors.extend(new_actors)

        print(f"[ConstructionLaneChange] ✅ 施工区生成完成，新actor数量: {len(new_actors)}")
        if self.construction_location:
            print(
                f"[ConstructionLaneChange] 施工位置: ({self.construction_location.x:.1f}, {self.construction_location.y:.1f})")

        # 6) ✅ 使用 TrafficFlowSpawner 生成相邻车道的高密度交通流
        if self.enable_traffic_flow:
            print(f"[ConstructionLaneChange] 开始生成相邻车道交通流...")

            # 找到相邻车道（左或右）
            adjacent_wp = None
            left_wp = start_wp.get_left_lane()
            right_wp = start_wp.get_right_lane()

            if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                adjacent_wp = left_wp
                self.adjacent_lane_id = "left"
                print(f"  - 使用左侧车道生成交通流")
            elif right_wp and right_wp.lane_type == carla.LaneType.Driving:
                adjacent_wp = right_wp
                self.adjacent_lane_id = "right"
                print(f"  - 使用右侧车道生成交通流")
            else:
                print(f"[ConstructionLaneChange] ⚠️ 没有找到相邻车道，跳过交通流生成")

            if adjacent_wp:
                try:
                    import sys
                    sys.path.insert(0, '/home/ajifang/DriveAdapter/tools')
                    from custom_eval import TrafficFlowSpawner

                    client = carla.Client("localhost", 2000)
                    client.set_timeout(5.0)
                    self.traffic_flow_spawner = TrafficFlowSpawner(client, self.world, self.tm_port)

                    # 收集施工区域的障碍物位置作为避让区域
                    avoid_locs = []
                    if self.construction_location:
                        avoid_locs.append(self.construction_location)

                    # 在相邻车道生成交通流
                    traffic_vehicles = self.traffic_flow_spawner.spawn_high_density_surrounding_flow(
                        base_wp=adjacent_wp,
                        lanes_num=1,
                        opposite_lanes_num=0,
                        enable_same_lane=True,
                        enable_left=True,
                        enable_right=True,
                        enable_opposite=False,
                        density_per_100m=self.traffic_density,
                        range_ahead=self.flow_range,
                        range_behind=self.flow_range,
                        speed_diff_pct=20.0,
                        disable_lane_change=True,
                        follow_dist=4.0,
                        ego_loc=self.ego_spawn_transform.location,
                        min_gap_to_ego=20.0,
                        avoid_centers=avoid_locs,
                        avoid_radius=25.0,
                        total_spawn_cap=80,
                    )

                    self.scenario_actors.extend(traffic_vehicles)
                    print(f"[ConstructionLaneChange] ✅ 交通流生成完成，车辆数量: {len(traffic_vehicles)}")

                except Exception as e:
                    print(f"[ConstructionLaneChange] ⚠️ 交通流生成失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 同步模式稳定几帧
        if self.world.get_settings().synchronous_mode:
            for _ in range(5):
                self.world.tick()

        print(f"[ConstructionLaneChange] ✅ 场景生成成功")
        print(f"  - ego spawn: ({self.ego_spawn_transform.location.x:.1f}, {self.ego_spawn_transform.location.y:.1f})")
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        return self.ego_spawn_transform

    def cleanup(self):
        """
        清理施工场景生成的所有actor（cones/水马/垃圾/行人/车流车辆等）
        """
        # 先停 TM 控制车辆（可选）
        for a in self.scenario_actors:
            try:
                if a.type_id.startswith("vehicle"):
                    a.set_autopilot(False)
            except:
                pass

        # 统一销毁
        for a in self.scenario_actors:
            try:
                a.destroy()
            except:
                pass
        self.scenario_actors = []
        self.ego_spawn_transform = None
        self.construction_location = None
        self.adjacent_lane_id = None

# ============================================================================
# 场景7: 车门突然打开场景（从 new_scenarios/vehicle_opens_door.py 转换）
# ============================================================================

class VehicleOpensDoorScenario(ScenarioBase):
    """
    车门突然打开场景（修正版）

    修复点：
    1) 停放车必须在“车道右侧路边”（相对于 parked_wp 的车道坐标系，而不是 ego yaw）
    2) 只选择支持 open_door 的车型；否则你触发了也看不到门动画
    3) 触发逻辑增加可观测 debug，避免你以为触发了但其实没进分支
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "vehicle_opens_door"
        self.scenario_description = "车门突然打开场景"

        self.door_vehicle_distance = float(getattr(config, "door_vehicle_distance", 30.0))
        self.door_trigger_distance = float(getattr(config, "door_trigger_distance", 15.0))

        # 你想要“右前方路边”，默认直接设 right（别用 random）
        self.door_side = str(getattr(config, "door_side", "right")).lower()
        if self.door_side not in ["left", "right", "random"]:
            self.door_side = "right"

        # 触发检查频率（默认每步都允许尝试）
        self._door_attempt_cooldown_steps = int(getattr(config, "door_attempt_cooldown_steps", 1))
        self._door_attempt_step_counter = 0

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.parked_vehicle: Optional[carla.Actor] = None

        self.door_opened: bool = False
        self._actual_side: str = "right"
        self._door_to_open = None  # carla.VehicleDoor enum

    def setup(self) -> bool:
        print(f"\n[VehicleOpensDoor] 开始生成场景...")
        print(f"  - 停放车辆距离: {self.door_vehicle_distance}m")
        print(f"  - 触发距离: {self.door_trigger_distance}m")
        print(f"  - 车门侧/停车侧: {self.door_side}")

        # ✅ 优先使用 XML 预定义位置（仅当地图匹配时）
        use_xml = False
        try:
            from .scenario_xml_parser import get_predefined_spawn_for_scenario
            map_name = self.map.name.split('/')[-1]
            predefined_spawn = get_predefined_spawn_for_scenario("vehicle_opens_door", map_name)
            if predefined_spawn:
                self.ego_spawn_transform = predefined_spawn
                print(f"  - ✅ 使用 XML 预定义位置（地图: {map_name}）")
                use_xml = True
        except Exception as e:
            print(f"  - ⚠️ 无法使用 XML 位置: {e}")
            use_xml = False

        if not use_xml:
            spawns = self.map.get_spawn_points()
            if not spawns:
                print("[VehicleOpensDoor] ❌ 地图没有spawn点")
                return False
            self.ego_spawn_transform = random.choice(spawns)
            print("  - 使用随机 spawn 点")

        # road waypoint
        start_wp = self.map.get_waypoint(
            self.ego_spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if not start_wp:
            print("[VehicleOpensDoor] ❌ 无法获取 start waypoint")
            return False

        print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, {start_wp.transform.location.y:.1f})")

        parked_wp = self._advance_waypoint(start_wp, self.door_vehicle_distance)

        # 决定 side
        if self.door_side == "random":
            self._actual_side = random.choice(["left", "right"])
        else:
            self._actual_side = self.door_side

        # 生成停放车（强制在车道 side 上 & 必须支持开门）
        v = self._spawn_parked_vehicle_strict(
            parked_wp=parked_wp,
            side=self._actual_side
        )
        if not v:
            print("[VehicleOpensDoor] ❌ 停放车辆生成失败（可能该路段太窄或附近碰撞体太多）")
            return False

        self.parked_vehicle = v
        self.scenario_actors.append(self.parked_vehicle)

        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        print("[VehicleOpensDoor] ✅ 场景生成成功")
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        return self.ego_spawn_transform

    def _advance_waypoint(self, wp: carla.Waypoint, distance: float) -> carla.Waypoint:
        traveled = 0.0
        step = 2.0
        while traveled < distance:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            traveled += step
        return wp

    # -----------------------------
    # 核心：强制右侧/左侧 + 必须支持开门
    # -----------------------------
    def _spawn_parked_vehicle_strict(self, parked_wp: carla.Waypoint, side: str) -> Optional[carla.Actor]:
        lib = self.world.get_blueprint_library()

        # 优先挑“更可能支持开门”的车型（减少试错次数）
        preferred = [
            "vehicle.audi.tt",
            "vehicle.audi.a2",
            "vehicle.bmw.grandtourer",
            "vehicle.mercedes.coupe",
            "vehicle.lincoln.mkz_2020",
            "vehicle.dodge.charger_2020",
        ]
        candidates = []
        for m in preferred:
            bp = lib.find(m)
            if bp:
                candidates.append(bp)

        if not candidates:
            candidates = list(lib.filter("vehicle.*"))

        if not candidates:
            return None

        # 车道坐标系：必须用 parked_wp 的 right_vec
        lane_right = parked_wp.transform.get_right_vector()
        road_center = parked_wp.transform.location

        want_right = (side == "right")

        def side_ok(v_loc: carla.Location) -> bool:
            dx = v_loc.x - road_center.x
            dy = v_loc.y - road_center.y
            dot = dx * lane_right.x + dy * lane_right.y
            # dot>0 表示在“车道右侧”
            is_right = (dot > 0)
            return (is_right == want_right)

        # 在多车型 + 多偏移尝试里，找一个：1) 生成成功 2) side 正确 3) 支持开门
        random.shuffle(candidates)

        for bp in candidates[:12]:
            for attempt in range(12):
                lane_width = parked_wp.lane_width

                # 把车推向路边：0.7~1.8 倍 lane_width
                offset = lane_width * (0.7 + 0.1 * attempt)
                if side == "left":
                    offset *= -1

                base = parked_wp.transform.location
                spawn_loc = carla.Location(
                    x=base.x + lane_right.x * offset,
                    y=base.y + lane_right.y * offset,
                    z=base.z + 0.8
                )
                spawn_tf = carla.Transform(spawn_loc, parked_wp.transform.rotation)

                v = self.world.try_spawn_actor(bp, spawn_tf)
                if not v:
                    continue

                # side 校验（不对就销毁重试）
                v_loc = v.get_location()
                if not side_ok(v_loc):
                    try:
                        v.destroy()
                    except Exception:
                        pass
                    continue

                # 停车状态：物理开 + 手刹锁死（更可能看到门动画）
                try:
                    v.set_simulate_physics(True)
                    v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                    v.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    v.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass

                # ✅ 探测是否支持开门（不支持就换车）
                door_enum = self._probe_vehicle_door_support(v, side)
                if door_enum is None:
                    try:
                        v.destroy()
                    except Exception:
                        pass
                    continue

                self._door_to_open = door_enum
                print(
                    f"  - 停放车辆生成: {v.type_id} ID={v.id}, side={side}, "
                    f"pos=({v_loc.x:.1f},{v_loc.y:.1f}), door_enum={door_enum}"
                )
                return v

        return None

    def _probe_vehicle_door_support(self, vehicle: carla.Actor, side: str):
        """
        返回一个可用的 VehicleDoor（比如 FR/RR 或 FL/RL）
        方式：open_door -> tick -> close_door（立刻关回去）
        """
        # API 检查
        if not hasattr(carla, "VehicleDoor"):
            print("[VehicleOpensDoor] ⚠️ 当前 CARLA API 没有 VehicleDoor，无法开门动画")
            return None
        if not hasattr(vehicle, "open_door"):
            return None

        # 右侧优先开前门 FR（开门更明显）
        if side == "right":
            door_list = [carla.VehicleDoor.FR, carla.VehicleDoor.RR]
        else:
            door_list = [carla.VehicleDoor.FL, carla.VehicleDoor.RL]

        for d in door_list:
            try:
                vehicle.open_door(d)
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                # 能关更好，保持初始状态
                if hasattr(vehicle, "close_door"):
                    vehicle.close_door(d)
                    if self.world.get_settings().synchronous_mode:
                        self.world.tick()
                return d
            except Exception:
                continue

        return None

    # -----------------------------
    # 触发开门（你在 env.step 每步调用）
    # -----------------------------
    def check_and_open_door(self, ego_location: carla.Location):
        if not self.parked_vehicle:
            return
        if self.door_opened:
            return

        self._door_attempt_step_counter += 1
        if self._door_attempt_step_counter < self._door_attempt_cooldown_steps:
            return
        self._door_attempt_step_counter = 0

        v_loc = self.parked_vehicle.get_location()
        distance = math.hypot(ego_location.x - v_loc.x, ego_location.y - v_loc.y)

        # ✅ 调试：你可以确认是否真的进入触发判定
        if distance < self.door_trigger_distance + 2.0:
            print(f"[VehicleOpensDoor] debug: dist={distance:.2f}, trigger={self.door_trigger_distance:.2f}, opened={self.door_opened}")

        if distance >= self.door_trigger_distance:
            return

        print(f"[VehicleOpensDoor] ✅ 触发条件满足：dist={distance:.2f} < trigger={self.door_trigger_distance:.2f}")

        # 触发开门
        try:
            if self._door_to_open is None:
                print("[VehicleOpensDoor] ⚠️ 没有可用 door_enum（车型可能不支持），无法开门")
                return

            # 再确保停车状态
            try:
                self.parked_vehicle.set_simulate_physics(True)
                self.parked_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.parked_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception:
                pass

            self.parked_vehicle.open_door(self._door_to_open)

            # tick 一下让动画刷新
            if self.world.get_settings().synchronous_mode:
                self.world.tick()

            self.door_opened = True
            print(f"[VehicleOpensDoor] ✅ 车门已打开：door={self._door_to_open}")

        except Exception as e:
            # 不要置 door_opened=True，让后续还能重试
            print(f"[VehicleOpensDoor] ⚠️ 开门失败（将继续尝试）：{e}")

    def cleanup(self):
        self.door_opened = False
        self._door_to_open = None
        self._door_attempt_step_counter = 0
        super().cleanup()


# ===============================================================
# 场景8: 切入场景（从 new_scenarios/cut_in.py 转换）
# ============================================================================

class CutInScenario(ScenarioBase):
    """
    切入场景

    场景描述：
    - 自车在道路上行驶
    - 相邻车道的车辆突然切入到自车前方
    - 自车需要减速避免碰撞
    - 考验自车的紧急制动能力和车辆检测

    场景布局：
    ```
    [左车道]      [自车车道]
         →→→→→→  |          (切入车辆)
                  |
                  |         (自车)
                  |    ↑
    ```

    配置参数：
    - cutin_vehicle_distance: 切入车辆初始距离（米，默认40.0）
    - cutin_trigger_distance: 触发距离（米，默认30.0）
    - cutin_direction: 切入方向（"left"/"right"/"random"，默认"random"）
    - cutin_vehicle_speed: 切入车辆速度（m/s，默认10.0）
    - cutin_lane_change_distance: 变道距离（米，默认15.0）

    训练价值：
    - 测试紧急制动能力
    - 测试车辆检测和跟踪
    - 测试速度控制
    - 真实场景常见（高速公路）

    难度：⭐⭐⭐⭐ 困难
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "cut_in"
        self.scenario_description = "切入场景"

        # 读取配置参数
        self.cutin_vehicle_distance = float(getattr(config, "cutin_vehicle_distance", 40.0))
        self.cutin_trigger_distance = float(getattr(config, "cutin_trigger_distance", 30.0))
        self.cutin_direction = str(getattr(config, "cutin_direction", "random"))
        self.cutin_vehicle_speed = float(getattr(config, "cutin_vehicle_speed", 10.0))
        self.cutin_lane_change_distance = float(getattr(config, "cutin_lane_change_distance", 15.0))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.cutin_vehicle: Optional[carla.Actor] = None

    def setup(self) -> bool:
        """生成切入场景"""
        print(f"\n[CutIn] 开始生成场景...")
        print(f"  - 切入车辆距离: {self.cutin_vehicle_distance}m")
        print(f"  - 触发距离: {self.cutin_trigger_distance}m")
        print(f"  - 切入方向: {self.cutin_direction}")
        print(f"  - 切入车辆速度: {self.cutin_vehicle_speed}m/s")

        # 1. 选择spawn点（需要多车道）
        spawns = self.map.get_spawn_points()
        if not spawns:
            print("[CutIn] ❌ 地图没有spawn点")
            return False

        # 尝试找到有相邻车道的spawn点
        suitable_spawn = None
        for spawn in random.sample(spawns, min(len(spawns), 20)):
            wp = self.map.get_waypoint(
                spawn.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if wp:
                # 检查是否有相邻车道
                left_lane = wp.get_left_lane()
                right_lane = wp.get_right_lane()
                if (left_lane and left_lane.lane_type == carla.LaneType.Driving) or \
                   (right_lane and right_lane.lane_type == carla.LaneType.Driving):
                    suitable_spawn = spawn
                    break

        if not suitable_spawn:
            print("[CutIn] ⚠️ 找不到合适的多车道位置，使用随机spawn点")
            suitable_spawn = random.choice(spawns)

        self.ego_spawn_transform = suitable_spawn

        # 2. 获取waypoint
        start_wp = self.map.get_waypoint(
            self.ego_spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not start_wp:
            print("[CutIn] ❌ 无法获取waypoint")
            return False

        print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, "
              f"{start_wp.transform.location.y:.1f})")

        # 3. 确定切入方向
        if self.cutin_direction == "random":
            # 检查哪侧有车道
            left_lane = start_wp.get_left_lane()
            right_lane = start_wp.get_right_lane()
            available_directions = []
            if left_lane and left_lane.lane_type == carla.LaneType.Driving:
                available_directions.append("left")
            if right_lane and right_lane.lane_type == carla.LaneType.Driving:
                available_directions.append("right")

            if not available_directions:
                print("[CutIn] ❌ 没有可用的相邻车道")
                return False

            actual_direction = random.choice(available_directions)
        else:
            actual_direction = self.cutin_direction

        print(f"  - 实际切入方向: {actual_direction}")

        # 4. 获取相邻车道
        if actual_direction == "left":
            adjacent_wp = start_wp.get_left_lane()
        else:
            adjacent_wp = start_wp.get_right_lane()

        if not adjacent_wp or adjacent_wp.lane_type != carla.LaneType.Driving:
            print(f"[CutIn] ❌ {actual_direction}侧没有可用车道")
            return False

        # 5. 前进到切入位置
        cutin_wp = self._advance_waypoint(adjacent_wp, self.cutin_vehicle_distance)

        # 6. 生成切入车辆
        self.cutin_vehicle = self._spawn_cutin_vehicle(cutin_wp)

        if not self.cutin_vehicle:
            print("[CutIn] ❌ 切入车辆生成失败")
            return False

        self.scenario_actors.append(self.cutin_vehicle)

        # 7. 设置车辆初始速度（不使用autopilot，直接设置速度）
        # 将速度从 m/s 转换为 CARLA 的速度向量
        forward_vec = cutin_wp.transform.get_forward_vector()
        velocity = carla.Vector3D(
            x=forward_vec.x * self.cutin_vehicle_speed,
            y=forward_vec.y * self.cutin_vehicle_speed,
            z=0.0
        )
        self.cutin_vehicle.set_target_velocity(velocity)

        print(f"  - 切入车辆速度设置: {self.cutin_vehicle_speed}m/s")

        # 8. 等待物理稳定
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        print(f"[CutIn] ✅ 场景生成成功")
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

    def _advance_waypoint(self, wp: carla.Waypoint, distance: float) -> carla.Waypoint:
        """前进指定距离"""
        traveled = 0.0
        step = 2.0
        while traveled < distance:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            traveled += step
        return wp

    def _spawn_cutin_vehicle(self, wp: carla.Waypoint) -> Optional[carla.Actor]:
        """
        生成切入车辆

        Args:
            wp: 生成位置

        Returns:
            carla.Actor: 切入车辆，失败返回None
        """
        lib = self.world.get_blueprint_library()

        # 获取车辆blueprint
        vehicle_bps = lib.filter("vehicle.*")
        vehicle_bp = random.choice(vehicle_bps)

        # 生成车辆
        spawn_tf = wp.transform
        spawn_tf.location.z += 0.5

        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
        if not vehicle:
            # 尝试更高的位置
            spawn_tf.location.z += 0.5
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

        if vehicle:
            v_loc = vehicle.get_location()
            print(f"  - 切入车辆生成: ID={vehicle.id}, 位置=({v_loc.x:.1f}, {v_loc.y:.1f})")

        return vehicle


# ============================================================================
# 场景9: 停车场出口场景（从 new_scenarios/parking_exit.py 转换）
# ============================================================================

class ParkingExitScenario(ScenarioBase):
    """
    停车场出口场景

    场景描述：
    - 自车在道路上行驶
    - 路边停车场有车辆突然驶出
    - 自车需要减速避让
    - 考验自车的侧方车辆检测和速度控制

    场景布局：
    ```
    [停车场]  |  [车道]  |
            |          |  (停车场车辆)
       ↓      |          |
       →→→→→→→|          |  (驶出)
              |        |  (自车接近)
              |    ↑     |
    ```

    配置参数：
    - parking_exit_distance: 停车场距离（米，默认35.0）
    - parking_vehicle_speed: 驶出车辆速度（m/s，默认3.0）
    - parking_trigger_distance: 触发距离（米，默认20.0）
    - parking_side: 停车场侧（"left"/"right"/"random"，默认"random"）

    训练价值：
    - 测试侧方车辆检测
    - 测试速度控制
    - 测试紧急制动
    - 真实场景常见（城市道路）

    难度：⭐⭐⭐ 中等
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "parking_exit"
        self.scenario_description = "停车场出口场景"

        # 读取配置参数
        self.parking_exit_distance = float(getattr(config, "parking_exit_distance", 35.0))
        self.parking_vehicle_speed = float(getattr(config, "parking_vehicle_speed", 3.0))
        self.parking_trigger_distance = float(getattr(config, "parking_trigger_distance", 20.0))
        self.parking_side = str(getattr(config, "parking_side", "random"))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.parking_vehicle: Optional[carla.Actor] = None
        self.triggered: bool = False

    def setup(self) -> bool:
        """生成停车场出口场景"""
        print(f"\n[ParkingExit] 开始生成场景...")
        print(f"  - 停车场距离: {self.parking_exit_distance}m")
        print(f"  - 驶出车辆速度: {self.parking_vehicle_speed}m/s")
        print(f"  - 触发距离: {self.parking_trigger_distance}m")
        print(f"  - 停车场侧: {self.parking_side}")

        # 1. 选择spawn点
        spawns = self.map.get_spawn_points()
        if not spawns:
            print("[ParkingExit] ❌ 地图没有spawn点")
            return False

        self.ego_spawn_transform = random.choice(spawns)

        # 2. 获取waypoint
        start_wp = self.map.get_waypoint(
            self.ego_spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not start_wp:
            print("[ParkingExit] ❌ 无法获取waypoint")
            return False

        print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, "
              f"{start_wp.transform.location.y:.1f})")

        # 3. 前进到停车场位置
        parking_wp = self._advance_waypoint(start_wp, self.parking_exit_distance)

        # 4. 确定停车场侧
        if self.parking_side == "random":
            actual_side = random.choice(["left", "right"])
        else:
            actual_side = self.parking_side

        # 5. 生成停车场车辆
        self.parking_vehicle = self._spawn_parking_vehicle(parking_wp, actual_side)

        if not self.parking_vehicle:
            print("[ParkingExit] ❌ 停车场车辆生成失败")
            return False

        self.scenario_actors.append(self.parking_vehicle)

        # 6. 初始时禁用物理模拟（车辆静止在停车场）
        self.parking_vehicle.set_simulate_physics(False)

        # 7. 等待物理稳定
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        print(f"[ParkingExit] ✅ 场景生成成功")
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

    def _advance_waypoint(self, wp: carla.Waypoint, distance: float) -> carla.Waypoint:
        """前进指定距离"""
        traveled = 0.0
        step = 2.0
        while traveled < distance:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            traveled += step
        return wp

    def _spawn_parking_vehicle(
        self,
        wp: carla.Waypoint,
        side: str
    ) -> Optional[carla.Actor]:
        """
        生成停车场车辆

        Args:
            wp: 停车场位置
            side: 停车场侧（"left"/"right"）

        Returns:
            carla.Actor: 停车场车辆，失败返回None
        """
        lib = self.world.get_blueprint_library()

        # 获取车辆blueprint
        vehicle_bps = lib.filter("vehicle.*")
        vehicle_bp = random.choice(vehicle_bps)

        # 计算停车场位置（路边外侧）
        lane_width = wp.lane_width
        offset = lane_width * 0.5 + 3.0  # 停车场距离车道中心

        right_vec = wp.transform.rotation.get_right_vector()
        if side == "left":
            offset *= -1

        spawn_loc = carla.Location(
            x=wp.transform.location.x + right_vec.x * offset,
            y=wp.transform.location.y + right_vec.y * offset,
            z=wp.transform.location.z + 0.5
        )

        # 调整朝向（垂直于道路）
        spawn_rot = carla.Rotation(
            pitch=wp.transform.rotation.pitch,
            yaw=wp.transform.rotation.yaw + (90 if side == "right" else -90),
            roll=wp.transform.rotation.roll
        )
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)

        # 生成车辆
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
        if not vehicle:
            # 尝试更高的位置
            spawn_loc.z += 0.5
            spawn_tf = carla.Transform(spawn_loc, spawn_rot)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

        if vehicle:
            # 初始设置为静止
            vehicle.set_simulate_physics(False)
            v_loc = vehicle.get_location()
            print(f"  - 停车场车辆生成: ID={vehicle.id}, 位置=({v_loc.x:.1f}, {v_loc.y:.1f}), 侧={side}")

        return vehicle

    def check_and_trigger_exit(self, ego_location: carla.Location):
        """
        检查自车距离并触发车辆驶出（需要在env.step()中调用）

        Args:
            ego_location: 自车当前位置

        使用方法：
        在 carla_env.py 的 step() 方法中添加：
        ```python
        if self.scenario_instance and hasattr(self.scenario_instance, 'check_and_trigger_exit'):
            ego_loc = self.ego.get_location()
            self.scenario_instance.check_and_trigger_exit(ego_loc)
        ```
        """
        if self.triggered or not self.parking_vehicle:
            return

        # 计算距离
        vehicle_loc = self.parking_vehicle.get_location()
        distance = math.hypot(ego_location.x - vehicle_loc.x, ego_location.y - vehicle_loc.y)

        # 如果自车接近到触发距离，启动车辆驶出
        if distance < self.parking_trigger_distance:
            try:
                # 启用物理模拟
                self.parking_vehicle.set_simulate_physics(True)

                # 设置车辆速度（不使用autopilot，直接设置速度）
                # 获取车辆朝向（驶入道路的方向）
                vehicle_transform = self.parking_vehicle.get_transform()
                forward_vec = vehicle_transform.get_forward_vector()

                # 设置目标速度
                velocity = carla.Vector3D(
                    x=forward_vec.x * self.parking_vehicle_speed,
                    y=forward_vec.y * self.parking_vehicle_speed,
                    z=0.0
                )
                self.parking_vehicle.set_target_velocity(velocity)

                self.triggered = True
                print(f"[ParkingExit] ✅ 车辆开始驶出！距离={distance:.1f}m, 速度={self.parking_vehicle_speed}m/s")
            except Exception as e:
                print(f"[ParkingExit] ⚠️ 触发驶出失败: {e}")


# ============================================================================
# 场景工厂 - 根据名称创建场景实例（必须在所有场景类定义之后）
# ============================================================================

class ScenarioFactory:
    """场景工厂 - 根据场景名称创建对应的场景实例"""

    # 场景注册表
    SCENARIOS = {
        "parked_obstacles": ParkedObstaclesScenario,
        "cones": ConesScenario,
        "jaywalker": JaywalkerScenario,
        "trimma": TrimmaScenario,
        "construction_lane_change": ConstructionLaneChangeScenario,
        # ✅ 新增场景（已实现）
        "vehicle_opens_door": VehicleOpensDoorScenario,
        "cut_in": CutInScenario,
        "parking_exit": ParkingExitScenario,
    }

    @staticmethod
    def create_scenario(
        scenario_name: str,
        world: carla.World,
        carla_map: carla.Map,
        config: Any
    ) -> Optional[ScenarioBase]:
        """
        创建场景实例

        Args:
            scenario_name: 场景名称
            world: CARLA世界对象
            carla_map: CARLA地图对象
            config: 配置对象

        Returns:
            ScenarioBase: 场景实例，如果场景名称不存在返回None
        """
        scenario_class = ScenarioFactory.SCENARIOS.get(scenario_name)

        if scenario_class is None:
            print(f"[ScenarioFactory] ❌ 未知场景: {scenario_name}")
            print(f"[ScenarioFactory] 可用场景: {list(ScenarioFactory.SCENARIOS.keys())}")
            return None

        return scenario_class(world, carla_map, config)

    @staticmethod
    def list_scenarios() -> List[str]:
        """列出所有可用的场景名称"""
        return list(ScenarioFactory.SCENARIOS.keys())

    @staticmethod
    def get_scenario_info(scenario_name: str) -> Optional[str]:
        """获取场景描述信息"""
        scenario_class = ScenarioFactory.SCENARIOS.get(scenario_name)
        if scenario_class is None:
            return None

        # 创建临时实例获取描述（不初始化world）
        try:
            return scenario_class.__doc__
        except Exception:
            return None