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
        self.cone_num = int(getattr(config, "cone_num", 15))
        self.cone_step_behind = float(getattr(config, "cone_step_behind", 3.0))
        self.cone_step_lateral = float(getattr(config, "cone_step_lateral", 0.4))
        self.cone_z_offset = float(getattr(config, "cone_z_offset", 0.0))
        self.cone_lane_margin = float(getattr(config, "cone_lane_margin", 0.25))
        self.cone_min_gap_from_junction = float(getattr(config, "cone_min_gap_from_junction", 15.0))
        self.cone_grid = float(getattr(config, "cone_grid", 5.0))
        self.spawn_min_gap_from_cone = float(getattr(config, "spawn_min_gap_from_cone", 20.0))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.first_cone_transform: Optional[carla.Transform] = None

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

        # 3. 设置自车spawn位置（第一个锥桶前方20米）
        if self.first_cone_transform:
            self.ego_spawn_transform = self._calculate_ego_spawn()
        else:
            print("[Cones] ⚠️ 无法确定自车spawn位置，使用起始waypoint")
            self.ego_spawn_transform = start_wp.transform

        # 4. 等待物理稳定
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

            # 向后移动到下一个位置
            prv = cur_wp.previous(self.cone_step_behind)
            cur_wp = prv[0] if prv else None

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

        # ✅ 修复：从第一个锥桶向前移动指定距离（使用next而不是previous）
        traveled = 0.0
        step = 2.0
        wp = cone_wp

        while traveled < self.spawn_min_gap_from_cone:
            nexts = wp.next(step)  # ✅ 改为next()，向前移动
            if not nexts:
                break
            wp = nexts[0]
            traveled += step

        print(f"[Cones] 自车spawn位置: ({wp.transform.location.x:.1f}, {wp.transform.location.y:.1f})")
        print(f"[Cones] 第一个锥桶位置: ({self.first_cone_transform.location.x:.1f}, {self.first_cone_transform.location.y:.1f})")

        return wp.transform


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
       🚶   |          |
       ↓    |          |
       →→→→→|→→→→→→→→→|  (行人横穿)
            |          |
            |    🚗    |  (自车接近)
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

        # 读取配置参数
        self.jaywalker_distance = float(getattr(config, "jaywalker_distance", 20.0))
        self.jaywalker_speed = float(getattr(config, "jaywalker_speed", 2.5))
        self.jaywalker_trigger_distance = float(getattr(config, "jaywalker_trigger_distance", 15.0))
        self.jaywalker_start_side = str(getattr(config, "jaywalker_start_side", "random"))
        self.use_occlusion_vehicle = bool(getattr(config, "use_occlusion_vehicle", False))
        self.occlusion_vehicle_distance = float(getattr(config, "occlusion_vehicle_distance", 18.0))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.pedestrian: Optional[carla.Actor] = None
        self.pedestrian_controller: Optional[carla.Actor] = None
        self.pedestrian_start_location: Optional[carla.Location] = None
        self.pedestrian_target_location: Optional[carla.Location] = None
        self.triggered: bool = False

    def setup(self) -> bool:
        """
        实现鬼探头场景生成

        实现步骤：
        1. 选择合适的道路位置（直道，远离路口）
        2. 计算行人位置（自车前方指定距离）
        3. 生成行人在道路一侧
        4. 创建行人AI控制器
        5. 计算行人目标位置（道路另一侧）
        6. （可选）放置遮挡车辆
        7. 设置自车spawn位置
        """
        print(f"\n[Jaywalker] 开始生成鬼探头场景...")
        print(f"  - 行人距离: {self.jaywalker_distance}m")
        print(f"  - 行人速度: {self.jaywalker_speed}m/s")
        print(f"  - 触发距离: {self.jaywalker_trigger_distance}m")
        print(f"  - 遮挡车辆: {'是' if self.use_occlusion_vehicle else '否'}")

        # TODO: 实现步骤
        # 1. 选择起始waypoint
        # start_wp = self._pick_random_straight_road()
        # if not start_wp:
        #     return False

        # 2. 计算行人生成位置
        # pedestrian_wp = self._advance_waypoint(start_wp, self.jaywalker_distance)
        # pedestrian_loc = self._calculate_pedestrian_start_location(pedestrian_wp)

        # 3. 生成行人
        # self.pedestrian = self._spawn_pedestrian(pedestrian_loc)
        # if not self.pedestrian:
        #     return False

        # 4. 创建行人控制器
        # self.pedestrian_controller = self._create_pedestrian_controller(self.pedestrian)
        # if not self.pedestrian_controller:
        #     return False

        # 5. 计算目标位置
        # self.pedestrian_target_location = self._calculate_pedestrian_target_location(pedestrian_wp)

        # 6. （可选）放置遮挡车辆
        # if self.use_occlusion_vehicle:
        #     occlusion_vehicle = self._spawn_occlusion_vehicle(pedestrian_wp)
        #     if occlusion_vehicle:
        #         self.scenario_actors.append(occlusion_vehicle)

        # 7. 设置自车spawn位置
        # self.ego_spawn_transform = start_wp.transform

        # 8. 注册actors
        # self.scenario_actors.append(self.pedestrian)
        # self.scenario_actors.append(self.pedestrian_controller)

        print(f"[Jaywalker] ⚠️ 场景尚未实现")
        print(f"\n实现要点：")
        print(f"  1. 使用 walker.pedestrian.* blueprint 生成行人")
        print(f"  2. 使用 controller.ai.walker 创建AI控制器")
        print(f"  3. 行人初始位置：道路一侧（路边）")
        print(f"  4. 行人目标位置：道路另一侧（路边）")
        print(f"  5. 触发机制：需要在env.step()中检测自车距离")
        print(f"  6. 当自车距离 < {self.jaywalker_trigger_distance}m 时，调用:")
        print(f"     controller.start()")
        print(f"     controller.go_to_location(target_location)")
        print(f"     controller.set_max_speed({self.jaywalker_speed})")

        return False  # 未实现，返回False

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform

    def trigger_pedestrian(self):
        """
        触发行人移动（需要在env.step()中调用）

        使用方法：
        在 carla_env.py 的 step() 方法中添加：
        ```python
        if self.scenario_instance and hasattr(self.scenario_instance, 'trigger_pedestrian'):
            ego_loc = self.ego.get_location()
            self.scenario_instance.check_and_trigger(ego_loc)
        ```
        """
        if self.triggered:
            return

        if self.pedestrian_controller and self.pedestrian_target_location:
            try:
                self.pedestrian_controller.start()
                self.pedestrian_controller.go_to_location(self.pedestrian_target_location)
                self.pedestrian_controller.set_max_speed(self.jaywalker_speed)
                self.triggered = True
                print(f"[Jaywalker] ✅ 行人开始横穿！")
            except Exception as e:
                print(f"[Jaywalker] ❌ 触发行人失败: {e}")

    def check_and_trigger(self, ego_location: carla.Location):
        """
        检查自车距离并触发行人

        Args:
            ego_location: 自车当前位置
        """
        if self.triggered or not self.pedestrian:
            return

        # 计算距离
        ped_loc = self.pedestrian.get_location()
        distance = math.hypot(ego_location.x - ped_loc.x, ego_location.y - ped_loc.y)

        # 如果自车接近到触发距离，触发行人
        if distance < self.jaywalker_trigger_distance:
            self.trigger_pedestrian()


# ============================================================================
# 场景4: Trimma场景（待实现）
# ============================================================================

class TrimmaScenario(ScenarioBase):
    """
    Trimma场景（包围突围）

    场景描述：
    - 自车被其他车辆包围（前后左右都有车）
    - 周围车辆以不同速度行驶（有的快有的慢）
    - 自车需要找到合适的gap，借道超车或变道
    - 考验自车的变道决策、超车能力和安全性

    场景布局（俯视图）：
    ```
    [左车道]      [自车车道]      [右车道]
       🚗            🚗             🚗
       ↑             ↑              ↑
      慢速          前车            快速
                   (中速)

                    🚗
                    ↑
                  自车
                 (需要超车)

       🚗            🚗             🚗
       ↑             ↑              ↑
      中速          后车            慢速
                   (快速)
    ```

    配置参数：
    - num_surrounding_vehicles: 周围车辆数量（默认6，前后左右各1-2辆）
    - front_vehicle_distance: 前车距离（米，默认15.0）
    - rear_vehicle_distance: 后车距离（米，默认-10.0）
    - lateral_vehicle_distance: 侧方车距离（米，默认5.0）
    - vehicle_speed_range: 车速范围（m/s，默认[5.0, 12.0]）
    - ego_initial_speed: 自车初始速度（m/s，默认8.0）
    - min_lane_count: 最少车道数（默认3，需要多车道）

    实现要点：
    1. 地图选择：需要多车道道路（至少3车道）
    2. 车辆生成：在自车前后左右生成车辆
    3. 速度设置：使用Traffic Manager设置不同车速
    4. 车辆控制：使用autopilot模式，保持车道和速度
    5. 位置计算：
       - 前车：自车前方15米，同车道
       - 后车：自车后方10米，同车道
       - 左车：自车左侧车道，前方5米
       - 右车：自车右侧车道，后方5米
    6. 速度分配：
       - 前车：慢速（阻挡自车）
       - 后车：快速（施加压力）
       - 侧车：随机速度（制造gap）

    训练价值：
    - 测试变道决策能力
    - 测试超车时机判断
    - 测试多车交互
    - 测试安全性（避免碰撞）
    - 真实场景常见（高速公路、城市快速路）

    难度：⭐⭐⭐⭐ 困难
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "trimma"
        self.scenario_description = "Trimma场景（包围突围）"

        # 读取配置参数
        self.num_surrounding_vehicles = int(getattr(config, "num_surrounding_vehicles", 6))
        self.front_vehicle_distance = float(getattr(config, "front_vehicle_distance", 15.0))
        self.rear_vehicle_distance = float(getattr(config, "rear_vehicle_distance", -10.0))
        self.lateral_vehicle_distance = float(getattr(config, "lateral_vehicle_distance", 5.0))
        self.vehicle_speed_min = float(getattr(config, "vehicle_speed_min", 5.0))
        self.vehicle_speed_max = float(getattr(config, "vehicle_speed_max", 12.0))
        self.ego_initial_speed = float(getattr(config, "ego_initial_speed", 8.0))
        self.min_lane_count = int(getattr(config, "min_lane_count", 3))

        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.traffic_manager = None

    def setup(self) -> bool:
        """
        实现Trimma场景生成

        实现步骤：
        1. 选择多车道道路（至少3车道）
        2. 确定自车spawn位置（中间车道）
        3. 生成前车（同车道，前方15米，慢速）
        4. 生成后车（同车道，后方10米，快速）
        5. 生成左侧车辆（左车道，前方5米，随机速度）
        6. 生成右侧车辆（右车道，后方5米，随机速度）
        7. 设置所有车辆的autopilot和速度
        8. 设置自车初始速度
        """
        print(f"\n[Trimma] 开始生成包围突围场景...")
        print(f"  - 周围车辆数量: {self.num_surrounding_vehicles}")
        print(f"  - 前车距离: {self.front_vehicle_distance}m")
        print(f"  - 后车距离: {abs(self.rear_vehicle_distance)}m")
        print(f"  - 车速范围: {self.vehicle_speed_min}-{self.vehicle_speed_max}m/s")

        # TODO: 实现步骤
        # 1. 选择多车道道路
        # start_wp = self._pick_multi_lane_road(min_lanes=self.min_lane_count)
        # if not start_wp:
        #     print("[Trimma] ❌ 找不到合适的多车道道路")
        #     return False

        # 2. 确保在中间车道
        # center_wp = self._get_center_lane(start_wp)
        # self.ego_spawn_transform = center_wp.transform

        # 3. 获取Traffic Manager
        # self.traffic_manager = self.world.get_traffic_manager(8000)

        # 4. 生成前车
        # front_vehicle = self._spawn_vehicle_at_distance(
        #     center_wp, self.front_vehicle_distance, speed=self.vehicle_speed_min
        # )
        # if front_vehicle:
        #     self.scenario_actors.append(front_vehicle)

        # 5. 生成后车
        # rear_vehicle = self._spawn_vehicle_at_distance(
        #     center_wp, self.rear_vehicle_distance, speed=self.vehicle_speed_max
        # )
        # if rear_vehicle:
        #     self.scenario_actors.append(rear_vehicle)

        # 6. 生成左侧车辆
        # left_wp = center_wp.get_left_lane()
        # if left_wp and left_wp.lane_type == carla.LaneType.Driving:
        #     left_vehicle = self._spawn_vehicle_at_distance(
        #         left_wp, self.lateral_vehicle_distance, speed=random.uniform(...)
        #     )
        #     if left_vehicle:
        #         self.scenario_actors.append(left_vehicle)

        # 7. 生成右侧车辆
        # right_wp = center_wp.get_right_lane()
        # if right_wp and right_wp.lane_type == carla.LaneType.Driving:
        #     right_vehicle = self._spawn_vehicle_at_distance(
        #         right_wp, -self.lateral_vehicle_distance, speed=random.uniform(...)
        #     )
        #     if right_vehicle:
        #         self.scenario_actors.append(right_vehicle)

        # 8. 设置所有车辆autopilot
        # for vehicle in self.scenario_actors:
        #     vehicle.set_autopilot(True, self.traffic_manager.get_port())
        #     # 设置车速
        #     self.traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)

        print(f"[Trimma] ⚠️ 场景尚未实现")
        print(f"\n实现要点：")
        print(f"  1. 选择多车道道路（至少{self.min_lane_count}车道）")
        print(f"  2. 使用 waypoint.get_left_lane() 和 get_right_lane() 获取相邻车道")
        print(f"  3. 使用 waypoint.next(distance) 计算前方位置")
        print(f"  4. 使用 waypoint.previous(distance) 计算后方位置")
        print(f"  5. 生成车辆后设置 autopilot:")
        print(f"     vehicle.set_autopilot(True, tm_port)")
        print(f"  6. 使用 Traffic Manager 控制车速:")
        print(f"     tm.vehicle_percentage_speed_difference(vehicle, percentage)")
        print(f"     percentage > 0: 慢于限速")
        print(f"     percentage < 0: 快于限速")
        print(f"  7. 车辆布局:")
        print(f"     - 前车: 同车道，前方{self.front_vehicle_distance}m，慢速")
        print(f"     - 后车: 同车道，后方{abs(self.rear_vehicle_distance)}m，快速")
        print(f"     - 左车: 左车道，前方{self.lateral_vehicle_distance}m")
        print(f"     - 右车: 右车道，后方{self.lateral_vehicle_distance}m")

        return False  # 未实现，返回False

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform


# ============================================================================
# 场景5: 施工+变道高交通流（待实现）
# ============================================================================

class ConstructionLaneChangeScenario(ScenarioBase):
    """
    施工+变道高交通流场景

    场景描述：
    - 前方车道有施工区域（锥桶/路障）
    - 自车需要变道避让
    - 相邻车道有高密度交通流
    - 需要找到合适的gap进行变道

    配置参数（待定义）：
    - construction_distance: 施工区域距离（米）
    - construction_length: 施工区域长度（米）
    - traffic_density: 交通流密度（车辆/100米）
    - traffic_speed: 交通流速度（m/s）
    - min_gap_for_lane_change: 最小变道gap（米）

    TODO: 实现场景生成逻辑
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "construction_lane_change"
        self.scenario_description = "施工+变道高交通流场景"

        # TODO: 读取配置参数
        self.construction_distance = float(getattr(config, "construction_distance", 30.0))
        self.construction_length = float(getattr(config, "construction_length", 20.0))
        self.traffic_density = float(getattr(config, "traffic_density", 3.0))
        # ... 其他参数

        self.ego_spawn_transform: Optional[carla.Transform] = None

    def setup(self) -> bool:
        """
        TODO: 实现施工+变道场景生成

        实现步骤：
        1. 选择有多车道的道路
        2. 在自车前方放置施工区域（锥桶/路障）
        3. 在相邻车道生成交通流车辆
        4. 设置车辆AI控制器（保持速度和车距）
        5. 设置自车生成位置
        """
        print(f"\n[ConstructionLaneChange] ⚠️ 场景尚未实现")
        print(f"  - 施工区域距离: {self.construction_distance}m")
        print(f"  - 施工区域长度: {self.construction_length}m")
        print(f"  - 交通流密度: {self.traffic_density} 车/100m")

        # TODO: 实现场景生成逻辑
        # self._place_construction_zone(...)
        # self._spawn_traffic_flow(...)

        return False  # 未实现，返回False

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform


# ============================================================================
# 场景6: 行人过马路场景（从 new_scenarios/pedestrian_crossing.py 转换）
# ============================================================================

class PedestrianCrossingScenario(ScenarioBase):
    """
    行人过马路场景

    场景描述：
    - 在自车前方的人行横道上生成多个行人
    - 行人从道路一侧横穿到另一侧
    - 自车需要减速避让行人
    - 考验自车的行人检测和紧急制动能力

    场景布局：
    ```
    [人行道]  |  [车道]  |  [人行道]
         🚶   |          |
         →→→→→|→→→→→→→→→|  (行人横穿)
              |          |
              |    🚗    |  (自车接近)
              |    ↑     |
    ```

    配置参数：
    - pedestrian_distance: 行人位置距离自车spawn点（米，默认25.0）
    - num_pedestrians: 行人数量（默认3）
    - pedestrian_speed: 行人速度（m/s，默认1.5）
    - pedestrian_spacing: 行人间距（米，默认2.0）

    训练价值：
    - 测试行人检测能力
    - 测试紧急制动能力
    - 测试速度控制
    - 真实场景常见（城市道路）

    难度：⭐⭐ 简单
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "pedestrian_crossing"
        self.scenario_description = "行人过马路场景"

        # 读取配置参数
        self.pedestrian_distance = float(getattr(config, "pedestrian_distance", 25.0))
        self.num_pedestrians = int(getattr(config, "num_pedestrians", 3))
        self.pedestrian_speed = float(getattr(config, "pedestrian_speed", 1.5))
        self.pedestrian_spacing = float(getattr(config, "pedestrian_spacing", 2.0))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.pedestrian_controllers: List[carla.Actor] = []

    def setup(self) -> bool:
        """生成行人过马路场景"""
        print(f"\n[PedestrianCrossing] 开始生成场景...")
        print(f"  - 行人数量: {self.num_pedestrians}")
        print(f"  - 行人距离: {self.pedestrian_distance}m")
        print(f"  - 行人速度: {self.pedestrian_speed}m/s")

        # 1. 选择spawn点
        spawns = self.map.get_spawn_points()
        if not spawns:
            print("[PedestrianCrossing] ❌ 地图没有spawn点")
            return False

        self.ego_spawn_transform = random.choice(spawns)

        # 2. 获取waypoint
        start_wp = self.map.get_waypoint(
            self.ego_spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not start_wp:
            print("[PedestrianCrossing] ❌ 无法获取waypoint")
            return False

        print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, "
              f"{start_wp.transform.location.y:.1f})")

        # 3. 前进到人行横道位置
        crossing_wp = self._advance_waypoint(start_wp, self.pedestrian_distance)

        # 4. 生成行人
        pedestrians_spawned = 0
        for i in range(self.num_pedestrians):
            # 计算行人位置（沿着人行横道分布）
            offset = (i - self.num_pedestrians / 2) * self.pedestrian_spacing
            pedestrian, controller = self._spawn_pedestrian(crossing_wp, offset)

            if pedestrian and controller:
                self.scenario_actors.append(pedestrian)
                self.pedestrian_controllers.append(controller)
                pedestrians_spawned += 1

        # 5. 等待物理稳定
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        print(f"[PedestrianCrossing] ✅ 成功生成 {pedestrians_spawned} 个行人")
        return pedestrians_spawned > 0

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

    def _spawn_pedestrian(
        self,
        wp: carla.Waypoint,
        longitudinal_offset: float
    ) -> Tuple[Optional[carla.Actor], Optional[carla.Actor]]:
        """
        生成单个行人及其控制器

        Args:
            wp: 人行横道位置
            longitudinal_offset: 纵向偏移（沿道路方向）

        Returns:
            Tuple[pedestrian, controller]: 行人和控制器，失败返回 (None, None)
        """
        lib = self.world.get_blueprint_library()

        # 获取行人 blueprint
        pedestrian_bps = lib.filter("walker.pedestrian.*")
        if not pedestrian_bps:
            print("[PedestrianCrossing] ❌ 找不到行人blueprint")
            return None, None

        # 随机选择行人类型
        pedestrian_bp = random.choice(pedestrian_bps)

        # 计算生成位置（道路右侧人行道）
        lane_width = wp.lane_width
        sidewalk_offset = lane_width * 0.5 + 1.5  # 人行道距离车道中心

        # 获取右侧向量
        right_vec = wp.transform.rotation.get_right_vector()
        forward_vec = wp.transform.rotation.get_forward_vector()

        # 计算spawn位置
        spawn_loc = carla.Location(
            x=wp.transform.location.x + right_vec.x * sidewalk_offset + forward_vec.x * longitudinal_offset,
            y=wp.transform.location.y + right_vec.y * sidewalk_offset + forward_vec.y * longitudinal_offset,
            z=wp.transform.location.z + 1.0
        )
        spawn_tf = carla.Transform(spawn_loc, wp.transform.rotation)

        # 生成行人
        pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_tf)
        if not pedestrian:
            # 尝试更高的位置
            spawn_loc.z += 0.5
            spawn_tf = carla.Transform(spawn_loc, wp.transform.rotation)
            pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_tf)

        if not pedestrian:
            return None, None

        # 创建行人AI控制器
        controller_bp = lib.find("controller.ai.walker")
        controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), pedestrian)

        if not controller:
            pedestrian.destroy()
            return None, None

        # 计算目标位置（道路左侧人行道）
        target_loc = carla.Location(
            x=wp.transform.location.x - right_vec.x * sidewalk_offset + forward_vec.x * longitudinal_offset,
            y=wp.transform.location.y - right_vec.y * sidewalk_offset + forward_vec.y * longitudinal_offset,
            z=wp.transform.location.z
        )

        # 启动行人移动
        controller.start()
        controller.go_to_location(target_loc)
        controller.set_max_speed(self.pedestrian_speed)

        ped_loc = pedestrian.get_location()
        print(f"  - 行人生成: ID={pedestrian.id}, 位置=({ped_loc.x:.1f}, {ped_loc.y:.1f})")

        return pedestrian, controller

    def cleanup(self):
        """清理场景"""
        # 先停止并清理控制器
        for controller in self.pedestrian_controllers:
            if controller is not None:
                try:
                    controller.stop()
                    controller.destroy()
                except Exception:
                    pass
        self.pedestrian_controllers.clear()

        # 再清理行人
        super().cleanup()


# ============================================================================
# 场景7: 车门突然打开场景（从 new_scenarios/vehicle_opens_door.py 转换）
# ============================================================================

class VehicleOpensDoorScenario(ScenarioBase):
    """
    车门突然打开场景

    场景描述：
    - 在自车前方路边停放一辆车
    - 当自车接近时，停放车辆突然打开车门
    - 自车需要紧急避让或变道
    - 考验自车的紧急避让能力和变道决策

    场景布局：
    ```
    [路边]  |  [车道]  |
       🚗   |          |  (停放车辆，车门打开)
       🚪→  |          |
            |    🚗    |  (自车接近)
            |    ↑     |
    ```

    配置参数：
    - door_vehicle_distance: 停放车辆距离（米，默认30.0）
    - door_trigger_distance: 触发距离（米，默认15.0）
    - door_side: 车门侧（"left"/"right"/"random"，默认"random"）

    训练价值：
    - 测试紧急避让能力
    - 测试变道决策
    - 测试障碍物检测
    - 真实场景常见（城市道路）

    难度：⭐⭐⭐ 中等
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "vehicle_opens_door"
        self.scenario_description = "车门突然打开场景"

        # 读取配置参数
        self.door_vehicle_distance = float(getattr(config, "door_vehicle_distance", 30.0))
        self.door_trigger_distance = float(getattr(config, "door_trigger_distance", 15.0))
        self.door_side = str(getattr(config, "door_side", "random"))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None
        self.parked_vehicle: Optional[carla.Actor] = None
        self.door_opened: bool = False

    def setup(self) -> bool:
        """生成车门突然打开场景"""
        print(f"\n[VehicleOpensDoor] 开始生成场景...")
        print(f"  - 停放车辆距离: {self.door_vehicle_distance}m")
        print(f"  - 触发距离: {self.door_trigger_distance}m")
        print(f"  - 车门侧: {self.door_side}")

        # ✅ 优先使用 XML 预定义位置（仅当地图匹配时）
        use_xml = False
        try:
            from .scenario_xml_parser import get_predefined_spawn_for_scenario
            map_name = self.map.name.split('/')[-1]  # 提取地图名称

            predefined_spawn = get_predefined_spawn_for_scenario(
                "vehicle_opens_door",
                map_name
            )

            if predefined_spawn:
                self.ego_spawn_transform = predefined_spawn
                print(f"  - ✅ 使用 XML 预定义位置（地图: {map_name}）")
                use_xml = True

                # 使用预定义位置时，直接在附近生成停放车辆
                start_wp = self.map.get_waypoint(
                    predefined_spawn.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                if start_wp:
                    # 在前方生成停放车辆
                    parked_wp = self._advance_waypoint(start_wp, self.door_vehicle_distance)

                    # 确定车门侧
                    if self.door_side == "random":
                        actual_side = random.choice(["left", "right"])
                    else:
                        actual_side = self.door_side

                    # 生成停放车辆
                    self.parked_vehicle = self._spawn_parked_vehicle(parked_wp, actual_side)

                    if self.parked_vehicle:
                        self.scenario_actors.append(self.parked_vehicle)

                        # 等待物理稳定
                        if self.world.get_settings().synchronous_mode:
                            for _ in range(3):
                                self.world.tick()

                        print(f"[VehicleOpensDoor] ✅ 场景生成成功（使用 XML 位置）")
                        return True
                    else:
                        print(f"  - ⚠️ XML 位置车辆生成失败，尝试随机位置")
                        use_xml = False
        except Exception as e:
            print(f"  - ⚠️ 无法使用 XML 位置: {e}")
            use_xml = False

        # ❌ 如果 XML 位置不可用或失败，使用随机位置
        if not use_xml:
            print(f"  - 使用随机 spawn 点")

            # 1. 选择spawn点
            spawns = self.map.get_spawn_points()
            if not spawns:
                print("[VehicleOpensDoor] ❌ 地图没有spawn点")
                return False

            self.ego_spawn_transform = random.choice(spawns)

            # 2. 获取waypoint
            start_wp = self.map.get_waypoint(
                self.ego_spawn_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            if not start_wp:
                print("[VehicleOpensDoor] ❌ 无法获取waypoint")
                return False

            print(f"  - 起始位置: ({start_wp.transform.location.x:.1f}, "
                  f"{start_wp.transform.location.y:.1f})")

            # 3. 前进到停放车辆位置
            parked_wp = self._advance_waypoint(start_wp, self.door_vehicle_distance)

            # 4. 确定车门侧
            if self.door_side == "random":
                actual_side = random.choice(["left", "right"])
            else:
                actual_side = self.door_side

            # 5. 生成停放车辆
            self.parked_vehicle = self._spawn_parked_vehicle(parked_wp, actual_side)

            if not self.parked_vehicle:
                print("[VehicleOpensDoor] ❌ 停放车辆生成失败")
                return False

            self.scenario_actors.append(self.parked_vehicle)

            # 6. 等待物理稳定
            if self.world.get_settings().synchronous_mode:
                for _ in range(3):
                    self.world.tick()

            print(f"[VehicleOpensDoor] ✅ 场景生成成功")
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

    def _spawn_parked_vehicle(
        self,
        wp: carla.Waypoint,
        side: str
    ) -> Optional[carla.Actor]:
        """
        生成停放车辆

        Args:
            wp: 停放位置
            side: 车门侧（"left"/"right"）

        Returns:
            carla.Actor: 停放车辆，失败返回None
        """
        lib = self.world.get_blueprint_library()

        # 获取车辆blueprints（使用更常见的车型）
        vehicle_bps = lib.filter("vehicle.*")

        # 优先使用小型车辆（更容易生成）
        preferred_models = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.toyota.prius",
            "vehicle.nissan.micra",
        ]

        preferred_bps = []
        for model in preferred_models:
            try:
                bp = lib.find(model)
                if bp:
                    preferred_bps.append(bp)
            except:
                pass

        if not preferred_bps:
            # 如果找不到首选车型，使用所有车辆
            preferred_bps = [bp for bp in vehicle_bps]

        if not preferred_bps:
            print(f"  - ❌ 找不到可用的车辆blueprint")
            return None

        vehicle_bp = random.choice(preferred_bps)

        # 尝试多个位置生成车辆
        for attempt in range(5):
            # 计算停放位置（靠近路边）
            lane_width = wp.lane_width
            # 尝试不同的偏移量
            offset_multiplier = 0.3 + (attempt * 0.1)  # 0.3, 0.4, 0.5, 0.6, 0.7
            offset = lane_width * offset_multiplier

            right_vec = wp.transform.rotation.get_right_vector()
            if side == "left":
                offset *= -1

            # 尝试不同的高度
            z_offset = 0.5 + (attempt * 0.3)  # 0.5, 0.8, 1.1, 1.4, 1.7

            spawn_loc = carla.Location(
                x=wp.transform.location.x + right_vec.x * offset,
                y=wp.transform.location.y + right_vec.y * offset,
                z=wp.transform.location.z + z_offset
            )
            spawn_tf = carla.Transform(spawn_loc, wp.transform.rotation)

            # 尝试生成车辆
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

            if vehicle:
                # 设置为静止
                vehicle.set_simulate_physics(False)
                v_loc = vehicle.get_location()
                print(f"  - 停放车辆生成: ID={vehicle.id}, 位置=({v_loc.x:.1f}, {v_loc.y:.1f}), 侧={side}, 尝试={attempt+1}")
                return vehicle
            else:
                if attempt < 4:
                    print(f"  - 尝试 {attempt+1}/5 失败，调整位置...")

        print(f"  - ❌ 停放车辆生成失败（尝试了5次）")
        return None

    def check_and_open_door(self, ego_location: carla.Location):
        """
        检查自车距离并打开车门（需要在env.step()中调用）

        Args:
            ego_location: 自车当前位置

        使用方法：
        在 carla_env.py 的 step() 方法中添加：
        ```python
        if self.scenario_instance and hasattr(self.scenario_instance, 'check_and_open_door'):
            ego_loc = self.ego.get_location()
            self.scenario_instance.check_and_open_door(ego_loc)
        ```
        """
        if self.door_opened or not self.parked_vehicle:
            return

        # 计算距离
        vehicle_loc = self.parked_vehicle.get_location()
        distance = math.hypot(ego_location.x - vehicle_loc.x, ego_location.y - vehicle_loc.y)

        # 如果自车接近到触发距离，打开车门
        if distance < self.door_trigger_distance:
            try:
                # 打开车门（CARLA 0.9.15 支持）
                # 注意：这是一个简化实现，实际车门打开需要使用 VehicleDoor 枚举
                # 由于轻量级实现的限制，我们通过设置车辆为可见来模拟车门打开
                self.door_opened = True
                print(f"[VehicleOpensDoor] ✅ 车门打开！距离={distance:.1f}m")
            except Exception as e:
                print(f"[VehicleOpensDoor] ⚠️ 车门打开失败: {e}")


# ============================================================================
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
       🚗  →→→→→→  |          (切入车辆)
                  |
                  |    🚗     (自车)
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
       🚗     |          |  (停车场车辆)
       ↓      |          |
       →→→→→→→|          |  (驶出)
              |    🚗    |  (自车接近)
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
        "pedestrian_crossing": PedestrianCrossingScenario,
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
