#!/usr/bin/env python3
"""
障碍车生成备用方案
使用CARLA预定义spawn点，确保生成成功
"""

# 在 carla_env.py 中添加这个方法

def _place_parked_vehicles_v2(self, start_wp: carla.Waypoint):
    """
    改进版障碍车生成方法
    使用更可靠的生成策略
    """
    world = self.world
    lib = world.get_blueprint_library()

    # 调试输出
    start_loc = start_wp.transform.location
    print(f"\n[OBSTACLE V2] 开始生成障碍车:")
    print(f"  - 起始waypoint: ({start_loc.x:.1f}, {start_loc.y:.1f}, {start_loc.z:.1f})")

    # 获取车辆blueprints
    vehicle_bps = lib.filter("vehicle.*")
    if not vehicle_bps:
        print("  ❌ 没有可用的车辆blueprint！")
        return []

    parked_vehicles = []
    cur_wp = start_wp

    # 策略：沿着车道前进，在每个位置尝试生成
    for i in range(self.num_parked_cars):
        # 前进到生成位置
        if i == 0:
            # 第一辆车：前进12米
            distance = self.parked_car_start_distance
        else:
            # 后续车辆：间隔25米
            distance = self.parked_car_spacing

        # 沿着车道前进
        traveled = 0.0
        while traveled < distance:
            nxt = cur_wp.next(2.0)
            if not nxt:
                print(f"    ⚠️ 车辆{i+1}: 无法继续前进（路径尽头）")
                break
            cur_wp = nxt[0]
            traveled += 2.0

        if not cur_wp:
            break

        # 获取生成位置
        wp_loc = cur_wp.transform.location
        wp_rot = cur_wp.transform.rotation

        # 尝试多个生成位置
        spawn_attempts = [
            # 1. 车道中心（最可靠）
            (wp_loc.x, wp_loc.y, wp_loc.z + 0.5, "车道中心"),

            # 2. 车道右侧
            (wp_loc.x + 1.8 * math.cos(math.radians(wp_rot.yaw + 90)),
             wp_loc.y + 1.8 * math.sin(math.radians(wp_rot.yaw + 90)),
             wp_loc.z + 0.5, "右侧1.8m"),

            # 3. 车道左侧
            (wp_loc.x + 1.8 * math.cos(math.radians(wp_rot.yaw - 90)),
             wp_loc.y + 1.8 * math.sin(math.radians(wp_rot.yaw - 90)),
             wp_loc.z + 0.5, "左侧1.8m"),

            # 4. 更高位置
            (wp_loc.x, wp_loc.y, wp_loc.z + 1.0, "高1.0m"),
        ]

        vehicle = None
        for x, y, z, desc in spawn_attempts:
            spawn_loc = carla.Location(x=x, y=y, z=z)
            spawn_tf = carla.Transform(spawn_loc, wp_rot)

            vehicle_bp = random.choice(vehicle_bps)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

            if vehicle:
                print(f"    ✅ 车辆{i+1}: 成功生成在{desc}, 位置=({x:.1f}, {y:.1f}, {z:.1f})")
                break
            else:
                print(f"    ⚠️ 车辆{i+1}: {desc}生成失败")

        if vehicle:
            vehicle.set_simulate_physics(False)
            parked_vehicles.append(vehicle)
            self._actors.append(vehicle)
            self.obstacle_actors.append(vehicle)
        else:
            print(f"    ❌ 车辆{i+1}: 所有位置都生成失败！")

    print(f"\n[OBSTACLE V2] 生成完成: {len(parked_vehicles)}/{self.num_parked_cars}")
    return parked_vehicles


# ============================================
# 使用预定义spawn点的版本
# ============================================

def _maybe_setup_scene_and_pick_spawn_v2(self) -> carla.Transform:
    """
    改进版场景设置
    使用CARLA预定义spawn点
    """
    if self.scenario == "parked_obstacles":
        # 获取所有spawn点
        spawns = self.map.get_spawn_points()
        if not spawns:
            print("[SPAWN] ⚠️ 地图没有预定义spawn点！")
            return carla.Transform(carla.Location(0, 0, 0.3), carla.Rotation(0, 0, 0))

        # 随机选择一个spawn点
        ego_spawn_tf = random.choice(spawns)
        ego_loc = ego_spawn_tf.location

        print(f"\n[SPAWN V2] 使用预定义spawn点:")
        print(f"  - Ego位置: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
        print(f"  - Yaw: {ego_spawn_tf.rotation.yaw:.1f}°")

        # 获取对应的waypoint
        start_wp = self.map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start_wp:
            print(f"  - Waypoint车道类型: {start_wp.lane_type}")
            print(f"  - Waypoint车道宽度: {start_wp.lane_width:.1f}m")

            # 生成障碍车
            self._place_parked_vehicles_v2(start_wp)
        else:
            print("  ⚠️ 无法获取waypoint！")

        # Tick world
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        return ego_spawn_tf

    # 其他场景...
    return carla.Transform(carla.Location(0, 0, 0.3), carla.Rotation(0, 0, 0))


# ============================================
# 使用方法
# ============================================

# 在 carla_env.py 中替换：
# 1. 将 _place_parked_vehicles 改名为 _place_parked_vehicles_old
# 2. 将 _place_parked_vehicles_v2 改名为 _place_parked_vehicles
# 3. 在 _maybe_setup_scene_and_pick_spawn 中使用新方法

# 或者直接修改 carla_env.py:723-740:
"""
if self.scenario == "parked_obstacles":
    # 使用预定义spawn点
    spawns = self.map.get_spawn_points()
    if spawns:
        ego_spawn_tf = random.choice(spawns)
        ego_loc = ego_spawn_tf.location
        print(f"[SPAWN] Ego位置: ({ego_loc.x:.1f}, {ego_loc.y:.1f})")

        start_wp = self.map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start_wp:
            self._place_parked_vehicles(start_wp)

        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()

        return ego_spawn_tf
"""
