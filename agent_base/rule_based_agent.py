# agents/rule_based/agent.py
from __future__ import annotations

import os
import sys
import time
import random
import argparse
import glob
import math
from typing import Optional, Tuple, List

# 你的 egg 路径（保留也没问题）
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

import carla


def _add_carla_pythonapi_paths():
    """
    加入 CARLA PythonAPI 路径，确保 agents/traffic_manager 等模块可用
    """
    candidates = []
    if "CARLA_ROOT" in os.environ:
        candidates.append(os.environ["CARLA_ROOT"])
    candidates.append("/home/ajifang/carla")

    for carla_root in candidates:
        pyapi = os.path.join(carla_root, "PythonAPI")
        if not os.path.isdir(pyapi):
            continue

        # egg
        dist_dir = os.path.join(pyapi, "carla", "dist")
        eggs = glob.glob(os.path.join(dist_dir, "carla-*.egg"))
        for e in eggs:
            if e not in sys.path:
                sys.path.append(e)

        # PythonAPI/carla (包含 agents, navigation 等)
        carla_pkg = os.path.join(pyapi, "carla")
        if carla_pkg not in sys.path:
            sys.path.append(carla_pkg)

        return True

    return False


if not _add_carla_pythonapi_paths():
    print("[WARN] Could not locate CARLA PythonAPI automatically. "
          "Please set CARLA_ROOT env var or edit fallback path.")


def set_sync_mode(world: carla.World, enable: bool, fixed_delta_seconds: float = 0.05):
    settings = world.get_settings()
    settings.synchronous_mode = enable
    settings.fixed_delta_seconds = fixed_delta_seconds if enable else None
    world.apply_settings(settings)


def spawn_ego(world: carla.World, blueprint_filter: str = "vehicle.tesla.model3") -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter(blueprint_filter)
    if not bps:
        bps = bp_lib.filter("vehicle.*")
    bp = random.choice(bps)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    random.shuffle(spawn_points)
    ego = None
    for sp in spawn_points:
        ego = world.try_spawn_actor(bp, sp)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle. Try clearing other vehicles or choose another map.")

    return ego


def spawn_npc_traffic(world: carla.World, traffic_manager: carla.TrafficManager,
                      num_vehicles: int = 30, seed: int = 0) -> List[carla.Actor]:
    """
    生成 NPC：全部 TM autopilot 控制
    """
    if num_vehicles <= 0:
        return []

    random.seed(seed)
    traffic_manager.set_synchronous_mode(world.get_settings().synchronous_mode)

    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    actors: List[carla.Actor] = []
    for sp in spawn_points:
        if len(actors) >= num_vehicles:
            break

        bp = random.choice(vehicle_bps)
        if bp.has_attribute("number_of_wheels"):
            if int(bp.get_attribute("number_of_wheels")) < 4:
                continue

        npc = world.try_spawn_actor(bp, sp)
        if npc is None:
            continue

        npc.set_autopilot(True, traffic_manager.get_port())
        actors.append(npc)

    for a in actors:
        traffic_manager.distance_to_leading_vehicle(a, 2.0)
        traffic_manager.vehicle_percentage_speed_difference(a, random.uniform(-10, 15))
        traffic_manager.auto_lane_change(a, True)

    return actors


def follow_ego_spectator(world: carla.World, ego: carla.Vehicle, height: float = 3.0, dist: float = 8.0):
    """
    tick 之后更新更稳定：后上方 look-at
    """
    spec = world.get_spectator()
    tf = ego.get_transform()
    fwd = tf.get_forward_vector()

    loc = tf.location - carla.Location(x=fwd.x * dist, y=fwd.y * dist, z=0.0)
    loc.z += height

    target = tf.location + carla.Location(z=1.2)
    yaw = math.degrees(math.atan2(target.y - loc.y, target.x - loc.x))
    rot = carla.Rotation(pitch=-12.0, yaw=yaw, roll=0.0)
    spec.set_transform(carla.Transform(loc, rot))


def tm_set_target_speed(tm: carla.TrafficManager, vehicle: carla.Vehicle, target_speed_mps: float):
    """
    让 TM 的速度更接近你的目标速度。
    优先用 tm.set_desired_speed（若存在）；否则用 vehicle_percentage_speed_difference 近似。
    """
    target_kmh = float(target_speed_mps) * 3.6

    # 1) 新版本/部分版本存在 set_desired_speed(vehicle, kmh)
    if hasattr(tm, "set_desired_speed"):
        try:
            tm.set_desired_speed(vehicle, target_kmh)
            return
        except Exception:
            pass

    # 2) 退化：按 speed limit 算百分比差
    try:
        speed_limit = float(vehicle.get_speed_limit())  # km/h
    except Exception:
        speed_limit = 50.0

    if speed_limit <= 1e-3:
        speed_limit = 50.0

    # desired = speed_limit * (1 - perc/100)
    perc = (1.0 - (target_kmh / speed_limit)) * 100.0
    perc = float(max(-80.0, min(80.0, perc)))  # 限幅

    tm.vehicle_percentage_speed_difference(vehicle, perc)


def tm_force_lane_change(tm: carla.TrafficManager, vehicle: carla.Vehicle, to_left: bool) -> bool:
    """
    兼容性封装：尽量调用 TM 的强制变道接口。
    CARLA 0.9.15 常见是 force_lane_change(vehicle, bool_left)
    若不存在，会打印可用方法，返回 False。
    """
    if hasattr(tm, "force_lane_change"):
        try:
            tm.force_lane_change(vehicle, to_left)
            return True
        except TypeError:
            # 有些版本签名可能不同
            try:
                tm.force_lane_change(vehicle, bool(to_left))
                return True
            except Exception:
                pass
        except Exception:
            pass

    # 兜底：有的版本没有 force_lane_change，只有 auto_lane_change 相关
    print("[TM] force_lane_change not available in this build.")
    # 打印 lane change 相关方法名，方便你确认版本暴露了什么 API
    names = [n for n in dir(tm) if "lane" in n.lower() or "change" in n.lower()]
    print("[TM] Available TM methods containing 'lane'/'change':", names)
    return False


def _actor_is_relevant_obstacle(actor: carla.Actor) -> bool:
    tid = (getattr(actor, "type_id", "") or "").lower()
    return tid.startswith("vehicle.") or tid.startswith("walker.") or tid.startswith("static.")


def _get_lane_width_safe(wp: Optional[carla.Waypoint]) -> float:
    if wp is None:
        return 3.5
    try:
        w = float(getattr(wp, "lane_width", 3.5))
        return w if w > 0.1 else 3.5
    except Exception:
        return 3.5


def detect_blocking_ahead(
    world: carla.World,
    ego: carla.Vehicle,
    max_forward_dist: float = 22.0,
    lane_margin: float = 0.7,
    fov_deg: float = 35.0,
) -> Tuple[bool, Optional[carla.Actor], float]:
    """
    纯 CARLA API 前方阻塞检测：
    - 优先看“同一车道中心线附近”的障碍
    - 结合扇区（FOV）避免把旁边/后方当阻塞
    返回：(blocked, actor, forward_dist)
    """
    amap = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_w = _get_lane_width_safe(ego_wp)

    cos_th = math.cos(math.radians(fov_deg))

    best_actor = None
    best_fd = 1e9

    for a in world.get_actors():
        try:
            if a.id == ego.id:
                continue
        except Exception:
            continue

        if not _actor_is_relevant_obstacle(a):
            continue

        try:
            aloc = a.get_transform().location
        except Exception:
            continue

        dx = aloc.x - ego_loc.x
        dy = aloc.y - ego_loc.y

        # 前向/侧向分量（ego坐标系）
        fd = fwd.x * dx + fwd.y * dy
        if fd <= 0.5 or fd > max_forward_dist:
            continue

        # 扇区角度（避免旁边很近但不在前方的）
        dist = math.hypot(dx, dy) + 1e-6
        dot = fd / dist
        if dot < cos_th:
            continue

        lat = right.x * dx + right.y * dy

        # 侧向阈值：车道宽/2 + margin + 障碍物半宽
        obs_half = 0.4
        try:
            bb = getattr(a, "bounding_box", None)
            if bb is not None:
                obs_half = float(bb.extent.y)
        except Exception:
            pass

        lat_th = 0.5 * lane_w + lane_margin + obs_half
        if abs(lat) > lat_th:
            continue

        # 同车道强化：若能拿到 actor waypoint，要求 road/lane 更一致
        try:
            awp = amap.get_waypoint(aloc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if ego_wp is not None and awp is not None:
                if (awp.road_id != ego_wp.road_id) or (awp.lane_id != ego_wp.lane_id):
                    # 不是同一车道：仍可能阻塞（比如路口），但降低优先级
                    fd = fd + 6.0
        except Exception:
            pass

        if fd < best_fd:
            best_fd = fd
            best_actor = a

    if best_actor is None:
        return False, None, 0.0
    return True, best_actor, float(best_fd)


def lane_exists(amap: carla.Map, ego_loc: carla.Location, to_left: bool) -> bool:
    wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return False
    nxt = wp.get_left_lane() if to_left else wp.get_right_lane()
    if nxt is None:
        return False
    return nxt.lane_type == carla.LaneType.Driving


def lane_is_clear(
    world: carla.World,
    ego: carla.Vehicle,
    to_left: bool,
    check_forward: float = 18.0,
    check_backward: float = 8.0,
    lateral_margin: float = 0.8,
) -> bool:
    """
    粗略判断目标车道是否空（前方/后方是否有车）
    - 用 ego 当前 waypoint 的左/右邻车道作为目标车道中心
    - 检查其他车辆投影在 ego 坐标系下的前后/侧向距离
    """
    amap = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return False

    tgt_wp = ego_wp.get_left_lane() if to_left else ego_wp.get_right_lane()
    if tgt_wp is None or tgt_wp.lane_type != carla.LaneType.Driving:
        return False

    # 目标车道中心相对于 ego 车道中心的侧向偏移（近似用 lane_width）
    lane_w = _get_lane_width_safe(ego_wp)
    center_lat = (+lane_w) if to_left else (-lane_w)

    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ego.id:
            continue
        try:
            aloc = a.get_transform().location
        except Exception:
            continue

        dx = aloc.x - ego_loc.x
        dy = aloc.y - ego_loc.y
        fd = fwd.x * dx + fwd.y * dy
        lat = right.x * dx + right.y * dy

        # 在目标车道中心附近？
        if abs(lat - center_lat) > (0.5 * lane_w + lateral_margin):
            continue

        # 前后范围内是否有车
        if -check_backward <= fd <= check_forward:
            return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sync", action="store_true", help="Enable synchronous mode")
    parser.add_argument("--dt", type=float, default=0.05, help="Fixed delta seconds (sync mode)")
    parser.add_argument("--npc", type=int, default=20, help="Number of NPC vehicles to spawn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spectator", action="store_true")
    parser.add_argument("--blueprint", default="vehicle.tesla.model3")

    # 行为参数
    parser.add_argument("--target-speed", type=float, default=10.0, help="m/s (TM tries to match via speed diff)")
    parser.add_argument("--detect-dist", type=float, default=22.0, help="Forward distance to detect obstacles (m)")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between forced lane changes")
    parser.add_argument("--lanechange-timeout", type=float, default=4.0, help="Timeout to consider lane change done")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()
    original_settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_random_device_seed(args.seed)

    actors: List[carla.Actor] = []
    ego: Optional[carla.Vehicle] = None
    npcs: List[carla.Actor] = []

    try:
        # 同步模式
        if args.sync:
            set_sync_mode(world, True, fixed_delta_seconds=args.dt)
            traffic_manager.set_synchronous_mode(True)

        # 生成 ego
        ego = spawn_ego(world, blueprint_filter=args.blueprint)
        actors.append(ego)
        print(f"[EGO] Spawned: id={ego.id}, type={ego.type_id}, loc={ego.get_location()}")

        # 生成 NPC
        npcs = spawn_npc_traffic(world, traffic_manager, num_vehicles=args.npc, seed=args.seed)
        actors.extend(npcs)
        print(f"[NPC] Spawned {len(npcs)} vehicles")

        # 把 ego 也交给 TM autopilot
        ego.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.auto_lane_change(ego, True)
        traffic_manager.distance_to_leading_vehicle(ego, 2.0)
        tm_set_target_speed(traffic_manager, ego, args.target_speed)

        # 一些可选：让 ego 更“守规矩”或更“激进”
        # traffic_manager.ignore_lights_percentage(ego, 0.0)
        # traffic_manager.ignore_signs_percentage(ego, 0.0)
        # traffic_manager.ignore_vehicles_percentage(ego, 0.0)

        amap = world.get_map()

        # 变道状态机
        lane_change_in_progress = False
        lane_change_start_lane_id = None
        lane_change_start_time = 0.0
        last_force_time = -1e9

        step = 0

        while True:
            # 推进仿真
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # spectator（tick 后）
            if args.spectator and ego is not None:
                follow_ego_spectator(world, ego)

            if ego is None:
                continue

            # 变道完成判定：lane_id 变化 或 超时
            ego_wp = amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            ego_lane_id = ego_wp.lane_id if ego_wp else None

            if lane_change_in_progress:
                done = False
                if lane_change_start_lane_id is not None and ego_lane_id is not None:
                    if ego_lane_id != lane_change_start_lane_id:
                        done = True
                if (time.time() - lane_change_start_time) > args.lanechange_timeout:
                    done = True

                if done:
                    lane_change_in_progress = False
                    lane_change_start_lane_id = None

            # 如果正在变道，就别频繁触发新命令
            if lane_change_in_progress:
                if step % 40 == 0:
                    print(f"[LC] in progress... ego_lane={ego_lane_id}")
                step += 1
                continue

            # 检测前方阻塞
            blocked, actor, fd = detect_blocking_ahead(
                world, ego,
                max_forward_dist=args.detect_dist,
                lane_margin=0.7,
                fov_deg=35.0,
            )

            if blocked:
                now = time.time()
                if now - last_force_time < args.cooldown:
                    # 冷却中：不重复触发
                    if step % 40 == 0:
                        print(f"[OBS] blocked by {actor.type_id if actor else 'unknown'} at {fd:.1f}m, cooldown...")
                    step += 1
                    continue

                # 选择变道方向：优先存在且更空的车道
                left_ok = lane_exists(amap, ego.get_location(), to_left=True) and lane_is_clear(world, ego, to_left=True)
                right_ok = lane_exists(amap, ego.get_location(), to_left=False) and lane_is_clear(world, ego, to_left=False)

                # 选择策略：左优先（你也可以改成右优先或更复杂）
                chosen = None
                if left_ok and right_ok:
                    chosen = True  # left
                elif left_ok:
                    chosen = True
                elif right_ok:
                    chosen = False

                if chosen is None:
                    print(f"[OBS] blocked at {fd:.1f}m but no safe adjacent lane. actor={actor.type_id if actor else 'unknown'}")
                    step += 1
                    continue

                ok = tm_force_lane_change(traffic_manager, ego, to_left=chosen)
                if ok:
                    last_force_time = now
                    lane_change_in_progress = True
                    lane_change_start_lane_id = ego_lane_id
                    lane_change_start_time = now
                    side = "LEFT" if chosen else "RIGHT"
                    print(f"[LC] FORCE {side} | blocked_by={actor.type_id if actor else 'unknown'} fd={fd:.1f}m "
                          f"ego_lane={ego_lane_id}")
                else:
                    print("[LC] Failed to call TM force lane change (API not available).")

            # 打印一些状态
            if step % 60 == 0:
                v = ego.get_velocity()
                speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                print(f"[STEP {step:06d}] v={speed:.2f}m/s lane={ego_lane_id} blocked={blocked} fd={fd:.1f}")

            step += 1

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt")
    finally:
        # 恢复 settings
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        # 清理 actors
        for a in actors[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        print("[CLEANUP] Destroyed actors, restored world settings.")


if __name__ == "__main__":
    main()
