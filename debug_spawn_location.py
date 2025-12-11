"""检查spawn位置和地形"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("Spawn位置诊断")
print("="*70)

config = Config()
config.scenario = "parked_obstacles"
config.render = False

env = CarlaEnv(config, 2000, 8000)

print("\n[1] 重置环境并检查spawn...")
env.reset()

# 获取ego信息
ego_tf = env.ego.get_transform()
ego_loc = ego_tf.location
ego_rot = ego_tf.rotation

print(f"\n自车信息:")
print(f"  位置: ({ego_loc.x:.2f}, {ego_loc.y:.2f}, {ego_loc.z:.2f})")
print(f"  旋转: pitch={ego_rot.pitch:.2f}, yaw={ego_rot.yaw:.2f}, roll={ego_rot.roll:.2f}")

# 获取waypoint信息
waypoint = env.map.get_waypoint(ego_loc, project_to_road=True)
print(f"\n最近的waypoint:")
print(f"  位置: ({waypoint.transform.location.x:.2f}, {waypoint.transform.location.y:.2f}, {waypoint.transform.location.z:.2f})")
print(f"  车道类型: {waypoint.lane_type}")
print(f"  车道宽度: {waypoint.lane_width:.2f}m")
print(f"  是否在路口: {waypoint.is_junction}")

# 检查前方路径
print(f"\n前方路径检查（每5米）:")
current_wp = waypoint
for i in range(10):
    next_wps = current_wp.next(5.0)
    if next_wps:
        current_wp = next_wps[0]
        loc = current_wp.transform.location
        print(f"  {(i+1)*5}m: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}), junction={current_wp.is_junction}")
    else:
        print(f"  {(i+1)*5}m: 无法继续")
        break

# 检查停车障碍位置
print(f"\n停车障碍位置:")
if hasattr(env, '_actors'):
    parked_count = 0
    for actor in env._actors:
        if 'vehicle' in actor.type_id.lower() and actor != env.ego:
            loc = actor.get_location()
            distance = np.sqrt((loc.x - ego_loc.x)**2 + (loc.y - ego_loc.y)**2)
            print(f"  停车 #{parked_count+1}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}), 距离={distance:.1f}m")
            parked_count += 1

# 测试物理
print(f"\n[2] 测试物理模拟（5步）:")
for step in range(5):
    # 大油门
    action = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)

    loc = env.ego.get_location()
    vel = env.ego.get_velocity()
    vel_scalar = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # 获取control
    control = env.ego.get_control()

    print(f"  Step {step+1}:")
    print(f"    位置: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
    print(f"    速度: {vel_scalar:.2f} m/s")
    print(f"    Control: throttle={control.throttle:.2f}, brake={control.brake:.2f}, steer={control.steer:.2f}")
    print(f"    手刹: {control.hand_brake}, 档位: {control.gear}")

print("\n" + "="*70)
env.close()
