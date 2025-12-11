"""验证档位修复"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("验证档位修复")
print("="*70)

config = Config()
config.scenario = "parked_obstacles"
config.render = False

env = CarlaEnv(config, 2000, 8000)

print("\n[1] 重置环境...")
env.reset()

ego_loc_start = env.ego.get_location()
print(f"起始位置: ({ego_loc_start.x:.2f}, {ego_loc_start.y:.2f}, {ego_loc_start.z:.2f})")

# 检查初始档位
control = env.ego.get_control()
print(f"初始档位: {control.gear}")

print("\n[2] 测试20步（大油门直行）:")
print("-"*70)

for step in range(20):
    action = np.array([0.8, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)

    loc = env.ego.get_location()
    vel = env.ego.get_velocity()
    vel_scalar = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    control = env.ego.get_control()

    # 计算移动距离
    distance = np.sqrt((loc.x - ego_loc_start.x)**2 + (loc.y - ego_loc_start.y)**2)

    if step % 5 == 0 or step < 3:
        print(f"Step {step+1:2d}: 速度={vel_scalar:5.2f} m/s, 移动={distance:5.2f}m, 档位={control.gear}")

print("-"*70)

# 最终统计
ego_loc_end = env.ego.get_location()
total_distance = np.sqrt((ego_loc_end.x - ego_loc_start.x)**2 + (ego_loc_end.y - ego_loc_start.y)**2)
final_vel = env.ego.get_velocity()
final_speed = np.sqrt(final_vel.x**2 + final_vel.y**2 + final_vel.z**2)

print(f"\n总结:")
print(f"  - 总移动距离: {total_distance:.2f}m")
print(f"  - 最终速度: {final_speed:.2f} m/s")
print(f"  - 平均速度: {total_distance / (20 * 0.05):.2f} m/s")

if total_distance > 10:
    print(f"\n✅ 修复成功！车辆正常移动")
else:
    print(f"\n❌ 仍有问题，车辆移动距离太短")

print("="*70)

env.close()
