"""检查车辆的物理控制设置"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

CARLA_ROOT = "/home/ajifang/carla"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-0.9.15-py3.7-linux-x86_64.egg"))

import carla
from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("物理控制详细检查")
print("="*70)

config = Config()
config.scenario = "plain"
config.render = False

env = CarlaEnv(config, 2000, 8000)
env.reset()

print("\n[1] 车辆基本信息:")
print(f"  类型: {env.ego.type_id}")
print(f"  物理模拟启用: {env.ego.is_alive}")

# 获取物理控制
physics = env.ego.get_physics_control()
print(f"\n[2] 物理参数:")
print(f"  质量: {physics.mass:.0f} kg")
print(f"  阻力系数: {physics.drag_coefficient:.3f}")
print(f"  重心: ({physics.center_of_mass.x:.2f}, {physics.center_of_mass.y:.2f}, {physics.center_of_mass.z:.2f})")

# 获取当前control
control = env.ego.get_control()
print(f"\n[3] 当前控制状态:")
print(f"  Throttle: {control.throttle:.2f}")
print(f"  Brake: {control.brake:.2f}")
print(f"  Steer: {control.steer:.2f}")
print(f"  Hand Brake: {control.hand_brake}")
print(f"  Reverse: {control.reverse}")
print(f"  Manual Gear Shift: {control.manual_gear_shift}")
print(f"  Gear: {control.gear}")

# 测试：应用大油门
print(f"\n[4] 应用大油门控制...")
test_control = carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0)
test_control.gear = 1
test_control.manual_gear_shift = True
test_control.hand_brake = False  # 🔧 明确释放手刹
env.ego.apply_control(test_control)

# Tick一次
if env.world.get_settings().synchronous_mode:
    env.world.tick()

# 检查control是否生效
actual_control = env.ego.get_control()
print(f"  应用后的实际control:")
print(f"    Throttle: {actual_control.throttle:.2f}")
print(f"    Brake: {actual_control.brake:.2f}")
print(f"    Gear: {actual_control.gear}")
print(f"    Hand Brake: {actual_control.hand_brake}")

# 测试5步
print(f"\n[5] 测试5步:")
start_loc = env.ego.get_location()

for step in range(5):
    # 应用control
    test_control = carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0)
    test_control.gear = 1
    test_control.manual_gear_shift = True
    test_control.hand_brake = False
    env.ego.apply_control(test_control)

    # Tick
    if env.world.get_settings().synchronous_mode:
        env.world.tick()

    loc = env.ego.get_location()
    vel = env.ego.get_velocity()
    vel_scalar = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    distance = np.sqrt((loc.x - start_loc.x)**2 + (loc.y - start_loc.y)**2)

    print(f"  Step {step+1}: 速度={vel_scalar:.2f} m/s, 移动={distance:.3f}m")

print("\n" + "="*70)
env.close()
