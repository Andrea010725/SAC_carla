"""测试不同车辆的物理表现"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("车辆物理测试")
print("="*70)

# 测试不同车辆
vehicles_to_test = [
    'tesla.cybertruck',
    'tesla.model3',
    'audi.a2',
    'toyota.prius'
]

for vehicle_name in vehicles_to_test:
    print(f"\n{'='*70}")
    print(f"测试车辆: {vehicle_name}")
    print(f"{'='*70}")

    config = Config()
    config.scenario = "plain"  # 使用plain场景，没有障碍物
    config.render = False
    config.vehicle_name = vehicle_name

    try:
        env = CarlaEnv(config, 2000, 8000)
        env.reset()

        start_loc = env.ego.get_location()

        # 测试10步
        for step in range(10):
            action = np.array([1.0, 0.0], dtype=np.float32)  # 全油门
            obs, reward, done, info = env.step(action)

        end_loc = env.ego.get_location()
        final_vel = env.ego.get_velocity()
        final_speed = np.sqrt(final_vel.x**2 + final_vel.y**2 + final_vel.z**2)
        distance = np.sqrt((end_loc.x - start_loc.x)**2 + (end_loc.y - start_loc.y)**2)

        # 获取车辆物理属性
        physics = env.ego.get_physics_control()
        mass = physics.mass

        print(f"  质量: {mass:.0f} kg")
        print(f"  10步后速度: {final_speed:.2f} m/s")
        print(f"  10步移动距离: {distance:.2f} m")
        print(f"  平均速度: {distance / (10 * 0.05):.2f} m/s")

        if distance > 5:
            print(f"  ✅ 物理表现正常")
        else:
            print(f"  ⚠️  移动距离太短")

        env.close()

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

print(f"\n{'='*70}")
print("测试完成")
print("="*70)
