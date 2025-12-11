"""详细打印每步的位置变化"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("位置变化详细诊断")
print("="*70)

config = Config()
config.scenario = "plain"  # 使用plain场景
config.render = False

env = CarlaEnv(config, 2000, 8000)
env.reset()

print("\n测试10步，打印每步的详细位置:")
print("-"*70)

prev_loc = env.ego.get_location()
print(f"起始: x={prev_loc.x:.4f}, y={prev_loc.y:.4f}, z={prev_loc.z:.4f}")

for step in range(10):
    action = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)

    loc = env.ego.get_location()
    vel = env.ego.get_velocity()
    vel_scalar = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # 计算位移
    dx = loc.x - prev_loc.x
    dy = loc.y - prev_loc.y
    dz = loc.z - prev_loc.z
    dist = np.sqrt(dx**2 + dy**2)

    print(f"Step {step+1:2d}:")
    print(f"  位置: x={loc.x:.4f}, y={loc.y:.4f}, z={loc.z:.4f}")
    print(f"  位移: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, dist={dist:.4f}m")
    print(f"  速度: {vel_scalar:.2f} m/s")

    prev_loc = loc

print("-"*70)
env.close()
