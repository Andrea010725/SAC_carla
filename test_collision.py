"""测试collision检测是否正常工作"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("Collision检测测试")
print("="*70)

# 创建配置
config = Config()
config.scenario = "parked_obstacles"
config.render = False

# 创建环境
print("\n[1] 创建环境...")
env = CarlaEnv(config, 2000, 8000)
print("✅ 环境创建成功")

# 重置环境
print("\n[2] 重置环境...")
env.reset()
print("✅ 环境重置成功")

# 测试几步
print("\n[3] 测试10步，观察collision...")
collision_detected = False

for step in range(10):
    # 随机action
    action = np.array([0.5, 0.0], dtype=np.float32)  # 直行

    # 执行
    obs, reward, done, info = env.step(action)

    # 检查collision
    collision_in_info = info.get('collision', 0.0) if info else 0.0
    collision_in_env = env.collision

    print(f"  Step {step+1}:")
    print(f"    - info['collision']: {collision_in_info}")
    print(f"    - env.collision: {collision_in_env}")
    print(f"    - done: {done}")
    print(f"    - reward: {reward:.2f}")

    if collision_in_info > 0:
        collision_detected = True
        print(f"    ✅ 检测到碰撞!")

    if done:
        print(f"    Episode结束")
        break

print("\n" + "="*70)
if collision_detected:
    print("✅ Collision检测正常工作")
else:
    print("⚠️  未检测到collision（可能没有撞到，或者检测有问题）")
print("="*70)

env.close()
