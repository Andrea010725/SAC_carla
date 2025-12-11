"""详细的collision诊断 - 监控每一步的collision状态"""
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config
from carla_base.carla_env import CarlaEnv

print("="*70)
print("Collision详细诊断")
print("="*70)

# 创建配置
config = Config()
config.scenario = "parked_obstacles"
config.render = False

# 创建环境
print("\n[1] 创建环境...")
env = CarlaEnv(config, 2000, 8000)
print("✅ 环境创建成功")

# 添加collision监控
collision_events = []
original_on_collision = env._on_collision

def monitored_on_collision(event):
    print(f"  🔴 COLLISION EVENT! Actor: {event.other_actor.type_id}")
    collision_events.append(event)
    original_on_collision(event)

env._on_collision = monitored_on_collision

# 重置环境
print("\n[2] 重置环境...")
env.reset()
print("✅ 环境重置成功")
print(f"  - collision_sensor存在: {env.collision_sensor is not None}")
print(f"  - 初始collision状态: {env.collision}")

# 测试20步
print("\n[3] 测试20步，强制直行撞向障碍物...")
print("-"*70)

for step in range(20):
    # 强制直行
    action = np.array([0.8, 0.0], dtype=np.float32)  # 大油门直行

    # 记录step前的状态
    collision_before = env.collision

    # 执行
    obs, reward, done, info = env.step(action)

    # 记录step后的状态
    collision_after_step = env.collision  # step内部会清零
    collision_in_info = info.get('collision', 0.0) if info else 0.0

    # 获取位置和速度
    loc = env.ego.get_location()
    vel = obs[8] if len(obs) > 8 else 0.0

    print(f"Step {step+1:2d}:")
    print(f"  位置: ({loc.x:.1f}, {loc.y:.1f}), 速度: {vel:.2f} m/s")
    print(f"  collision_before: {collision_before}")
    print(f"  collision_after_step: {collision_after_step}")
    print(f"  info['collision']: {collision_in_info}")
    print(f"  reward: {reward:.2f}, done: {done}")
    print(f"  collision_events数量: {len(collision_events)}")

    if collision_in_info > 0:
        print(f"  ✅ info中检测到collision!")

    if len(collision_events) > 0:
        print(f"  ✅ collision_sensor触发了 {len(collision_events)} 次")

    if done:
        print(f"  Episode结束")
        break

    print()

print("-"*70)
print(f"\n总结:")
print(f"  - 总步数: {step+1}")
print(f"  - collision_sensor触发次数: {len(collision_events)}")
print(f"  - 最终done: {done}")

if len(collision_events) > 0:
    print(f"\n✅ Collision sensor正常工作")
    print(f"  触发的collision事件:")
    for i, event in enumerate(collision_events):
        print(f"    {i+1}. 撞到: {event.other_actor.type_id}")
else:
    print(f"\n⚠️  Collision sensor未触发")
    print(f"  可能原因:")
    print(f"    1. 没有撞到任何东西（停车障碍物太远）")
    print(f"    2. Collision sensor没有正确attach")
    print(f"    3. Collision sensor的listen没有工作")

print("="*70)

env.close()
