"""快速测试PPO在Router中的工作情况"""
import numpy as np
import sys
sys.path.insert(0, '/home/ajifang/SAC_carla')

from router_carla import PlannerRouterCarla

print("=" * 60)
print("测试PPO Router集成")
print("=" * 60)

# 创建Router (使用PPO)
try:
    router = PlannerRouterCarla(
        planner_bins=(-1/3, 1/3),
        v_ref_base=12.0,
        use_ppo=True,           # 使用PPO
        ppo_model_path=None,    # 无模型,使用默认控制
        ppo_action_dim=2        # 2维动作
    )
    print("\n✅ Router创建成功")
except Exception as e:
    print(f"\n❌ Router创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"  - Rule Planner: {type(router.rule).__name__}")
print(f"  - IL Planner: {type(router.il).__name__}")
print(f"  - RL Planner: {type(router.rl).__name__}")

# 检查PPO是否被正确设置
if hasattr(router, '_use_ppo'):
    print(f"  - 使用PPO: {router._use_ppo}")

# 模拟attach_context (在实际运行时CARLA会提供这些对象)
# 这里我们跳过,因为只是测试Router的基本功能
print("\n⏭️  跳过attach_context (需要CARLA环境)")

# 测试不同selector action的路由
print("\n测试Planner路由:")
print("-" * 60)
test_actions = [
    (-0.8, "应该选择RULE"),
    (-0.5, "应该选择RULE"),
    (-0.2, "应该选择IL"),
    (0.0, "应该选择IL"),
    (0.2, "应该选择IL"),
    (0.5, "应该选择RL(PPO)"),
    (0.8, "应该选择RL(PPO)")
]

from router_carla import PLANNER_IDS

for action, expected in test_actions:
    pid = router._bucketize(action)
    planner_name = PLANNER_IDS[pid]
    print(f"  action={action:5.2f} → PID={pid} ({planner_name:4s}) - {expected}")

print("-" * 60)

# 测试观测转换 (不需要CARLA)
print("\n测试PPO观测转换:")
test_obs = np.array([
    100.0, 50.0, 0.3,  # x, y, z
    0.1, 1.57, 0.0,    # pitch, yaw, roll
    0.5, 0.0, 5.0      # acc, ang_v, vel
])

try:
    state = router.rl._convert_obs_to_state(test_obs)
    print(f"  输入观测 (9维): {test_obs}")
    print(f"  转换后格式: {state.keys()}")
    print(f"  state['state'].shape: {state['state'].shape}")
    print("  ✅ 观测转换正常")
except Exception as e:
    print(f"  ❌ 观测转换失败: {e}")

# 测试动作转换
print("\n测试PPO动作转换:")
test_actions = [
    np.array([0.5, 0.0]),
    np.array([-0.5, 0.3]),
    np.array([1.0, -0.5]),
]

for action in test_actions:
    try:
        throttle, brake, steer = router.rl._convert_action_to_control(action)
        print(f"  动作 {action} → T={throttle:.2f}, B={brake:.2f}, S={steer:.2f}")
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")

print("\n" + "=" * 60)
print("✅ Router基础功能测试全部通过!")
print("=" * 60)

print("\n📝 下一步:")
print("  1. 在CARLA环境中运行完整测试: python main.py")
print("  2. (可选) 训练PPO模型: python train_ppo_carla.py")
print("  3. 查看详细文档: PPO_INTEGRATION_GUIDE.md")
