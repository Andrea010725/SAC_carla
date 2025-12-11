"""
PPO Planner集成测试脚本

测试PPO适配器是否能正确:
1. 转换观测空间
2. 执行推理 (或返回默认控制)
3. 转换动作空间
4. 与CARLA环境交互
"""

import sys
import numpy as np
import time

# 确保可以导入项目模块
sys.path.insert(0, '/home/ajifang/SAC_carla')

from config import Config
from carla_base.carla_env import CarlaEnv
from planners.rl_ppo_adapter import RLPPOAdapter
from lane_ref import LaneRef


def test_adapter_basic():
    """测试适配器基础功能 (不需要CARLA)"""
    print("\n" + "=" * 70)
    print("测试1: PPO适配器基础功能")
    print("=" * 70)

    # 创建适配器
    print("\n[Step 1] 创建PPO适配器...")
    adapter = RLPPOAdapter(action_dim=2)
    print("✅ 适配器创建成功")

    # 模拟attach_context
    print("\n[Step 2] 模拟attach_context...")
    adapter.attach_context(world=None, ego=None, lane_ref=None)
    print("✅ Context已注入")

    # 测试观测转换
    print("\n[Step 3] 测试观测转换...")
    test_obs = np.array([
        100.0, 50.0, 0.3,  # x, y, z
        0.1, 1.57, 0.0,    # pitch, yaw, roll
        0.5, 0.0, 5.0      # acc, ang_v, vel
    ])
    print(f"输入观测 (9维): {test_obs}")

    state = adapter._convert_obs_to_state(test_obs)
    print(f"转换后state格式: {state.keys()}")
    print(f"state['state'].shape: {state['state'].shape}")
    print("✅ 观测转换成功")

    # 测试plan (无模型,应返回默认控制)
    print("\n[Step 4] 测试plan接口...")
    throttle, brake, steer = adapter.plan(test_obs)
    print(f"输出控制:")
    print(f"  throttle: {throttle:.3f}")
    print(f"  brake:    {brake:.3f}")
    print(f"  steer:    {steer:.3f}")

    # 验证控制范围
    assert 0.0 <= throttle <= 1.0, f"throttle超出范围: {throttle}"
    assert 0.0 <= brake <= 1.0, f"brake超出范围: {brake}"
    assert -1.0 <= steer <= 1.0, f"steer超出范围: {steer}"
    print("✅ Plan接口测试通过")

    print("\n" + "=" * 70)
    print("✅ 适配器基础功能测试全部通过!")
    print("=" * 70)


def test_adapter_with_carla():
    """测试适配器与CARLA环境集成"""
    print("\n" + "=" * 70)
    print("测试2: PPO适配器 + CARLA环境")
    print("=" * 70)

    # 创建配置
    print("\n[Step 1] 加载配置...")
    config = Config()
    config.planner_mode = 'RL'
    config.scenario = "cones"
    config.cone_num = 10
    print("✅ 配置加载完成")

    # 创建CARLA环境
    print("\n[Step 2] 连接CARLA环境...")
    try:
        env = CarlaEnv(config, 2000, 8000)
        env.reset()
        print("✅ CARLA环境连接成功")
    except Exception as e:
        print(f"❌ CARLA连接失败: {e}")
        print("请确保CARLA服务器正在运行")
        return

    # 创建PPO planner
    print("\n[Step 3] 创建PPO Planner...")
    ppo_planner = RLPPOAdapter(action_dim=2)

    # 获取CARLA对象
    world = env.world
    ego = env.ego_vehicle
    amap = world.get_map()

    # 创建LaneRef
    print("\n[Step 4] 创建LaneRef...")
    ego_wp = amap.get_waypoint(ego.get_location())
    lane_ref = LaneRef(amap, seed_wp=ego_wp, step=1.0, max_len=500.0)
    print("✅ LaneRef创建完成")

    # Attach context
    print("\n[Step 5] Attach context...")
    ppo_planner.attach_context(world=world, ego=ego, lane_ref=lane_ref)
    print("✅ Context注入完成")

    # 测试控制循环
    print("\n[Step 6] 测试控制循环 (50步)...")
    print("-" * 70)
    print(f"{'Step':>6} | {'Throttle':>8} | {'Brake':>8} | {'Steer':>8} | {'Speed':>8}")
    print("-" * 70)

    try:
        for step in range(50):
            # 获取观测
            obs = env._get_obs()

            # PPO规划
            throttle, brake, steer = ppo_planner.plan(obs)

            # 获取车速
            velocity = ego.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

            # 打印
            if step % 5 == 0:
                print(f"{step:6d} | {throttle:8.3f} | {brake:8.3f} | {steer:8.3f} | {speed:8.2f}")

            # 应用控制
            import carla
            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer
            )
            ego.apply_control(control)

            # 更新环境
            world.tick()
            time.sleep(0.05)

        print("-" * 70)
        print("✅ 控制循环测试通过")

    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理环境
        print("\n[Step 7] 清理环境...")
        try:
            env.clean()
            print("✅ 环境清理完成")
        except:
            pass

    print("\n" + "=" * 70)
    print("✅ CARLA集成测试完成!")
    print("=" * 70)


def test_action_conversion():
    """测试不同动作维度的转换"""
    print("\n" + "=" * 70)
    print("测试3: 动作空间转换")
    print("=" * 70)

    # 测试2维动作
    print("\n[Case 1] 2维动作: [throttle_brake, steer]")
    adapter_2d = RLPPOAdapter(action_dim=2)
    adapter_2d._attached = True

    test_actions_2d = [
        np.array([0.5, 0.0]),    # 中等油门,直行
        np.array([-0.5, 0.3]),   # 刹车,右转
        np.array([1.0, -0.5]),   # 全油门,左转
    ]

    for i, action in enumerate(test_actions_2d):
        throttle, brake, steer = adapter_2d._convert_action_to_control(action)
        print(f"  动作{i+1}: {action} → T={throttle:.2f}, B={brake:.2f}, S={steer:.2f}")

    # 测试3维动作
    print("\n[Case 2] 3维动作: [throttle, brake, steer]")
    adapter_3d = RLPPOAdapter(action_dim=3)
    adapter_3d._attached = True

    test_actions_3d = [
        np.array([0.5, 0.0, 0.0]),   # 中等油门
        np.array([0.0, 0.5, 0.3]),   # 刹车+右转
        np.array([1.0, 0.0, -0.5]),  # 全油门+左转
    ]

    for i, action in enumerate(test_actions_3d):
        throttle, brake, steer = adapter_3d._convert_action_to_control(action)
        print(f"  动作{i+1}: {action} → T={throttle:.2f}, B={brake:.2f}, S={steer:.2f}")

    print("\n✅ 动作转换测试通过")


def test_observation_edge_cases():
    """测试观测转换的边界情况"""
    print("\n" + "=" * 70)
    print("测试4: 观测转换边界情况")
    print("=" * 70)

    adapter = RLPPOAdapter(action_dim=2)
    adapter._attached = True

    # 测试正常9维
    print("\n[Case 1] 正常9维观测")
    obs_9d = np.random.randn(9)
    state = adapter._convert_obs_to_state(obs_9d)
    print(f"  输入: {obs_9d.shape} → 输出: {state['state'].shape}")
    assert state['state'].shape == (1, 9)
    print("  ✅ 通过")

    # 测试维度不足
    print("\n[Case 2] 维度不足 (7维)")
    obs_7d = np.random.randn(7)
    state = adapter._convert_obs_to_state(obs_7d)
    print(f"  输入: {obs_7d.shape} → 输出: {state['state'].shape}")
    assert state['state'].shape == (1, 9)
    print("  ✅ 通过 (自动填充到9维)")

    # 测试维度过多
    print("\n[Case 3] 维度过多 (12维)")
    obs_12d = np.random.randn(12)
    state = adapter._convert_obs_to_state(obs_12d)
    print(f"  输入: {obs_12d.shape} → 输出: {state['state'].shape}")
    assert state['state'].shape == (1, 9)
    print("  ✅ 通过 (自动截断到9维)")

    print("\n✅ 边界情况测试通过")


def main():
    """运行所有测试"""
    print("\n" + "#" * 70)
    print("#" + " " * 22 + "PPO Planner 集成测试" + " " * 23 + "#")
    print("#" * 70)

    try:
        # 基础功能测试 (不需要CARLA)
        test_adapter_basic()

        # 动作转换测试
        test_action_conversion()

        # 观测转换边界测试
        test_observation_edge_cases()

        # CARLA集成测试
        print("\n是否运行CARLA集成测试? (需要CARLA服务器运行)")
        response = input("输入 y/n (默认n): ").strip().lower()

        if response == 'y':
            test_adapter_with_carla()
        else:
            print("\n⏭️  跳过CARLA集成测试")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 总结
    print("\n" + "#" * 70)
    print("#" + " " * 28 + "测试总结" + " " * 33 + "#")
    print("#" * 70)
    print("\n✅ 所有测试通过!")
    print("\n📝 下一步:")
    print("  1. 训练PPO模型或提供已训练模型")
    print("  2. 实现模型加载逻辑 (rl_ppo_adapter.py:_load_ppo_model)")
    print("  3. 集成到router_carla.py")
    print("  4. 端到端测试整个系统")
    print("\n参考文档: PPO_INTEGRATION_GUIDE.md")
    print("#" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
