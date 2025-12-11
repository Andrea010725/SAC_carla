#!/usr/bin/env python
"""
快速测试锥桶场景是否正常工作
"""
import sys
sys.path.append("/home/ajifang/carla/PythonAPI/carla/")
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

from config import Config
from carla_base.carla_env import CarlaEnv
import time

def test_cones_scene():
    print("=" * 60)
    print("锥桶场景测试")
    print("=" * 60)

    config = Config()
    print(f"\n配置信息:")
    print(f"  场景模式: {config.scenario}")
    print(f"  地图: {config.map_name}")
    print(f"  锥桶数量: {config.cone_num}")
    print(f"  纵向间距: {config.cone_step_behind} m")
    print(f"  横向推进: {config.cone_step_lateral} m")

    print("\n创建环境...")
    env = CarlaEnv(config, config.test_carla_port, config.test_carla_tm_port)

    print("\n重置环境（放置锥桶）...")
    obs = env.reset()

    print("\n✅ 环境重置完成！")
    print(f"初始观测: {obs}")

    # 检查是否有锥桶被放置
    if hasattr(env, '_actors') and env._actors:
        cone_count = sum(1 for actor in env._actors if 'cone' in actor.type_id.lower())
        print(f"\n🎯 场景中的锥桶数量: {cone_count}")

        if cone_count > 0:
            print("✅ 锥桶已成功放置！")
            print("\n提示: 在CARLA窗口中应该能看到橙色的锥桶")
            print("      如果看不到，可能是相机角度问题，尝试调整视角")
        else:
            print("⚠️  没有检测到锥桶，可能放置失败")
    else:
        print("⚠️  无法检查锥桶（_actors属性不存在）")

    print("\n保持环境运行10秒，请在CARLA窗口中查看...")
    for i in range(10):
        time.sleep(1)
        print(f"  {10-i} 秒...")

    print("\n关闭环境...")
    env.close()
    print("✅ 测试完成！")

if __name__ == "__main__":
    try:
        test_cones_scene()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
