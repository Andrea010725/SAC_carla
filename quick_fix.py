#!/usr/bin/env python
"""
快速修复Traffic Manager端口冲突
"""
import sys
sys.path.append("/home/ajifang/carla/PythonAPI/carla/")
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

import carla
import time

def fix_tm_port(carla_port=2000, tm_port=3000):
    """重置Traffic Manager"""
    try:
        print(f"连接到CARLA (端口 {carla_port})...")
        client = carla.Client("127.0.0.1", carla_port)
        client.set_timeout(10.0)

        world = client.get_world()
        print(f"当前地图: {world.get_map().name}")

        # 尝试获取TM（这会自动创建或连接）
        print(f"重置Traffic Manager (端口 {tm_port})...")
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(False)
        time.sleep(0.5)
        tm.set_synchronous_mode(True)

        print("✅ Traffic Manager重置成功！")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Traffic Manager端口修复工具")
    print("=" * 60)

    carla_port = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    tm_port = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    success = fix_tm_port(carla_port, tm_port)

    if success:
        print("\n现在可以运行测试了:")
        print("  python main.py")
    else:
        print("\n如果还是失败，请运行:")
        print("  ./cleanup_ports.sh")
        print("  然后重启CARLA服务器")

    sys.exit(0 if success else 1)
