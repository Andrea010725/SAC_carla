#!/usr/bin/env python
"""
快速切换CARLA地图的工具脚本
用法: python load_map.py Town05
"""
import sys
sys.path.append("/home/ajifang/carla/PythonAPI/carla/")
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

import carla
import time

def load_map(map_name="Town05", port=2000):
    """加载指定地图"""
    try:
        print(f"连接到CARLA服务器 (端口 {port})...")
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(60.0)

        world = client.get_world()
        current_map = world.get_map().name
        print(f"当前地图: {current_map}")

        if map_name in current_map:
            print(f"地图已是 {map_name}，无需切换")
            return True

        print(f"正在加载地图: {map_name} (这可能需要20-40秒)...")
        start = time.time()
        world = client.load_world(map_name)
        elapsed = time.time() - start

        print(f"✅ 地图加载成功！耗时 {elapsed:.1f} 秒")
        print(f"新地图: {world.get_map().name}")
        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    map_name = sys.argv[1] if len(sys.argv) > 1 else "Town05"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    print("=" * 60)
    print("CARLA地图加载工具")
    print("=" * 60)

    success = load_map(map_name, port)
    sys.exit(0 if success else 1)
