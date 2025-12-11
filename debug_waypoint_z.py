"""检查waypoint的z坐标"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 添加CARLA路径
CARLA_ROOT = "/home/ajifang/carla"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-0.9.15-py3.7-linux-x86_64.egg"))

import carla

print("="*70)
print("Waypoint Z坐标检查")
print("="*70)

# 连接CARLA
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

print(f"\n当前地图: {carla_map.name}")

# 获取一些spawn points
spawn_points = carla_map.get_spawn_points()
print(f"\n检查前5个spawn points:")

for i, spawn in enumerate(spawn_points[:5]):
    loc = spawn.location

    # 获取对应的waypoint
    wp = carla_map.get_waypoint(loc, project_to_road=True)

    print(f"\nSpawn {i+1}:")
    print(f"  原始spawn.location.z: {loc.z:.4f}")
    print(f"  waypoint.transform.location.z: {wp.transform.location.z:.4f}")
    print(f"  差值: {loc.z - wp.transform.location.z:.4f}")

print("\n" + "="*70)
print("结论:")
print("  - 如果waypoint.z接近0，说明waypoint是正确的")
print("  - 如果spawn.z比waypoint.z高很多，说明spawn points本身就在空中")
print("="*70)
