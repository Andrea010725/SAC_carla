# 🚗 新场景实现完整指南

## 📋 目录
1. [架构分析](#架构分析)
2. [接口规范](#接口规范)
3. [实现的场景](#实现的场景)
4. [使用方法](#使用方法)

---

## 🏗️ 架构分析

### **现状说明**

你的 `new_scenarios/` 文件夹包含 **30+ 个 ScenarioRunner 框架的场景**，但这些场景与你现有的 RL 训练系统**完全不兼容**。

### **两套系统对比**

| 对比项 | 你的系统 (scenario_manager.py) | new_scenarios/ (ScenarioRunner) |
|--------|-------------------------------|--------------------------------|
| **基类** | `ScenarioBase` | `BasicScenario` |
| **依赖** | 仅 CARLA API | py_trees + ScenarioRunner 框架 |
| **初始化方法** | `setup()` | `_initialize_actors()` + `_create_behavior()` |
| **控制方式** | 直接生成静态/动态 actors | 行为树 (Behavior Tree) |
| **配置对象** | 简单 `Config` 类 | `ScenarioConfiguration` 类 |
| **与 RL 集成** | ✅ 已集成 | ❌ 需要大量改造 |

---

## 🔌 接口规范（必须遵守）

### **输入接口**

```python
def __init__(self, world: carla.World, carla_map: carla.Map, config: Config):
    """
    Args:
        world: CARLA 世界对象
        carla_map: CARLA 地图对象
        config: 配置对象（来自 config.py 的 Config 类）
    """
```

### **输出接口**

```python
def setup(self) -> bool:
    """
    场景初始化，生成所有障碍物/行人/车辆

    Returns:
        bool: 是否成功初始化
    """

def get_spawn_transform(self) -> Optional[carla.Transform]:
    """
    返回自车生成位置

    Returns:
        carla.Transform: 自车的 spawn 位置
    """

def get_obstacle_actors(self) -> List[carla.Actor]:
    """
    返回场景中的障碍物列表（用于观测）

    Returns:
        List[carla.Actor]: 障碍物列表
    """

def cleanup(self):
    """
    清理场景中生成的所有 actors
    """
```

### **关键要求**

1. ✅ **输入参数必须是** `(world, carla_map, config)`
2. ✅ **config 必须是** `Config` 类实例（不是 `ScenarioConfiguration`）
3. ✅ **必须实现** `setup()` 和 `get_spawn_transform()`
4. ✅ **所有生成的 actors 必须添加到** `self.scenario_actors` 列表
5. ✅ **cleanup() 必须清理所有 actors**

---

## 🎯 已实现的场景

### **1. 行人过马路场景 (PedestrianCrossingScenario)**

**场景描述：**
- 在自车前方的人行横道上生成多个行人
- 行人从道路一侧横穿到另一侧
- 自车需要减速避让行人

**配置参数：**
```python
# config.py
self.pedestrian_distance = 25.0      # 行人距离自车的距离（米）
self.num_pedestrians = 3             # 行人数量
self.pedestrian_speed = 1.5          # 行人速度（m/s）
self.pedestrian_spacing = 2.0        # 行人间距（米）
```

**训练价值：**
- ⭐⭐ 难度：简单
- 测试行人检测能力
- 测试紧急制动能力
- 真实场景常见

---

### **2. 车门突然打开场景 (VehicleOpensDoorScenario)**

**场景描述：**
- 在自车前方路边停放一辆车
- 当自车接近时，停放车辆突然打开车门
- 自车需要紧急避让或变道

**配置参数：**
```python
# config.py
self.door_vehicle_distance = 30.0    # 停放车辆距离（米）
self.door_trigger_distance = 15.0    # 触发距离（米）
self.door_side = "left"              # 车门侧（"left"/"right"/"random"）
```

**训练价值：**
- ⭐⭐⭐ 难度：中等
- 测试紧急避让能力
- 测试变道决策
- 真实场景常见（城市道路）

---

### **3. 切入场景 (CutInScenario)**

**场景描述：**
- 自车在道路上行驶
- 相邻车道的车辆突然切入到自车前方
- 自车需要减速避免碰撞

**配置参数：**
```python
# config.py
self.cutin_vehicle_distance = 40.0   # 切入车辆初始距离（米）
self.cutin_trigger_distance = 30.0   # 触发距离（米）
self.cutin_direction = "left"        # 切入方向（"left"/"right"/"random"）
self.cutin_vehicle_speed = 10.0      # 切入车辆速度（m/s）
```

**训练价值：**
- ⭐⭐⭐⭐ 难度：困难
- 测试紧急制动能力
- 测试车辆检测和跟踪
- 真实场景常见（高速公路）

---

### **4. 停车场出口场景 (ParkingExitScenario)**

**场景描述：**
- 自车在道路上行驶
- 路边停车场有车辆突然驶出
- 自车需要减速避让

**配置参数：**
```python
# config.py
self.parking_exit_distance = 35.0    # 停车场距离（米）
self.parking_vehicle_speed = 3.0     # 驶出车辆速度（m/s）
self.parking_trigger_distance = 20.0 # 触发距离（米）
```

**训练价值：**
- ⭐⭐⭐ 难度：中等
- 测试侧方车辆检测
- 测试速度控制
- 真实场景常见

---

## 📝 实现代码

所有场景的实现代码已添加到 `carla_base/scenario_manager.py` 文件中。

### **场景注册**

```python
# carla_base/scenario_manager.py

class ScenarioFactory:
    SCENARIOS = {
        # 已有场景
        "parked_obstacles": ParkedObstaclesScenario,
        "cones": ConesScenario,
        "jaywalker": JaywalkerScenario,
        "trimma": TrimmaScenario,
        "construction_lane_change": ConstructionLaneChangeScenario,

        # ✅ 新增场景
        "pedestrian_crossing": PedestrianCrossingScenario,
        "vehicle_opens_door": VehicleOpensDoorScenario,
        "cut_in": CutInScenario,
        "parking_exit": ParkingExitScenario,
    }
```

---

## 🚀 使用方法

### **步骤 1：配置场景参数**

在 `config.py` 中添加场景配置：

```python
# config.py

class Config:
    def __init__(self):
        # ... 现有配置 ...

        # ===== Scenario =====
        self.scenario = "pedestrian_crossing"  # 选择场景
        self.random_scenario = True
        self.scenario_pool = [
            "parked_obstacles",
            "cones",
            "pedestrian_crossing",  # ✅ 新场景
            "vehicle_opens_door",   # ✅ 新场景
            "cut_in",               # ✅ 新场景
            "parking_exit",         # ✅ 新场景
        ]

        # ===== Pedestrian Crossing 配置 =====
        self.pedestrian_distance = 25.0
        self.num_pedestrians = 3
        self.pedestrian_speed = 1.5
        self.pedestrian_spacing = 2.0

        # ===== Vehicle Opens Door 配置 =====
        self.door_vehicle_distance = 30.0
        self.door_trigger_distance = 15.0
        self.door_side = "random"

        # ===== Cut In 配置 =====
        self.cutin_vehicle_distance = 40.0
        self.cutin_trigger_distance = 30.0
        self.cutin_direction = "random"
        self.cutin_vehicle_speed = 10.0

        # ===== Parking Exit 配置 =====
        self.parking_exit_distance = 35.0
        self.parking_vehicle_speed = 3.0
        self.parking_trigger_distance = 20.0
```

### **步骤 2：测试单个场景**

```bash
# 创建测试脚本
python test_new_scenario.py --scenario pedestrian_crossing
```

### **步骤 3：开始训练**

```bash
# 使用现有训练脚本，会自动从 scenario_pool 中随机选择场景
python train_ppo_with_wandb.py
```

---

## 🔍 场景测试脚本

创建 `test_new_scenario.py`：

```python
import carla
import time
from config import Config
from carla_base.scenario_manager import ScenarioFactory

def test_scenario(scenario_name):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario_name}")
    print(f"{'='*60}\n")

    # 连接 CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    # 创建配置
    config = Config()
    config.scenario = scenario_name

    # 创建场景
    scenario = ScenarioFactory.create_scenario(
        scenario_name=scenario_name,
        world=world,
        carla_map=carla_map,
        config=config
    )

    if scenario is None:
        print(f"❌ 场景 {scenario_name} 不存在")
        return False

    # 初始化场景
    print(f"🔄 正在初始化场景...")
    success = scenario.setup()

    if success:
        print(f"✅ 场景初始化成功！")

        # 获取spawn位置
        spawn_tf = scenario.get_spawn_transform()
        print(f"   📍 自车spawn位置: ({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f})")

        # 获取障碍物
        obstacles = scenario.get_obstacle_actors()
        print(f"   🚧 障碍物数量: {len(obstacles)}")

        # 生成自车（用于观察）
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego = world.try_spawn_actor(ego_bp, spawn_tf)

        if ego:
            print(f"   🚗 自车已生成，ID={ego.id}")

            # 设置观察者视角
            spectator = world.get_spectator()
            spectator_tf = carla.Transform(
                spawn_tf.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(spectator_tf)

            # 等待观察
            print(f"\n⏳ 等待 10 秒观察场景...")
            for i in range(10):
                world.tick()
                time.sleep(1)
                print(f"   {10-i} 秒...")

            # 清理自车
            ego.destroy()

        # 清理场景
        print(f"\n🧹 正在清理场景...")
        scenario.cleanup()
        print(f"✅ 场景清理完成")
        return True
    else:
        print(f"❌ 场景初始化失败")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='pedestrian_crossing',
                       help='场景名称')
    args = parser.parse_args()

    test_scenario(args.scenario)
```

---

## 📊 场景优先级推荐

### **第一批（简单，建议先实现）：**
1. ✅ `pedestrian_crossing` - 行人过马路
2. ✅ `vehicle_opens_door` - 车门突然打开
3. ✅ `parking_exit` - 停车场出口

### **第二批（中等难度）：**
4. ✅ `cut_in` - 切入场景
5. `signalized_junction_left_turn` - 信号灯左转
6. `signalized_junction_right_turn` - 信号灯右转

### **第三批（困难）：**
7. `highway_cut_in` - 高速公路切入
8. `change_lane` - 变道场景
9. `actor_flow` - 交通流场景

---

## ⚠️ 注意事项

### **1. 接口一致性**
- ✅ 所有场景必须继承 `ScenarioBase`
- ✅ 必须实现 `setup()` 和 `get_spawn_transform()`
- ✅ 输入参数必须是 `(world, carla_map, config)`

### **2. Actor 管理**
- ✅ 所有生成的 actors 必须添加到 `self.scenario_actors`
- ✅ `cleanup()` 必须清理所有 actors
- ✅ 行人控制器也需要清理

### **3. 配置参数**
- ✅ 使用 `getattr(config, "param_name", default_value)` 读取配置
- ✅ 在 `config.py` 中添加对应的配置参数
- ✅ 提供合理的默认值

### **4. 场景注册**
- ✅ 在 `ScenarioFactory.SCENARIOS` 中注册新场景
- ✅ 在 `config.scenario_pool` 中添加场景名称

---

## 🎓 实现模板

```python
class YourNewScenario(ScenarioBase):
    """场景描述"""

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "your_scenario_name"
        self.scenario_description = "场景描述"

        # 读取配置参数
        self.param1 = float(getattr(config, "param1", default_value))

        # 内部状态
        self.ego_spawn_transform: Optional[carla.Transform] = None

    def setup(self) -> bool:
        """场景初始化"""
        # 1. 选择spawn点
        # 2. 生成障碍物/行人/车辆
        # 3. 等待物理稳定
        return True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform
```

---

## 📞 支持

如果遇到问题，请检查：
1. CARLA 服务器是否正常运行
2. 配置参数是否正确
3. 场景是否已注册到 `ScenarioFactory`
4. 接口是否符合规范

---

**祝训练顺利！🚀**
