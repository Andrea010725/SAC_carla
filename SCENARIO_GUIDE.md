# 🎬 场景管理系统使用指南

## 📋 概述

新的场景管理系统提供了规范的接口来创建和管理Corner Cases场景。所有场景代码统一放在 `carla_base/scenario_manager.py` 中。

---

## 🗂️ 场景列表

### 已实现的场景

#### 1. **parked_obstacles** - 停放车辆避让 ✅
- **描述**: 在自车前方12-20米范围内生成多辆静止车辆
- **难度**: 中等
- **配置参数**:
  ```python
  config.scenario = "parked_obstacles"
  config.num_parked_cars = 4                      # 车辆数量
  config.parked_car_spacing = 8.0                 # 车辆间距（米）
  config.parked_car_start_distance_min = 12.0     # 第一辆车最小距离
  config.parked_car_start_distance_max = 20.0     # 第一辆车最大距离
  config.parked_car_offset = 0.0                  # 横向偏移（0=车道中心）
  ```

### 待实现的场景

#### 2. **cones** - 锥桶场景 ⏳
- **描述**: 车道一侧放置锥桶，逐渐向中心靠近
- **难度**: 简单
- **TODO**: 实现 `ConesScenario.setup()` 方法
- **配置参数**（待定义）:
  ```python
  config.scenario = "cones"
  config.cone_num = 15                    # 锥桶数量
  config.cone_step_behind = 3.0           # 纵向间距（米）
  config.cone_step_lateral = 0.4          # 横向递进（米）
  config.cone_z_offset = 0.0              # 高度偏移
  config.cone_lane_margin = 0.25          # 距车道边缘最小距离
  ```

#### 3. **jaywalker** - 鬼探头 ⏳
- **描述**: 行人从路边或停放车辆后突然冲出
- **难度**: 困难
- **TODO**: 实现 `JaywalkerScenario.setup()` 方法
- **配置参数**（待定义）:
  ```python
  config.scenario = "jaywalker"
  config.jaywalker_distance = 15.0        # 行人出现距离（米）
  config.jaywalker_speed = 2.0            # 行人速度（m/s）
  config.jaywalker_trigger_distance = 20.0 # 触发距离
  config.use_occlusion = True             # 是否使用遮挡物
  ```

#### 4. **trimma** - Trimma场景 ⏳
- **描述**: TODO - 请补充场景描述
- **难度**: 待定
- **TODO**:
  1. 定义场景描述
  2. 定义配置参数
  3. 实现 `TrimmaScenario.setup()` 方法

#### 5. **construction_lane_change** - 施工+变道高交通流 ⏳
- **描述**: 前方车道施工，需要变道避让，相邻车道有高密度交通流
- **难度**: 非常困难
- **TODO**: 实现 `ConstructionLaneChangeScenario.setup()` 方法
- **配置参数**（待定义）:
  ```python
  config.scenario = "construction_lane_change"
  config.construction_distance = 30.0     # 施工区域距离（米）
  config.construction_length = 20.0       # 施工区域长度（米）
  config.traffic_density = 3.0            # 交通流密度（车/100米）
  config.traffic_speed = 10.0             # 交通流速度（m/s）
  config.min_gap_for_lane_change = 15.0   # 最小变道gap（米）
  ```

---

## 🏗️ 场景架构

### 类继承关系

```
ScenarioBase (基类)
├── ParkedObstaclesScenario (已实现)
├── ConesScenario (待实现)
├── JaywalkerScenario (待实现)
├── TrimmaScenario (待实现)
└── ConstructionLaneChangeScenario (待实现)
```

### 核心接口

每个场景类必须实现：

```python
class YourScenario(ScenarioBase):
    def setup(self) -> bool:
        """
        场景初始化 - 生成障碍物、设置环境等

        Returns:
            bool: 是否成功初始化
        """
        # 1. 选择合适的起始位置
        # 2. 生成障碍物/行人/车辆
        # 3. 设置自车生成位置
        # 4. 返回成功/失败
        pass

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """
        获取自车生成位置

        Returns:
            carla.Transform: 自车生成的Transform
        """
        return self.ego_spawn_transform
```

---

## 📝 如何添加新场景

### 步骤1: 在 `scenario_manager.py` 中创建场景类

```python
class YourNewScenario(ScenarioBase):
    """
    你的新场景

    场景描述：
    - 详细描述场景内容
    - 说明难点和挑战

    配置参数：
    - param1: 参数1说明
    - param2: 参数2说明
    """

    def __init__(self, world: carla.World, carla_map: carla.Map, config: Any):
        super().__init__(world, carla_map, config)
        self.scenario_name = "your_new_scenario"
        self.scenario_description = "你的新场景描述"

        # 读取配置参数
        self.param1 = float(getattr(config, "param1", 10.0))
        self.param2 = int(getattr(config, "param2", 5))

        self.ego_spawn_transform: Optional[carla.Transform] = None

    def setup(self) -> bool:
        """实现场景生成逻辑"""
        print(f"\n[YourNewScenario] 开始生成场景...")

        # TODO: 实现你的场景生成逻辑
        # 1. 选择起始位置
        # 2. 生成障碍物
        # 3. 设置自车spawn位置

        return True  # 成功返回True

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """返回自车生成位置"""
        return self.ego_spawn_transform
```

### 步骤2: 在 `ScenarioFactory` 中注册场景

```python
class ScenarioFactory:
    SCENARIOS = {
        "parked_obstacles": ParkedObstaclesScenario,
        "cones": ConesScenario,
        "jaywalker": JaywalkerScenario,
        "trimma": TrimmaScenario,
        "construction_lane_change": ConstructionLaneChangeScenario,
        "your_new_scenario": YourNewScenario,  # 添加这一行
    }
```

### 步骤3: 在 `config.py` 中添加默认配置

```python
# 在 Config 类的 __init__ 中添加：
self.scenario = "your_new_scenario"
self.param1 = 10.0
self.param2 = 5
```

### 步骤4: 在训练脚本中使用

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "your_new_scenario"
config.param1 = 15.0
config.param2 = 10
```

---

## 🔧 实现场景的技巧

### 1. 选择合适的起始位置

```python
# 方法1: 使用地图预定义spawn点
spawns = self.map.get_spawn_points()
self.ego_spawn_transform = random.choice(spawns)

# 方法2: 使用waypoint
start_wp = self._pick_random_waypoint()
self.ego_spawn_transform = start_wp.transform

# 方法3: 手动指定位置
self.ego_spawn_transform = carla.Transform(
    carla.Location(x=100.0, y=200.0, z=0.3),
    carla.Rotation(yaw=90.0)
)
```

### 2. 生成静态障碍物（车辆）

```python
# 获取blueprint
lib = self.world.get_blueprint_library()
vehicle_bp = lib.filter("vehicle.tesla.model3")[0]

# 生成车辆
spawn_tf = carla.Transform(...)
vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

if vehicle:
    vehicle.set_simulate_physics(False)  # 静止车辆
    self.scenario_actors.append(vehicle)  # 注册到场景
```

### 3. 生成动态障碍物（行人）

```python
# 获取行人blueprint
pedestrian_bp = lib.filter("walker.pedestrian.*")[0]

# 生成行人
pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_tf)

if pedestrian:
    # 生成AI控制器
    walker_controller_bp = lib.find("controller.ai.walker")
    controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), pedestrian)

    # 设置行人移动
    controller.start()
    controller.go_to_location(target_location)
    controller.set_max_speed(2.0)  # m/s

    self.scenario_actors.append(pedestrian)
    self.scenario_actors.append(controller)
```

### 4. 生成锥桶/路障

```python
# 获取锥桶blueprint
cone_bp = lib.find("static.prop.trafficcone01")

# 生成锥桶
cone = self.world.try_spawn_actor(cone_bp, spawn_tf)

if cone:
    self.scenario_actors.append(cone)
```

### 5. 沿着道路放置障碍物

```python
cur_wp = start_wp
for i in range(num_obstacles):
    # 前进一定距离
    nxt = cur_wp.next(spacing)
    if not nxt:
        break
    cur_wp = nxt[0]

    # 在当前waypoint生成障碍物
    obstacle = self._spawn_obstacle(cur_wp)
    if obstacle:
        self.scenario_actors.append(obstacle)
```

---

## 🧪 测试场景

### 方法1: 单独测试场景生成

创建测试脚本 `test_scenario.py`:

```python
import carla
from carla_base.scenario_manager import ScenarioFactory
from config import Config

# 连接CARLA
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# 创建配置
config = Config()
config.scenario = "your_new_scenario"
config.param1 = 15.0

# 创建场景
scenario = ScenarioFactory.create_scenario(
    scenario_name=config.scenario,
    world=world,
    carla_map=carla_map,
    config=config
)

# 初始化场景
if scenario:
    success = scenario.setup()
    if success:
        print("✅ 场景生成成功")
        print(f"障碍物数量: {len(scenario.get_obstacle_actors())}")
        print(f"自车spawn位置: {scenario.get_spawn_transform()}")
    else:
        print("❌ 场景生成失败")

    # 清理
    scenario.cleanup()
```

### 方法2: 在训练中测试

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "your_new_scenario"
episodes = 1  # 只跑1个episode测试
```

---

## 📊 场景调试

### 1. 打印调试信息

在 `setup()` 方法中添加详细的打印：

```python
def setup(self) -> bool:
    print(f"\n[{self.scenario_name}] 开始生成场景...")
    print(f"  - 参数1: {self.param1}")
    print(f"  - 参数2: {self.param2}")

    # ... 生成逻辑 ...

    print(f"  - 成功生成 {len(self.scenario_actors)} 个actors")
    return True
```

### 2. 使用可视化

开启调试绘制查看障碍物位置：

```python
config.enable_debug_drawing = True
config.draw_obstacle_boxes = True
```

### 3. 检查spawn位置

```python
spawn_tf = scenario.get_spawn_transform()
if spawn_tf:
    print(f"Spawn位置: ({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f})")
    print(f"Spawn朝向: {spawn_tf.rotation.yaw:.1f}°")
```

---

## 🎯 最佳实践

### 1. 场景参数化

所有场景参数都应该从 `config` 读取，方便调整：

```python
# ✅ 好的做法
self.distance = float(getattr(config, "obstacle_distance", 15.0))

# ❌ 不好的做法
self.distance = 15.0  # 写死
```

### 2. 错误处理

生成失败时应该返回 `False` 并打印错误信息：

```python
if not spawns:
    print(f"[{self.scenario_name}] ❌ 没有可用的spawn点")
    return False
```

### 3. 清理资源

所有生成的actors都应该注册到 `self.scenario_actors`：

```python
vehicle = self.world.try_spawn_actor(...)
if vehicle:
    self.scenario_actors.append(vehicle)  # 重要！
```

### 4. 物理稳定

生成后等待几帧让物理稳定：

```python
if self.world.get_settings().synchronous_mode:
    for _ in range(3):
        self.world.tick()
```

---

## 📚 参考示例

### 完整示例: ParkedObstaclesScenario

查看 `scenario_manager.py` 中的 `ParkedObstaclesScenario` 类，这是一个完整实现的示例。

关键点：
1. 从config读取参数
2. 选择spawn点
3. 沿着道路生成车辆
4. 注册到 `scenario_actors`
5. 返回自车spawn位置

---

## 🆘 常见问题

### Q: 场景生成失败怎么办？
A: 检查：
1. spawn点是否有效
2. 障碍物是否与其他物体碰撞
3. 是否在同步模式下tick了几帧

### Q: 障碍物没有被检测到？
A: 确保：
1. 障碍物注册到了 `self.scenario_actors`
2. 障碍物在检测范围内（50米）
3. `config.obs_use_obstacles = True`

### Q: 如何调试场景？
A:
1. 开启可视化 (`config.enable_debug_drawing = True`)
2. 打印详细日志
3. 单独测试场景生成（不运行训练）

---

**祝场景开发顺利！🚀**
