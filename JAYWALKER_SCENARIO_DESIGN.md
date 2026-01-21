# 🚶 Jaywalker场景（鬼探头）详细设计

## 📋 场景概述

**场景名称**: Jaywalker（鬼探头）

**核心概念**: 行人突然从道路一侧横穿到另一侧，自车需要紧急制动

**难度等级**: ⭐⭐⭐⭐⭐ 非常困难

**训练价值**: ⭐⭐⭐⭐⭐ 极高（城市道路最危险场景之一）

---

## 🎬 场景描述

### 视觉效果（俯视图）

```
自车行驶方向: ↑

═══════════════════════════════════════
[路边/人行道]  |  [车道]  |  [路边/人行道]
               |          |
          🚶   |          |
          ↓    |          |
          →→→→→|→→→→→→→→→|  ← 行人横穿轨迹
               |          |
               |          |
               |    🚗    |  ← 自车（距离15米）
               |    ↑     |
               |    |     |
               |    |     |
═══════════════════════════════════════
```

### 时间线

```
T=0s: 场景初始化
  - 行人站在道路左侧（路边）
  - 自车spawn在行人后方20米
  - 行人状态：静止，等待触发

T=1-2s: 自车接近
  - 自车加速前进
  - 距离行人：15-20米
  - 行人状态：静止

T=2.5s: 触发！
  - 自车距离行人 < 15米（触发距离）
  - 行人开始横穿马路
  - 行人速度：2.5 m/s

T=3-4s: 紧急情况
  - 自车距离行人：5-10米
  - 行人在车道中间
  - 自车需要：紧急制动！

T=5s: 结果
  - 成功：自车停在行人前方，避免碰撞
  - 失败：碰撞行人，episode结束
```

---

## 🏗️ 实现设计

### 步骤1: 选择道路位置

```python
def _pick_random_straight_road(self):
    """选择直道，远离路口"""
    candidates = [
        wp for wp in self.map.generate_waypoints(5.0)
        if wp.lane_type == carla.LaneType.Driving
        and not wp.is_junction
    ]

    # 过滤：检查前后30米是否有路口
    for wp in candidates:
        if not self._is_near_junction(wp, 30.0):
            return wp

    return None
```

**要求**：
- 直道（不是弯道）
- 远离路口（至少30米）
- 行车道类型

---

### 步骤2: 计算行人位置

```python
def _calculate_pedestrian_positions(self, start_wp):
    """计算行人起始和目标位置"""

    # 1. 行人waypoint（自车前方20米）
    pedestrian_wp = start_wp
    for _ in range(int(self.jaywalker_distance / 2.0)):
        nxt = pedestrian_wp.next(2.0)
        if nxt:
            pedestrian_wp = nxt[0]

    # 2. 获取道路宽度和方向
    lane_width = pedestrian_wp.lane_width
    wp_tf = pedestrian_wp.transform
    right_vec = wp_tf.get_right_vector()

    # 3. 决定起始侧
    if self.jaywalker_start_side == "random":
        start_side = random.choice(["left", "right"])
    else:
        start_side = self.jaywalker_start_side

    # 4. 计算起始位置（路边）
    if start_side == "left":
        offset = -(lane_width / 2 + 1.0)  # 左侧路边，车道外1米
    else:
        offset = (lane_width / 2 + 1.0)   # 右侧路边，车道外1米

    start_loc = carla.Location(
        x=wp_tf.location.x + right_vec.x * offset,
        y=wp_tf.location.y + right_vec.y * offset,
        z=wp_tf.location.z + 1.0  # 抬高1米，确保在地面上
    )

    # 5. 计算目标位置（对侧路边）
    target_offset = -offset  # 相反侧
    target_loc = carla.Location(
        x=wp_tf.location.x + right_vec.x * target_offset,
        y=wp_tf.location.y + right_vec.y * target_offset,
        z=wp_tf.location.z + 1.0
    )

    return start_loc, target_loc, pedestrian_wp
```

**位置示意**：
```
车道宽度 = 3.5米

[路边] | [车道] | [路边]
  -4.5m   0m    +4.5m

起始位置: -4.5m (左侧路边，车道外1米)
目标位置: +4.5m (右侧路边，车道外1米)
横穿距离: 9米
```

---

### 步骤3: 生成行人

```python
def _spawn_pedestrian(self, location):
    """生成行人"""
    lib = self.world.get_blueprint_library()

    # 获取行人blueprint（随机选择）
    pedestrian_bps = lib.filter("walker.pedestrian.*")
    if not pedestrian_bps:
        print("[Jaywalker] ❌ 找不到行人blueprint")
        return None

    pedestrian_bp = random.choice(pedestrian_bps)

    # 生成行人
    spawn_tf = carla.Transform(
        location,
        carla.Rotation(yaw=0.0)  # 朝向不重要，控制器会调整
    )

    pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_tf)

    if pedestrian:
        print(f"[Jaywalker] ✅ 行人生成成功")
        print(f"   - 位置: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
        return pedestrian
    else:
        print(f"[Jaywalker] ❌ 行人生成失败")
        return None
```

---

### 步骤4: 创建行人控制器

```python
def _create_pedestrian_controller(self, pedestrian):
    """创建行人AI控制器"""
    lib = self.world.get_blueprint_library()

    # 获取控制器blueprint
    controller_bp = lib.find("controller.ai.walker")

    # 生成控制器（附加到行人）
    controller = self.world.spawn_actor(
        controller_bp,
        carla.Transform(),
        attach_to=pedestrian
    )

    if controller:
        print(f"[Jaywalker] ✅ 行人控制器创建成功")
        return controller
    else:
        print(f"[Jaywalker] ❌ 控制器创建失败")
        return None
```

---

### 步骤5: 触发机制

```python
def check_and_trigger(self, ego_location):
    """
    检查自车距离，触发行人移动

    在 carla_env.step() 中每步调用：
    if self.scenario_instance:
        self.scenario_instance.check_and_trigger(ego_loc)
    """
    if self.triggered:
        return

    if not self.pedestrian:
        return

    # 计算距离
    ped_loc = self.pedestrian.get_location()
    distance = math.hypot(
        ego_location.x - ped_loc.x,
        ego_location.y - ped_loc.y
    )

    # 触发条件：自车距离 < 触发距离
    if distance < self.jaywalker_trigger_distance:
        self.trigger_pedestrian()

def trigger_pedestrian(self):
    """触发行人横穿"""
    if self.triggered:
        return

    if not self.pedestrian_controller or not self.pedestrian_target_location:
        return

    try:
        # 启动控制器
        self.pedestrian_controller.start()

        # 设置目标位置
        self.pedestrian_controller.go_to_location(self.pedestrian_target_location)

        # 设置速度
        self.pedestrian_controller.set_max_speed(self.jaywalker_speed)

        self.triggered = True
        print(f"[Jaywalker] 🚨 行人开始横穿！距离={distance:.1f}m")

    except Exception as e:
        print(f"[Jaywalker] ❌ 触发失败: {e}")
```

---

### 步骤6: （可选）遮挡车辆

```python
def _spawn_occlusion_vehicle(self, pedestrian_wp):
    """
    在行人前方放置遮挡车辆

    目的：增加难度，行人被车辆遮挡，更难发现
    """
    lib = self.world.get_blueprint_library()
    vehicle_bp = lib.filter("vehicle.*")[0]

    # 在行人前方2米放置车辆
    occlusion_wp = pedestrian_wp
    for _ in range(int(self.occlusion_vehicle_distance / 2.0)):
        nxt = occlusion_wp.next(2.0)
        if nxt:
            occlusion_wp = nxt[0]

    # 生成车辆
    vehicle = self.world.try_spawn_actor(vehicle_bp, occlusion_wp.transform)

    if vehicle:
        vehicle.set_simulate_physics(False)  # 静止车辆
        print(f"[Jaywalker] ✅ 遮挡车辆已放置")
        return vehicle

    return None
```

**遮挡效果**：
```
自车视角：

    🚗 遮挡车辆
    |
    | (行人在车后，看不见)
    🚶
    |
    |
    🚗 自车
```

---

## 📊 配置参数详解

### 基础参数

```python
config.scenario = "jaywalker"

# 行人位置
config.jaywalker_distance = 20.0        # 行人在自车前方20米

# 行人速度
config.jaywalker_speed = 2.5            # 2.5 m/s (约9 km/h，快走速度)

# 触发距离
config.jaywalker_trigger_distance = 15.0 # 自车距离15米时，行人开始移动

# 起始侧
config.jaywalker_start_side = "random"  # "left"/"right"/"random"

# 遮挡车辆
config.use_occlusion_vehicle = False    # 是否使用遮挡车辆
config.occlusion_vehicle_distance = 18.0 # 遮挡车辆距离自车18米
```

### 难度调节

#### 简单模式
```python
config.jaywalker_distance = 25.0        # 更远
config.jaywalker_speed = 2.0            # 更慢
config.jaywalker_trigger_distance = 20.0 # 更早触发
config.use_occlusion_vehicle = False    # 无遮挡
```

**效果**：
- 反应时间：5秒
- 制动距离：充足
- 难度：⭐⭐⭐

#### 中等模式（默认）
```python
config.jaywalker_distance = 20.0
config.jaywalker_speed = 2.5
config.jaywalker_trigger_distance = 15.0
config.use_occlusion_vehicle = False
```

**效果**：
- 反应时间：3秒
- 制动距离：紧张
- 难度：⭐⭐⭐⭐

#### 困难模式
```python
config.jaywalker_distance = 15.0        # 更近
config.jaywalker_speed = 3.5            # 更快（跑步速度）
config.jaywalker_trigger_distance = 10.0 # 更晚触发
config.use_occlusion_vehicle = True     # 有遮挡
```

**效果**：
- 反应时间：1-2秒
- 制动距离：极限
- 难度：⭐⭐⭐⭐⭐

---

## 🎯 自车面临的挑战

### 挑战1: 检测行人

```
初始状态：
- 行人在路边（车道外）
- 距离：20米
- 状态：静止

问题：
- 行人很小（0.5米）
- 可能被遮挡
- 静止时不明显
```

### 挑战2: 预测行人行为

```
触发前：
- 行人静止
- 无法预测是否会移动

触发后：
- 行人突然横穿
- 速度：2.5 m/s
- 横穿时间：约3-4秒
```

### 挑战3: 紧急制动

```
假设自车速度：8 m/s (28.8 km/h)
触发距离：15米
行人速度：2.5 m/s

计算：
- 自车到达行人位置时间：15m / 8m/s = 1.875秒
- 行人横穿到车道中心时间：(3.5m/2) / 2.5m/s = 0.7秒

结果：碰撞风险极高！必须紧急制动！

制动要求：
- 反应时间：< 0.5秒
- 制动加速度：> 4 m/s²
- 停止距离：< 10米
```

### 挑战4: 遮挡物（困难模式）

```
有遮挡车辆时：

    🚗 遮挡车辆（18米）
    |
    | 行人在车后，看不见！
    🚶 (20米)
    |
    |
    🚗 自车

问题：
- 行人被完全遮挡
- 触发时才能看到
- 反应时间更短
```

---

## 📐 位置计算详解

### 行人起始位置

```python
# 假设：
# - 车道宽度 = 3.5米
# - 起始侧 = 左侧
# - 行人waypoint位置 = (100.0, 200.0, 0.3)

lane_width = 3.5
offset = -(lane_width / 2 + 1.0) = -(1.75 + 1.0) = -2.75米

行人起始位置 = waypoint位置 + right_vec × offset
             = (100.0, 200.0) + right_vec × (-2.75)

如果道路是南北向（right_vec指向东）：
行人位置 = (100.0 - 2.75, 200.0, 1.0) = (97.25, 200.0, 1.0)
```

**位置说明**：
- 在车道左侧
- 距离车道中心线：2.75米
- 距离车道边缘：1.0米（在人行道上）

### 行人目标位置

```python
target_offset = -offset = +2.75米

行人目标位置 = waypoint位置 + right_vec × (+2.75)
             = (102.75, 200.0, 1.0)
```

**横穿距离**：
```
起始: (97.25, 200.0)
目标: (102.75, 200.0)
距离: 5.5米

横穿时间 = 5.5m / 2.5m/s = 2.2秒
```

---

## 🎮 场景执行流程

### 详细时间线（假设自车速度8m/s）

```
T=0.0s: 场景初始化
  - 行人位置: (97.25, 200.0) 左侧路边
  - 自车位置: (100.0, 180.0) 后方20米
  - 自车速度: 0 m/s
  - 行人状态: 静止 ⏸️

T=1.0s: 自车启动
  - 自车速度: 4 m/s
  - 距离行人: 16米
  - 行人状态: 静止 ⏸️

T=2.0s: 自车加速
  - 自车速度: 7 m/s
  - 距离行人: 9米
  - 行人状态: 静止 ⏸️

T=2.3s: 触发！🚨
  - 自车速度: 8 m/s
  - 距离行人: 14.5米 < 15米（触发距离）
  - 行人状态: 开始横穿！▶️
  - 行人速度: 2.5 m/s

T=2.5s: 行人进入车道
  - 自车速度: 8 m/s
  - 距离行人: 13米
  - 行人位置: 车道边缘
  - 自车应该: 开始制动！🛑

T=3.0s: 危险时刻
  - 自车速度: 6 m/s (如果制动)
  - 距离行人: 9米
  - 行人位置: 车道中心
  - 碰撞风险: 高！

T=3.5s: 结果
  - 成功: 自车停在行人前方5米 ✅
  - 失败: 碰撞行人 ❌
```

---

## 🏆 成功标准

### 必须满足
- ❌ 不能碰撞行人
- ✅ 必须在行人前方停车
- ✅ 停车距离 > 2米（安全距离）

### 理想表现
- ✅ 提前检测到行人（距离15-20米）
- ✅ 触发后立即制动（反应时间 < 0.5秒）
- ✅ 平滑制动（不是急刹）
- ✅ 停在行人前方5-8米（安全且舒适）

---

## 🔧 实现技术细节

### 1. 行人生成

```python
# CARLA行人blueprints
walker.pedestrian.0001  # 成年男性
walker.pedestrian.0002  # 成年女性
walker.pedestrian.0003  # 老年人
walker.pedestrian.0004  # 儿童
...

# 随机选择
pedestrian_bp = random.choice(lib.filter("walker.pedestrian.*"))
```

### 2. 行人控制器

```python
# 控制器API
controller.start()                          # 启动控制器
controller.stop()                           # 停止控制器
controller.go_to_location(target_location)  # 移动到目标位置
controller.set_max_speed(speed)             # 设置最大速度（m/s）
```

### 3. 触发检测

```python
# 在 carla_env.step() 中添加：
if self.scenario_instance and hasattr(self.scenario_instance, 'check_and_trigger'):
    ego_loc = self.ego.get_location()
    self.scenario_instance.check_and_trigger(ego_loc)
```

### 4. 行人状态查询

```python
# 检查行人是否还活着
if pedestrian.is_alive:
    location = pedestrian.get_location()
    velocity = pedestrian.get_velocity()
```

---

## 📊 与其他场景对比

| 特性 | parked_obstacles | cones | jaywalker |
|------|-----------------|-------|-----------|
| 障碍物类型 | 静态车辆 | 静态锥桶 | 动态行人 |
| 障碍物大小 | 大（4-5米） | 小（0.5米） | 小（0.5米） |
| 运动模式 | 静止 | 静止 | 横向移动 |
| 触发机制 | 无 | 无 | 有（距离触发） |
| 反应时间 | 充足 | 充足 | 极短（1-2秒） |
| 制动要求 | 低 | 低 | 极高 |
| 避让方式 | 横向+减速 | 横向+减速 | 紧急制动 |
| 难度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 训练价值 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎓 训练建议

### 训练顺序

1. **阶段1**: parked_obstacles（静态避让）
2. **阶段2**: cones（连续避让）
3. **阶段3**: jaywalker（动态避让+紧急制动）

### 训练策略

#### 策略1: 渐进式难度
```python
# Episode 1-100: 简单模式
config.jaywalker_distance = 30.0
config.jaywalker_speed = 2.0
config.jaywalker_trigger_distance = 25.0

# Episode 101-200: 中等模式
config.jaywalker_distance = 20.0
config.jaywalker_speed = 2.5
config.jaywalker_trigger_distance = 15.0

# Episode 201+: 困难模式
config.jaywalker_distance = 15.0
config.jaywalker_speed = 3.5
config.jaywalker_trigger_distance = 10.0
```

#### 策略2: 混合训练
```python
# 与其他场景混合
config.random_scenario = True
config.scenario_pool = ["parked_obstacles", "cones", "jaywalker"]
```

---

## ⚠️ 实现注意事项

### 1. 行人物理

```python
# 行人需要物理模拟
pedestrian.set_simulate_physics(True)  # 不能设为False！

# 否则行人会悬浮或穿模
```

### 2. 控制器生命周期

```python
# 控制器必须在行人销毁前停止
controller.stop()
controller.destroy()
pedestrian.destroy()

# 顺序很重要！
```

### 3. 触发时机

```python
# 触发距离要合理
# 太远：行人走太久，不真实
# 太近：自车来不及反应

推荐：15米（自车速度8m/s时，约2秒反应时间）
```

### 4. 行人速度

```python
# 真实行人速度参考
慢走: 1.0-1.5 m/s
正常走: 1.5-2.0 m/s
快走: 2.0-2.5 m/s
慢跑: 2.5-3.5 m/s
快跑: 3.5-5.0 m/s

推荐：2.5 m/s（快走，突然冲出的感觉）
```

---

## 🎯 实现优先级

### 必须实现（核心功能）
1. ✅ 行人生成
2. ✅ 行人控制器
3. ✅ 触发机制
4. ✅ 横向移动

### 可选实现（增强功能）
5. ⏳ 遮挡车辆
6. ⏳ 多个行人
7. ⏳ 行人动画（跑步姿势）
8. ⏳ 声音效果

---

## 📚 CARLA API参考

### 行人相关API

```python
# 生成行人
pedestrian = world.spawn_actor(pedestrian_bp, transform)

# 创建控制器
controller_bp = lib.find("controller.ai.walker")
controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)

# 控制行人
controller.start()
controller.go_to_location(carla.Location(x, y, z))
controller.set_max_speed(2.5)  # m/s

# 查询状态
location = pedestrian.get_location()
velocity = pedestrian.get_velocity()
is_alive = pedestrian.is_alive

# 清理
controller.stop()
controller.destroy()
pedestrian.destroy()
```

---

## 🎉 总结

**Jaywalker场景**是最具挑战性的场景：

1. **核心机制**: 行人突然横穿，触发紧急制动
2. **关键技术**: 行人生成、AI控制、触发机制
3. **训练价值**: 极高，测试紧急情况处理能力
4. **实现难度**: 高，需要动态触发和行人控制
5. **真实意义**: 城市道路最危险场景之一

**这是一个非常值得实现的高价值场景！** 🚀
