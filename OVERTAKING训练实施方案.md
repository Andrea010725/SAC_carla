# Overtaking 场景训练实施方案

## 🎯 目标
将 overtaking 场景（45条预定义路线）用于 PPO Agent 训练

---

## ⚠️ 核心挑战

### 1. **架构不兼容**
- **当前训练**: `CarlaEnv` + `gym.Env` 接口
- **Overtaking**: `Leaderboard` + `ScenarioRunner` 框架

### 2. **两种可能的方案**

#### 方案 A: 轻量级改造（推荐）✅
**思路**: 在 CarlaEnv 中模拟 overtaking 场景的核心元素

**优点**:
- ✅ 不需要 Leaderboard 框架
- ✅ 保持快速训练速度
- ✅ 可以自由调整场景难度

**缺点**:
- ⚠️ 场景简化，不如真实 overtaking 复杂

#### 方案 B: 深度集成（复杂）❌
**思路**: 完全集成 Leaderboard 框架到训练循环

**优点**:
- ✅ 使用完整的 overtaking 场景

**缺点**:
- ❌ 需要大量重构
- ❌ 训练速度极慢
- ❌ 实现复杂度高

---

## 💡 推荐方案：方案 A（轻量级改造）

### 核心思路
在当前 cones 场景基础上，添加 overtaking 的核心元素：

1. **静止障碍车辆**（模拟 ParkedObstacle）
2. **动态障碍**（模拟 Accident）
3. **施工区域**（使用锥桶 + 车辆组合）
4. **对向车流**（可选）

### 实现步骤

#### 步骤 1: 创建新场景类型 `overtaking_simple`

在 `config.py` 中添加配置：
```python
# config.py
self.scenario = "overtaking_simple"  # 新场景类型

# Overtaking 场景参数
self.overtaking_parked_cars = 2      # 停车障碍数量
self.overtaking_accident_cars = 1    # 事故车辆数量
self.overtaking_cones = 10           # 施工锥桶数量
self.overtaking_oncoming = False     # 是否添加对向车流（初期false）
```

#### 步骤 2: 在 `carla_env.py` 中实现场景

**位置**: `carla_base/carla_env.py`

**添加新方法**:
```python
def _setup_overtaking_simple_scenario(self):
    \"\"\"
    创建简化版 overtaking 训练场景
    包含: 停车障碍 + 事故车辆 + 施工区域
    \"\"\"
    # 1. 选择起始点
    start_wp = self._pick_random_start_waypoint(...)

    # 2. 放置停车障碍（车辆停在路边）
    parked_cars = []
    for i in range(self.overtaking_parked_cars):
        # 在路线上某个位置放置静止车辆
        spawn_wp = start_wp.next(20 + i * 30)[0]  # 间隔30米
        parked_car = self._spawn_parked_vehicle(spawn_wp)
        parked_cars.append(parked_car)

    # 3. 放置事故场景（车辆横在路中）
    accident_wp = start_wp.next(100)[0]
    accident_car = self._spawn_accident_vehicle(accident_wp)

    # 4. 放置施工锥桶
    construction_wp = start_wp.next(150)[0]
    construction_cones = self._place_construction_cones(construction_wp)

    # 5. 计算自车起始位置
    ego_spawn = start_wp.transform

    return ego_spawn, {
        'parked_cars': parked_cars,
        'accident_car': accident_car,
        'cones': construction_cones
    }

def _spawn_parked_vehicle(self, waypoint):
    \"\"\"在路边生成停车障碍\"\"\"
    # 获取路边位置（向右偏移）
    right_offset = waypoint.transform.get_right_vector() * 1.5  # 1.5米偏移
    spawn_location = waypoint.transform.location + right_offset

    spawn_transform = carla.Transform(
        spawn_location,
        waypoint.transform.rotation
    )

    # 随机选择车辆类型
    vehicle_bp = random.choice(
        self.world.get_blueprint_library().filter('vehicle.*')
    )

    vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)

    # 设置为静止
    vehicle.set_simulate_physics(False)

    return vehicle

def _spawn_accident_vehicle(self, waypoint):
    \"\"\"生成事故车辆（横在路中）\"\"\"
    spawn_transform = waypoint.transform

    # 旋转车辆（模拟事故）
    spawn_transform.rotation.yaw += 45  # 旋转45度

    vehicle_bp = random.choice(
        self.world.get_blueprint_library().filter('vehicle.*')
    )

    vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
    vehicle.set_simulate_physics(False)

    return vehicle

def _place_construction_cones(self, start_wp):
    \"\"\"放置施工区域锥桶\"\"\"
    cones = []
    for i in range(self.overtaking_cones):
        cone_wp = start_wp.next(i * 2.0)[0]

        # 锥桶放在左侧车道
        left_offset = cone_wp.transform.get_right_vector() * -1.0
        cone_location = cone_wp.transform.location + left_offset

        cone_transform = carla.Transform(
            cone_location,
            cone_wp.transform.rotation
        )

        cone_bp = self.world.get_blueprint_library().find('static.prop.trafficcone01')
        cone = self.world.spawn_actor(cone_bp, cone_transform)
        cones.append(cone)

    return cones
```

#### 步骤 3: 修改 reset() 方法

```python
def reset(self):
    # ... 现有代码 ...

    # 处理 overtaking_simple 场景
    if self.scenario == "overtaking_simple":
        print(f"[CarlaEnv] 🎯 设置 overtaking_simple 场景...")
        ego_spawn, scenario_actors = self._setup_overtaking_simple_scenario()

        # 保存场景对象以便cleanup
        self._scenario_actors = scenario_actors

        # spawn自车
        self.ego = self.world.spawn_actor(ego_bp, ego_spawn)
        # ...
```

#### 步骤 4: 清理场景对象

```python
def _clear_actors(self):
    \"\"\"清理所有生成的对象\"\"\"
    # 清理 overtaking 场景对象
    if hasattr(self, '_scenario_actors'):
        for actor_list in self._scenario_actors.values():
            if isinstance(actor_list, list):
                for actor in actor_list:
                    if actor is not None:
                        actor.destroy()
            elif actor_list is not None:
                actor_list.destroy()
        self._scenario_actors = {}

    # 清理其他对象...
```

---

## 🚀 快速实现版本

如果想快速测试效果，可以先实现一个**最简单的版本**：

### 最小实现（30分钟内完成）

只添加 **停车障碍** 场景：

```python
# config.py
self.scenario = "parked_obstacles"
self.num_parked_cars = 3
self.parked_car_spacing = 40  # 米

# carla_env.py - 在 reset() 中
if self.scenario == "parked_obstacles":
    start_wp = self._pick_random_start_waypoint(...)

    # 简单：沿路线放置3辆停车
    for i in range(self.num_parked_cars):
        distance = 30 + i * self.parked_car_spacing
        wp = start_wp.next(distance)[0]

        # 右侧路边停车
        offset = wp.transform.get_right_vector() * 1.8
        spawn_loc = wp.transform.location + offset
        spawn_tf = carla.Transform(spawn_loc, wp.transform.rotation)

        # spawn车辆
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.*'))
        vehicle = self.world.spawn_actor(bp, spawn_tf)
        vehicle.set_simulate_physics(False)

        # 保存以便清理
        self._parked_vehicles.append(vehicle)

    # spawn自车在起始点前方
    ego_spawn = start_wp.transform
```

---

## 📊 效果预期

### 训练效果对比

| 场景 | 复杂度 | 训练速度 | 学习内容 |
|------|--------|---------|---------|
| Cones | ⭐ | ⭐⭐⭐⭐⭐ | 基础避障 |
| Parked Obstacles | ⭐⭐ | ⭐⭐⭐⭐ | 变道超车 |
| Overtaking Simple | ⭐⭐⭐ | ⭐⭐⭐ | 多种场景组合 |
| Overtaking Full | ⭐⭐⭐⭐⭐ | ⭐ | 完整真实场景 |

### 建议训练计划

1. **Phase 1** (100-200 episodes): Parked Obstacles
   - 学习变道超车基础

2. **Phase 2** (200-300 episodes): Overtaking Simple
   - 学习应对多种障碍

3. **Phase 3** (评估): Overtaking Full
   - 在真实场景上测试

---

## ⚡ 立即可执行的方案

### 选项 1: 最快速度（推荐）✅

**实现"停车障碍"场景**（最简单的 overtaking 元素）

**时间**: 30分钟

**步骤**:
1. 修改 `config.py` 添加 `parked_obstacles` 场景配置
2. 修改 `carla_env.py` 添加停车车辆生成代码
3. 修改 `train_ppo_standalone.py` 使用新场景
4. 开始训练

**效果**: 比 cones 更接近真实，但保持快速训练

---

### 选项 2: 完整 Overtaking 训练（复杂）

**时间**: 2-3天

**需要**:
1. 创建 Leaderboard 适配器
2. 修改训练循环
3. 处理 XML 路线加载
4. 实现场景切换逻辑

**效果**: 完全使用真实 overtaking 场景，但训练极慢

---

## 🎯 我的推荐

**立即行动方案**：

1. **今天**：实现"停车障碍"场景（选项1）
   - 30分钟实现
   - 立即开始训练
   - 看看效果是否比 cones 好

2. **如果效果好**：逐步添加更多元素
   - 添加事故车辆
   - 添加施工锥桶
   - 添加对向车流

3. **最终目标**：训练效果接近真实 overtaking 场景

---

## 🛠️ 实现代码

我可以立即帮你实现选项1（停车障碍场景），需要修改3个文件：
1. `config.py` - 添加配置
2. `carla_env.py` - 实现场景生成
3. `train_ppo_standalone.py` - 使用新场景

你想要我现在就实现吗？

---

**创建时间**: 2025-12-10
**方案**: 轻量级改造
**预计实现时间**: 30分钟
**效果**: 明显优于 cones，接近真实 overtaking
