# 🔍 当前代码场景能力分析报告

## 📊 执行摘要

根据代码分析，你的项目**当前已经实现**了以下场景功能：

### ✅ 已实现的场景（3个）

| 场景名称 | 实现位置 | 状态 | 可用性 |
|---------|---------|------|--------|
| **parked_obstacles** | `scenario_manager.py` | ✅ 完整实现 | 可直接使用 |
| **cones** | `carla_env.py` (旧代码) | ⚠️ 部分实现 | 需要迁移 |
| **cones_xml** | `carla_env.py` (旧代码) | ⚠️ 部分实现 | 需要迁移 |

---

## 🎯 详细场景分析

### 1️⃣ **parked_obstacles** - 停放车辆避让 ✅

#### 实现状态
- **新系统**: ✅ 已在 `scenario_manager.ParkedObstaclesScenario` 中完整实现
- **旧系统**: ⚠️ `carla_env._place_parked_vehicles()` 已标记为废弃

#### 功能描述
```
在自车前方12-20米范围内生成多辆静止车辆
- 第一辆车：随机距离（12-20米）
- 后续车辆：按固定间距（默认8米）排列
- 车辆停放在车道中心或指定横向偏移位置
```

#### 配置参数
```python
config.scenario = "parked_obstacles"
config.num_parked_cars = 4                      # 车辆数量
config.parked_car_spacing = 8.0                 # 车辆间距（米）
config.parked_car_start_distance_min = 12.0     # 第一辆车最小距离
config.parked_car_start_distance_max = 20.0     # 第一辆车最大距离
config.parked_car_offset = 0.0                  # 横向偏移（0=车道中心）
```

#### 实现细节
- ✅ 使用地图预定义spawn点
- ✅ 沿着waypoint前进生成车辆
- ✅ 多种生成策略（车道中心、抬高、偏移）
- ✅ 自动注册到 `obstacle_actors`
- ✅ 详细的调试输出

#### 使用方式
```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "parked_obstacles"
config.num_parked_cars = 4
```

---

### 2️⃣ **cones** - 锥桶场景 ⚠️

#### 实现状态
- **新系统**: ⏳ 接口已预留，但 `setup()` 未实现
- **旧系统**: ✅ `carla_env._place_cones_conditionally_behind()` 已实现

#### 功能描述
```
在车道一侧放置一系列锥桶，逐渐向车道中心靠近
- 自动检测左右车道情况，选择放置侧
- 锥桶从车道边缘开始，逐渐向中心递进
- 形成"收窄"效果，需要避让或变道
```

#### 已实现的功能（旧代码）
```python
def _place_cones_conditionally_behind(
    start_wp: carla.Waypoint,
    num_cones: int = 10,              # 锥桶数量
    step_behind: float = 3.0,         # 纵向间距（米）
    step_lateral_per_cone: float = 0.35,  # 横向递进（米）
    z_offset: float = 0.0,            # 高度偏移
    lane_margin: float = 0.25,        # 距车道边缘最小距离
):
    # 1. 检测左右车道类型
    # 2. 决定放置侧（左/右）
    # 3. 沿着waypoint向后放置锥桶
    # 4. 每个锥桶逐渐向车道中心靠近
```

#### 实现逻辑
1. **检测车道情况**:
   ```python
   left_lane_wp = start_wp.get_left_lane()
   right_lane_wp = start_wp.get_right_lane()
   is_left_driving = left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving
   is_right_driving = right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving
   ```

2. **决定放置侧**:
   - 如果只有左侧是行车道 → 放左侧
   - 如果只有右侧是行车道 → 放右侧
   - 如果两侧都是行车道 → 随机选择

3. **计算锥桶位置**:
   ```python
   # 起始偏移（车道边缘）
   start_offset = (lane_width/2 - lane_margin) * direction

   # 每个锥桶的递进偏移
   progression_offset = i * step_lateral_per_cone * direction

   # 最终位置（限制在车道内）
   actual_offset = clamp(start_offset + progression_offset, -max, +max)
   ```

4. **生成锥桶**:
   ```python
   cone_bp = lib.find("static.prop.trafficcone01")
   cone = world.try_spawn_actor(cone_bp, cone_transform)
   ```

#### 配置参数（旧系统）
```python
config.scenario = "cones"  # 或 "cones_old"
config.cone_num = 15
config.cone_step_behind = 3.0
config.cone_step_lateral = 0.4
config.cone_z_offset = 0.0
config.cone_lane_margin = 0.25
```

#### 需要做的工作
将旧代码迁移到新系统：
```python
# 在 scenario_manager.ConesScenario 中：
def setup(self) -> bool:
    # 1. 选择起始waypoint
    start_wp = self._pick_random_start_waypoint()

    # 2. 调用旧的锥桶生成逻辑（或重写）
    cones = self._place_cones_conditionally_behind(
        start_wp=start_wp,
        num_cones=self.cone_num,
        step_behind=self.cone_step_behind,
        step_lateral_per_cone=self.cone_step_lateral,
        z_offset=self.cone_z_offset,
        lane_margin=self.cone_lane_margin
    )

    # 3. 注册到 scenario_actors
    self.scenario_actors.extend(cones)

    # 4. 设置自车spawn位置
    self.ego_spawn_transform = self._calculate_ego_spawn(start_wp)

    return True
```

---

### 3️⃣ **cones_xml** - 基于XML路径的锥桶场景 ⚠️

#### 实现状态
- **新系统**: ⏳ 未实现
- **旧系统**: ✅ `carla_env._maybe_setup_scene_and_pick_spawn()` 中已实现

#### 功能描述
```
从XML文件读取预定义路径，沿着路径放置锥桶
- 支持多个XML文件随机选择
- 自动切换到XML指定的地图
- 沿着XML路径放置锥桶
```

#### 已实现的功能
```python
# 1. 读取XML文件
xml_path = self._pick_random_xml_file()
town, waypoints = self._parse_xml_waypoints(xml_path)

# 2. 切换地图
if town:
    self._load_map_if_needed(town)

# 3. 选择路径上的随机点
pick = random.choice(waypoints)
start_loc = carla.Location(x=pick["x"], y=pick["y"], z=pick["z"])

# 4. 放置锥桶
start_wp = map.get_waypoint(start_loc, project_to_road=True)
self._place_cones_conditionally_behind(start_wp, ...)
```

#### 配置参数
```python
config.scenario = "cones_xml"
config.xml_file = "/path/to/waypoints.xml"  # 单个XML文件
config.xml_dir = "/path/to/xml/dir"         # XML文件目录（随机选择）
config.randomize_town = True                # 是否随机切换地图
config.town_pool = ["Town01", "Town03", "Town05"]
```

#### XML文件格式
```xml
<route town="Town05">
  <waypoints>
    <position x="100.0" y="200.0" z="0.3" yaw="90.0"/>
    <position x="110.0" y="200.0" z="0.3" yaw="90.0"/>
    ...
  </waypoints>
</route>
```

---

## 🚧 未实现的场景（2个）

### 4️⃣ **jaywalker** - 鬼探头 ❌

#### 状态
- **新系统**: ⏳ 接口已预留，`setup()` 未实现
- **旧系统**: ❌ 无相关代码

#### 需要实现的功能
```python
def setup(self) -> bool:
    # 1. 选择合适的道路位置
    # 2. （可选）放置遮挡车辆
    # 3. 在路边生成行人
    # 4. 设置行人AI控制器
    # 5. 配置触发条件（自车距离）
    # 6. 设置自车spawn位置
```

#### 实现难点
- 行人生成和AI控制
- 触发机制（需要监控自车距离）
- 遮挡物放置
- 行人移动速度和路径

---

### 5️⃣ **trimma** - Trimma场景 ❌

#### 状态
- **新系统**: ⏳ 接口已预留，`setup()` 未实现
- **旧系统**: ❌ 无相关代码

#### 需要补充
- 场景描述（Trimma是什么场景？）
- 配置参数定义
- 实现逻辑

---

### 6️⃣ **construction_lane_change** - 施工+变道高交通流 ❌

#### 状态
- **新系统**: ⏳ 接口已预留，`setup()` 未实现
- **旧系统**: ❌ 无相关代码

#### 需要实现的功能
```python
def setup(self) -> bool:
    # 1. 选择有多车道的道路
    # 2. 在自车前方放置施工区域（锥桶/路障）
    # 3. 在相邻车道生成交通流车辆
    # 4. 设置车辆AI控制器（保持速度和车距）
    # 5. 设置自车spawn位置
```

#### 实现难点
- 多车道检测
- 交通流生成和管理
- 车辆AI控制（保持车距、速度）
- 施工区域布置

---

## 📚 从B2Drive可以借鉴的场景

根据 `B2DRIVE_SCENARIOS_ANALYSIS.md`，B2Drive项目有10个场景：

| B2Drive场景 | 你的项目对应 | 可迁移性 |
|------------|-------------|---------|
| route_obstacles (路障/事故) | `parked_obstacles` | ✅ 已实现 |
| actor_flow (交通流) | `construction_lane_change` | ⚠️ 可参考 |
| change_lane (变道超车) | - | ⚠️ 可参考 |
| highway_cut_in (高速插队) | - | ⚠️ 可参考 |
| parking_exit (停车出库) | - | ⚠️ 可参考 |
| left_turn_enter_flow (左转汇入) | - | ⚠️ 可参考 |
| signalized_junction_* (信号灯) | - | ⚠️ 可参考 |
| cross_bicycle_flow (自行车流) | `jaywalker` | ⚠️ 可参考 |

**关键问题**: B2Drive使用 **ScenarioRunner** 框架，你的项目没有这个依赖。

---

## 🎯 总结与建议

### 当前可用的场景

1. **parked_obstacles** ✅
   - 完全可用
   - 新系统已实现
   - 可直接用于训练

2. **cones** ⚠️
   - 旧代码已实现核心逻辑
   - 需要迁移到新系统（工作量小）
   - 迁移后即可使用

3. **cones_xml** ⚠️
   - 旧代码已实现
   - 需要迁移到新系统（工作量中等）
   - 需要准备XML文件

### 优先级建议

#### 🔥 高优先级（立即可做）
1. **迁移 cones 场景** - 代码已有，只需整合
2. **测试 parked_obstacles** - 确保新系统工作正常

#### 🟡 中优先级（需要开发）
3. **实现 jaywalker** - 需要学习行人生成和控制
4. **实现 construction_lane_change** - 需要交通流管理

#### 🔵 低优先级（需要明确需求）
5. **定义 trimma 场景** - 需要先明确场景内容
6. **迁移 cones_xml** - 如果需要XML路径功能

---

## 🛠️ 快速迁移指南

### 迁移 cones 场景（30分钟）

```python
# 在 scenario_manager.ConesScenario 中：

def setup(self) -> bool:
    print(f"\n[Cones] 开始生成锥桶场景...")

    # 1. 选择起始waypoint
    start_wp = self._pick_random_start_waypoint(
        min_gap_from_junction=15.0,
        grid=5.0
    )
    if not start_wp:
        return False

    # 2. 生成锥桶（复用旧代码逻辑）
    cones = self._place_cones(start_wp)
    self.scenario_actors.extend(cones)

    # 3. 设置自车spawn位置（锥桶前方20米）
    self.ego_spawn_transform = self._calculate_ego_spawn(start_wp)

    # 4. 等待物理稳定
    if self.world.get_settings().synchronous_mode:
        for _ in range(3):
            self.world.tick()

    print(f"[Cones] ✅ 成功生成 {len(cones)} 个锥桶")
    return True

def _place_cones(self, start_wp):
    """复制 carla_env._place_cones_conditionally_behind 的逻辑"""
    # ... 复制旧代码 ...
    pass

def _pick_random_start_waypoint(self, min_gap_from_junction, grid):
    """复制 carla_env._pick_random_start_waypoint 的逻辑"""
    # ... 复制旧代码 ...
    pass
```

---

## 📊 场景能力对比表

| 场景类型 | 当前状态 | 实现难度 | 训练价值 | 推荐优先级 |
|---------|---------|---------|---------|-----------|
| 停放车辆避让 | ✅ 已实现 | - | ⭐⭐⭐⭐ | 立即使用 |
| 锥桶避让 | ⚠️ 需迁移 | ⭐ 简单 | ⭐⭐⭐ | 高 |
| 鬼探头 | ❌ 未实现 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 中 |
| 施工+变道 | ❌ 未实现 | ⭐⭐⭐⭐ 困难 | ⭐⭐⭐⭐⭐ | 中 |
| Trimma | ❌ 未定义 | ❓ 未知 | ❓ 未知 | 低 |
| XML路径锥桶 | ⚠️ 需迁移 | ⭐⭐ 简单 | ⭐⭐ | 低 |

---

**结论**: 你的项目已经有了不错的基础（3个场景的代码），只需要少量工作就能让 `cones` 场景可用。建议先完成迁移，然后再考虑实现新场景。
