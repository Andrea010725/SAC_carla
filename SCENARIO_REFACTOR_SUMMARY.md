# ✅ 场景系统重构完成总结

## 📋 完成的工作

### 1. 创建了规范的场景管理系统

**新文件**: `carla_base/scenario_manager.py`

包含：
- `ScenarioBase` - 场景基类，定义统一接口
- `ParkedObstaclesScenario` - 停放车辆场景（已实现）
- `ConesScenario` - 锥桶场景（接口已预留）
- `JaywalkerScenario` - 鬼探头场景（接口已预留）
- `TrimmaScenario` - Trimma场景（接口已预留）
- `ConstructionLaneChangeScenario` - 施工+变道场景（接口已预留）
- `ScenarioFactory` - 场景工厂，统一创建场景实例

### 2. 更新了 `carla_env.py`

- 导入场景管理器
- 修改 `_maybe_setup_scene_and_pick_spawn()` 使用新系统
- 保留旧函数用于向后兼容（标记为废弃）

### 3. 创建了详细文档

- `SCENARIO_GUIDE.md` - 场景开发指南
- `VISUALIZATION_CONFIG.md` - 可视化配置说明
- `VISUALIZATION_GUIDE.md` - 可视化使用指南
- `TROUBLESHOOTING.md` - 故障排除指南

---

## 🏗️ 场景系统架构

### 核心接口

每个场景类必须实现两个方法：

```python
class YourScenario(ScenarioBase):
    def setup(self) -> bool:
        """
        场景初始化 - 生成障碍物、设置环境等
        Returns: 是否成功初始化
        """
        pass

    def get_spawn_transform(self) -> Optional[carla.Transform]:
        """
        获取自车生成位置
        Returns: 自车生成的Transform
        """
        pass
```

### 使用方式

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "parked_obstacles"  # 或其他场景名称
config.num_parked_cars = 4
config.parked_car_spacing = 8.0
# ... 其他配置参数
```

---

## 📊 场景列表

| 场景名称 | 状态 | 描述 | 难度 |
|---------|------|------|------|
| `parked_obstacles` | ✅ 已实现 | 停放车辆避让 | 中等 |
| `cones` | ⏳ 接口已预留 | 锥桶避让 | 简单 |
| `jaywalker` | ⏳ 接口已预留 | 鬼探头（行人突然横穿） | 困难 |
| `trimma` | ⏳ 接口已预留 | Trimma场景 | 待定 |
| `construction_lane_change` | ⏳ 接口已预留 | 施工+变道高交通流 | 非常困难 |

---

## 🎯 下一步工作

### 对于每个待实现的场景，需要：

#### 1. 定义场景描述和参数

在对应的场景类中补充：
```python
"""
场景描述：
- 详细描述场景内容
- 说明难点和挑战

配置参数：
- param1: 参数1说明
- param2: 参数2说明
"""
```

#### 2. 实现 `setup()` 方法

```python
def setup(self) -> bool:
    # 1. 选择合适的起始位置
    # 2. 生成障碍物/行人/车辆
    # 3. 设置自车生成位置
    # 4. 返回成功/失败
    pass
```

#### 3. 测试场景

```python
# 单独测试
python test_scenario.py

# 在训练中测试
config.scenario = "your_scenario"
episodes = 1  # 只跑1个episode
```

---

## 📝 实现场景的步骤

### 以 `cones` 场景为例：

#### 步骤1: 补充场景描述

在 `ConesScenario` 类的docstring中：
```python
"""
锥桶场景

场景描述：
- 在车道一侧放置一系列锥桶
- 锥桶逐渐向车道中心靠近，形成"收窄"效果
- 自车需要避让锥桶，可能需要变道

配置参数：
- cone_num: 锥桶数量（默认15）
- cone_step_behind: 锥桶纵向间距（米，默认3.0）
- cone_step_lateral: 锥桶横向递进距离（米，默认0.4）
- cone_z_offset: 锥桶高度偏移（米，默认0.0）
- cone_lane_margin: 锥桶距离车道边缘的最小距离（米，默认0.25）
"""
```

#### 步骤2: 实现 `setup()` 方法

```python
def setup(self) -> bool:
    print(f"\n[Cones] 开始生成锥桶场景...")

    # 1. 选择起始waypoint（远离路口）
    start_wp = self._pick_random_start_waypoint()
    if not start_wp:
        print("[Cones] ❌ 无法找到合适的起始位置")
        return False

    # 2. 确定锥桶放置的一侧（左侧或右侧）
    place_on_left = random.choice([True, False])

    # 3. 沿着车道放置锥桶
    cur_wp = start_wp
    for i in range(self.cone_num):
        # 计算锥桶位置
        cone_tf = self._calculate_cone_transform(cur_wp, i, place_on_left)

        # 生成锥桶
        cone = self._spawn_cone(cone_tf)
        if cone:
            self.scenario_actors.append(cone)

        # 前进到下一个位置
        nxt = cur_wp.next(self.cone_step_behind)
        if not nxt:
            break
        cur_wp = nxt[0]

    # 4. 设置自车生成位置（锥桶前方20米）
    self.ego_spawn_transform = self._calculate_ego_spawn(start_wp)

    # 5. 等待物理稳定
    if self.world.get_settings().synchronous_mode:
        for _ in range(3):
            self.world.tick()

    print(f"[Cones] ✅ 成功生成 {len(self.scenario_actors)} 个锥桶")
    return True
```

#### 步骤3: 实现辅助方法

```python
def _pick_random_start_waypoint(self):
    """选择远离路口的waypoint"""
    # TODO: 实现逻辑
    pass

def _calculate_cone_transform(self, wp, index, place_on_left):
    """计算锥桶的Transform"""
    # TODO: 实现逻辑
    pass

def _spawn_cone(self, transform):
    """生成单个锥桶"""
    lib = self.world.get_blueprint_library()
    cone_bp = lib.find("static.prop.trafficcone01")
    cone = self.world.try_spawn_actor(cone_bp, transform)
    return cone

def _calculate_ego_spawn(self, start_wp):
    """计算自车生成位置"""
    # TODO: 实现逻辑
    pass
```

#### 步骤4: 测试

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "cones"
config.cone_num = 15
config.cone_step_behind = 3.0
config.cone_step_lateral = 0.4

# 只跑1个episode测试
episodes = 1
```

---

## 🔧 实用工具函数

### 1. 选择远离路口的waypoint

```python
def _pick_random_start_waypoint(self, min_gap_from_junction=15.0):
    """选择远离路口的waypoint"""
    cands = [wp for wp in self.map.generate_waypoints(5.0)
             if wp.lane_type == carla.LaneType.Driving]
    random.shuffle(cands)

    for wp in cands:
        if not wp.is_junction and not self._is_near_junction(wp, min_gap_from_junction):
            return wp
    return None

def _is_near_junction(self, wp, dist=15.0):
    """检查waypoint是否靠近路口"""
    cur = wp
    for _ in range(int(dist)):
        nxt = cur.next(1.0)
        if not nxt:
            break
        cur = nxt[0]
        if cur.is_junction:
            return True
    return False
```

### 2. 沿着道路前进

```python
def _advance_waypoint(self, wp, distance):
    """沿着道路前进指定距离"""
    cur_wp = wp
    traveled = 0.0
    step = 2.0

    while traveled < distance:
        nxt = cur_wp.next(step)
        if not nxt:
            break
        cur_wp = nxt[0]
        traveled += step

    return cur_wp
```

### 3. 计算横向偏移位置

```python
def _get_lateral_offset_location(self, wp, offset):
    """计算横向偏移后的位置"""
    wp_tf = wp.transform
    right_vec = wp_tf.get_right_vector()

    offset_loc = carla.Location(
        x=wp_tf.location.x + right_vec.x * offset,
        y=wp_tf.location.y + right_vec.y * offset,
        z=wp_tf.location.z
    )
    return offset_loc
```

---

## 📚 参考资料

### 文档

- `SCENARIO_GUIDE.md` - 详细的场景开发指南
- `VISUALIZATION_CONFIG.md` - 可视化配置
- `TROUBLESHOOTING.md` - 问题排查

### 代码示例

- `scenario_manager.py` 中的 `ParkedObstaclesScenario` - 完整实现示例
- `carla_env.py` 中的旧函数（标记为废弃） - 可以参考但不推荐使用

### CARLA文档

- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
- [Waypoints](https://carla.readthedocs.io/en/latest/core_map/#waypoints)
- [Actors](https://carla.readthedocs.io/en/latest/core_actors/)

---

## ✅ 验证清单

实现新场景后，确保：

- [ ] 场景类继承自 `ScenarioBase`
- [ ] 实现了 `setup()` 方法
- [ ] 实现了 `get_spawn_transform()` 方法
- [ ] 所有actors注册到 `self.scenario_actors`
- [ ] 在 `ScenarioFactory.SCENARIOS` 中注册
- [ ] 添加了详细的docstring
- [ ] 配置参数从 `config` 读取
- [ ] 测试通过（至少跑1个episode）
- [ ] 可视化正常（能看到障碍物）

---

## 🎉 总结

新的场景系统提供了：

1. **规范的接口** - 所有场景统一继承 `ScenarioBase`
2. **清晰的结构** - 场景代码集中在 `scenario_manager.py`
3. **易于扩展** - 添加新场景只需实现两个方法
4. **向后兼容** - 旧代码仍然可以工作
5. **详细文档** - 完整的开发指南和示例

现在你可以：
- ✅ 使用已实现的 `parked_obstacles` 场景
- ✅ 根据指南实现其他4个场景
- ✅ 轻松添加新的自定义场景

**祝开发顺利！🚀**
