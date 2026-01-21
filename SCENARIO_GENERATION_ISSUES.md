# 🐛 场景生成失败问题说明

## ❌ 问题描述

在训练时，`vehicle_opens_door`、`cut_in` 和 `parking_exit` 场景可能会初始化失败，显示：

```
[VehicleOpensDoor] ❌ 停放车辆生成失败
[CarlaEnv] ⚠️ 场景 vehicle_opens_door 初始化失败，使用默认spawn
```

---

## 🔍 问题原因

### **1. 车辆生成位置冲突**
- 生成位置可能与其他物体（建筑、车辆、路障等）重叠
- CARLA 的 `try_spawn_actor()` 会拒绝在冲突位置生成

### **2. 地图限制**
- 某些地图（如 Town05）的路边可能没有足够空间
- 车道宽度不足以放置停放车辆

### **3. 高度问题**
- 生成高度不合适（太低或太高）
- 地形起伏导致车辆悬空或嵌入地面

### **4. 多车道要求**
- `cut_in` 场景需要多车道道路
- 某些 spawn 点可能是单车道

---

## ✅ 已采取的措施

### **1. 改进车辆生成逻辑**

在 `scenario_manager.py` 中改进了 `_spawn_parked_vehicle()` 方法：

```python
# 改进前：只尝试2次
vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
if not vehicle:
    spawn_loc.z += 0.5
    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)

# 改进后：尝试5次，调整位置和高度
for attempt in range(5):
    offset_multiplier = 0.3 + (attempt * 0.1)  # 0.3, 0.4, 0.5, 0.6, 0.7
    z_offset = 0.5 + (attempt * 0.3)  # 0.5, 0.8, 1.1, 1.4, 1.7
    # ... 尝试生成
```

**改进点：**
- ✅ 增加重试次数（2次 → 5次）
- ✅ 动态调整横向偏移量
- ✅ 动态调整高度
- ✅ 使用更常见的小型车辆
- ✅ 添加详细的调试信息

### **2. 暂时禁用不稳定场景**

在 `train_ppo_with_wandb.py` 中暂时禁用了容易失败的场景：

```python
config.scenario_pool = [
    "parked_obstacles",      # ✅ 稳定
    "cones",                 # ✅ 稳定
    "pedestrian_crossing",   # ✅ 稳定
    # "vehicle_opens_door",    # ⚠️ 暂时禁用
    # "cut_in",                # ⚠️ 暂时禁用
    # "parking_exit",          # ⚠️ 暂时禁用
]
```

---

## 🎯 当前训练配置

### **使用的场景（3个稳定场景）：**

| # | 场景名称 | 状态 | 说明 |
|---|---------|------|------|
| 1 | parked_obstacles | ✅ 启用 | 静态车辆，生成成功率高 |
| 2 | cones | ✅ 启用 | 静态锥桶，生成成功率高 |
| 3 | pedestrian_crossing | ✅ 启用 | 动态行人，生成成功率高 |

### **暂时禁用的场景（3个）：**

| # | 场景名称 | 状态 | 原因 |
|---|---------|------|------|
| 4 | vehicle_opens_door | ⚠️ 禁用 | 车辆生成可能失败 |
| 5 | cut_in | ⚠️ 禁用 | 需要多车道，某些位置不满足 |
| 6 | parking_exit | ⚠️ 禁用 | 车辆生成可能失败 |

---

## 🔧 如何启用更多场景？

### **方法 1：测试场景稳定性**

先单独测试每个场景，看看成功率：

```bash
# 测试 vehicle_opens_door 场景
python test_new_scenarios.py --scenario vehicle_opens_door

# 测试 cut_in 场景
python test_new_scenarios.py --scenario cut_in

# 测试 parking_exit 场景
python test_new_scenarios.py --scenario parking_exit
```

如果某个场景测试成功率高（>80%），可以启用它。

### **方法 2：更换地图**

某些地图对这些场景更友好：

```python
# config.py 或 train_ppo_with_wandb.py
config.map_name = "Town03"  # 或 Town01, Town02, Town04
```

不同地图特点：
- **Town01**: 小镇，道路简单，适合基础场景
- **Town03**: 城市，多车道，适合 cut_in
- **Town04**: 高速公路，多车道，适合 cut_in
- **Town05**: 城市（当前），复杂，某些位置不适合停车

### **方法 3：调整场景参数**

减少生成难度：

```python
# config.py
# Vehicle Opens Door 场景
self.door_vehicle_distance = 40.0  # 增加距离，避开复杂区域
self.door_side = "right"  # 固定侧，避免随机失败

# Cut In 场景
self.cutin_vehicle_distance = 50.0  # 增加距离
self.cutin_direction = "left"  # 固定方向

# Parking Exit 场景
self.parking_exit_distance = 45.0  # 增加距离
self.parking_side = "right"  # 固定侧
```

### **方法 4：逐步启用**

先启用一个，测试稳定后再启用下一个：

```python
# 第一步：启用 vehicle_opens_door
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",  # ← 先启用这个
]

# 训练一段时间，观察是否稳定

# 第二步：启用 parking_exit
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",
    "parking_exit",  # ← 再启用这个
]

# 第三步：启用 cut_in
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",
    "parking_exit",
    "cut_in",  # ← 最后启用这个
]
```

---

## 📊 场景生成成功率（估计）

基于 Town05 地图：

| 场景 | 成功率 | 说明 |
|------|--------|------|
| parked_obstacles | ~95% | 非常稳定 |
| cones | ~98% | 非常稳定 |
| pedestrian_crossing | ~90% | 稳定 |
| vehicle_opens_door | ~60% | 不稳定，取决于位置 |
| cut_in | ~50% | 不稳定，需要多车道 |
| parking_exit | ~55% | 不稳定，取决于位置 |

---

## 🎯 推荐策略

### **当前阶段（训练初期）：**

使用 3 个稳定场景：
```python
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
]
```

**优点：**
- ✅ 训练稳定，不会因场景失败中断
- ✅ 包含静态和动态障碍物
- ✅ 难度适中

### **进阶阶段（训练中期）：**

测试并启用 1-2 个动态场景：
```python
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",  # 如果测试成功率 >70%
]
```

### **高级阶段（训练后期）：**

启用所有场景（如果稳定性改善）：
```python
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",
    "cut_in",
    "parking_exit",
]
```

---

## 🔍 调试技巧

### **1. 查看场景初始化日志**

训练时注意观察：
```
[VehicleOpensDoor] 开始生成场景...
  - 停放车辆距离: 30.0m
  - 触发距离: 15.0m
  - 车门侧: random
  - 起始位置: (35.1, 137.3)
  - 尝试 1/5 失败，调整位置...
  - 尝试 2/5 失败，调整位置...
  - 停放车辆生成: ID=123, 位置=(40.2, 135.1), 侧=right, 尝试=3
[VehicleOpensDoor] ✅ 场景生成成功
```

### **2. 统计失败率**

运行一段时间后，统计各场景的失败次数：
```bash
grep "初始化失败" carla_training.log | sort | uniq -c
```

### **3. 测试特定位置**

如果某个场景总是失败，可以固定 spawn 点测试：
```python
# carla_env.py
self.initial_spawn_tf = {
    "x": 100.0,
    "y": 50.0,
    "z": 0.3,
    "yaw": 0.0
}
```

---

## ✅ 总结

### **当前状态：**
- ✅ 使用 3 个稳定场景训练
- ✅ 改进了车辆生成逻辑
- ✅ 暂时禁用不稳定场景

### **下一步：**
1. 先用 3 个稳定场景训练
2. 测试其他场景的成功率
3. 根据测试结果逐步启用更多场景
4. 考虑更换地图以提高成功率

### **长期改进：**
- 实现更智能的 spawn 点选择
- 添加场景生成失败的自动重试
- 为每个地图预定义适合的场景

---

**现在可以开始训练了！** 🚀

```bash
python train_ppo_with_wandb.py
```

训练会使用 3 个稳定的场景，不会因为场景生成失败而中断。

---

**创建时间：** 2026-01-14
**问题：** 动态场景生成失败
**解决方案：** 改进生成逻辑 + 暂时禁用不稳定场景
**当前场景数：** 3 个（稳定）
