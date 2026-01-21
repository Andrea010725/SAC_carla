# ✅ 随机场景训练已配置

## 📋 功能说明

现在训练会在每个Episode开始时**随机选择**场景：
- 50%概率：`parked_obstacles`（停放车辆避让）
- 50%概率：`cones`（锥桶避让）

这样可以让模型学习应对多种场景，提高泛化能力。

---

## ⚙️ 配置说明

### 1. Config.py 中的配置

```python
# ✅ 随机场景配置
self.random_scenario = False         # 是否启用随机场景
self.scenario_pool = ["parked_obstacles", "cones"]  # 场景池
```

### 2. train_ppo_with_wandb.py 中的配置

```python
# ✅ 启用随机场景训练
config.random_scenario = True  # 开启随机场景
config.scenario_pool = ["parked_obstacles", "cones"]  # 场景池

# Parked obstacles场景参数
config.num_parked_cars = 4
config.parked_car_spacing = 8.0

# Cones场景参数
config.cone_num = 15
config.cone_step_behind = 3.0
config.cone_step_lateral = 0.4
```

---

## 🎯 运行效果

### 终端输出示例

```
======================================================================
🎬 Episode 1 - 场景初始化
======================================================================

[RandomScenario] 本次Episode场景: parked_obstacles

📍 地图: Town05
🎭 场景类型: parked_obstacles

[ParkedObstacles] 开始生成停放车辆...
[ParkedObstacles] ✅ 成功生成 4 辆障碍车

🚧 障碍物信息 (共4个):
   [1] vehicle.tesla.model3
       距离自车: 15.3m
   ...

======================================================================
🎬 Episode 2 - 场景初始化
======================================================================

[RandomScenario] 本次Episode场景: cones

📍 地图: Town05
🎭 场景类型: cones

[Cones] 开始生成锥桶场景...
[Cones] ✅ 成功生成 15 个锥桶

🚧 障碍物信息 (共15个):
   [1] static.prop.trafficcone01
       距离自车: 20.0m
   ...

======================================================================
🎬 Episode 3 - 场景初始化
======================================================================

[RandomScenario] 本次Episode场景: parked_obstacles

...
```

每个Episode开始时会显示：
```
[RandomScenario] 本次Episode场景: xxx
```

---

## 🎮 使用方式

### 方式1: 随机场景训练（当前配置）

```python
# 在 train_ppo_with_wandb.py 中：
config.random_scenario = True
config.scenario_pool = ["parked_obstacles", "cones"]
```

**效果**：每个Episode随机选择场景

### 方式2: 固定单一场景

```python
# 只训练parked_obstacles
config.random_scenario = False
config.scenario = "parked_obstacles"

# 或只训练cones
config.random_scenario = False
config.scenario = "cones"
```

**效果**：所有Episode使用同一场景

### 方式3: 自定义场景池

```python
# 可以添加更多场景（未来实现后）
config.random_scenario = True
config.scenario_pool = ["parked_obstacles", "cones", "jaywalker"]
```

---

## 📊 训练策略建议

### 策略1: 分阶段训练

#### 阶段1: 单场景训练（Episode 1-100）
```python
config.random_scenario = False
config.scenario = "parked_obstacles"
```
**目的**：先学会基本的避让能力

#### 阶段2: 切换场景（Episode 101-200）
```python
config.random_scenario = False
config.scenario = "cones"
```
**目的**：学习不同类型的避让

#### 阶段3: 随机场景（Episode 201+）
```python
config.random_scenario = True
config.scenario_pool = ["parked_obstacles", "cones"]
```
**目的**：提高泛化能力

### 策略2: 直接随机训练（推荐）

```python
# 从一开始就使用随机场景
config.random_scenario = True
config.scenario_pool = ["parked_obstacles", "cones"]
```

**优点**：
- 模型从一开始就学习应对多种场景
- 泛化能力更强
- 不容易过拟合单一场景

**缺点**：
- 初期学习可能较慢
- 需要更多训练时间

---

## 🔍 监控训练

### 1. 终端输出

每个Episode开始时会显示：
```
[RandomScenario] 本次Episode场景: xxx
```

可以看到场景切换情况。

### 2. Wandb日志

可以在wandb中添加场景类型的记录：

```python
# 在 train_with_logging() 中添加：
if wandb_run:
    wandb_run.log({
        "episode/scenario": config.scenario,  # 记录当前场景
        "episode/reward": episode_reward,
        ...
    })
```

### 3. 统计场景分布

可以统计每种场景的出现次数：

```python
scenario_counts = {"parked_obstacles": 0, "cones": 0}

# 在每个episode后：
scenario_counts[current_scenario] += 1

print(f"场景统计: {scenario_counts}")
```

---

## ⚙️ 高级配置

### 1. 调整场景概率

如果想让某个场景出现更频繁：

```python
# 方法1: 在场景池中重复添加
config.scenario_pool = ["parked_obstacles", "parked_obstacles", "cones"]
# 结果：parked_obstacles 66.7%, cones 33.3%

# 方法2: 修改carla_env.py中的选择逻辑
# 使用weighted random choice
```

### 2. 动态调整场景池

```python
# 根据训练进度调整场景
if episode < 100:
    config.scenario_pool = ["parked_obstacles"]
elif episode < 200:
    config.scenario_pool = ["cones"]
else:
    config.scenario_pool = ["parked_obstacles", "cones"]
```

### 3. 场景难度递增

```python
# 简单 → 中等 → 困难
if episode < 100:
    config.num_parked_cars = 2
    config.cone_num = 10
elif episode < 200:
    config.num_parked_cars = 3
    config.cone_num = 15
else:
    config.num_parked_cars = 4
    config.cone_num = 20
```

---

## 🐛 调试技巧

### 1. 验证随机性

运行几个episode，检查场景是否真的在变化：

```bash
python train_ppo_with_wandb.py | grep "RandomScenario"
```

应该看到：
```
[RandomScenario] 本次Episode场景: parked_obstacles
[RandomScenario] 本次Episode场景: cones
[RandomScenario] 本次Episode场景: parked_obstacles
[RandomScenario] 本次Episode场景: cones
...
```

### 2. 测试单一场景

如果某个场景有问题，可以临时关闭随机：

```python
config.random_scenario = False
config.scenario = "cones"  # 只测试cones
```

### 3. 检查场景参数

确保两个场景的参数都正确配置：

```python
# Parked obstacles
config.num_parked_cars = 4  ✅
config.parked_car_spacing = 8.0  ✅

# Cones
config.cone_num = 15  ✅
config.cone_step_behind = 3.0  ✅
```

---

## 📈 预期训练效果

### 单场景训练 vs 随机场景训练

| 指标 | 单场景 | 随机场景 |
|------|--------|---------|
| 初期学习速度 | 快 | 慢 |
| 收敛速度 | 快 | 慢 |
| 单场景性能 | 高 | 中 |
| 泛化能力 | 低 | 高 |
| 过拟合风险 | 高 | 低 |
| 推荐用途 | 快速验证 | 实际部署 |

### 训练曲线预期

**单场景训练**：
```
Reward
  ^
  |     /----
  |    /
  |   /
  |  /
  | /
  +-----------> Episode
  快速上升，但只在单一场景表现好
```

**随机场景训练**：
```
Reward
  ^
  |         /---
  |       /
  |     /
  |   /
  | /
  +-----------> Episode
  上升较慢，但在多种场景都表现好
```

---

## 🎯 当前配置总结

```python
✅ 随机场景: 开启
✅ 场景池: ["parked_obstacles", "cones"]
✅ 可视化: 开启
✅ Spectator: chase模式

场景参数:
- Parked obstacles: 4辆车，间距8米
- Cones: 15个锥桶，间距3米
```

---

## 🚀 运行训练

```bash
# 1. 启动CARLA
cd /home/ajifang/carla
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600

# 2. 运行训练
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

观察终端输出，应该看到场景在 `parked_obstacles` 和 `cones` 之间随机切换！

---

**随机场景训练已配置完成！祝训练顺利！** 🎉
