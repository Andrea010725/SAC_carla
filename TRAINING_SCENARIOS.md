# 🎮 训练场景完整说明

## 📌 回答你的问题

**Q: 所以现在的训练场景是唯一的一种吗？**

**A: 不是！代码支持4种场景，但当前训练脚本只使用了1种。**

---

## 🎯 支持的4种场景

### 1️⃣ **plain** (空场景)

**特点**：
- 纯净道路，无任何障碍
- 自车随机spawn在地图上
- 适合学习基础驾驶技能

**配置**：
```python
config.scenario = "plain"
```

**用途**：
- 基础训练
- 验证环境是否正常
- 学习车道保持、速度控制

---

### 2️⃣ **cones** (锥桶场景)

**特点**：
- 在自车前方放置一排锥桶
- 锥桶从车道一侧逐渐推进到另一侧
- 模拟窄道/避障场景

**配置**：
```python
config.scenario = "cones"
config.cone_num = 15               # 锥桶数量
config.cone_step_behind = 3.0      # 纵向间距（米）
config.cone_step_lateral = 0.4     # 横向推进步长
config.cone_min_gap_from_junction = 15.0  # 距路口最小距离
```

**场景示例**：
```
自车spawn点 (后方20米)
    ↓
🚗━━━━━━→ 20m → 🔺━3m→🔺━3m→🔺━3m→🔺━3m→...
                 ↓0.4m  ↓0.4m  ↓0.4m
            从路肩逐渐推进到车道中心
```

**代码位置**：`carla_env.py:635-654`

**用途**：
- 学习避障
- 学习精确转向
- 适合窄道场景

---

### 3️⃣ **cones_xml** (XML路线锥桶)

**特点**：
- 从XML文件读取预定义路线
- 在路线上随机选点放置锥桶
- 支持多地图、多路线

**配置**：
```python
config.scenario = "cones_xml"
config.xml_file = "/path/to/waypoints.xml"  # 单个XML
# 或
config.xml_dir = "/path/to/xml_folder"      # 多个XML随机选
config.randomize_town = True                # 随机切换地图
config.town_pool = ["Town01", "Town03", "Town05"]
```

**XML格式**：
```xml
<routes>
  <route town="Town05">
    <waypoints>
      <position x="100.0" y="200.0" z="0.5" yaw="90.0"/>
      <position x="105.0" y="202.0" z="0.5" yaw="95.0"/>
      ...
    </waypoints>
  </route>
</routes>
```

**代码位置**：`carla_env.py:604-632`

**用途**：
- 测试特定路线
- 多样化训练（多地图）
- 可重复的场景（相同XML）

---

### 4️⃣ **parked_obstacles** (停车障碍 - 当前使用) ✅

**特点**：
- 在自车前方路边放置4辆停车
- 模拟Overtaking（超车）场景
- 需要变道、超车、回归策略

**配置**：
```python
config.scenario = "parked_obstacles"
config.num_parked_cars = 4              # 停车数量
config.parked_car_spacing = 50.0        # 停车间距（米）
config.parked_car_offset = 1.8          # 横向偏移（米，路边）
config.parked_car_start_distance = 30.0 # 第一辆车距起点（米）
```

**场景布局**：
```
自车spawn点
    ↓
🚗━━━━━━→ 30m → 🚙 ━━ 50m ━━→ 🚙 ━━ 50m ━━→ 🚙 ━━ 50m ━━→ 🚙
                 ↑1.8m       ↑1.8m       ↑1.8m       ↑1.8m
              (路边停车，不模拟物理)
```

**停车特点**：
- 横向偏移1.8米（模拟路边停车）
- `set_simulate_physics(False)` - 静止不动
- 随机车型（Tesla, Audi, BMW, Toyota）

**代码位置**：`carla_env.py:657-674, 791-862`

**用途**：
- 学习超车策略 ⭐
- 变道决策
- 速度控制（减速-加速）
- 安全距离判断

---

## 🔄 场景选择逻辑

### 代码中的优先级（`carla_env.py:581-682`）

```python
def _maybe_setup_scene_and_pick_spawn(self):
    # 优先级1: 强制指定spawn点
    if self.initial_spawn_tf is not None:
        return self.initial_spawn_tf

    # 优先级2: cones_xml场景
    if self.scenario == "cones_xml":
        # 读取XML，放置锥桶，基于锥桶选spawn
        ...

    # 优先级3: cones场景
    if self.scenario == "cones":
        # 随机选点，放置锥桶，基于锥桶选spawn
        ...

    # 优先级4: parked_obstacles场景 ✅（当前使用）
    if self.scenario == "parked_obstacles":
        # 随机选点，放置停车，在起点spawn
        ...

    # 兜底: plain场景（随机spawn）
    return random.choice(self.map.get_spawn_points())
```

---

## 📊 当前训练配置

### train_ppo_with_wandb.py (line 458-461)

```python
config = Config()
config.scenario = "parked_obstacles"  # ✅ 只使用这一种
config.num_parked_cars = 4
config.render = True
```

**所以，当前训练确实只使用 `parked_obstacles` 一种场景！**

---

## 🔧 如何切换/混合场景

### 方法1: 单一场景训练

**切换到锥桶场景**：
```python
# train_ppo_with_wandb.py line 459
config.scenario = "cones"
config.cone_num = 15
```

**切换到空场景**：
```python
config.scenario = "plain"
```

---

### 方法2: 混合场景训练（推荐）

**每个episode随机选择场景**：

```python
# 在 train_ppo_with_wandb.py 的训练循环中修改

import random

# 在 env.reset() 之前
scenarios = ["parked_obstacles", "cones", "plain"]
config.scenario = random.choice(scenarios)

# 如果选中cones，设置参数
if config.scenario == "cones":
    config.cone_num = random.randint(10, 20)
    config.cone_step_lateral = random.uniform(0.3, 0.5)

# 如果选中parked_obstacles，随机化参数
elif config.scenario == "parked_obstacles":
    config.num_parked_cars = random.randint(3, 6)
    config.parked_car_spacing = random.uniform(40.0, 60.0)
    config.parked_car_start_distance = random.uniform(20.0, 40.0)

obs = env.reset()
```

**优点**：
- ✅ 更好的泛化能力
- ✅ 防止过拟合单一场景
- ✅ 学到更鲁棒的策略

**缺点**：
- ⚠️ 训练可能更慢（多任务学习）
- ⚠️ Reward scale可能不一致

---

### 方法3: 渐进式训练（Curriculum Learning）

**先简单，后复杂**：

```python
# 阶段1: Episode 0-200 → plain场景
if episode < 200:
    config.scenario = "plain"

# 阶段2: Episode 200-500 → cones场景
elif episode < 500:
    config.scenario = "cones"
    config.cone_num = min(10 + (episode - 200) // 20, 20)  # 逐渐增加难度

# 阶段3: Episode 500+ → parked_obstacles场景
else:
    config.scenario = "parked_obstacles"
    config.num_parked_cars = min(2 + (episode - 500) // 100, 6)
```

**优点**：
- ✅ 符合人类学习规律
- ✅ 更稳定的训练
- ✅ 更高的成功率

---

## 🎯 场景难度对比

| 场景 | 难度 | 主要挑战 | 适合阶段 |
|-----|------|----------|---------|
| plain | ⭐ | 车道保持、速度控制 | 初期（0-100 episodes）|
| cones | ⭐⭐⭐ | 避障、精确转向 | 中期（100-300 episodes）|
| parked_obstacles | ⭐⭐⭐⭐ | 超车、变道、决策 | 后期（300+ episodes）✅ 当前 |
| cones_xml | ⭐⭐⭐⭐⭐ | 多样化场景、泛化 | 评估/测试 |

---

## 💡 场景设计建议

### 建议1: 增加场景随机性

**当前问题**：
- 停车位置相对固定（前方30米）
- 间距固定（50米）
- 数量固定（4辆）

**改进**：
```python
# carla_env.py reset() 中添加
self.parked_car_start_distance = random.uniform(20.0, 50.0)
self.parked_car_spacing = random.uniform(40.0, 70.0)
self.num_parked_cars = random.randint(2, 6)
```

**效果**：
- ✅ 防止过拟合
- ✅ 更好的泛化
- ✅ 更鲁棒的策略

---

### 建议2: 混合场景训练

**实现**：在 `train_ppo_with_wandb.py` 中：

```python
# line 459 修改为
scenario_pool = [
    ("parked_obstacles", 0.5),  # 50%概率
    ("cones", 0.3),             # 30%概率
    ("plain", 0.2)              # 20%概率
]

import numpy as np
scenarios, probs = zip(*scenario_pool)
config.scenario = np.random.choice(scenarios, p=probs)
```

**权重说明**：
- 50% parked_obstacles - 主要训练目标（overtaking）
- 30% cones - 辅助训练（避障）
- 20% plain - 基础训练（车道保持）

---

### 建议3: 创建自定义场景

**新场景：Dynamic Obstacles（动态障碍）**

```python
# 在 carla_env.py 中添加

if self.scenario == "dynamic_obstacles":
    # 放置移动的车辆（使用autopilot）
    for i in range(3):
        npc_vehicle = spawn_npc_vehicle(...)
        npc_vehicle.set_autopilot(True, self.tm_port)
        npc_vehicle.set_target_velocity(carla.Vector3D(5.0, 0, 0))  # 慢速行驶
        self._actors.append(npc_vehicle)
```

**挑战**：
- 需要预测NPC行为
- 动态决策（何时超车）
- 更接近真实场景

---

## 📈 推荐训练策略

### 策略A: 专注Overtaking（当前方案）✅

```python
config.scenario = "parked_obstacles"  # 单一场景
# 训练1000 episodes
```

**优点**：
- 快速收敛
- 针对性强
- 适合论文/demo

**缺点**：
- 泛化能力弱
- 过拟合风险

---

### 策略B: 混合训练（推荐）⭐

```python
# 每个episode随机选择
scenarios = ["parked_obstacles", "cones", "plain"]
config.scenario = random.choice(scenarios)
# 训练1500 episodes
```

**优点**：
- 泛化能力强
- 鲁棒性高
- 适合实际部署

**缺点**：
- 收敛较慢
- Reward可能波动

---

### 策略C: 渐进式训练（最佳）🏆

```python
# Episode 0-300: plain
# Episode 300-600: cones
# Episode 600-1000: parked_obstacles
# Episode 1000+: 混合训练
```

**优点**：
- 稳定训练
- 高成功率
- 最佳泛化

**缺点**：
- 训练时间长
- 需要调整curriculum

---

## 🔍 场景代码位置索引

| 场景 | 设置位置 | 代码位置 |
|-----|---------|---------|
| plain | `config.scenario = "plain"` | `carla_env.py:676-682` |
| cones | `config.scenario = "cones"` | `carla_env.py:635-654, 719-788` |
| cones_xml | `config.scenario = "cones_xml"` | `carla_env.py:604-632` |
| parked_obstacles | `config.scenario = "parked_obstacles"` | `carla_env.py:657-674, 791-862` |

**场景选择主函数**：`carla_env.py:581-682 (_maybe_setup_scene_and_pick_spawn)`

---

## ✅ 总结

### 当前状态
- ✅ 代码支持4种场景
- ✅ 当前只用1种（parked_obstacles）
- ✅ 适合快速训练overtaking技能

### 改进方向
1. **短期**：增加场景随机性（停车位置、数量、间距）
2. **中期**：混合场景训练（parked_obstacles + cones + plain）
3. **长期**：渐进式训练 + 动态障碍场景

### 推荐配置

**快速验证（当前）**：
```python
config.scenario = "parked_obstacles"
episodes = 1000
```

**生产训练（推荐）**：
```python
# 混合场景
scenarios = ["parked_obstacles", "cones", "plain"]
config.scenario = random.choice(scenarios)
episodes = 1500
```

**最佳性能**：
```python
# 渐进式训练
if episode < 300:
    config.scenario = "plain"
elif episode < 600:
    config.scenario = "cones"
else:
    config.scenario = random.choice(["parked_obstacles", "cones", "plain"])
episodes = 2000
```

---

**你可以根据训练目标选择合适的场景策略！** 🚀
