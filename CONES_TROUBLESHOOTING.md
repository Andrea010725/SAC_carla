# 🔍 Cones场景问题诊断

## 问题描述

运行 `python train_ppo_with_wandb.py` 后，自车前面没有看到锥桶。

---

## 🛠️ 诊断步骤

### 步骤1: 运行诊断脚本

我创建了一个专门的诊断脚本来测试cones场景。

```bash
# 1. 确保CARLA正在运行
cd /home/ajifang/carla
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600

# 2. 在新终端运行诊断脚本
cd /home/ajifang/SAC_carla
python test_cones_scenario.py
```

**预期输出**：
```
🔍 Cones场景诊断
======================================================================
[1/5] 连接CARLA服务器...
✅ 连接成功
   - 地图: Town05

[2/5] 创建配置...
✅ 配置创建成功
   - 场景: cones
   - 锥桶数量: 15

[3/5] 创建场景实例...
✅ 场景实例创建成功

[4/5] 初始化场景（生成锥桶）...
[Cones] 开始生成锥桶场景...
  - 锥桶数量: 15
  - 纵向间距: 3.0m
  - 横向递进: 0.4m
  - 起始位置: (123.4, 56.7)
  - 放置策略: 从左侧向右侧移动
[Cones] ✅ 成功生成 15 个锥桶
✅ 场景初始化成功

[5/5] 检查生成结果...
   - 障碍物数量: 15
✅ 成功生成 15 个锥桶

   前3个锥桶位置:
   [1] (123.4, 56.7, 0.0)
   [2] (123.4, 59.7, 0.0)
   [3] (123.4, 62.7, 0.0)

   自车spawn位置:
   - 位置: (123.4, 36.7, 0.3)
   - 距离第一个锥桶: 20.0m

✅ 诊断完成！Cones场景工作正常
```

---

### 步骤2: 检查训练脚本输出

运行 `python train_ppo_with_wandb.py` 时，请仔细查看终端输出：

#### 关键信息1: 场景类型
```
🎭 场景类型: cones  ← 应该显示 "cones"
```

如果显示的是 `parked_obstacles` 或其他，说明配置没有生效。

#### 关键信息2: 场景初始化
```
[Cones] 开始生成锥桶场景...  ← 应该有这行
[Cones] ✅ 成功生成 15 个锥桶  ← 应该有这行
```

如果没有这些输出，说明场景没有被调用。

#### 关键信息3: 障碍物数量
```
🚧 障碍物信息 (共15个):  ← 应该是15个，不是0个
   [1] static.prop.trafficcone01
       距离自车: 20.0m
```

如果显示 `🚧 障碍物: 无`，说明锥桶没有生成或没有注册。

---

### 步骤3: 检查CARLA窗口

在CARLA窗口中：

#### 检查1: 是否有锥桶模型
- 锥桶是橙白相间的交通锥
- 应该有15个
- 排列成一条线

#### 检查2: 是否开启了调试绘制
如果看不到边界框，可能是调试绘制没开启：

```python
# 在 train_ppo_with_wandb.py 中确认：
config.enable_debug_drawing = True
config.draw_obstacle_boxes = True
```

#### 检查3: 自车位置
- 自车应该在锥桶前方
- 不应该在锥桶后方或侧方

---

## 🔍 可能的原因

### 原因1: 配置没有生效

**检查**：`train_ppo_with_wandb.py` 第698行
```python
config.scenario = "cones"  # 确认是 "cones" 不是 "parked_obstacles"
```

**解决**：确保修改已保存，重新运行。

---

### 原因2: 场景生成失败

**症状**：终端显示
```
[Cones] ❌ 无法找到合适的起始位置
```

**原因**：
- 地图上没有合适的waypoint
- 所有waypoint都靠近路口

**解决**：
```python
# 降低路口距离要求
config.cone_min_gap_from_junction = 10.0  # 从15改为10
```

或者切换地图：
```python
# 在 config.py 中修改
self.town = "Town03"  # 或 "Town01", "Town05"
```

---

### 原因3: 锥桶生成了但看不到

**症状**：
- 终端显示生成成功
- 但CARLA窗口看不到

**可能原因**：
1. 锥桶在自车后方
2. 锥桶在很远的地方
3. 相机角度问题

**解决**：
```python
# 调整自车距离
config.spawn_min_gap_from_cone = 15.0  # 从20改为15，更近一些

# 或者使用fixed模式看得更清楚
config.spectator_mode = "fixed"  # 从"chase"改为"fixed"
```

---

### 原因4: obstacle_actors没有正确注册

**检查**：在 `scenario_manager.py` 的 `ConesScenario.setup()` 中

第369行应该有：
```python
self.scenario_actors.extend(cones)
```

**验证**：运行诊断脚本，看 `障碍物数量` 是否为15。

---

## 📊 对比测试

### 测试1: 切换回parked_obstacles

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "parked_obstacles"
config.num_parked_cars = 4
```

运行后，如果能看到4辆车，说明：
- 场景系统工作正常
- 问题出在cones场景本身

### 测试2: 减少锥桶数量

```python
config.cone_num = 5  # 从15改为5
```

如果5个能看到，15个看不到，可能是：
- 生成位置问题
- 性能问题

---

## 🐛 调试技巧

### 技巧1: 添加打印输出

在 `scenario_manager.py` 的 `ConesScenario._place_cones_conditionally_behind()` 中添加：

```python
# 在第527行生成锥桶后添加：
if cone_actor:
    cones_spawned.append(cone_actor)
    loc = cone_actor.get_location()
    print(f"    ✅ 锥桶{i}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")  # 添加这行
```

这样可以看到每个锥桶的具体位置。

### 技巧2: 检查spawn位置

在 `scenario_manager.py` 的 `ConesScenario._calculate_ego_spawn()` 中添加：

```python
# 在第566行返回前添加：
print(f"[DEBUG] 自车spawn: ({wp.transform.location.x:.1f}, {wp.transform.location.y:.1f})")
print(f"[DEBUG] 第一个锥桶: ({self.first_cone_transform.location.x:.1f}, {self.first_cone_transform.location.y:.1f})")
return wp.transform
```

### 技巧3: 使用CARLA的spectator手动查看

在CARLA窗口中：
1. 按 `F1` 切换到自由视角
2. 用鼠标和键盘移动
3. 查找锥桶在哪里

---

## 📝 请提供以下信息

为了帮你更好地诊断，请提供：

### 1. 诊断脚本输出
```bash
python test_cones_scenario.py
```
完整输出是什么？

### 2. 训练脚本输出
```bash
python train_ppo_with_wandb.py
```
特别是：
- `🎭 场景类型:` 显示什么？
- 是否有 `[Cones] 开始生成锥桶场景...`？
- `🚧 障碍物信息` 显示多少个？

### 3. CARLA窗口情况
- 能看到自车吗？
- 自车周围有什么？
- 能看到任何锥桶吗？

---

## 🎯 快速修复尝试

如果上面的都不行，尝试这个最简单的配置：

```python
# 在 train_ppo_with_wandb.py 中：
config.scenario = "cones"
config.cone_num = 5                     # 减少到5个
config.cone_step_behind = 5.0           # 增加间距
config.spawn_min_gap_from_cone = 10.0   # 减少距离
config.spectator_mode = "fixed"         # 使用固定视角
```

---

**请先运行诊断脚本，然后告诉我输出结果！** 🔍
