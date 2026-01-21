# 🎨 可视化配置说明

## 📊 当前绘制的所有元素

### 1. 🔵 **青色圆圈** (检测范围)
- **位置**: 自车周围
- **半径**: 50米
- **作用**: 显示障碍物检测范围
- **开关**: `config.draw_detection_range = True/False`

### 2. 🟢 **绿色箭头** (前向方向)
- **起点**: 自车位置（高度+1米）
- **长度**: 10米
- **作用**: 显示自车的前向方向（朝哪开）
- **开关**: `config.draw_ego_direction = True/False`

### 3. 🔴🟠🟡 **彩色边界框** (障碍物)
- **红色**: 距离 < 10米（危险！）
- **橙色**: 距离 10-20米（警告）
- **黄色**: 距离 > 20米（安全）
- **作用**: 标记障碍物位置和危险程度
- **开关**: `config.draw_obstacle_boxes = True/False`

### 4. ⚪ **白色虚线** (车道中心)
- **位置**: 自车前方40米
- **作用**: 显示车道中心线，判断是否偏离
- **开关**: `config.draw_lane_center = True/False`

---

## ⚙️ 如何自定义可视化

### 方法1: 在 `train_ppo_with_wandb.py` 中配置

```python
# 在 train_ppo() 函数中，config 创建后添加：

# 全局开关（关闭所有调试绘制）
config.enable_debug_drawing = True  # False = 关闭所有

# 细节开关（单独控制每个元素）
config.draw_detection_range = True   # 青色圆圈
config.draw_ego_direction = True     # 绿色箭头
config.draw_obstacle_boxes = True    # 障碍物边界框
config.draw_lane_center = True       # 白色车道线

# 绘制频率（降低可提升性能）
config.debug_draw_interval = 5       # 每5步绘制一次
```

### 方法2: 在 `config.py` 中设置默认值

```python
# 在 Config 类的 __init__ 中添加：
self.enable_debug_drawing = True
self.draw_detection_range = True
self.draw_ego_direction = True
self.draw_obstacle_boxes = True
self.draw_lane_center = True
self.debug_draw_interval = 5
```

---

## 🎯 常见配置场景

### 场景1: 只看障碍物（最简洁）
```python
config.draw_detection_range = False  # 关闭圆圈
config.draw_ego_direction = False    # 关闭箭头
config.draw_obstacle_boxes = True    # 保留障碍物
config.draw_lane_center = False      # 关闭车道线
```

### 场景2: 只看方向和车道（调试转向）
```python
config.draw_detection_range = False
config.draw_ego_direction = True     # 保留箭头
config.draw_obstacle_boxes = False
config.draw_lane_center = True       # 保留车道线
```

### 场景3: 完全关闭（最快性能）
```python
config.enable_debug_drawing = False  # 一键关闭所有
```

### 场景4: 全部开启（完整调试）
```python
config.enable_debug_drawing = True
config.draw_detection_range = True
config.draw_ego_direction = True
config.draw_obstacle_boxes = True
config.draw_lane_center = True
config.debug_draw_interval = 5
```

---

## 🔍 可视化元素详解

### 绿色箭头的作用

**为什么需要它？**
- 显示自车的**实际朝向**（不是速度方向）
- 帮助判断转向是否正确
- 在静止时也能看到车头方向

**如何使用？**
- 箭头应该指向道路前方
- 如果箭头指向路边 → 转向有问题
- 如果箭头和车道线平行 → 方向正确

**示例**：
```
正常情况:
  车道线: ————————————→
  绿色箭头: ————→ (平行)

转向过度:
  车道线: ————————————→
  绿色箭头:    ↗ (偏离)
```

### 青色圆圈的作用

**为什么需要它？**
- 显示**障碍物检测范围**（50米）
- 帮助判断障碍物是否在检测范围内
- 验证观测配置是否正确

**如何使用？**
- 障碍物在圆圈内 → 应该被检测到
- 障碍物在圆圈外 → 不会被检测到
- 如果圆圈内的障碍物没有边界框 → 检测有问题

### 彩色边界框的作用

**为什么需要它？**
- 显示**已检测到的障碍物**
- 颜色表示**危险程度**
- 距离标签显示**精确距离**

**颜色含义**：
- 🔴 **红色** (< 10米): 非常危险，应该紧急避让
- 🟠 **橙色** (10-20米): 需要注意，准备避让
- 🟡 **黄色** (> 20米): 安全距离，正常行驶

### 白色车道线的作用

**为什么需要它？**
- 显示**车道中心线**（前方40米）
- 帮助判断是否**偏离车道**
- 验证车道保持功能

**如何使用？**
- 自车应该在车道线附近
- 如果自车偏离车道线 → 车道保持有问题
- 如果车道线弯曲 → 自车应该跟随弯曲

---

## 📊 性能影响

### 绘制频率对比

| 配置 | FPS影响 | 适用场景 |
|------|---------|---------|
| `debug_draw_interval = 1` | -30% | 详细调试 |
| `debug_draw_interval = 5` | -10% | 正常训练（推荐）|
| `debug_draw_interval = 10` | -5% | 快速训练 |
| `enable_debug_drawing = False` | 0% | 最快训练 |

### 元素数量对比

| 配置 | 绘制元素数 | 性能影响 |
|------|-----------|---------|
| 全部开启 | ~100条线 | 中等 |
| 只开障碍物 | ~20条线 | 较小 |
| 只开箭头 | 1条线 | 极小 |
| 全部关闭 | 0条线 | 无 |

---

## 🛠️ 快速修改示例

### 示例1: 关闭绿色箭头

在 `train_ppo_with_wandb.py` 的 `train_ppo()` 函数中，找到：
```python
config = Config()
config.scenario = "parked_obstacles"
config.num_parked_cars = 4
config.render = True
config.spectator_mode = "chase"
```

添加：
```python
config.draw_ego_direction = False  # 关闭绿色箭头
```

### 示例2: 只保留障碍物边界框

```python
config.draw_detection_range = False
config.draw_ego_direction = False
config.draw_obstacle_boxes = True   # 只保留这个
config.draw_lane_center = False
```

### 示例3: 降低绘制频率（提升性能）

```python
config.debug_draw_interval = 10  # 从5改为10
```

---

## 🎨 可视化效果预览

### 完整可视化（全部开启）
```
     🟢 (绿色箭头，指向前方)
      ↑
     🚗 (自车)
    ╱   ╲
   ╱     ╲  🔵 (青色圆圈，50米)
  ╱       ╲
 ╱    🟡   ╲  (黄色障碍物，远)
╱     🟠    ╲ (橙色障碍物，中)
      🔴      (红色障碍物，近)
————————————— (白色车道线)
```

### 最简可视化（只有障碍物）
```
     🚗 (自车)

     🟡 (障碍物)
     🟠
     🔴
```

---

## 📝 调试技巧

### 1. 验证场景是否正确

**检查清单**：
- [ ] 能看到4个障碍物边界框（如果配置了4辆车）
- [ ] 障碍物距离在12-20米范围内
- [ ] 绿色箭头指向道路前方
- [ ] 白色车道线沿着道路延伸
- [ ] 青色圆圈覆盖障碍物

### 2. 诊断训练问题

**如果自车频繁碰撞**：
- 开启 `draw_obstacle_boxes` 查看障碍物位置
- 开启 `draw_ego_direction` 查看转向是否正确
- 检查红色边界框是否频繁出现

**如果自车偏离车道**：
- 开启 `draw_lane_center` 查看车道线
- 开启 `draw_ego_direction` 查看朝向
- 检查自车是否跟随车道线

**如果自车不避让障碍物**：
- 开启 `draw_detection_range` 查看检测范围
- 开启 `draw_obstacle_boxes` 查看是否检测到
- 检查障碍物是否在青色圆圈内

---

## 🆘 常见问题

### Q: 为什么看不到绿色箭头？
A: 可能被关闭了，设置 `config.draw_ego_direction = True`

### Q: 为什么看不到障碍物边界框？
A: 检查：
1. `config.draw_obstacle_boxes = True`
2. 障碍物是否在检测范围内（青色圆圈）
3. `obstacle_actors` 是否正确注册

### Q: 可视化太卡怎么办？
A: 降低绘制频率：`config.debug_draw_interval = 10` 或 `20`

### Q: 如何完全关闭可视化？
A: `config.enable_debug_drawing = False`

---

**现在你可以根据需要自定义可视化了！🎨**
