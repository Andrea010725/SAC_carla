# ✅ Cones场景迁移完成

## 📋 迁移总结

Cones场景已成功迁移到新的场景管理系统！

---

## 🎯 如何使用

### 方法1: 在训练脚本中使用

编辑 `train_ppo_with_wandb.py`：

```python
config = Config()
config.scenario = "cones"  # 切换到cones场景
config.render = True
config.spectator_mode = "chase"

# Cones场景参数（可选，使用默认值）
config.cone_num = 15                    # 锥桶数量
config.cone_step_behind = 3.0           # 纵向间距（米）
config.cone_step_lateral = 0.4          # 横向递进（米）
config.cone_z_offset = 0.0              # 高度偏移
config.cone_lane_margin = 0.25          # 距边缘距离
config.cone_min_gap_from_junction = 15.0 # 距路口距离
config.spawn_min_gap_from_cone = 20.0   # 自车距第一个锥桶距离
```

### 方法2: 在config.py中修改默认值

编辑 `config.py`：

```python
# 在 Config 类的 __init__ 中修改：
self.scenario = "cones"  # 从 "parked_obstacles" 改为 "cones"
```

---

## 🎮 运行训练

```bash
# 1. 启动CARLA服务器
cd /home/ajifang/carla
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600

# 2. 在新终端运行训练
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

---

## 📊 预期输出

### 终端输出示例

```
======================================================================
🎬 Episode 1 - 场景初始化
======================================================================
📍 地图: Town05
🎭 场景类型: cones

[Cones] 开始生成锥桶场景...
  - 锥桶数量: 15
  - 纵向间距: 3.0m
  - 横向递进: 0.4m
  - 起始位置: (123.4, 56.7)
  - 放置策略: 从左侧向右侧移动（右侧是非行车道）
[Cones] ✅ 成功生成 15 个锥桶

🚗 自车信息:
   - 位置: (123.4, 36.7, 0.3)
   - 朝向: Yaw=90.0°
   - 车型: tesla.cybertruck

🚧 障碍物信息 (共15个):
   [1] static.prop.trafficcone01
       位置: (123.4, 56.7, 0.0)
       距离自车: 20.0m
   [2] static.prop.trafficcone01
       位置: (123.4, 59.7, 0.0)
       距离自车: 23.0m
   ...

👁️  观测配置:
   - 类型: state_lane_obstacles
   - 维度: 30
   - 障碍物检测数量: 5
   - 检测范围: 50.0m

🎨 可视化:
   - Pygame渲染: ✅
   - Spectator模式: chase
   - 调试绘制: ✅
======================================================================
```

### CARLA窗口效果

你会看到：
- 🚗 自车在车道中心
- 🔶 15个锥桶从车道一侧逐渐向另一侧移动
- 🟢 绿色箭头（自车方向）
- 🔵 青色圆圈（50米检测范围）
- 🟡 黄色边界框（锥桶，距离较远）
- ⚪ 白色车道线

---

## ⚙️ 参数调整

### 简单模式（适合初期训练）

```python
config.cone_num = 10                    # 减少锥桶
config.cone_step_behind = 4.0           # 增加间距
config.cone_step_lateral = 0.3          # 减慢收窄
```

**效果**：
- 总长度: 40米
- 收窄速度: 慢
- 难度: ⭐⭐

### 中等模式（默认）

```python
config.cone_num = 15
config.cone_step_behind = 3.0
config.cone_step_lateral = 0.4
```

**效果**：
- 总长度: 45米
- 收窄速度: 中等
- 难度: ⭐⭐⭐

### 困难模式（适合后期训练）

```python
config.cone_num = 20                    # 增加锥桶
config.cone_step_behind = 2.5           # 减小间距
config.cone_step_lateral = 0.5          # 加快收窄
```

**效果**：
- 总长度: 50米
- 收窄速度: 快
- 难度: ⭐⭐⭐⭐

---

## 🔍 验证场景正确性

### 检查清单

运行训练后，检查以下内容：

- [ ] 终端显示 `[Cones] ✅ 成功生成 15 个锥桶`
- [ ] CARLA窗口中能看到15个锥桶
- [ ] 锥桶从车道一侧逐渐向另一侧移动
- [ ] 自车spawn在第一个锥桶前方约20米
- [ ] 青色圆圈覆盖锥桶区域
- [ ] 锥桶显示黄色/橙色边界框（根据距离）

### 常见问题

#### Q1: 看不到锥桶？
```
检查：
1. config.enable_debug_drawing = True
2. config.draw_obstacle_boxes = True
3. 锥桶是否在检测范围内（50米）
```

#### Q2: 锥桶生成失败？
```
终端会显示：
[Cones] ❌ 无法找到合适的起始位置

解决：
- 切换到其他地图（Town03, Town05）
- 减小 cone_min_gap_from_junction（从15改为10）
```

#### Q3: 自车spawn位置不对？
```
检查：
- spawn_min_gap_from_cone 参数（默认20米）
- 第一个锥桶是否成功生成
```

---

## 📈 与parked_obstacles场景对比

| 特性 | parked_obstacles | cones |
|------|-----------------|-------|
| 障碍物类型 | 静态车辆 | 静态锥桶 |
| 障碍物大小 | 大（4-5米） | 小（0.5米） |
| 数量 | 4个 | 15个 |
| 分布模式 | 离散（间隔8米） | 连续（间隔3米） |
| 横向变化 | 无 | 有（逐渐移动） |
| 总长度 | 32米 | 45米 |
| 避让难度 | ⭐⭐⭐ | ⭐⭐⭐ |
| 训练价值 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 训练建议

### 训练顺序

1. **第1阶段**: 使用 `parked_obstacles` 场景
   - 学习基本的避让能力
   - 训练50-100个episodes

2. **第2阶段**: 切换到 `cones` 场景（简单模式）
   - 学习连续避让
   - 训练100-200个episodes

3. **第3阶段**: `cones` 场景（中等模式）
   - 提升横向控制精度
   - 训练200-300个episodes

4. **第4阶段**: `cones` 场景（困难模式）
   - 挑战极限
   - 训练300+个episodes

### 混合训练

也可以混合使用两个场景：

```python
# 在训练循环中随机切换场景
import random

for episode in range(total_episodes):
    if random.random() < 0.5:
        config.scenario = "parked_obstacles"
    else:
        config.scenario = "cones"

    # 训练...
```

---

## 🐛 调试技巧

### 1. 单独测试场景生成

创建测试脚本 `test_cones.py`：

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
config.scenario = "cones"
config.cone_num = 15

# 创建场景
scenario = ScenarioFactory.create_scenario(
    scenario_name="cones",
    world=world,
    carla_map=carla_map,
    config=config
)

# 初始化场景
if scenario:
    success = scenario.setup()
    if success:
        print("✅ 场景生成成功")
        print(f"锥桶数量: {len(scenario.get_obstacle_actors())}")
        print(f"自车spawn位置: {scenario.get_spawn_transform()}")

        # 保持场景10秒
        import time
        time.sleep(10)

        # 清理
        scenario.cleanup()
    else:
        print("❌ 场景生成失败")
```

### 2. 查看详细日志

在 `scenario_manager.py` 的 `ConesScenario.setup()` 中已经有详细的打印输出。

### 3. 可视化调试

```python
# 开启所有可视化
config.render = True
config.spectator_mode = "chase"
config.enable_debug_drawing = True
config.draw_detection_range = True
config.draw_ego_direction = True
config.draw_obstacle_boxes = True
config.draw_lane_center = True
```

---

## 📚 相关文档

- `CONES_SCENARIO_DETAILED.md` - 场景详细设计文档
- `CONES_CODE_ANALYSIS.md` - 代码逐行分析
- `SCENARIO_GUIDE.md` - 场景开发指南
- `VISUALIZATION_CONFIG.md` - 可视化配置说明

---

## ✅ 迁移完成清单

- [x] 创建 `ConesScenario` 类
- [x] 实现 `setup()` 方法
- [x] 实现 `_pick_random_start_waypoint()` 方法
- [x] 实现 `_is_near_junction()` 方法
- [x] 实现 `_place_cones_conditionally_behind()` 方法
- [x] 实现 `_calculate_ego_spawn()` 方法
- [x] 在 `ScenarioFactory` 中注册
- [x] 标记旧代码为废弃
- [x] 配置参数已在 `config.py` 中定义
- [x] 创建使用文档

---

**Cones场景已经可以使用了！祝训练顺利！** 🚀
