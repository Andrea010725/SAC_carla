# 4场景随机切换 - 修改指南

## 📋 概述

本指南说明如何将训练从单一场景改为4个场景随机切换。

**4个场景：**
1. **ConesScenario** - 锥桶避让场景
2. **JaywalkerScenario** - 鬼探头（行人横穿）场景
3. **TrimmaScenario** - 包围突围（左右夹击）场景
4. **ConstructionLaneChangeScenario** - 施工变道场景

---

## ✅ 已完成的修改

### 1. 新训练文件：`train_ppo_4scenarios.py`

**位置：** `/home/ajifang/SAC_carla/train_ppo_4scenarios.py`

**核心修改：**

```python
# 场景池配置（第692-697行）
config.random_scenario = True
config.scenario_pool = [
    "cones",                      # 锥桶场景
    "jaywalker",                  # 鬼探头（行人横穿）
    "jaywalker",                  # 鬼探头（行人横穿）
    "trimma",                     # 包围突围（左右夹击）
    "construction_lane_change",   # 施工变道
]
```

**新增功能：**
- ✅ 场景触发逻辑支持（第155-169行）
- ✅ JaywalkerScenario 的 `check_and_trigger()` 和 `tick_update()` 调用
- ✅ 4个场景的完整参数配置（第710-734行）
- ✅ Wandb 标签更新为 4scenarios

---

## 🔧 需要的额外修改

### 修改1：carla_env.py - 添加场景触发逻辑

**文件：** `carla_base/carla_env.py`

**位置：** `step()` 方法中，约第564行（`self._tick_once()` 之后）

**添加代码：**

```python
def step(self, action):
    # ... 现有代码 ...

    snapshot, display_image = self._tick_once(timeout=15.0)

    # ===== ✅ 新增：场景触发逻辑 =====
    if self.scenario_instance:
        # JaywalkerScenario: 检查并触发行人横穿
        if hasattr(self.scenario_instance, 'check_and_trigger'):
            try:
                ego_loc = self.ego.get_location()
                self.scenario_instance.check_and_trigger(ego_loc)
            except Exception as e:
                print(f"⚠️ check_and_trigger failed: {e}")

        # JaywalkerScenario: 更新行人移动
        if hasattr(self.scenario_instance, 'tick_update'):
            try:
                self.scenario_instance.tick_update()
            except Exception as e:
                print(f"⚠️ tick_update failed: {e}")

    # ✅ 绘制调试信息（每N步绘制一次，避免性能影响）
    if self.episode_steps % self.debug_draw_interval == 0:
        self._draw_debug_info()

    # ... 继续现有代码 ...
```

**说明：**
- `check_and_trigger()`: 检测自车距离，触发行人开始横穿
- `tick_update()`: 每帧更新行人移动（使用 WalkerControl）
- 其他场景（cones, trimma, construction）不需要这些调用

---

## 📊 场景参数配置

### Cones场景（锥桶）
```python
config.cone_num = 15                    # 锥桶数量
config.cone_step_behind = 3.0           # 纵向间距（米）
config.cone_step_lateral = 0.4          # 横向递进（米）
config.spawn_min_gap_from_cone = 20.0   # 自车距第一个锥桶距离
```

### Jaywalker场景（鬼探头）
```python
config.jaywalker_distance = 20.0        # 行人位置距离自车（米）
config.jaywalker_speed = 2.0            # 行人速度（m/s）
config.jaywalker_trigger_distance = 15.0 # 触发距离（米）
config.jaywalker_start_side = "random"  # 行人起始侧
```

### Trimma场景（包围突围）
```python
config.front_vehicle_distance = 18.0    # 前车距离（米）
config.side_vehicle_offset = 3.0        # 左右车偏移（米）
config.front_speed_diff_pct = -60.0     # 前车速度差（负=更快）
config.side_speed_diff_pct = +80.0      # 左右车速度差（正=更慢）
config.disable_lane_change = True       # 禁止变道
```

### Construction场景（施工变道）
```python
config.construction_distance = 30.0     # 施工区距离（米）
config.construction_length = 20.0       # 施工区长度（米）
config.traffic_density = 3.0            # 交通密度（辆/100m）
config.traffic_speed = 8.0              # 交通流速度（m/s）
config.construction_type = "construction1" # 施工类型
```

---

## 🚀 运行方法

### 1. 确保 CARLA 服务器运行
```bash
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen
```

### 2. 运行新的训练脚本
```bash
cd /home/ajifang/SAC_carla
python train_ppo_4scenarios.py
```

### 3. 查看训练日志
- **本地日志：** `training_log_4scenarios.json`
- **Wandb：** 自动上传到项目 `SAC-CARLA-PPO`
- **权重保存：** `./weights/ppo-carla-4scenarios/`

---

## 📈 Wandb 监控指标

### 场景相关指标
- `episode/done_reason`: 终止原因（可以看到不同场景的表现）
- `episode/success`: 成功率
- `episode/collision_count`: 碰撞次数

### 训练指标
- `training/policy_loss`: 策略损失
- `training/value_loss`: 价值损失
- `training/entropy`: 熵（探索程度）

### 早停指标
- `early/success_rate_W`: 滑动窗口成功率
- `early/collision_rate_W`: 滑动窗口碰撞率
- `early/plateau_count`: 平台期计数

---

## ⚠️ 注意事项

### 1. 场景难度差异
- **Cones**: ⭐⭐ 简单（横向避让）
- **Jaywalker**: ⭐⭐⭐⭐⭐ 非常困难（紧急制动）
- **Trimma**: ⭐⭐⭐⭐ 困难（变道超车）
- **Construction**: ⭐⭐⭐⭐ 困难（施工变道）

**建议：** 训练初期可能在 Jaywalker 场景失败率较高，这是正常的。

### 2. 训练时间
- 4场景训练比单场景慢约 **1.5-2倍**
- 每个 episode 约 **30-60秒**（取决于场景复杂度）
- 300 episodes 预计 **4-6小时**

### 3. 收敛标准
```python
TARGET_SUCCESS = 0.80       # 成功率 >= 80%
TARGET_RAN_FULL = 0.80      # 完整运行率 >= 80%
MAX_COLLISION = 0.10        # 碰撞率 <= 10%
```

由于场景难度不同，可能需要调整这些阈值。

### 4. 场景特殊处理
- **Jaywalker**: 需要 `check_and_trigger()` 和 `tick_update()`
- **Trimma**: 使用 Traffic Manager 控制周围车辆
- **Construction**: 使用 `ahead_obstacle_scenario` 生成施工区
- **Cones**: 无需特殊处理

---

## 🐛 故障排查

### 问题1：Jaywalker 场景行人不移动
**原因：** 缺少 `tick_update()` 调用

**解决：** 确保在 `carla_env.py` 的 `step()` 中添加了场景触发逻辑

### 问题2：Trimma 场景车辆不动
**原因：** Traffic Manager 未正确初始化

**检查：**
```python
# scenario_manager.py 第1307行
self.traffic_manager = client.get_trafficmanager(self.tm_port)
```

### 问题3：Construction 场景生成失败
**原因：** `tiny_scenarios_obstacle.py` 或 `carla_data_provider.py` 缺失

**检查：**
```bash
ls carla_base/tiny_scenarios_obstacle.py
ls carla_base/carla_data_provider.py
```

### 问题4：训练过程中 CARLA 崩溃
**原因：** 场景生成的 actors 过多

**解决：**
- 减少 `traffic_density`（Construction 场景）
- 减少 `cone_num`（Cones 场景）
- 增加 CARLA 服务器内存

---

## 📝 与原版本的对比

| 项目 | 原版本 (train_ppo_with_wandb.py) | 新版本 (train_ppo_4scenarios.py) |
|------|----------------------------------|----------------------------------|
| 场景数量 | 3个（parked_obstacles, cones, pedestrian_crossing） | 4个（cones, jaywalker, trimma, construction） |
| 场景触发 | 无 | ✅ 支持 JaywalkerScenario 触发 |
| 权重目录 | `./weights/ppo-carla-obs30` | `./weights/ppo-carla-4scenarios` |
| 日志文件 | `training_log.json` | `training_log_4scenarios.json` |
| Wandb标签 | `parked_obstacles` | `4scenarios` |

---

## 🎯 下一步

1. **运行训练：** `python train_ppo_4scenarios.py`
2. **监控 Wandb：** 查看 4 个场景的表现差异
3. **调整参数：** 根据训练结果调整场景难度
4. **评估模型：** 在每个场景上单独测试性能

---

## 📚 相关文件

- **训练脚本：** `train_ppo_4scenarios.py`
- **场景管理：** `carla_base/scenario_manager.py`
- **环境封装：** `carla_base/carla_env.py`
- **配置文件：** `config.py`
- **场景实现：**
  - ConesScenario: `scenario_manager.py:344-614`
  - JaywalkerScenario: `scenario_manager.py:621-1023`
  - TrimmaScenario: `scenario_manager.py:1029-1391`
  - ConstructionLaneChangeScenario: `scenario_manager.py:1396-1714`

---

**生成时间：** 2026-01-27
**版本：** v1.0
