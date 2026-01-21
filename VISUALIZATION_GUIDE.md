# 🎨 CARLA训练可视化指南

## 📋 概述

本指南说明如何使用增强的可视化功能来监控训练过程，确保场景配置正确。

---

## 🚀 快速开始

### 1. 启动CARLA服务器
```bash
cd /home/ajifang/carla
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

### 2. 运行训练（带可视化）
```bash
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

---

## 🎯 可视化功能说明

### 1️⃣ **Pygame窗口** (400x300)
显示自车第一人称相机视角，包含：
- 实时画面
- 控制信息面板（左上角）
  - Planner模式
  - Throttle/Steer/Brake数值
  - 速度（km/h）

### 2️⃣ **CARLA Spectator视角** (第三人称跟随)
在CARLA窗口中显示：
- **绿色箭头**: 自车前向方向（10米）
- **青色圆圈**: 障碍物检测范围（50米）
- **彩色边界框**: 障碍物
  - 🔴 红色: 距离 < 10米（危险）
  - 🟠 橙色: 距离 10-20米（警告）
  - 🟡 黄色: 距离 > 20米（安全）
- **白色虚线**: 车道中心线（前方40米）
- **距离标签**: 每个障碍物上方显示距离

### 3️⃣ **终端输出**
每个Episode开始时打印：
```
======================================================================
🎬 Episode 1 - 场景初始化
======================================================================
📍 地图: Town05
🎭 场景类型: parked_obstacles

🚗 自车信息:
   - 位置: (123.4, 56.7, 0.3)
   - 朝向: Yaw=90.0°
   - 车型: tesla.cybertruck

🚧 障碍物信息 (共4个):
   [1] vehicle.tesla.model3
       位置: (145.2, 58.3, 0.5)
       距离自车: 15.3m
   [2] vehicle.audi.a2
       位置: (153.1, 59.1, 0.5)
       距离自车: 23.8m
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

---

## ⚙️ 配置选项

### 在 `train_ppo_with_wandb.py` 中修改：

```python
config.render = True                    # 开启/关闭Pygame窗口
config.spectator_mode = "chase"         # "none" / "fixed" / "chase"
config.enable_debug_drawing = True      # 开启/关闭CARLA调试绘制
config.debug_draw_interval = 5          # 每N步绘制一次（降低性能影响）
```

### Spectator模式说明：
- **"none"**: 不移动观察者（默认俯视图）
- **"fixed"**: 固定角度跟随（后方28米，高7米）
- **"chase"**: 追逐视角（后方23米，高9米，俯角-12°）

---

## 🔍 如何验证场景正确性

### ✅ 检查清单

1. **自车生成位置**
   - 终端输出显示自车坐标
   - CARLA窗口中能看到自车（Cybertruck）
   - 绿色箭头指向前方

2. **障碍物生成**
   - 终端显示障碍物数量和位置
   - CARLA窗口中能看到彩色边界框
   - 距离标签显示正确（12-20米范围内）

3. **检测范围**
   - 青色圆圈覆盖50米范围
   - 障碍物在圆圈内时应该被检测到

4. **车道线**
   - 白色虚线沿着道路延伸
   - 自车应该在车道中心附近

### ❌ 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 看不到障碍物边界框 | `obstacle_actors`为空 | 检查`_place_parked_vehicles()`是否正确注册 |
| 距离标签显示错误 | 坐标计算问题 | 检查`_draw_debug_info()`中的距离计算 |
| Pygame窗口黑屏 | 相机未正确附加 | 检查`camera_display`是否成功spawn |
| Spectator不跟随 | 模式设置错误 | 确认`spectator_mode="chase"` |

---

## 🎮 实时监控指标

### 在训练过程中观察：

1. **Pygame面板**
   - Throttle应该在0-1之间波动
   - Steer应该根据道路弯曲调整
   - 速度应该逐渐增加（训练后期）

2. **CARLA窗口**
   - 自车应该沿着白色车道线行驶
   - 接近障碍物时应该减速/转向
   - 不应该频繁碰撞（红色边界框）

3. **终端输出**
   - 每个Episode的reward应该逐渐增加
   - `done_reason`应该从`collision`变为`time_limit`
   - 障碍物距离应该保持在安全范围

---

## 🛠️ 性能优化

如果可视化导致训练变慢：

```python
# 方案1: 降低绘制频率
config.debug_draw_interval = 10  # 从5改为10

# 方案2: 关闭部分可视化
config.enable_debug_drawing = False  # 只保留Pygame和Spectator

# 方案3: 完全关闭可视化（最快）
config.render = False
config.spectator_mode = "none"
config.enable_debug_drawing = False
```

---

## 📊 与Wandb配合使用

可视化 + Wandb监控 = 完整的训练观测：

- **本地可视化**: 实时查看场景和行为
- **Wandb面板**: 查看reward曲线、loss等指标
- **终端输出**: 快速诊断问题

建议工作流：
1. 前几个Episode开启全部可视化，确认场景正确
2. 确认无误后，关闭部分可视化加速训练
3. 定期开启可视化检查训练效果

---

## 🎯 修改场景后的验证流程

当你修改场景配置后（如增加障碍物、改变距离等）：

1. **开启全部可视化**
   ```python
   config.render = True
   config.spectator_mode = "chase"
   config.enable_debug_drawing = True
   ```

2. **运行1个Episode**
   ```python
   # 在train_ppo_with_wandb.py中临时修改
   episodes=1  # 只跑1个episode验证
   ```

3. **检查终端输出**
   - 障碍物数量是否正确
   - 距离是否在预期范围
   - 观测维度是否匹配

4. **检查CARLA窗口**
   - 障碍物位置是否合理
   - 自车能否正常行驶
   - 检测范围是否覆盖障碍物

5. **确认无误后恢复训练**
   ```python
   episodes=300  # 恢复正常训练
   ```

---

## 📝 调试技巧

### 1. 截图保存
在CARLA窗口按 `F12` 可以截图保存到：
```
/home/ajifang/carla/Saved/Screenshots/
```

### 2. 暂停训练
在终端按 `Ctrl+C` 可以优雅地停止训练（会保存模型）

### 3. 查看详细日志
```bash
# 实时查看训练日志
tail -f training_log.json

# 查看Wandb日志
wandb sync wandb/  # 同步离线日志
```

---

## 🎨 可视化效果预览

### 正常场景示例：
```
- 自车在车道中心
- 绿色箭头指向前方
- 障碍物显示橙色/黄色边界框（距离适中）
- 白色车道线清晰可见
- Pygame显示平滑的控制输入
```

### 异常场景示例：
```
❌ 自车偏离车道（车道线在侧方）
❌ 障碍物显示红色边界框（距离过近）
❌ 绿色箭头指向障碍物（即将碰撞）
❌ Pygame显示剧烈抖动的控制输入
```

---

## 📞 问题反馈

如果遇到可视化问题，请检查：
1. CARLA服务器是否正常运行
2. Pygame是否正确安装 (`pip install pygame`)
3. 显示器是否正确连接（如果是远程服务器，需要X11转发）

---

**祝训练顺利！🚀**
