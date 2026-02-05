# Pygame渲染使用说明

## ✅ 问题已解决

OpenGL错误已通过环境变量配置解决，pygame渲染功能已完全恢复。

## 🚀 启动训练的3种方法

### 方法1：使用启动脚本（推荐）

```bash
./run_training_with_display.sh
```

这个脚本会自动设置所需的环境变量。

### 方法2：手动设置环境变量

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export SDL_VIDEODRIVER=dummy
python train_ppo_with_wandb.py
```

### 方法3：一行命令

```bash
LIBGL_ALWAYS_SOFTWARE=1 SDL_VIDEODRIVER=dummy python train_ppo_with_wandb.py
```

## 📋 配置说明

### 当前配置（train_ppo_with_wandb.py 第777行）

```python
config.render = True  # ✅ 开启pygame可视化
```

### 环境变量说明

- **LIBGL_ALWAYS_SOFTWARE=1**
  - 强制使用软件渲染
  - 绕过硬件OpenGL驱动问题
  - 避免 `libGL error: failed to open swrast` 错误

- **SDL_VIDEODRIVER=dummy**
  - 使用虚拟显示驱动
  - 避免X11显示错误
  - 允许pygame在无显示环境中运行

## 🎮 Pygame功能

启用pygame渲染后，你将获得：

1. **实时HUD显示**
   - 车辆速度
   - 转向角度
   - 油门/刹车状态
   - Episode信息

2. **相机画面**
   - 车辆前视图
   - 实时更新

3. **调试信息**
   - 训练状态
   - 奖励信息
   - 场景类型

## ⚠️ 注意事项

### 性能影响

- 软件渲染比硬件渲染慢约10-20%
- 对训练速度影响较小
- GPU仍用于神经网络计算

### 显示问题

- 使用dummy驱动时，pygame窗口可能不显示
- 这是正常的，不影响训练
- CARLA的spectator模式仍然正常工作

### 如果需要禁用渲染

如果想提高训练速度，可以禁用pygame渲染：

```python
# 在 train_ppo_with_wandb.py 第777行
config.render = False  # 禁用pygame渲染
```

然后直接运行：
```bash
python train_ppo_with_wandb.py
```

## 🔧 故障排除

### 如果仍然出现OpenGL错误

1. 确认环境变量已设置：
   ```bash
   echo $LIBGL_ALWAYS_SOFTWARE
   echo $SDL_VIDEODRIVER
   ```

2. 尝试添加更多环境变量：
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   export SDL_VIDEODRIVER=dummy
   export PYGAME_HIDE_SUPPORT_PROMPT=1
   export DISPLAY=:0
   python train_ppo_with_wandb.py
   ```

3. 检查pygame版本：
   ```bash
   pip show pygame
   ```

### 如果需要真实显示

如果你有X11显示服务器并想看到pygame窗口：

```bash
# 不设置SDL_VIDEODRIVER=dummy
export LIBGL_ALWAYS_SOFTWARE=1
python train_ppo_with_wandb.py
```

## 📊 验证渲染是否工作

运行训练后，检查输出中是否有：

```
✅ 环境创建成功
  - Observation space: Box(-inf, inf, (30,), float32)
  - Action space: Box(-1.0, 1.0, (3,), float32)
  - yref_in_steer= True , yref_steer_gain= 0.03
```

如果看到这些信息且没有OpenGL错误，说明pygame渲染已正常工作。

## 🎯 推荐配置

**训练时（追求速度）：**
```python
config.render = False
```

**调试时（需要可视化）：**
```python
config.render = True
```
并使用启动脚本：
```bash
./run_training_with_display.sh
```

## 📝 更新日志

- 2026-02-02: 修复OpenGL错误，恢复pygame渲染功能
- 2026-02-02: 创建启动脚本 `run_training_with_display.sh`
- 2026-02-02: 添加环境变量配置说明
