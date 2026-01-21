# 🔧 CARLA训练故障排除指南

## ❌ 错误: `_queue.Empty` (同步模式超时)

### 问题描述
```
File "/home/ajifang/SAC_carla/carla_base/carla_sync_mode.py", line 109, in _retrieve_data
    item = sensor_queue.get(timeout=min(remaining, 0.5))
_queue.Empty
```

### 原因分析
1. **CARLA服务器启动参数错误** - 使用了 `-RenderOffScreen` 但代码要求渲染
2. **服务器未完全启动** - 训练脚本启动太快
3. **端口冲突** - 2000端口被占用
4. **同步模式配置问题** - 传感器数据未正确同步

---

## ✅ 解决方案

### 方案1: 使用正确的启动脚本（推荐）

#### 步骤1: 关闭所有CARLA进程
```bash
pkill -9 -f CarlaUE4
```

#### 步骤2: 使用新的启动脚本
```bash
cd /home/ajifang/SAC_carla
./start_carla_with_render.sh
```

这个脚本会：
- ✅ 自动关闭旧进程
- ✅ 使用正确的参数启动CARLA
- ✅ 等待服务器完全初始化
- ✅ 显示启动状态

#### 步骤3: 等待CARLA完全启动
看到以下信息后再继续：
```
✅ CARLA服务器运行正常
🎬 CARLA服务器已就绪！
```

#### 步骤4: 测试连接（可选）
```bash
python test_carla_connection.py
```

应该看到：
```
✅ 连接成功
✅ 当前地图: Town05
✅ 渲染模式正常
✅ 车辆生成成功
✅ 相机传感器生成成功
✅ 相机数据接收成功
✅ 诊断完成！CARLA服务器运行正常
```

#### 步骤5: 运行训练
```bash
python train_ppo_with_wandb.py
```

---

### 方案2: 手动启动CARLA

如果启动脚本不工作，手动启动：

```bash
# 1. 关闭旧进程
pkill -9 -f CarlaUE4

# 2. 切换到CARLA目录
cd /home/ajifang/carla

# 3. 启动CARLA（带渲染）
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600

# 4. 等待20-30秒，直到看到CARLA窗口完全加载

# 5. 在新终端运行训练
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

---

### 方案3: 无渲染模式（最快，但无可视化）

如果你只想快速训练，不需要可视化：

#### 修改 `train_ppo_with_wandb.py`:
```python
config.render = False                    # 关闭Pygame
config.spectator_mode = "none"           # 关闭Spectator
config.enable_debug_drawing = False      # 关闭调试绘制
```

#### 启动CARLA（无渲染）:
```bash
cd /home/ajifang/carla
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

---

## 🔍 常见问题诊断

### 问题1: CARLA窗口打开但黑屏

**原因**: 显卡驱动问题或资源不足

**解决**:
```bash
# 降低画质
./CarlaUE4.sh -quality-level=Low -windowed -ResX=640 -ResY=480
```

### 问题2: 端口被占用

**检查**:
```bash
lsof -i :2000
```

**解决**:
```bash
# 杀死占用进程
kill -9 <PID>

# 或者使用其他端口
./CarlaUE4.sh -carla-rpc-port=2002
```

然后修改 `config.py`:
```python
self.carla_port = 2002
```

### 问题3: Pygame窗口不显示

**原因**: 远程服务器没有X11转发

**解决**:
```bash
# SSH时启用X11转发
ssh -X user@server

# 或者使用VNC/远程桌面
```

### 问题4: 训练启动后立即崩溃

**检查**:
```bash
# 查看CARLA日志
tail -f /home/ajifang/carla/Saved/Logs/CarlaUE4.log
```

**常见原因**:
- 内存不足（需要至少8GB）
- GPU驱动问题
- CARLA版本不匹配

---

## 📊 性能优化

### 如果训练很慢：

#### 1. 降低可视化频率
```python
# 在 train_ppo_with_wandb.py
config.debug_draw_interval = 20  # 从5改为20
```

#### 2. 降低CARLA画质
```bash
./CarlaUE4.sh -quality-level=Low
```

#### 3. 减小Pygame窗口
修改 `carla_env.py`:
```python
self.screen = pygame.display.set_mode((320, 240), ...)  # 从400x300改为320x240
```

#### 4. 关闭部分可视化
```python
config.enable_debug_drawing = False  # 关闭CARLA调试绘制
config.spectator_mode = "none"       # 关闭Spectator跟随
```

---

## 🧪 测试清单

运行训练前，确保以下都正常：

- [ ] CARLA服务器已启动（看到CARLA窗口）
- [ ] `test_carla_connection.py` 全部通过
- [ ] 没有端口冲突（`lsof -i :2000` 只显示CARLA）
- [ ] 内存充足（`free -h` 显示至少4GB可用）
- [ ] GPU正常（`nvidia-smi` 显示GPU状态）

---

## 📝 调试技巧

### 1. 查看详细错误
```bash
python train_ppo_with_wandb.py 2>&1 | tee training_debug.log
```

### 2. 单步调试
在 `carla_env.py` 的 `reset()` 中添加：
```python
print(f"[DEBUG] sync_mode: {self.sync_mode}")
print(f"[DEBUG] camera_display: {self.camera_display}")
print(f"[DEBUG] ego: {self.ego}")
```

### 3. 测试最小配置
临时修改 `train_ppo_with_wandb.py`:
```python
episodes=1  # 只跑1个episode
timesteps=10  # 只跑10步
```

---

## 🆘 仍然无法解决？

### 收集以下信息：

1. **CARLA版本**:
```bash
cat /home/ajifang/carla/VERSION
```

2. **Python版本**:
```bash
python --version
```

3. **GPU信息**:
```bash
nvidia-smi
```

4. **完整错误日志**:
```bash
python train_ppo_with_wandb.py 2>&1 | tee error.log
```

5. **CARLA日志**:
```bash
tail -100 /home/ajifang/carla/Saved/Logs/CarlaUE4.log
```

---

## ✅ 成功标志

训练正常启动后，你应该看到：

### 终端输出:
```
======================================================================
🎬 Episode 1 - 场景初始化
======================================================================
📍 地图: Town05
🎭 场景类型: parked_obstacles
...
```

### CARLA窗口:
- 自车在道路上
- 障碍物有彩色边界框
- 青色检测范围圆圈
- 白色车道线

### Pygame窗口:
- 显示相机视角
- 左上角有控制面板
- 画面流畅更新

---

**祝调试顺利！🚀**
