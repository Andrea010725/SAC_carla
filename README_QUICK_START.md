# 快速开始指南

## 问题已修复 ✅

**Pygame 显示问题已解决！** 详细信息请查看 `PYGAME_DISPLAY_FIX.md`

## 快速验证

### 1. 测试环境是否正常

```bash
python test_pygame_display.py
```

**预期输出**：
```
[CarlaEnv] ✅ Pygame显示已启用（真实窗口模式）
✅ Pygame显示测试通过！
```

### 2. 启动训练

**方法1：使用启动脚本（推荐）**
```bash
./start_training.sh
```

**方法2：直接运行**
```bash
python train_ppo_with_wandb.py
```

**方法3：后台运行**
```bash
nohup python train_ppo_with_wandb.py > training.log 2>&1 &

# 查看进度
tail -f training.log
```

## 如何查看 Pygame 窗口

### 本地机器
- 窗口会自动显示（800x600）

### 远程服务器
1. **使用 VNC/远程桌面**（推荐）
   - 连接到服务器的远程桌面
   - 运行训练脚本
   - 在远程桌面中查看窗口

2. **使用 X11 转发**
   ```bash
   ssh -X user@server
   python train_ppo_with_wandb.py
   ```

3. **无头模式**
   - 如果不需要看窗口，代码会自动使用虚拟模式
   - 功能完全正常

## 训练状态检查

### 正常的训练日志示例

```
Episode 1 terminated after 108 timesteps in 10.514s with reward -314.065.
Episode 2 terminated after 95 timesteps in 9.234s with reward -287.123.
Episode 3 terminated after 112 timesteps in 11.023s with reward -298.456.
...
```

### 关键指标

✅ **正常**：
- Episode 完成 50+ 步
- Reward 有变化
- 每个 episode 10-30 秒
- 有障碍物检测日志

❌ **异常**：
- Episode 立即结束（0-1步）
- Reward 一直为 0 或 NaN
- 大量错误信息

## 训练速度说明

**训练很快是正常的！**

- 每个 episode: 10-30 秒
- 每步: 0.1-0.2 秒
- 100 episodes: 20-50 分钟

这是因为：
1. RTX 4090 GPU 性能强大
2. CARLA 同步模式不受实时限制
3. 代码经过优化

## 监控训练进度

### 方法1：查看日志文件

```bash
# 实时查看
tail -f training.log

# 或查看最近的日志
tail -f logs/training_*.log
```

### 方法2：查看 Reward 图表

```bash
# 查看生成的图表
ls -lh reward_plots/

# 使用图片查看器打开
eog reward_plots/reward_components_latest.png
```

### 方法3：WandB（如果配置了）

访问 WandB 网页查看实时图表

## 常见问题

### Q: 看不到 Pygame 窗口？

**A**: 检查日志中的消息：
- "真实窗口模式" → 使用远程桌面查看
- "虚拟模式" → 正常，功能不受影响

### Q: 训练是否太快？

**A**: 不是！这是正常速度，是优点不是问题。

### Q: 如何停止训练？

**A**: 
- 前台运行：按 `Ctrl+C`
- 后台运行：`pkill -f train_ppo_with_wandb.py`

### Q: 如何恢复训练？

**A**: 
- 检查 `weights/ppo-carla-obs30/` 目录
- 如果有权重文件，训练会自动加载
- 或修改 `train_ppo_with_wandb.py` 中的 `load_existing` 参数

## 文件说明

- `train_ppo_with_wandb.py` - 主训练脚本
- `test_pygame_display.py` - 测试脚本
- `start_training.sh` - 启动脚本
- `PYGAME_DISPLAY_FIX.md` - 详细修复说明
- `carla_base/carla_env.py` - 环境代码（已修复）

## 下一步

1. ✅ 运行测试脚本验证环境
2. ✅ 启动训练
3. ✅ 监控训练进度
4. ✅ 等待训练完成（300 episodes 约 2-4 小时）

## 需要帮助？

- 查看 `PYGAME_DISPLAY_FIX.md` 获取详细信息
- 检查训练日志中的错误信息
- 运行测试脚本诊断问题

---

**最后更新**: 2026-02-03
**状态**: ✅ Pygame 显示问题已修复，训练功能正常
