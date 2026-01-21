# 快速训练模式指南

## ✅ 已配置完成

### 当前配置
```python
# train_ppo_with_wandb.py:684
config.render = False  # 禁用渲染，快速训练
```

**效果**：
- ⚡ **训练速度提升 50-100倍**
- ❌ 无pygame窗口（看不到画面）
- ✅ 只有collision传感器（< 0.1秒/帧）
- 📊 可通过Wandb监控训练进度

### 性能对比

| 模式 | 每帧耗时 | Episode耗时 | 训练1000 episodes |
|------|---------|-------------|-------------------|
| render=True (800x600) | 5-6秒 | ~2500秒 | 694小时 |
| render=True (400x300) | 2-3秒 | ~1000秒 | 278小时 |
| **render=False** | **< 0.1秒** | **~25秒** | **7小时** |

## 🚀 启动训练

### 方法1: 使用快速启动脚本（推荐）

```bash
./start_fast_training.sh
```

### 方法2: 手动启动

```bash
# 1. 确保CARLA在运行
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound

# 2. 另一个终端运行训练
cd /home/ajifang/SAC_carla
./start_ppo_training_wandb.sh
```

## 📊 监控训练

### Wandb实时监控

训练开始后，访问：
```
https://wandb.ai/andrea23/SAC-CARLA-PPO
```

**关键指标**：
- `episode_reward`: 每个episode的总奖励
- `episode_steps`: 每个episode的步数
- `fps`: 实际训练速度（应该 > 10 FPS）
- `collision_rate`: 碰撞率（应该逐渐降低）
- `speed`: 车辆平均速度

### 终端输出

```bash
# 快速训练模式下的输出（无调试信息）
Episode 1: reward=123.45, steps=234, time=25s
Episode 2: reward=145.67, steps=456, time=23s
...
```

### 日志文件

```bash
# 查看实时日志
tail -f logs/training_*.log

# 查看训练数据
cat training_log.json | jq '.'
```

## 🎯 训练目标

### 完整训练计划

根据 `config.py` 配置：
- `train_total_steps = 5e10` (500亿步)
- `max_episode_steps = 500` (每个episode最多500步)

**实际训练量**（建议）：
- **初步训练**: 1,000 episodes (~7小时)
- **中等训练**: 10,000 episodes (~70小时)
- **完整训练**: 100,000 episodes (~700小时)

### 保存间隔

- `save_interval_steps = 4e4` (每40,000步保存一次)
- `test_interval_steps = 2e3` (每2,000步测试一次)

每个episode约200-500步，所以：
- 每 **80-200 episodes** 保存一次模型
- 每 **4-10 episodes** 测试一次

## 🛑 停止训练

### 安全停止

```bash
# 方法1: Ctrl+C（推荐）
# 训练脚本会捕获信号并保存模型

# 方法2: 查找并终止进程
ps aux | grep train_ppo
kill <PID>
```

### 恢复训练

训练会自动从最新的checkpoint恢复：
```bash
./start_fast_training.sh
# 输出会显示: "✓ 从epoch XXX恢复训练"
```

## 🎨 训练完成后测试

训练完成后，如果想**看到训练效果**：

### 1. 切换到渲染模式

```python
# train_ppo_with_wandb.py:684
config.render = True  # 改回True
```

### 2. 创建测试脚本

```bash
# test_trained_model.py
from config import Config
from carla_base.carla_env import CarlaEnv

config = Config()
config.render = True  # 启用渲染看效果
config.max_episode_steps = 1000  # 更长的测试时间

# 加载训练好的模型
model = load_ppo_model("weights/ppo-carla-standalone/")

# 测试多个episodes
for i in range(10):
    test_episode(model, config)
```

## 📈 训练技巧

### 监控过拟合

如果发现：
- `episode_reward` 不再上升
- `collision_rate` 不再下降
- 测试性能变差

→ 可能过拟合，停止训练

### 调整学习率

如果训练不稳定：
```python
# config.py
self.actor_lr = 3e-4  # 降低到 1e-4
self.critic_lr = 3e-4  # 降低到 1e-4
```

### 调整场景难度

训练顺序（由易到难）：
1. `scenario = "plain"` - 空场景
2. `scenario = "cones"` - 锥桶障碍
3. `scenario = "parked_obstacles"` - 停车障碍（当前）

## 🐛 常见问题

### Q: 训练卡住不动

```bash
# 检查CARLA是否响应
ps aux | grep CarlaUE4

# 重启CARLA
./clean_memory.sh
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen ...
```

### Q: 训练速度仍然慢

检查 `config.render` 是否真的是 `False`：
```python
python -c "from config import Config; c = Config(); print(f'render={c.render}')"
```

### Q: 想要边训练边看画面

**不推荐**，因为会慢100倍。如果坚持：
```python
config.render = True
# 并降低分辨率（已修改为400x300）
```

## 📁 重要文件

- `weights/ppo-carla-standalone/` - 模型权重
- `logs/training_*.log` - 训练日志
- `training_log.json` - 结构化训练数据
- `result/reward_logs/` - 奖励监控数据

## 🎉 训练完成后

```bash
# 查看最终模型
ls -lh weights/ppo-carla-standalone/

# 查看训练统计
cat training_log.json | jq '.episodes | length'

# 备份模型（可选）
cp -r weights/ppo-carla-standalone weights/ppo-carla-backup-$(date +%Y%m%d)
```

---

## 🚀 现在开始训练！

```bash
./start_fast_training.sh
```

预期输出：
```
✅ PPO Agent创建成功
[4] 开始训练...
Wandb监控已启用
   实时查看: https://wandb.ai/andrea23/SAC-CARLA-PPO/runs/...

Episode 1/∞, Steps: 234, Reward: 123.45, Time: 25s
Episode 2/∞, Steps: 456, Reward: 145.67, Time: 23s
...
```

**祝训练顺利！** 🎯

---

修改时间: 2025-12-25 01:10
