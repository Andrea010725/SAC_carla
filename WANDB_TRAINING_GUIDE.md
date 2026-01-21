# PPO训练 - Wandb实时监控使用指南

## 🎯 功能特性

新的训练脚本 `train_ppo_with_wandb.py` 提供了完整的Wandb实时监控，包括：

### 📊 监控指标

#### 1. Episode级别指标
- `episode/total_reward`: 总reward
- `episode/steps`: episode步数
- `episode/duration`: episode持续时间（秒）
- `episode/avg_speed`: 平均速度（m/s）
- `episode/collision_count`: 碰撞次数
- `episode/success`: 是否成功（无碰撞且步数>100）

#### 2. Reward分解（每个episode的平均值）
- `reward/base_reward`: RewardMonitor基础reward
- `reward/r_wp`: Waypoint进度奖励
- `reward/r_speed`: 速度维持奖励
- `reward/r_lane`: 车道保持惩罚
- `reward/r_collision`: 碰撞惩罚
- `reward/r_idle`: 站桩惩罚

#### 3. Training指标
- `training/policy_loss`: Policy网络loss
- `training/value_loss`: Value网络loss
- `training/entropy`: 策略熵（探索程度）
- `training/update_count`: 网络更新次数
- `training/action_bias`: 当前action bias值

#### 4. Step级别指标（每10步记录一次）
- `step/reward`: 当前step reward
- `step/speed`: 当前速度
- `step/action_throttle`: Throttle action
- `step/action_steer`: Steer action
- `step/applied_bias`: 应用的bias值
- `step/forced_throttle`: 是否强制最小throttle

---

## 🚀 快速开始

### 步骤1: 安装Wandb

```bash
# 安装wandb
pip install wandb

# 登录wandb（首次使用）
wandb login
```

如果没有wandb账号，去 https://wandb.ai 注册（免费）。

### 步骤2: 启动CARLA服务器

```bash
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low
```

### 步骤3: 开始训练

```bash
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

### 步骤4: 实时监控

训练开始后，终端会输出：

```
✅ Wandb已初始化
   项目: SAC-CARLA-PPO
   Run名称: ppo-standalone-20231223-143025
   Run URL: https://wandb.ai/your-username/SAC-CARLA-PPO/runs/xxxxx
```

点击URL即可实时查看训练进度。

---

## 📈 Wandb Dashboard使用

### 查看训练曲线

在Wandb界面中，你可以看到：

1. **Overview标签页**
   - 显示所有关键指标的实时曲线
   - 自动对比不同run

2. **Charts标签页**
   - 自定义图表
   - 推荐的图表配置：
     - **Reward分解图**: 把所有 `reward/*` 指标放在一个图中
     - **Training Loss**: `training/policy_loss` 和 `training/value_loss`
     - **Success Rate**: `episode/success` (rolling average)
     - **Speed vs Collision**: scatter plot

3. **System标签页**
   - GPU/CPU使用率
   - 内存使用情况

### 推荐的自定义图表

#### 1. Reward分解堆叠图
```
Y轴: reward/base_reward, reward/r_wp, reward/r_speed, reward/r_lane
类型: Line chart (stacked)
```

#### 2. 训练健康度监控
```
Y轴: training/entropy, training/policy_loss
类型: Line chart (dual axis)
```

#### 3. Episode性能面板
```
指标: episode/total_reward, episode/avg_speed, episode/collision_count
类型: Parallel coordinates
```

---

## 🔍 如何诊断训练问题

### 1. 车辆不动
**症状**: `episode/avg_speed` 始终<1 m/s

**检查**:
- `training/action_bias` 是否>0.5 (前500 episodes)
- `step/forced_throttle` 是否经常为1

**解决**: 增加action bias强度或延长衰减周期

---

### 2. 碰撞率过高
**症状**: `episode/collision_count` >3, `episode/success` <0.1

**检查**:
- `reward/r_collision` 是否足够负 (<-5)
- `training/entropy` 是否过高 (>1.0，表示随机探索太多)

**解决**:
- 降低熵正则化系数
- 增加碰撞惩罚权重

---

### 3. Reward不收敛
**症状**: `episode/total_reward` 在100+ episodes后仍剧烈波动

**检查**:
- `training/policy_loss` 是否持续>0.1
- `training/value_loss` 是否持续>0.5
- `reward/r_wp` 和 `reward/r_speed` 是否都接近0

**解决**:
- 降低学习率
- 检查reward权重配置
- 简化场景（减少停车数量）

---

### 4. NaN/Inf问题
**症状**: 训练突然中断，日志显示 "检测到NaN/Inf"

**检查**:
- `training/policy_loss` 是否在NaN前突然激增
- `training/value_loss` 是否异常大 (>10)

**解决**:
- 降低学习率（policy_lr: 5e-5 -> 3e-5）
- 增加梯度裁剪强度

---

## 📊 Wandb高级功能

### 1. 对比不同run

在Wandb界面：
1. 选择多个run
2. 点击 "Compare"
3. 查看指标差异

**用例**: 对比不同学习率、不同reward配置的效果

---

### 2. 保存checkpoint到Wandb

脚本已自动配置，每100 episodes会上传checkpoint到Wandb：

```python
wandb_run.save(os.path.join(agent.base_path, 'policy_net*'))
wandb_run.save(os.path.join(agent.base_path, 'value_net*'))
```

恢复checkpoint:
```bash
wandb restore policy_net.index --run your-run-id
```

---

### 3. 添加自定义指标

在 `train_ppo_with_wandb.py` 中，你可以轻松添加新的监控指标：

```python
# 在 wandb_metrics 字典中添加
wandb_metrics['custom/my_metric'] = my_value

# 例如：监控waypoint距离
wandb_metrics['debug/wp_distance'] = dist_to_wp
```

---

### 4. Wandb Alerts

设置自动告警（在Wandb界面）：

1. 进入Run详情页
2. 点击 "Alerts"
3. 添加条件，例如：
   - `episode/collision_count > 5` -> 发送邮件
   - `training/policy_loss > 1.0` -> 发送Slack消息

---

## 🎨 推荐的Wandb Report模板

创建一个Report来总结训练：

### Report结构

**标题**: PPO Training - Parked Obstacles Scenario

**Section 1: Training Overview**
- Run名称、开始时间、总episodes
- 超参数表格（从config中自动提取）

**Section 2: Performance Metrics**
- `episode/total_reward` 曲线
- `episode/success` 曲线（rolling average 50 episodes）
- `episode/avg_speed` vs `episode/collision_count` scatter

**Section 3: Reward Analysis**
- Reward分解堆叠图
- 表格显示各reward组件的均值和标准差

**Section 4: Training Dynamics**
- `training/policy_loss` 和 `training/value_loss`
- `training/entropy` 曲线
- `training/action_bias` 衰减曲线

**Section 5: Conclusions**
- 最佳episode的视频（如果录制）
- 下一步改进方向

---

## ⚙️ 配置选项

### 修改Wandb项目名称

在 `train_ppo_with_wandb.py` 中：

```python
wandb_run = wandb.init(
    project="YOUR-PROJECT-NAME",  # 修改这里
    name=f"ppo-standalone-{time.strftime('%Y%m%d-%H%M%S')}",
    ...
)
```

### 禁用Wandb（临时）

如果想临时禁用Wandb但不修改代码：

```bash
WANDB_MODE=disabled python train_ppo_with_wandb.py
```

### 离线模式

如果网络不稳定：

```bash
WANDB_MODE=offline python train_ppo_with_wandb.py

# 训练完成后同步
wandb sync wandb/run-xxxxx
```

---

## 📝 最佳实践

### 1. Run命名规范

使用有意义的run名称：

```python
name=f"ppo-lr{policy_lr}-gamma{gamma}-{scenario}-{timestamp}"
```

例如: `ppo-lr5e-5-gamma0.99-parked4-20231223-143025`

### 2. 使用Tags

给run添加标签便于筛选：

```python
tags=["ppo", "carla", "parked_obstacles", "aggressive_reward", "experiment_v2"]
```

### 3. 记录代码版本

Wandb会自动记录git commit，确保代码已提交：

```bash
git add train_ppo_with_wandb.py
git commit -m "Add wandb monitoring"
```

### 4. 定期Review

建议每50-100 episodes review一次Wandb dashboard，及时发现问题。

---

## 🐛 故障排除

### Wandb无法连接

```bash
# 检查网络
ping wandb.ai

# 重新登录
wandb login --relogin
```

### Logging过慢

如果logging影响训练速度，减少记录频率：

```python
# Step级别从每10步改为每50步
if self._step_count % 50 == 0:
    self.wandb_run.log(...)
```

### 内存占用过高

Wandb会缓存数据，如果内存不足：

```bash
# 减少缓存
export WANDB_CACHE_DIR=/tmp/wandb_cache
```

---

## 📚 学习资源

- [Wandb文档](https://docs.wandb.ai/)
- [Wandb RL教程](https://wandb.ai/site/articles/rl-tutorial)
- [Wandb最佳实践](https://docs.wandb.ai/guides/track/best-practices)

---

## 🎯 总结

使用 `train_ppo_with_wandb.py` 你可以：

✅ 实时监控所有训练指标
✅ 详细的reward分解分析
✅ 自动保存checkpoint到云端
✅ 对比不同实验效果
✅ 诊断训练问题
✅ 生成专业的训练报告

**现在就开始训练吧！祝训练顺利！🚀**
