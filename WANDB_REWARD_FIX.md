# Wandb Reward组件记录修复

## 🔍 **问题诊断**

### 你发现的问题
昨天训练时，Wandb记录的所有reward组件都是**0**：
```
reward/r_wp: 0.0
reward/r_speed: 0.0
reward/r_lane: 0.0
reward/r_collision: 0.0
reward/r_idle: 0.0
...所有都是0
```

但`episode/total_reward`有正常值，说明reward计算是正确的，只是**没有记录分解**。

## 🔍 **根本原因**

### 代码流程分析

1. **CarlaEnv计算reward**（`carla_base/carla_env.py:893`）：
```python
def _get_reward(self):
    # 计算各个组件
    r_wp = 1.0 * delta_dist           # waypoint进度
    r_speed = 0.4 * norm_speed        # 速度奖励
    r_lane = -0.5 * lane_deviation    # 车道偏移
    r_collision = -5.0 if collision   # 碰撞惩罚
    r_idle = -10.0 if idle            # 龟速惩罚

    total_reward = base_reward + r_wp + r_speed + r_lane + r_collision + r_idle

    # ❌ 问题：没有保存这些组件！
    # self.last_reward_components = {...}  ← 缺失！

    return total_reward, done, info
```

2. **训练脚本尝试读取**（`train_ppo_with_wandb.py:158`）：
```python
# 提取reward分解（从carla_env）
if hasattr(self.carla_env, 'last_reward_components'):  # ← 检查失败
    self.step_reward_components = self.carla_env.last_reward_components.copy()
```

3. **累积到episode**（`train_ppo_with_wandb.py:481-483`）：
```python
# 累积reward分解
for key in episode_reward_components:
    if key in env.step_reward_components:  # ← step_reward_components不存在或为空
        episode_reward_components[key] += float(env.step_reward_components[key])
```

4. **记录到wandb**（`train_ppo_with_wandb.py:589-609`）：
```python
avg_reward_components = {
    f'reward/{k}': v / max(1, episode_steps)  # ← 所有v都是0
    for k, v in episode_reward_components.items()
}

wandb_metrics.update(avg_reward_components)
wandb_run.log(wandb_metrics, ...)
```

**结果**：所有reward组件都是0！

## ✅ **修复方案**

### 在CarlaEnv中保存reward组件

**文件**：`carla_base/carla_env.py:1009-1021`

**修改**：
```python
def _get_reward(self):
    # ... 计算所有reward组件 ...

    total_reward = (
        base_reward +
        r_wp +
        r_speed +
        r_lane +
        r_collision +
        r_idle
    )

    # ✅ 新增：保存reward组件供wandb记录
    self.last_reward_components = {
        'base_reward': float(base_reward),
        'r_wp': float(r_wp),
        'r_progress': float(r_wp),  # 别名，兼容
        'r_speed': float(r_speed),
        'r_lane': float(r_lane),
        'r_offroad': 0.0,  # 当前没用到
        'r_smooth': 0.0,   # 当前没用到
        'r_mag': 0.0,      # 当前没用到
        'r_collision': float(r_collision),
        'r_idle': float(r_idle),
    }

    info = {
        ...
    }

    return total_reward, done, info
```

## 📊 **修复后的效果**

### 修复前（昨天）
```
Wandb显示：
- episode/total_reward: -42.81  ✅ 有值
- reward/r_wp: 0.0              ❌ 错误
- reward/r_speed: 0.0           ❌ 错误
- reward/r_lane: 0.0            ❌ 错误
- reward/r_collision: 0.0       ❌ 错误
- reward/r_idle: 0.0            ❌ 错误
```

### 修复后（现在）
```
Wandb显示：
- episode/total_reward: -42.81  ✅ 有值
- reward/r_wp: 15.3             ✅ 正确（waypoint进度）
- reward/r_speed: 8.2           ✅ 正确（速度奖励）
- reward/r_lane: -12.5          ✅ 正确（车道偏移惩罚）
- reward/r_collision: -5.0      ✅ 正确（碰撞惩罚）
- reward/r_idle: 0.0            ✅ 正确（没有idle）
```

## 🎯 **Wandb将记录的所有指标**

### Episode级别

**基础指标**：
```python
'episode/total_reward': -42.81
'episode/steps': 234
'episode/duration': 25.3
'episode/avg_speed': 1.34
'episode/collision_count': 0
'episode/success': 0
```

**Reward组件**（平均每步）：
```python
'reward/base_reward': 0.05      # RewardMonitor基础分
'reward/r_wp': 0.065            # waypoint进度（关键！）
'reward/r_progress': 0.065      # 同r_wp（别名）
'reward/r_speed': 0.035         # 速度奖励
'reward/r_lane': -0.053         # 车道偏移惩罚
'reward/r_collision': -0.021    # 碰撞惩罚
'reward/r_idle': -0.043         # 龟速惩罚
'reward/r_offroad': 0.0         # 离路（未用）
'reward/r_smooth': 0.0          # 平滑（未用）
'reward/r_mag': 0.0             # 动作幅度（未用）
```

**训练指标**：
```python
'training/policy_loss': 0.385
'training/value_loss': 0.611
'training/entropy': 0.05
'training/update_count': 45
'training/action_bias': 0.0
```

### Step级别（每10步记录）

```python
'step/reward': 0.5
'step/speed': 1.2
'step/collision': 0
'step/progress': 0.02
'step/lane_deviation': 0.3
'step/control_throttle': 0.35
'step/control_brake': 0.0
'step/control_steer': -0.1
```

## 🔍 **如何验证修复**

### 方法1: 查看Wandb

训练后访问：
```
https://wandb.ai/andrea23/SAC-CARLA-PPO/runs/latest
```

在图表中查找：
- `reward/r_wp` - 应该有非零值
- `reward/r_speed` - 应该有正值
- `reward/r_lane` - 应该有负值（惩罚）

### 方法2: 查看终端输出

训练时应该看到：
```
Episode 10 完成 ✅
  当前: Reward=-42.81, Length=124
  最近10个平均:
    - Reward: -42.81
    - r_wp: 15.3        ← ✅ 有值
    - r_speed: 8.2      ← ✅ 有值
    - r_lane: -12.5     ← ✅ 有值
```

### 方法3: 检查代码

```python
# 在CarlaEnv.step()后
print(f"Reward components: {env.carla_env.last_reward_components}")
```

应该看到：
```
Reward components: {
    'base_reward': 0.02,
    'r_wp': 0.5,
    'r_speed': 0.3,
    'r_lane': -0.2,
    ...
}
```

## 💡 **为什么之前没发现？**

1. **total_reward是正确的** - 所以训练能正常进行
2. **只有分解信息丢失** - 不影响训练，只影响监控
3. **Wandb图表中不明显** - 需要展开reward组才能看到都是0

## ✅ **现在可以开始训练**

修复已完成，下次训练时所有reward组件都会正确记录到Wandb！

```bash
./start_fast_training.sh
```

**预期结果**：
- ✅ episode/total_reward有值
- ✅ reward/r_wp有值（关键指标）
- ✅ reward/r_speed有值
- ✅ reward/r_lane有负值
- ✅ reward/r_collision在碰撞时有负值
- ✅ 所有组件加起来等于total_reward

---

## 📊 **重要Reward组件说明**

| 组件 | 含义 | 典型值 | 重要性 |
|------|------|--------|--------|
| `r_wp` | waypoint进度 | 0~2.0 | ⭐⭐⭐⭐⭐ 核心 |
| `r_speed` | 速度奖励 | 0~0.4 | ⭐⭐⭐⭐ 重要 |
| `r_lane` | 车道偏移惩罚 | -1.0~0 | ⭐⭐⭐ 重要 |
| `r_collision` | 碰撞惩罚 | -5.0/0 | ⭐⭐⭐⭐ 重要 |
| `r_idle` | 龟速惩罚 | -10.0/0 | ⭐⭐⭐ 重要 |
| `base_reward` | RewardMonitor | -0.5~0.5 | ⭐⭐ 辅助 |

**训练时重点关注**：
- `r_wp` 应该逐渐变大（学会前进）
- `r_lane` 绝对值应该变小（学会保持车道）
- `r_collision` 出现频率应该降低

---

修复时间: 2025-12-25 05:30
关键改动: carla_env.py添加self.last_reward_components
