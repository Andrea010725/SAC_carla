# 🔍 Wandb Reward数据流分析

## ❓ 你的问题

**Wandb输出的reward是：**
1. **训练reward**（用于梯度更新的）？
2. **评测reward**（不放入梯度的）？

---

## ✅ 答案：是训练Reward（用于梯度更新的）

Wandb记录的是 **训练过程中实际使用的reward**，这个reward会被：
- ✅ 累积到 `episode_reward`
- ✅ 存入 `agent.memory`
- ✅ 用于计算梯度和更新网络

**不是评测reward！当前训练脚本没有分离训练/评测。**

---

## 📊 完整数据流追踪

### 1️⃣ Reward生成（carla_env.py）

```python
# carla_env.py line 897-1037
def _get_reward(self):
    # ... 计算各种reward组件 ...

    # 汇总reward
    total_reward = (
        base_reward +
        r_wp +
        r_speed +
        r_lane +
        r_collision +
        r_idle
    )

    # 📊 保存分解（供wandb读取）
    self.last_reward_components = {
        'base_reward': base_reward,
        'r_wp': r_wp,
        'r_speed': r_speed,
        'r_lane': r_lane,
        'r_collision': r_collision,
        'r_idle': r_idle,
    }

    return total_reward, done, info  # ✅ 返回total_reward
```

**这个 `total_reward` 就是训练使用的reward！**

---

### 2️⃣ Reward传递（CarlaGymEnv包装器）

```python
# train_ppo_with_wandb.py line 90-122
class CarlaGymEnv:
    def step(self, action):
        # ... action bias处理 ...

        # ✅ 调用CarlaEnv.step() 获取原始reward
        obs, reward, done, info = self.carla_env.step(action)

        # 📊 提取reward分解（从carla_env）
        if hasattr(self.carla_env, 'last_reward_components'):
            self.step_reward_components = self.carla_env.last_reward_components.copy()

        # ✅ 返回相同的reward（没有修改）
        return obs, reward, done, info
```

**注意**：
- ✅ reward没有被修改
- ✅ 直接传递给训练循环
- 📊 分解只是拷贝，不影响训练

---

### 3️⃣ Reward累积（训练循环）

```python
# train_ppo_with_wandb.py line 218-250

for t in range(1, timesteps + 1):
    # Agent预测action
    action, mean, std, log_prob, value = agent.predict(state)
    action_env = agent.convert_action(action)

    # ✅ 执行action，获取reward
    next_state, reward, done, info = env.step(action_env)

    # ✅ 累积到episode_reward（用于wandb）
    episode_reward += reward
    episode_steps = t

    # 📊 累积reward分解（用于wandb）
    for key in episode_reward_components:
        if key in env.step_reward_components:
            episode_reward_components[key] += env.step_reward_components[key]

    # ✅✅ 关键：存入agent.memory（用于梯度计算）
    agent.memory.append(state, action, reward, value, log_prob)

    state = next_state
```

**核心逻辑**：
1. `env.step()` 返回reward → 训练reward
2. `episode_reward += reward` → 累积用于wandb
3. `agent.memory.append(..., reward, ...)` → 存储用于梯度

**同一个reward值，既用于wandb记录，也用于训练更新！**

---

### 4️⃣ Reward用于梯度更新

```python
# train_ppo_with_wandb.py line 276-312

if episode % agent.update_frequency == 0:
    # ✅ 从memory中提取数据（包含刚才存入的reward）
    value_batches = agent.get_value_batches()
    policy_batches = agent.get_policy_batches()

    # Policy优化
    for data_batch in policy_batches:
        # ✅✅ 计算梯度（内部使用memory中的reward）
        total_loss, policy_grads = agent.get_policy_gradients(data_batch)

        # ✅✅ 更新网络
        agent.update_policy(policy_grads)

    # Value优化
    for data_batch in value_batches:
        # ✅✅ 计算梯度（内部使用memory中的reward）
        value_loss, value_grads = agent.get_value_gradients(data_batch)

        # ✅✅ 更新网络
        agent.update_value(value_grads)
```

**PPO内部逻辑**（简化）：
```python
# rl_agent_only/agents/ppo.py

def get_policy_gradients(self, batch):
    states, actions, old_log_probs, advantages, ... = batch

    # advantages是基于reward计算的
    # 计算policy loss
    loss = -advantages * new_log_probs + ...

    # 计算梯度
    grads = tape.gradient(loss, policy_params)

    return loss, grads
```

**Advantage计算**（GAE-λ）：
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...

其中 δ_t = reward_t + γV(s_{t+1}) - V(s_t)
```

**所以reward直接影响梯度更新！**

---

### 5️⃣ Reward记录到Wandb

```python
# train_ppo_with_wandb.py line 355-389

if wandb_run is not None:
    # 计算平均reward分解
    avg_reward_components = {
        f'reward/{k}': v / episode_steps if episode_steps > 0 else 0.0
        for k, v in episode_reward_components.items()
    }

    wandb_metrics = {
        # ✅ 记录累积的episode_reward
        'episode/total_reward': episode_reward,

        'episode/steps': episode_steps,
        'episode/duration': episode_duration,
        'episode/avg_speed': avg_speed,
        'episode/collision_count': episode_collision_count,

        'training/policy_loss': policy_loss_tracker,
        'training/value_loss': value_loss_tracker,
        'training/entropy': entropy_tracker,
    }

    # 添加reward分解
    wandb_metrics.update(avg_reward_components)

    # ✅ 上传到Wandb
    wandb_run.log(wandb_metrics)
```

---

## 📈 数据流图

```
┌──────────────────────────────────────────────────────────────┐
│                    CarlaEnv._get_reward()                    │
│  计算：base + wp + speed + lane + collision + idle          │
│  返回：total_reward                                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ↓ (total_reward)
┌──────────────────────────────────────────────────────────────┐
│                   CarlaEnv.step(action)                      │
│  调用：_get_reward()                                         │
│  返回：obs, reward, done, info                               │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ↓ (reward)
┌──────────────────────────────────────────────────────────────┐
│                  CarlaGymEnv.step(action)                    │
│  包装层：添加action bias                                     │
│  返回：obs, reward, done, info （reward未修改）              │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ↓ (reward)
┌──────────────────────────────────────────────────────────────┐
│                     训练主循环                                │
│  next_state, reward, done, info = env.step(action_env)      │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │ episode_reward  │         │  agent.memory   │           │
│  │  += reward      │         │  .append(       │           │
│  │                 │         │    ...,         │           │
│  │  ✅ 用于Wandb   │         │    reward,      │           │
│  │     记录        │         │    ...          │           │
│  │                 │         │  )              │           │
│  │                 │         │                 │           │
│  │                 │         │  ✅ 用于梯度    │           │
│  │                 │         │     计算        │           │
│  └─────────────────┘         └─────────────────┘           │
│           │                           │                     │
│           │                           │                     │
└───────────┼───────────────────────────┼─────────────────────┘
            │                           │
            ↓                           ↓
    ┌───────────────┐       ┌──────────────────────┐
    │  Wandb记录    │       │  梯度更新             │
    │  episode/     │       │  agent.get_policy_   │
    │  total_reward │       │    _gradients()      │
    │               │       │  agent.update_       │
    │  reward/r_wp  │       │    _policy()         │
    │  reward/...   │       │                      │
    └───────────────┘       └──────────────────────┘
```

---

## 🎯 关键结论

### ✅ Wandb记录的是训练Reward

**证据**：

1. **同一个变量**
   ```python
   next_state, reward, done, info = env.step(action_env)
   episode_reward += reward          # ← wandb用这个
   agent.memory.append(..., reward)  # ← 梯度用这个
   ```
   **同一个 `reward` 变量！**

2. **相同的数据源**
   - 来源：`carla_env._get_reward()`
   - 传递：`carla_env.step()` → `CarlaGymEnv.step()` → 训练循环
   - 用途：累积到 `episode_reward` + 存入 `agent.memory`

3. **没有分离训练/评测**
   - 代码中没有 `eval_mode`
   - 没有单独的评测循环
   - 所有episode都是训练episode

---

### ⚠️ 当前训练脚本的特点

#### ✅ 优点
- 简洁明了
- reward可追溯
- wandb监控的就是训练实际使用的数值

#### ❌ 缺点
- **缺少评测（evaluation）环节**
- 无法评估真实性能（训练reward可能过拟合）
- 无法验证泛化能力

---

## 💡 建议改进：添加评测模式

### 方法1: 每N个episode做一次评测

```python
# 在训练循环中添加

EVAL_INTERVAL = 50  # 每50 episodes评测一次
EVAL_EPISODES = 3   # 每次评测3个episode

if episode % EVAL_INTERVAL == 0:
    print(f"\n{'='*60}")
    print(f"[Evaluation] Episode {episode}")
    print(f"{'='*60}")

    eval_rewards = []
    eval_success_count = 0

    for eval_ep in range(EVAL_EPISODES):
        # 评测模式：不更新网络，不添加action bias
        state = env.reset()
        eval_reward = 0.0
        eval_steps = 0

        for t in range(1, timesteps + 1):
            # ✅ 使用确定性策略（mean，不采样）
            action = agent.predict_deterministic(state)
            action_env = agent.convert_action(action)

            # ✅ 执行action（不添加bias）
            next_state, reward, done, info = env.step_eval(action_env)
            eval_reward += reward
            eval_steps = t

            state = next_state
            if done:
                break

        eval_rewards.append(eval_reward)
        if info.get('collision', 0.0) == 0 and eval_steps > 100:
            eval_success_count += 1

    # 📊 记录评测结果到Wandb
    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    eval_success_rate = eval_success_count / EVAL_EPISODES

    if wandb_run is not None:
        wandb_run.log({
            'eval/avg_reward': avg_eval_reward,
            'eval/success_rate': eval_success_rate,
            'eval/best_reward': max(eval_rewards),
            'eval/worst_reward': min(eval_rewards),
            'episode_num': episode,
        })

    print(f"[Evaluation] Avg Reward: {avg_eval_reward:.2f}")
    print(f"[Evaluation] Success Rate: {eval_success_rate*100:.1f}%")
    print(f"{'='*60}\n")
```

### 方法2: 区分训练/评测reward

```python
# 添加一个标志区分训练/评测
class CarlaGymEnv:
    def __init__(self, ...):
        ...
        self.eval_mode = False  # 默认训练模式

    def step(self, action):
        # 🔧 评测模式不添加action bias
        if not self.eval_mode:
            # 训练模式：添加bias
            if self.current_episode < 500:
                applied_bias = ...
                action[0] = np.clip(action[0] + applied_bias, -1.0, 1.0)

        obs, reward, done, info = self.carla_env.step(action)
        return obs, reward, done, info

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode
```

---

## 📊 Wandb中应该看到的指标

### 当前（只有训练）
```
episode/total_reward  ← 训练reward（带bias）
reward/r_wp
reward/r_speed
...
training/policy_loss
training/value_loss
```

### 改进后（训练+评测）
```
# 训练指标
episode/total_reward      ← 训练reward（带bias）
training/policy_loss
training/value_loss

# 评测指标（每50 episodes）
eval/avg_reward           ← 评测reward（无bias，确定性策略）
eval/success_rate
eval/best_reward
eval/worst_reward
```

**好处**：
- ✅ 可以对比训练vs评测性能
- ✅ 检测过拟合（训练reward高但评测reward低）
- ✅ 验证泛化能力

---

## ✅ 总结

### 你的问题答案

**Q: Wandb输出的reward是训练的还是评测的？**

**A: 是训练reward（用于梯度更新的）**

具体来说：
1. ✅ 来自 `carla_env._get_reward()`
2. ✅ 传递到训练循环
3. ✅ 累积到 `episode_reward` → wandb记录
4. ✅ 存入 `agent.memory` → 梯度计算
5. ✅ 用于更新policy和value网络

**是同一个reward值，既用于wandb监控，也用于训练！**

### 建议

1. **短期**：继续当前方式（简单、可追溯）
2. **中期**：添加定期评测（每50 episodes）
3. **长期**：完整的训练/评测分离 + 最佳模型保存

**当前wandb监控的就是你想要的训练数据！** ✅
