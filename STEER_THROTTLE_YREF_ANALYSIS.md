# Steer、Throttle 和 Y_ref 关系分析

**日期**: 2026-01-08
**问题**: steer 和 throttle 是否由 y_ref 计算得出？
**答案**: ❌ **不是！三者是独立的，但 y_ref 可以影响 steer**

---

## 核心结论

### 简短回答

**steer 和 throttle 不是由 y_ref 计算得出的！**

- **throttle**: 完全独立，由 PPO 输出的 `action[0]` 决定
- **steer**: 主要由 PPO 输出的 `action[1]` 决定，但可以**可选地**叠加 y_ref 的影响
- **y_ref**: 由 PPO 输出的 `action[2]`，是一个**辅助信号**

### 当前配置

```python
# config.py:99
use_yref_mapping = False  # ❌ 当前关闭！
```

**结论**: 在你当前的配置下，**y_ref 完全不影响 steer**！三者完全独立。

---

## 详细分析

### 1. Action 空间定义

**PPO 输出 3 维动作**:
```python
action = [a0, a1, a2]  # 范围都是 [-1, 1]

a0 = action[0]  # throttle/brake
a1 = action[1]  # steer_raw
a2 = action[2]  # y_ref
```

**代码位置**: `train_ppo_with_wandb.py:144-146`

---

### 2. Throttle 计算（完全独立）

**代码位置**: `carla_env.py:408-416`

```python
a0 = action[0]  # 范围 [-1, 1]

if a0 >= 0:
    # 正值 → 油门
    throttle = clip(a0, 0.0, 1.0)
    brake = 0.0
else:
    # 负值 → 刹车
    throttle = 0.0
    brake = clip(-a0, 0.0, 1.0)
```

**关键点**:
- ✅ throttle **完全由 a0 决定**
- ❌ **不受 y_ref 影响**
- ❌ **不受 steer 影响**

---

### 3. Steer 计算（可选受 y_ref 影响）

**代码位置**: `carla_env.py:418-423`

```python
steer_raw = clip(action[1], -1.0, 1.0)
y_ref = clip(action[2], -1.0, 1.0)

if self.use_yref_in_steer:
    # 模式1: 使用 y_ref（当前关闭）
    steer = clip(steer_raw + yref_steer_gain * y_ref, -1.0, 1.0)
else:
    # 模式2: 不使用 y_ref（当前使用）
    steer = steer_raw
```

**当前配置**:
```python
# config.py:99
use_yref_mapping = False  # ❌ 关闭

# 因此实际执行:
steer = steer_raw  # 完全由 action[1] 决定
```

**关键点**:
- ✅ steer **主要由 action[1] 决定**
- ⚠️ **可选地**叠加 y_ref 的影响（当前关闭）
- ❌ **不受 throttle 影响**

---

### 4. Y_ref 的作用

#### 4.1 当 `use_yref_mapping = False` (当前配置)

**y_ref 完全无用！**

```python
# PPO 输出
action = [0.5, 0.2, 0.8]
         ^^^  ^^^  ^^^
         a0   a1   a2 (y_ref)

# 实际使用
throttle = 0.5  # 来自 a0
steer = 0.2     # 来自 a1
y_ref = 0.8     # ❌ 被忽略！
```

**结论**: y_ref 被 PPO 输出，但**完全不影响车辆控制**！

#### 4.2 当 `use_yref_mapping = True` (如果开启)

**y_ref 会影响 steer**:

```python
# PPO 输出
action = [0.5, 0.2, 0.8]
         ^^^  ^^^  ^^^
         a0   a1   a2 (y_ref)

# 实际使用
throttle = 0.5  # 来自 a0
steer_raw = 0.2 # 来自 a1
y_ref = 0.8     # 来自 a2

# steer 计算
yref_steer_gain = 0.15  # config.py:100
steer = clip(steer_raw + yref_steer_gain * y_ref, -1.0, 1.0)
      = clip(0.2 + 0.15 * 0.8, -1.0, 1.0)
      = clip(0.2 + 0.12, -1.0, 1.0)
      = 0.32
```

**结论**: y_ref 作为**辅助信号**，微调 steer。

---

## 数据流图

### 当前配置 (use_yref_mapping = False)

```
PPO Agent
    ↓
[a0, a1, a2]
 ↓   ↓   ↓
 ↓   ↓   └─→ y_ref (0.8) ❌ 被忽略
 ↓   ↓
 ↓   └─→ steer_raw (0.2) → steer = 0.2 ✅
 ↓
 └─→ throttle/brake (0.5) → throttle = 0.5 ✅

最终控制:
- throttle = 0.5
- steer = 0.2
- y_ref 无影响
```

### 如果开启 (use_yref_mapping = True)

```
PPO Agent
    ↓
[a0, a1, a2]
 ↓   ↓   ↓
 ↓   ↓   └─→ y_ref (0.8) ✅ 参与计算
 ↓   ↓       ↓
 ↓   └───────┴─→ steer = steer_raw + 0.15 * y_ref
 ↓                     = 0.2 + 0.12 = 0.32 ✅
 ↓
 └─→ throttle/brake (0.5) → throttle = 0.5 ✅

最终控制:
- throttle = 0.5
- steer = 0.32 (受 y_ref 影响)
- y_ref 影响 steer
```

---

## 代码追踪

### 流程1: train_ppo_with_wandb.py

```python
# 1. PPO 输出动作
action = agent.predict(obs)  # shape: (3,)
# action = [a0, a1, a2]

# 2. Wrapper 处理（几乎不改动作）
action3 = np.array([a0_applied, a1, a2], dtype=np.float32)
# a0_applied 可能有 bias，但 a1 和 a2 不变

# 3. 传递给 CarlaEnv
obs, reward, done, info = self.carla_env.step(action3)
```

**代码位置**: `train_ppo_with_wandb.py:129-162`

### 流程2: carla_env.py

```python
# 1. 接收动作
def step(self, action):
    a0 = action[0]
    a1 = action[1]
    a2 = action[2]

# 2. 计算 throttle/brake
if a0 >= 0:
    throttle = clip(a0, 0.0, 1.0)
    brake = 0.0
else:
    throttle = 0.0
    brake = clip(-a0, 0.0, 1.0)

# 3. 计算 steer
steer_raw = clip(a1, -1.0, 1.0)
y_ref = clip(a2, -1.0, 1.0)

if self.use_yref_in_steer:  # ❌ 当前 False
    steer = clip(steer_raw + self.yref_steer_gain * y_ref, -1.0, 1.0)
else:
    steer = steer_raw  # ✅ 当前执行这个

# 4. 应用控制
control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
self.ego.apply_control(control)
```

**代码位置**: `carla_env.py:408-438`

---

## 配置文件详解

### config.py

```python
# 行99-101
self.use_yref_mapping = False  # ❌ 关闭 y_ref 映射
self.yref_gain = 0.15          # y_ref 增益（当前不使用）
self.yref_penalty = 0.0        # y_ref 惩罚（当前不使用）
```

### train_ppo_with_wandb.py

```python
# 行77-78: 将配置传递给 CarlaEnv
self.carla_env.use_yref_in_steer = bool(getattr(config, "use_yref_mapping", False))
self.carla_env.yref_steer_gain = float(getattr(config, "yref_gain", 0.15))
```

**映射关系**:
```
config.use_yref_mapping → carla_env.use_yref_in_steer
config.yref_gain        → carla_env.yref_steer_gain
```

---

## 实际示例

### 示例1: 当前配置 (use_yref_mapping = False)

**PPO 输出**:
```python
action = [0.6, -0.3, 0.5]
```

**计算过程**:
```python
# Throttle
a0 = 0.6
throttle = 0.6  # ✅ 直接使用
brake = 0.0

# Steer
a1 = -0.3
steer_raw = -0.3
y_ref = 0.5  # ❌ 被忽略

use_yref_in_steer = False
steer = steer_raw = -0.3  # ✅ 直接使用

# 最终控制
VehicleControl(throttle=0.6, brake=0.0, steer=-0.3)
```

**结论**: y_ref (0.5) 完全无影响！

### 示例2: 如果开启 (use_yref_mapping = True)

**PPO 输出**:
```python
action = [0.6, -0.3, 0.5]
```

**计算过程**:
```python
# Throttle (不变)
throttle = 0.6
brake = 0.0

# Steer (受 y_ref 影响)
steer_raw = -0.3
y_ref = 0.5
yref_steer_gain = 0.15

use_yref_in_steer = True
steer = clip(steer_raw + yref_steer_gain * y_ref, -1.0, 1.0)
      = clip(-0.3 + 0.15 * 0.5, -1.0, 1.0)
      = clip(-0.3 + 0.075, -1.0, 1.0)
      = -0.225  # ✅ 受 y_ref 影响

# 最终控制
VehicleControl(throttle=0.6, brake=0.0, steer=-0.225)
```

**结论**: y_ref (0.5) 使 steer 从 -0.3 变为 -0.225！

---

## Y_ref 的设计意图

### 为什么有 y_ref？

**设计思想**: 分层控制

1. **steer_raw (a1)**: 主要转向信号
2. **y_ref (a2)**: 辅助微调信号

**优点**:
- 可以学习更细腻的转向控制
- 主信号 + 微调信号的组合

**缺点**:
- 增加动作空间维度
- 可能不收敛（如果 PPO 不知道如何使用）

### 为什么当前关闭？

**可能原因**:
1. 简化训练：3维动作空间 → 实际只用2维
2. 避免混淆：让 PPO 专注于学习 throttle 和 steer
3. 调试方便：减少变量

---

## 如何验证？

### 方法1: 查看日志

```bash
# 运行训练
python train_ppo_with_wandb.py

# 查看输出
grep "use_yref_in_steer" training.log
```

**预期输出**:
```
[DEBUG] CarlaEnv y_ref switch after wrapper init:
  carla_env.use_yref_in_steer = False
  carla_env.yref_steer_gain   = 0.15
```

### 方法2: 查看 Wandb

在 Wandb 中查看:
- `debug/yref_used`: 应该是 0.0 (表示未使用)
- `step/action_steer_raw`: 应该等于实际 steer
- `step/y_ref`: 有值但不影响控制

### 方法3: 修改配置测试

```python
# config.py:99
self.use_yref_mapping = True  # 改为 True

# 重新运行
python train_ppo_with_wandb.py

# 观察 steer 是否变化
```

---

## 总结

### 核心结论

| 变量 | 来源 | 是否独立 | 当前是否使用 |
|------|------|---------|------------|
| **throttle** | action[0] | ✅ 完全独立 | ✅ 使用 |
| **steer** | action[1] | ⚠️ 主要独立 | ✅ 使用 |
| **y_ref** | action[2] | ✅ 独立输出 | ❌ 不使用 |

### 关系图

```
action[0] ──────────────→ throttle ✅
action[1] ──────────────→ steer ✅
action[2] ──────────────→ y_ref ❌ (当前被忽略)
```

### 当前配置

```python
use_yref_mapping = False  # ❌ 关闭

# 因此:
throttle = f(action[0])  # ✅ 独立
steer = action[1]        # ✅ 独立
y_ref = action[2]        # ❌ 无影响
```

### 如果开启

```python
use_yref_mapping = True  # ✅ 开启

# 则:
throttle = f(action[0])                      # ✅ 独立
steer = action[1] + 0.15 * action[2]        # ⚠️ 受 y_ref 影响
y_ref = action[2]                            # ✅ 影响 steer
```

---

## 建议

### 当前训练

**如果 y_ref 不使用，建议**:
1. 减少动作空间到 2 维: `[throttle_brake, steer]`
2. 或保持 3 维但告知 PPO 第 3 维无用

**优点**:
- 减少探索空间
- 加快收敛
- 避免混淆

### 如果想使用 y_ref

**修改配置**:
```python
# config.py:99
self.use_yref_mapping = True
self.yref_gain = 0.15
```

**注意**:
- 需要重新训练
- PPO 需要学习如何使用 y_ref
- 可能需要更多 episodes

---

**分析日期**: 2026-01-08
**结论**: ✅ 三者独立，y_ref 当前不影响 steer 和 throttle
