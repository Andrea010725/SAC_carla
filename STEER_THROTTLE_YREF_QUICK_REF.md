# Steer、Throttle、Y_ref 关系 - 快速参考

## ❓ 问题

**steer 和 throttle 是由 y_ref 算出来的吗？**

## ✅ 答案

**❌ 不是！三者是独立的！**

在你当前的配置下，y_ref **完全不影响** steer 和 throttle。

---

## 🔍 当前配置

```python
# config.py:99
use_yref_mapping = False  # ❌ 关闭
```

---

## 📊 数据流图

### 当前配置 (y_ref 关闭)

```
PPO Agent 输出
    ↓
[a0,  a1,  a2]
 ↓    ↓    ↓
 ↓    ↓    └─→ y_ref ❌ 被忽略
 ↓    ↓
 ↓    └─→ steer_raw → steer ✅
 ↓
 └─→ throttle_brake → throttle/brake ✅

结论: 三者完全独立！
```

### 如果开启 (y_ref 开启)

```
PPO Agent 输出
    ↓
[a0,  a1,  a2]
 ↓    ↓    ↓
 ↓    ↓    └─→ y_ref ✅
 ↓    ↓         ↓
 ↓    └─────────┴─→ steer = steer_raw + 0.15 * y_ref ⚠️
 ↓
 └─→ throttle_brake → throttle/brake ✅

结论: y_ref 影响 steer，但不影响 throttle
```

---

## 💻 代码位置

### 1. Throttle 计算
**位置**: `carla_env.py:408-416`
```python
if a0 >= 0:
    throttle = a0  # ✅ 完全由 action[0] 决定
    brake = 0.0
else:
    throttle = 0.0
    brake = -a0
```

### 2. Steer 计算
**位置**: `carla_env.py:418-423`
```python
steer_raw = action[1]
y_ref = action[2]

if use_yref_in_steer:  # ❌ 当前 False
    steer = steer_raw + 0.15 * y_ref
else:
    steer = steer_raw  # ✅ 当前执行这个
```

---

## 📈 实际示例

### 示例: PPO 输出 [0.6, -0.3, 0.8]

**当前配置 (y_ref 关闭)**:
```python
action = [0.6, -0.3, 0.8]

throttle = 0.6   # ✅ 来自 action[0]
steer = -0.3     # ✅ 来自 action[1]
y_ref = 0.8      # ❌ 被忽略

最终控制: throttle=0.6, steer=-0.3
```

**如果开启 (y_ref 开启)**:
```python
action = [0.6, -0.3, 0.8]

throttle = 0.6   # ✅ 来自 action[0]
steer_raw = -0.3
y_ref = 0.8
steer = -0.3 + 0.15 * 0.8 = -0.18  # ⚠️ 受 y_ref 影响

最终控制: throttle=0.6, steer=-0.18
```

---

## 🎯 结论

### 当前配置

| 变量 | 来源 | 计算公式 | 是否独立 |
|------|------|---------|---------|
| throttle | action[0] | `clip(a0, 0, 1)` if a0≥0 | ✅ 完全独立 |
| brake | action[0] | `clip(-a0, 0, 1)` if a0<0 | ✅ 完全独立 |
| steer | action[1] | `a1` | ✅ 完全独立 |
| y_ref | action[2] | `a2` | ❌ 不使用 |

### 关系总结

```
throttle ←─ action[0] (独立)
steer    ←─ action[1] (独立)
y_ref    ←─ action[2] (无影响)

三者完全独立！
```

---

## 💡 建议

### 如果不使用 y_ref

**建议**: 减少动作空间到 2 维

```python
# config.py
self.action_dim = 2  # 从 3 改为 2
```

**优点**:
- 减少探索空间
- 加快收敛
- 避免浪费

### 如果想使用 y_ref

**建议**: 开启映射

```python
# config.py
self.use_yref_mapping = True
self.yref_gain = 0.15
```

**注意**:
- 需要重新训练
- 收敛可能更慢
- 但可能学到更细腻的控制

---

## 🔍 验证方法

### 查看配置
```bash
grep "use_yref_mapping" config.py
# 应该看到: use_yref_mapping = False
```

### 查看日志
```bash
python train_ppo_with_wandb.py 2>&1 | grep "use_yref_in_steer"
# 应该看到: use_yref_in_steer = False
```

### 查看 Wandb
- `debug/yref_used`: 0.0 (未使用)
- `step/action_steer_raw`: 等于实际 steer
- `step/y_ref`: 有值但无影响

---

**分析日期**: 2026-01-08
**结论**: ✅ 三者独立，y_ref 当前不影响控制
