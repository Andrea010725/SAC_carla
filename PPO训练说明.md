# PPO 训练说明

## ✅ 当前状态

### 最新权重
- **位置**: `weights/ppo-carla-standalone/`
- **保存时间**: 2025-12-10 02:16
- **Episode**: ~200
- **文件**:
  - `policy_net.index` + `policy_net.data-00000-of-00001` (13.5 KB)
  - `value_net.index` + `value_net.data-00000-of-00001` (15 KB)

---

## 🚀 如何训练

### 方式 1：从最新权重继续训练（推荐）

```bash
python train_ppo_standalone.py
```

**效果**:
- ✅ 自动检测并加载 Episode 200 的权重
- ✅ 从 Episode 201 开始继续训练
- ✅ 保持之前学到的策略

**预期输出**:
```
[2] 创建PPO Agent...

✅ 发现已有权重，将从以下路径加载:
  - Policy: ./weights/ppo-carla-standalone/policy_net.index
  - Value: ./weights/ppo-carla-standalone/value_net.index

loading weights...

✅ PPO Agent创建成功
  ✅ 已加载权重: Episode ~200 (12月10日 02:16)
  - Policy LR: 5e-5 (大幅降低，防止发散)
  - Value LR: 1e-4 (大幅降低，防止NaN)
  ...
```

---

### 方式 2：从头开始训练

如果想从头训练（丢弃已有权重）：

```bash
# 备份旧权重
mv weights/ppo-carla-standalone weights/ppo-carla-standalone.bak

# 重新训练
python train_ppo_standalone.py
```

**效果**:
- ⚠️  随机初始化网络
- ⚠️  从 Episode 1 开始
- ⚠️  丢失之前的学习进度

---

## 🔧 最新修复（12月10日 03:00+）

### 1. NaN 错误修复
- ✅ `reward_monitor.py`: 添加 NaN/Inf 检测，防止 matplotlib 崩溃
- ✅ `train_ppo_standalone.py`: 添加 action NaN 检测和安全默认值
- ✅ 降低学习率防止网络发散

### 2. 激进驾驶配置
- ✅ Action bias: 0.7（持续 500 episodes）
- ✅ 最小 throttle: 0.2（防止静止）
- ✅ Reward 惩罚: w_speed_pen=5.0, w_progress=3.0

### 3. 自动权重加载
- ✅ 启动时自动检测权重文件
- ✅ 存在则加载，否则从头训练
- ✅ 明确提示加载状态

---

## 📊 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| Policy LR | 5e-5 | 降低防止发散 |
| Value LR | 1e-4 | 降低防止 NaN |
| Entropy | 0.05 | 提高探索 |
| Batch Size | 256 | 稳定训练 |
| Action Bias | 0.7 (500 eps) | 强制前进 |
| Min Throttle | 0.2 | 防止静止 |

---

## 🎯 预期行为

### Episode 1-200（已完成）
- Throttle: 0.5-0.7
- Speed: 8-12 m/s
- 学会基本前进和避障

### Episode 201-500（当前）
- Throttle: 0.6-0.8（更激进）
- Speed: 10-15 m/s
- 学会快速通过锥桶

### Episode 500+（未来）
- Action bias 逐渐减弱
- 完全依赖学到的策略
- 应该能稳定快速驾驶

---

## ⚠️ 常见问题

### Q1: 如何确认是否加载了权重？
运行 `python check_weights.py` 或查看训练启动时的输出。

### Q2: 权重损坏怎么办？
备份文件在 `weights/ppo-carla-standalone.bak`（如果你做了备份）。

### Q3: 训练崩溃怎么办？
- NaN 错误：已修复，会自动处理
- 端口冲突：重启 CARLA 服务器
- 内存不足：降低 batch_size

---

## 📝 检查清单

运行训练前：
- [ ] 运行 `python check_weights.py` 确认权重状态
- [ ] 确认 CARLA 服务器运行在端口 2000
- [ ] 确认是否要继续训练（Yes）还是重新开始（No）

---

**最后更新**: 2025-12-10 03:30
