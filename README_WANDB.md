# 🚀 PPO训练增强版 - Wandb实时监控

## 📁 新增文件

### 1. **train_ppo_with_wandb.py** - 带Wandb监控的训练脚本
完全集成Wandb实时监控的PPO训练脚本，相比原版 `train_ppo_standalone.py` 的增强：

✅ **Wandb实时监控**
  - Episode级别指标（reward, steps, speed, collision）
  - Reward详细分解（6个组成部分）
  - Training指标（policy loss, value loss, entropy）
  - Step级别指标（每10步记录）
  - Action bias跟踪

✅ **优雅的降级处理**
  - 如果wandb未安装，自动跳过在线监控
  - 不影响正常训练流程

✅ **自动checkpoint上传**
  - 每100 episodes自动上传模型到Wandb云端

### 2. **WANDB_TRAINING_GUIDE.md** - 完整使用指南
详细的Wandb使用教程，包括：
  - 快速开始
  - Dashboard使用
  - 问题诊断
  - 高级功能
  - 最佳实践

### 3. **start_ppo_training_wandb.sh** - 一键启动脚本
自动化检查和启动训练：
  - 检查CARLA服务器状态
  - 检查Python环境
  - 自动安装/配置Wandb
  - 交互式确认
  - 自动记录训练日志

### 4. **carla_base/carla_env.py** (已修改)
添加了reward分解导出功能：
  - 新增 `last_reward_components` 属性
  - 记录每步的6个reward组成部分
  - 供训练脚本读取并上传到Wandb

---

## 🎯 快速开始

### 方法1: 使用启动脚本（推荐）

```bash
# 1. 启动CARLA（在另一个终端）
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20

# 2. 运行启动脚本
cd /home/ajifang/SAC_carla
./start_ppo_training_wandb.sh
```

脚本会自动：
- ✅ 检查CARLA是否运行
- ✅ 检查Python环境
- ✅ 检查/安装/登录Wandb
- ✅ 创建必要目录
- ✅ 显示训练配置
- ✅ 开始训练并记录日志

### 方法2: 直接运行Python脚本

```bash
# 1. 启动CARLA
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20

# 2. 安装wandb（首次使用）
pip install wandb
wandb login  # 输入API key

# 3. 开始训练
cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
```

---

## 📊 Wandb监控指标

### Episode级别（每个episode结束后记录）

| 指标 | 说明 | 期望值 |
|-----|------|--------|
| `episode/total_reward` | 总reward | >300 |
| `episode/steps` | episode步数 | >400 |
| `episode/avg_speed` | 平均速度（m/s） | >8 |
| `episode/collision_count` | 碰撞次数 | 0 |
| `episode/success` | 是否成功 | 1 |

### Reward分解（平均值）

| 指标 | 说明 | 典型值 |
|-----|------|--------|
| `reward/base_reward` | RewardMonitor基础 | 0.02~0.05 |
| `reward/r_wp` | Waypoint进度 | 0.3~0.6 |
| `reward/r_speed` | 速度维持 | 0.2~0.4 |
| `reward/r_lane` | 车道保持 | -0.3~0 |
| `reward/r_collision` | 碰撞惩罚 | -5 or 0 |
| `reward/r_idle` | 站桩惩罚 | -10 or 0 |

### Training指标

| 指标 | 说明 | 健康值 |
|-----|------|--------|
| `training/policy_loss` | Policy loss | <0.05 |
| `training/value_loss` | Value loss | <0.3 |
| `training/entropy` | 策略熵 | 0.3~0.8 |
| `training/action_bias` | 当前bias | 0.7→0 |

---

## 🎨 Wandb Dashboard示例

训练开始后，访问Wandb URL，你会看到：

### Charts标签页

**推荐创建的图表：**

1. **Training Progress (3x1面板)**
   ```
   [Episode Reward]  [Average Speed]  [Success Rate]
   ```

2. **Reward Decomposition (堆叠面积图)**
   ```
   所有reward/*指标叠加显示
   ```

3. **Training Health (双轴图)**
   ```
   左轴: policy_loss, value_loss
   右轴: entropy
   ```

4. **Action Monitoring**
   ```
   step/action_throttle 和 step/action_steer 的分布
   ```

### System标签页

监控系统资源：
- GPU使用率
- CPU使用率
- 内存占用
- 网络流量

---

## 🔍 如何使用Wandb诊断问题

### 问题1: 车辆不动

**检查**:
1. 打开 `training/action_bias` 图表
2. 检查前100 episodes是否>0.5
3. 查看 `step/forced_throttle` 是否经常为1

**诊断**:
- 如果 `action_bias` 过早衰减 → 延长衰减周期
- 如果 `forced_throttle` 很少为1 → 网络学到了错误的策略

---

### 问题2: 碰撞率高

**检查**:
1. 对比 `episode/collision_count` 和 `reward/r_collision`
2. 查看 `training/entropy` 是否过高（>1.0）
3. 对比不同episode的 `episode/avg_speed`

**诊断**:
- 如果 `r_collision` 惩罚不够 → 增加碰撞权重
- 如果 `entropy` 过高 → 降低熵正则化
- 如果速度过快 → 调整激进版reward配置

---

### 问题3: Reward不收敛

**检查**:
1. 查看 `episode/total_reward` 是否有上升趋势
2. 对比 `training/policy_loss` 和 `training/value_loss`
3. 查看 `reward/r_wp` 是否增长

**诊断**:
- 如果loss持续>0.1 → 降低学习率
- 如果 `r_wp` 接近0 → reward设计有问题
- 如果前100 episodes无改善 → 考虑简化场景

---

## 📈 预期训练曲线

### Episode 0-100（探索期）
```
episode/total_reward:  -50 → 50
episode/collision_rate: 60% → 40%
episode/avg_speed:     3 m/s → 6 m/s
training/entropy:      1.0 → 0.8
```

### Episode 100-300（学习期）
```
episode/total_reward:  50 → 150
episode/collision_rate: 40% → 20%
episode/avg_speed:     6 m/s → 8 m/s
training/entropy:      0.8 → 0.5
```

### Episode 300-600（优化期）
```
episode/total_reward:  150 → 250
episode/collision_rate: 20% → 10%
episode/avg_speed:     8 m/s → 10 m/s
training/entropy:      0.5 → 0.3
```

### Episode 600-1000（精炼期）
```
episode/total_reward:  250 → 350
episode/collision_rate: 10% → 5%
episode/avg_speed:     10 m/s → 11 m/s
training/entropy:      0.3 → 0.2
```

---

## 🛠️ 故障排除

### Wandb连接失败

```bash
# 重新登录
wandb login --relogin

# 或使用离线模式
WANDB_MODE=offline python train_ppo_with_wandb.py

# 训练完成后同步
wandb sync wandb/run-xxxxx
```

### 内存不足

Wandb会缓存数据，如果内存紧张：

```bash
# 设置临时缓存目录
export WANDB_CACHE_DIR=/tmp/wandb_cache

# 或禁用缓存
export WANDB_DISABLE_CODE=true
```

### 训练速度慢

如果Wandb logging影响速度：

1. 减少step级别记录频率（train_ppo_with_wandb.py line 114）：
   ```python
   if self._step_count % 50 == 0:  # 改为50或100
   ```

2. 或暂时禁用Wandb：
   ```bash
   WANDB_MODE=disabled python train_ppo_with_wandb.py
   ```

---

## 📚 参考资源

### 文档
- [Wandb官方文档](https://docs.wandb.ai/)
- [Wandb RL教程](https://wandb.ai/site/articles/rl-tutorial)
- [本地训练指南](./WANDB_TRAINING_GUIDE.md)

### 相关文件
- 原版训练脚本: `train_ppo_standalone.py`
- 简化版训练脚本: `train_ppo_simple.py`
- SAC训练脚本: `train_sac_with_frozen_planners.py`

---

## ✅ 检查清单

训练前确认：

- [ ] CARLA服务器已启动（端口2000）
- [ ] Python环境正常（TensorFlow 2.4.0）
- [ ] Wandb已安装并登录
- [ ] 磁盘空间充足（>5GB）
- [ ] GPU可用（推荐但非必需）

训练中监控：

- [ ] Wandb dashboard正常更新
- [ ] episode/total_reward 有上升趋势
- [ ] episode/collision_rate 逐渐下降
- [ ] training/entropy 逐渐收敛
- [ ] 无NaN/Inf警告

训练后检查：

- [ ] 模型已保存到 `weights/ppo-carla-standalone/`
- [ ] 日志已记录到 `training_log.json`
- [ ] Wandb run已完成（绿色勾号）
- [ ] 下载最佳checkpoint

---

## 🎯 下一步

训练完成后：

1. **评估模型性能**
   ```bash
   python main.py --test --model_path weights/ppo-carla-standalone/
   ```

2. **集成到SAC**
   - 在 `router.py` 中加载训练好的PPO模型
   - 作为RL planner使用

3. **继续优化**
   - 分析Wandb数据找到瓶颈
   - 调整超参数
   - 尝试不同场景

---

**现在你已经准备好开始训练了！🚀**

有任何问题，查看 `WANDB_TRAINING_GUIDE.md` 获取详细帮助。
