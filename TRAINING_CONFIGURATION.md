# 训练配置说明 - 你的PPO训练设置

## 📊 **核心训练参数**

### Episode是什么？
**1个Episode = 车辆完成1次完整的驾驶任务**

每个Episode包含：
- **起点**：在Town05地图随机选择一个起始点
- **场景**：parked_obstacles（路边有4辆停车，需要绕过）
- **最大步数**：最多500步（每步0.05秒 = 最长25秒）
- **结束条件**：
  - ✅ 碰撞 → Episode结束
  - ✅ 达到500步 → Episode超时结束
  - ✅ 长时间不动（>50步低于0.2m/s）→ Episode结束

---

## 🎯 **你的训练规模**

### 当前配置

| 参数 | 值 | 说明 |
|------|------|------|
| **总Episodes** | **1000** | 总共要完成1000次驾驶任务 |
| **每Episode最大步数** | **512** | 每次任务最多512步 |
| **保存频率** | 每100 episodes | 第100, 200, 300...保存模型 |
| **Config最大步数** | 500 | CarlaEnv的最大步数限制 |

**注意**：训练代码用的是512步，但CarlaEnv配置的是500步，以**较小值500步**为准。

---

## 🧮 **训练量计算**

### 总训练步数
```
1000 episodes × 500 steps/episode = 500,000 步
```

### 总训练时间（仿真时间）
```
500,000 steps × 0.05秒/step = 25,000秒 = 6.94小时（仿真时间）
```

### 总训练时间（实际时间，无渲染）
**每步耗时估算**：
- 无渲染模式：约0.5-1秒/step（取决于CPU）
- 有渲染模式：约2-5秒/step

**无渲染模式下**：
```
500,000 steps × 0.75秒/step ≈ 375,000秒 ≈ 104小时 ≈ 4.3天
```

**但考虑到**：
- Episode可能提前结束（碰撞、idle）
- 平均每episode约200-300步（不是500步）

**实际预计**：
```
1000 episodes × 250 steps/episode × 0.75秒/step
= 187,500秒 ≈ 52小时 ≈ 2.2天
```

---

## 📈 **训练进度里程碑**

### 短期（前100 episodes）
- **目标**：学会基本的前进和转向
- **时间**：约5小时
- **预期**：碰撞率逐渐降低

### 中期（100-500 episodes）
- **目标**：学会避障和绕行
- **时间**：约20-25小时
- **预期**：成功率>50%

### 长期（500-1000 episodes）
- **目标**：优化策略，提高平滑度
- **时间**：约25-30小时
- **预期**：成功率>70%

---

## 🎮 **每个Episode的详细流程**

### 1. Reset阶段（约2秒）
```python
env.reset()
```
- 清理上一个episode的actors
- 随机选择Town05的一个起始点
- 在起始点前方30米开始放置4辆停车（间隔50米）
- 生成自车（Tesla Cybertruck）
- 创建传感器（碰撞检测，如果有渲染则加相机）

### 2. Step循环（最多500步）
```python
for t in range(1, 512 + 1):  # 但CarlaEnv限制是500步
    action = agent.predict(state)       # PPO预测动作 [throttle/brake, steer]
    state, reward, done, info = env.step(action)

    if done:  # 碰撞/超时/idle
        break
```

每一步：
- PPO输出动作：`[throttle_brake, steer]`
- CARLA执行动作
- 计算reward（包括progress, speed, lane, collision, idle等）
- 检查是否结束

### 3. 训练更新（每episode结束后）
```python
if episode % agent.update_frequency == 0:  # update_frequency=1
    # Policy网络更新（10次优化步）
    # Value网络更新（10次优化步）
```

### 4. 保存（每100 episodes）
```python
if episode % 100 == 0:
    agent.save()  # 保存到 ./weights/ppo-carla-standalone/
```

---

## 📁 **训练输出**

### 模型保存位置
```
./weights/ppo-carla-standalone/
├── policy_net.data-00000-of-00001
├── policy_net.index
├── value_net.data-00000-of-00001
└── value_net.index
```

### 日志文件
```
./training_log.json           # TrainingLogger输出
./wandb/                       # Wandb训练曲线
```

### Wandb监控指标
**Episode级别**：
- `episode/total_reward`
- `episode/steps`
- `episode/duration`
- `episode/avg_speed`
- `episode/collision_count`
- `episode/success`

**Step级别**（每10步记录一次）：
- `step/reward`
- `step/speed`
- `step/action_throttle_brake`
- `step/action_steer`

**训练指标**：
- `training/policy_loss`
- `training/value_loss`
- `training/entropy`

---

## 🎯 **预期训练效果**

### Episode 1-100
```
Episode reward: -50 到 0
Collision rate: ~80%
Avg steps: ~100
主要问题: 乱撞、不会转向
```

### Episode 100-300
```
Episode reward: 0 到 30
Collision rate: ~50%
Avg steps: ~200
主要问题: 能前进但绕障不稳定
```

### Episode 300-600
```
Episode reward: 30 到 60
Collision rate: ~30%
Avg steps: ~300
主要问题: 偶尔撞车、转向不够平滑
```

### Episode 600-1000
```
Episode reward: 60 到 80+
Collision rate: <20%
Avg steps: ~400
表现: 能稳定绕过障碍物
```

---

## ⚙️ **如果想修改训练规模**

### 增加训练量
**文件**：`train_ppo_with_wandb.py:587`
```python
# 改为2000 episodes
episodes=2000,

# 或者10000 episodes（约20-25天）
episodes=10000,
```

### 减少训练量（快速测试）
```python
# 只训练100 episodes（约5小时）
episodes=100,
```

### 修改Episode长度
**文件**：`config.py:77`
```python
# 更长的episode（1000步 = 50秒）
self.max_episode_steps = 1000

# 更短的episode（200步 = 10秒）
self.max_episode_steps = 200
```

**注意**：也要同步修改 `train_ppo_with_wandb.py:588` 的 `timesteps` 参数。

---

## 📊 **总结**

| 项目 | 值 |
|------|------|
| **总Episodes** | 1000 |
| **每Episode最大步数** | 500 |
| **总训练步数** | ~250,000（平均每episode 250步） |
| **预计总时间（无渲染）** | 约50-60小时（2-2.5天） |
| **保存频率** | 每100 episodes |
| **场景** | Town05 + 4辆停车障碍 |
| **目标** | 学会绕过路边停车 |

---

**当前配置是合理的训练规模，足够让PPO学会基本的避障驾驶！** 🚗
