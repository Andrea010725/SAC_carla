# 🚀 快速开始 - 4场景训练

## 一键启动指南

### 步骤1：启动 CARLA 服务器
```bash
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen
```

### 步骤2：运行训练
```bash
cd /home/ajifang/SAC_carla
python train_ppo_4scenarios.py
```

---

## 📊 训练输出示例

```
======================================================================
PPO训练 - 4场景随机切换版本
场景池: cones, jaywalker, trimma, construction_lane_change
======================================================================

✅ Wandb已初始化
   Run名称: ppo-4scenarios-20260127-143052
   Run URL: https://wandb.ai/...

[场景配置] 4场景随机切换模式:
  1. cones
  2. jaywalker
  3. trimma
  4. construction_lane_change

[0] 加载激进版Reward配置...
✅ 激进版Reward已加载

[1] 创建训练日志记录器...
✅ 日志记录器创建成功

[2] 创建CARLA Gym环境...
✅ 环境创建成功
  - Observation space: Box(-inf, inf, (30,), float32)
  - Action space: Box(-1.0, 1.0, (3,), float32)

[3] 创建PPO Agent...
✅ 4场景训练：使用新权重目录，从头开始训练
✅ PPO Agent创建成功
  - 模型保存路径: ./weights/ppo-carla-4scenarios

[4] 开始训练...
----------------------------------------------------------------------

[RandomScenario] 本次Episode场景: cones

[Cones] 开始生成锥桶场景...
  - 锥桶数量: 15
  - 纵向间距: 3.0m
  - 横向递进: 0.4m
  - 起始位置: (123.4, 56.7)
  - 放置策略: 从左向右（两侧都是行车道，随机选择）
[Cones] ✅ 成功生成 15 个锥桶
[Cones] 自车spawn位置: (145.2, 58.3)

Episode 1 terminated after 512 timesteps in 28.456s with reward 234.567.

[RandomScenario] 本次Episode场景: jaywalker

[Jaywalker] 开始生成鬼探头场景...
  - 行人距离: 20.0m
  - 行人速度: 2.0m/s
  - 触发距离: 15.0m
  - 遮挡车辆: 否
[Jaywalker] ✅ 场景生成成功
  - ego spawn: (234.5, 123.4)
  - ped spawn: (254.3, 125.6) side=right
  - ped target: (254.3, 121.2)

Episode 2 terminated after 387 timesteps in 21.234s with reward 189.234.
...
```

---

## 🎯 预期训练效果

### 初期（Episode 1-50）
- **成功率：** 20-40%
- **碰撞率：** 40-60%
- **主要问题：** Jaywalker 场景紧急制动不及时

### 中期（Episode 51-150）
- **成功率：** 50-70%
- **碰撞率：** 20-30%
- **改善：** 开始学会避让和减速

### 后期（Episode 151-300）
- **成功率：** 70-85%
- **碰撞率：** 5-15%
- **稳定：** 各场景表现趋于稳定

---

## 📈 Wandb 监控要点

### 关键指标
1. **episode/success** - 成功率曲线（应该上升）
2. **episode/collision_count** - 碰撞次数（应该下降）
3. **early/success_rate_W** - 滑动窗口成功率（20 episodes）
4. **training/entropy** - 探索程度（应该逐渐下降）

### 场景分析
通过 `episode/done_reason` 可以看到：
- `time_limit`: 正常完成（好）
- `collision`: 碰撞终止（需要改进）
- `offroad`: 驶出道路（需要改进）
- `no_progress`: 卡住不动（需要改进）

---

## ⚡ 性能优化建议

### 如果训练太慢
```python
# 减少场景复杂度
config.cone_num = 10              # 从 15 减少到 10
config.traffic_density = 2.0      # 从 3.0 减少到 2.0
config.construction_length = 15.0 # 从 20.0 减少到 15.0
```

### 如果内存不足
```python
# 减少 batch size
agent = PPOAgent(
    ...
    batch_size=128,  # 从 256 减少到 128
    ...
)
```

### 如果想加快收敛
```python
# 增加学习率（谨慎使用）
agent = PPOAgent(
    ...
    policy_lr=2e-4,  # 从 1e-4 增加到 2e-4
    value_lr=4e-4,   # 从 2e-4 增加到 4e-4
    ...
)
```

---

## 🐛 常见问题

### Q1: 训练卡在某个场景
**A:** 检查该场景的参数配置，可能需要降低难度。例如：
```python
# Jaywalker 太难？增加触发距离
config.jaywalker_trigger_distance = 20.0  # 从 15.0 增加到 20.0
```

### Q2: CARLA 服务器崩溃
**A:** 重启 CARLA 并减少场景复杂度：
```bash
# 杀掉旧进程
pkill -9 CarlaUE4

# 重新启动
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen
```

### Q3: Wandb 上传失败
**A:** 检查网络连接，或使用离线模式：
```python
# 在 train_ppo_4scenarios.py 中修改
wandb_run = wandb.init(
    ...
    mode="offline",  # 添加这一行
)
```

### Q4: 训练不收敛
**A:** 可能需要调整早停参数：
```python
# 在 train_with_logging() 中修改
TARGET_SUCCESS = 0.70       # 从 0.80 降低到 0.70
MAX_COLLISION = 0.15        # 从 0.10 增加到 0.15
```

---

## 📊 训练完成后

### 1. 查看最终权重
```bash
ls -lh ./weights/ppo-carla-4scenarios/
```

### 2. 测试模型
```python
# 创建测试脚本 test_4scenarios.py
from train_ppo_4scenarios import CarlaGymEnv
from config import Config
from rl_agent_only.agents.ppo import PPOAgent

config = Config()
config.random_scenario = True
config.scenario_pool = ["cones", "jaywalker", "trimma", "construction_lane_change"]

env = CarlaGymEnv(config)
agent = PPOAgent(env=env, name="ppo-carla-4scenarios", load=True)

# 测试 10 个 episodes
for ep in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _, _, _, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {ep+1}: Reward={total_reward:.2f}, Scenario={env.carla_env.scenario}")

env.close()
```

### 3. 分析场景表现
```python
# 统计各场景成功率
import json

with open("training_log_4scenarios.json", "r") as f:
    log = json.load(f)

scenario_stats = {}
for episode in log["episodes"]:
    scenario = episode.get("scenario", "unknown")
    success = episode.get("success", False)

    if scenario not in scenario_stats:
        scenario_stats[scenario] = {"total": 0, "success": 0}

    scenario_stats[scenario]["total"] += 1
    if success:
        scenario_stats[scenario]["success"] += 1

for scenario, stats in scenario_stats.items():
    success_rate = stats["success"] / stats["total"] * 100
    print(f"{scenario}: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
```

---

## 🎓 进阶技巧

### 1. 场景权重调整
如果某个场景太难，可以减少它的出现频率：
```python
# 在 config.py 中
config.scenario_pool = [
    "cones",
    "cones",                      # 重复2次，出现概率更高
    "jaywalker",                  # 只出现1次
    "trimma",
    "construction_lane_change",
]
```

### 2. 课程学习
先训练简单场景，再逐步增加难度：
```python
# 阶段1：只训练 cones（100 episodes）
config.scenario_pool = ["cones"]

# 阶段2：添加 trimma（100 episodes）
config.scenario_pool = ["cones", "trimma"]

# 阶段3：添加 construction（100 episodes）
config.scenario_pool = ["cones", "trimma", "construction_lane_change"]

# 阶段4：添加 jaywalker（100 episodes）
config.scenario_pool = ["cones", "trimma", "construction_lane_change", "jaywalker"]
```

### 3. 迁移学习
从单场景模型开始：
```python
# 先加载单场景权重
agent = PPOAgent(
    env=env,
    name="ppo-carla-obs30",  # 旧模型
    load=True
)

# 然后在4场景上继续训练
# （需要确保 obs_dim 一致）
```

---

## 📞 获取帮助

如果遇到问题：
1. 查看 `4SCENARIOS_MODIFICATION_GUIDE.md` 详细文档
2. 检查 CARLA 日志：`/home/ajifang/carla/Saved/Logs/`
3. 查看训练日志：`training_log_4scenarios.json`
4. 检查 Wandb 面板的错误信息

---

**祝训练顺利！🎉**
