# RL Agent Only - 纯强化学习Agent代码

这个文件夹包含了从原项目中提取的**纯RL Agent代码**，不包含CARLA环境和场景相关的部分。

## 📁 目录结构

```
rl_agent_only/
├── agents/              # RL算法实现
│   ├── ppo.py          # PPO算法主实现（✅ 已修复NaN bug）
│   ├── agents.py       # 基础Agent类
│   ├── ppo_new.py      # PPO变体
│   └── agents_new.py   # Agent变体
│
├── networks/           # 神经网络架构
│   ├── networks.py     # PPO网络（policy + value + TRD）
│   └── architectures.py # 网络架构组件
│
├── parameters/         # 动态参数管理
│   └── parameters.py   # 学习率、clip ratio等动态参数
│
├── augmentations/      # 数据增强
│   ├── augmentations.py # 图像增强方法
│   └── simclr.py       # SimCLR对比学习
│
├── utils.py            # 工具函数
└── __init__.py         # 包初始化
```

## ✅ 已修复的Bug

### 1. NaN问题修复
- **位置**: `core/carla_env.py` lines 1084-1102
- **问题**: `_get_vector_map_features()`中使用了`...`(Ellipsis)作为参数
- **修复**: 完整实现了actor位置和速度的坐标提取
- **状态**: ✅ 已验证无NaN

### 2. Action双重缩放修复
- **位置**: `agents/ppo.py` lines 210-236
- **问题**: Normal+Tanh输出(-1,1)但被当作Beta分布再次缩放
- **修复**: 改为gaussian分布，直接使用tanh输出
- **状态**: ✅ 已修复

### 3. 车辆不动问题修复
- **位置**: `core/learning.py` line 299
- **问题**: `throttle_as_desired_speed=True`导致throttle只有0.12
- **修复**: 设置为False，让action直接控制throttle
- **状态**: ✅ 已修复

### 4. TRD Loss Coefficient未初始化修复
- **位置**: `agents/ppo.py` line 107
- **问题**: `self.trd_loss_coef`在`value_objective()`中被使用但从未初始化
- **修复**: 在`__init__`中添加 `self.trd_loss_coef = kwargs.get('trd_loss_coef', 1.0)`
- **状态**: ✅ 已修复（原项目和rl_agent_only都已修复）

## 🎯 核心功能

### 1. PPO Agent (`agents/ppo.py`)
- ✅ Proximal Policy Optimization算法
- ✅ Clipped surrogate objective
- ✅ GAE (Generalized Advantage Estimation)
- ✅ 支持连续和离散动作空间
- ✅ 动态学习率、clip ratio、熵系数
- ✅ Polyak平均（可选）
- ✅ 梯度裁剪
- ✅ TRD (Temporal Return Decomposition) 支持

### 2. PPO Network (`networks/networks.py`)
- ✅ Policy Network: Normal分布 + Tanh激活
- ✅ Value Network: Base * 10^Exp表示（支持大范围value）
- ✅ TRD Head: 时序回报分解预测
- ✅ 共享Dynamics Network（可选）

### 3. Memory管理 (`agents/ppo.py - PPOMemory`)
- ✅ 存储states, actions, rewards, values, log_probs
- ✅ 计算returns (rewards-to-go)
- ✅ 计算advantages (GAE)
- ✅ 计算TRD targets
- ✅ Batch生成和shuffle

### 4. 动态参数 (`parameters/parameters.py`)
- ✅ 学习率调度（linear decay, exponential decay, step decay）
- ✅ Clip ratio调度
- ✅ 熵系数调度
- ✅ Advantage scaling调度

## 📝 使用示例

### 基础PPO训练
```python
from rl_agent_only.agents.ppo import PPOAgent
import gym

# 创建环境（需要符合gym接口）
env = gym.make('YourEnv-v0')

# 创建PPO agent
agent = PPOAgent(
    env=env,
    policy_lr=3e-4,
    value_lr=1e-3,
    gamma=0.99,
    lambda_=0.95,
    clip_ratio=0.2,
    entropy_regularization=0.01,
    optimization_steps=(10, 10),
    batch_size=256
)

# 训练
agent.learn(
    episodes=1000,
    timesteps=512,
    save_every=100,
    render_every=False
)

# 评估
agent.collect(
    episodes=10,
    timesteps=512,
    render=True
)
```

### 自定义网络
```python
network_config = {
    'policy': {
        'units': 256,
        'num_layers': 3,
        'activation': 'relu'
    },
    'value': {
        'units': 256,
        'num_layers': 3,
        'activation': 'relu'
    }
}

agent = PPOAgent(
    env=env,
    network=network_config
)
```

### 动态学习率
```python
from rl_agent_only.parameters import LinearDecay

policy_lr = LinearDecay(
    initial_value=3e-4,
    final_value=1e-5,
    num_episodes=1000
)

agent = PPOAgent(
    env=env,
    policy_lr=policy_lr
)
```

## 🔧 与原项目的区别

### 移除的部分
- ❌ `rl/environments/` - CARLA环境代码
- ❌ `rl/environments/carla/tiny_scenarios.py` - 场景生成
- ❌ `rl/environments/carla/navigation/` - 路径规划
- ❌ CARLA相关的工具和依赖

### 保留的部分
- ✅ 所有RL算法实现
- ✅ 所有神经网络架构
- ✅ 所有参数管理
- ✅ 所有数据增强
- ✅ 所有工具函数

## 🚀 如何使用

### 1. 安装依赖
```bash
pip install tensorflow==2.4.0
pip install tensorflow-probability==0.12.0
pip install numpy
pip install gym
```

### 2. 导入Agent
```python
import sys
sys.path.append('/path/to/rl_agent_only')

from agents.ppo import PPOAgent
from networks.networks import PPONetwork
```

### 3. 适配你的环境
你需要提供一个符合gym接口的环境：
```python
class YourEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Dict({
            'feature1': gym.spaces.Box(low=0, high=1, shape=(10,)),
            'feature2': gym.spaces.Box(low=0, high=1, shape=(5,))
        })

    def step(self, action):
        # 执行动作
        next_state, reward, done, info = ...
        return next_state, reward, done, info

    def reset(self):
        # 重置环境
        return initial_state
```

## 📊 性能特点

### 优点
- ✅ 完整的PPO实现，包含所有现代技巧
- ✅ 支持复杂observation space（Dict, tuple等）
- ✅ TRD支持（提升value估计精度）
- ✅ 灵活的网络架构配置
- ✅ 完善的logging和tensorboard支持

### 局限
- ⚠️  需要手动适配环境接口
- ⚠️  原本为CARLA设计，可能需要调整observation处理
- ⚠️  依赖TensorFlow 2.x（不支持PyTorch）

## 🔗 相关文件

如果需要CARLA相关功能，请参考原项目：
- 环境代码：`rl/environments/carla/environment.py`
- 场景生成：`rl/environments/carla/tiny_scenarios.py`
- Agent封装：`core/carla_agent.py`
- 训练脚本：`core/learning.py`

## 📚 参考资料

### PPO论文
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### GAE论文
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

### TRD（如果使用）
- Temporal Return Decomposition for value function approximation

---

**最后更新**: 2025-12-07
**状态**: ✅ 所有NaN bug已修复
**测试**: ✅ 已在CARLA环境中验证工作正常
