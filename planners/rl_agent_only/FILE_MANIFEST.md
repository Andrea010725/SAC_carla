# RL Agent Only 文件清单

## 📋 完整文件列表及说明

### 根目录文件
```
rl_agent_only/
├── __init__.py              # 包初始化文件
├── utils.py                 # 通用工具函数（tensor转换、数据处理等）
├── README.md                # 详细使用文档
├── requirements.txt         # Python依赖列表
├── example_usage.py         # 完整使用示例
└── FILE_MANIFEST.md         # 本文件 - 文件清单
```

---

## 🤖 agents/ - RL算法实现

### 核心文件
| 文件 | 说明 | 状态 | 推荐使用 |
|------|------|------|----------|
| `ppo.py` | **PPO主实现** - 包含完整的PPO算法、Memory管理、TRD支持 | ✅ 已修复NaN bug | ⭐⭐⭐⭐⭐ |
| `agents.py` | 基础Agent抽象类 - 定义了agent的通用接口和方法 | ✅ 正常 | ⭐⭐⭐⭐ |
| `__init__.py` | 包初始化 - 导出主要类 | ✅ 正常 | - |

### 变体文件（可选）
| 文件 | 说明 | 状态 | 推荐使用 |
|------|------|------|----------|
| `ppo_new.py` | PPO变体实现 | ⚠️ 未测试 | ⭐⭐ |
| `agents_new.py` | Agent变体实现 | ⚠️ 未测试 | ⭐⭐ |
| `ppo-Copy1.py` | PPO备份版本 | ⚠️ 可能过时 | ⭐ |

### ppo.py 详细功能
- **PPOAgent类** (line 83+)
  - 完整的PPO算法实现
  - 支持连续和离散动作空间
  - Clipped surrogate objective
  - GAE (Generalized Advantage Estimation)
  - 动态参数调度
  - Gradient clipping
  - Polyak averaging（可选）
  - TRD (Temporal Return Decomposition)

- **PPOMemory类** (line 834+)
  - 存储transitions
  - 计算returns (rewards-to-go)
  - 计算advantages (GAE)
  - 计算TRD targets
  - Batch生成和shuffle

---

## 🧠 networks/ - 神经网络架构

### 核心文件
| 文件 | 说明 | 状态 | 推荐使用 |
|------|------|------|----------|
| `networks.py` | **PPO神经网络** - Policy、Value、TRD网络实现 | ✅ 正常 | ⭐⭐⭐⭐⭐ |
| `architectures.py` | 网络架构组件 - MLP、CNN等基础模块 | ✅ 正常 | ⭐⭐⭐⭐ |
| `__init__.py` | 包初始化 | ✅ 正常 | - |

### networks.py 详细功能
- **PPONetwork类**
  - Policy Network: Normal分布 + Tanh激活（输出连续动作）
  - Value Network: Base * 10^Exp表示（支持大范围value）
  - TRD Head: 时序回报分解预测（可选）
  - 支持共享Dynamics Network
  - 灵活的网络架构配置

- **核心方法**
  - `act()`: 根据policy采样动作
  - `predict()`: 预测value和action
  - `policy()`: 返回action distribution
  - `value()`: 预测state value
  - `value_and_trd()`: 同时预测value和TRD

---

## ⚙️ parameters/ - 动态参数管理

### 核心文件
| 文件 | 说明 | 状态 | 推荐使用 |
|------|------|------|----------|
| `parameters.py` | **动态参数类** - 学习率调度、参数衰减等 | ✅ 正常 | ⭐⭐⭐⭐⭐ |
| `__init__.py` | 包初始化 | ✅ 正常 | - |

### parameters.py 详细功能
- **DynamicParameter基类**
  - 支持固定值、函数、schedule
  - Episode级别的更新

- **内置Schedule**
  - `LinearDecay`: 线性衰减
  - `ExponentialDecay`: 指数衰减
  - `StepDecay`: 阶梯衰减
  - `CosineAnnealing`: 余弦退火

- **使用场景**
  - Learning rate scheduling
  - Clip ratio annealing
  - Entropy coefficient decay
  - Advantage scaling adjustment

---

## 🎨 augmentations/ - 数据增强

### 核心文件
| 文件 | 说明 | 状态 | 推荐使用 |
|------|------|------|----------|
| `augmentations.py` | **图像增强方法** - 各种数据增强技术 | ✅ 正常 | ⭐⭐⭐ |
| `simclr.py` | SimCLR对比学习 | ✅ 正常 | ⭐⭐ |
| `__init__.py` | 包初始化 | ✅ 正常 | - |

### augmentations.py 功能
- 随机裁剪 (Random Crop)
- 随机翻转 (Random Flip)
- 颜色抖动 (Color Jitter)
- 高斯模糊 (Gaussian Blur)
- 灰度变换 (Grayscale)
- 等等...

**注意**: 如果你的环境不使用图像输入，可以忽略这个模块。

---

## 🛠️ utils.py - 工具函数

### 核心功能
- **Tensor转换**
  - `to_tensor()`: 将numpy转为TensorFlow tensor
  - `to_float()`: 类型转换
  - `replace_nans()`: NaN替换

- **数值计算**
  - `rewards_to_go()`: 计算累积回报
  - `gae()`: Generalized Advantage Estimation
  - `clip_gradients()`: 梯度裁剪
  - `polyak_averaging()`: Polyak平均

- **数据处理**
  - `data_to_batches()`: 生成训练batch
  - `decompose_number()`: 数值分解（用于value network）

---

## 📊 文件统计

### 总计
- **总文件数**: 17个Python文件
- **总代码行数**: 约8000+ 行
- **核心文件**: 6个（agents/ppo.py, networks/networks.py等）
- **可选文件**: 11个

### 按模块统计
| 模块 | 文件数 | 核心度 |
|------|--------|--------|
| agents | 6 | ⭐⭐⭐⭐⭐ |
| networks | 3 | ⭐⭐⭐⭐⭐ |
| parameters | 2 | ⭐⭐⭐⭐ |
| augmentations | 3 | ⭐⭐ |
| utils | 1 | ⭐⭐⭐⭐ |
| 其他 | 2 | - |

---

## 🎯 最小使用集合

如果你只想要最核心的PPO功能，只需要这些文件：

```
rl_agent_only/
├── agents/
│   ├── ppo.py          # ⭐⭐⭐⭐⭐ 必需
│   ├── agents.py       # ⭐⭐⭐⭐⭐ 必需
│   └── __init__.py     # 必需
├── networks/
│   ├── networks.py     # ⭐⭐⭐⭐⭐ 必需
│   ├── architectures.py # ⭐⭐⭐⭐ 建议
│   └── __init__.py     # 必需
├── parameters/
│   ├── parameters.py   # ⭐⭐⭐⭐ 建议
│   └── __init__.py     # 必需
├── utils.py            # ⭐⭐⭐⭐⭐ 必需
└── __init__.py         # 必需
```

**最小集合共9个文件，约5000行代码**

---

## 🚀 快速开始

### 1. 最简使用
```python
from agents.ppo import PPOAgent

agent = PPOAgent(env=your_gym_env)
agent.learn(episodes=100, timesteps=512)
```

### 2. 标准使用
参考 `example_usage.py`

### 3. 高级定制
阅读 `README.md` 中的详细文档

---

## ⚠️ 移除的内容

为了保持代码纯粹，以下内容已移除：

### 不包含的模块
- ❌ `rl/environments/` - CARLA环境
- ❌ `rl/environments/carla/` - CARLA工具
- ❌ `rl/environments/carla/navigation/` - 路径规划
- ❌ `rl/environments/carla/tools/` - CARLA工具
- ❌ `tiny_scenarios.py` - 场景生成

### 不包含的依赖
- ❌ CARLA Python API
- ❌ Pygame（用于CARLA渲染）
- ❌ XML解析（用于route加载）

---

## 🔗 与原项目的关系

### 如何回到完整版本？
原项目位置：`/home/ajifang/RL4RL/carla-driving-rl-agent-master/`

### 如何使用CARLA环境？
参考原项目中的：
- `core/carla_env.py` - CARLA环境封装
- `core/carla_agent.py` - CARLA专用agent
- `core/learning.py` - 训练脚本

### 如何合并改动？
如果你在`rl_agent_only`中做了改进，可以这样同步回原项目：
```bash
# 复制改进的agent代码
cp rl_agent_only/agents/ppo.py rl/agents/ppo.py
cp rl_agent_only/networks/networks.py rl/networks/networks.py
```

---

## 📝 版本历史

### v1.0 (2025-12-07)
- ✅ 初始版本
- ✅ 修复NaN bug
- ✅ 修复action双重缩放
- ✅ 修复window_size属性问题
- ✅ 添加完整文档和示例

---

## 🤝 贡献

如果你发现bug或有改进建议，欢迎：
1. 修改代码
2. 添加测试
3. 更新文档
4. 提交改动到原项目

---

**最后更新**: 2025-12-07
**维护者**: [Your Name]
**原项目**: carla-driving-rl-agent
