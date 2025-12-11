#!/usr/bin/env python3
"""
RL Agent使用示例
展示如何在自定义环境中使用PPO Agent
"""
import gym
import numpy as np
from .ppo import PPOAgent
from .parameters import LinearDecay

# ============================================
# 1. 定义你的自定义环境
# ============================================
class SimpleEnv(gym.Env):
    """简单的示例环境"""

    def __init__(self):
        super().__init__()

        # 定义动作空间：连续动作 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 定义观测空间：字典形式
        self.observation_space = gym.spaces.Dict({
            'vector_feature': gym.spaces.Box(
                low=0.0, high=1.0, shape=(10,), dtype=np.float32
            ),
            'scalar_feature': gym.spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
        })

        self.timestep = 0
        self.max_timesteps = 100

    def reset(self):
        """重置环境"""
        self.timestep = 0
        return {
            'vector_feature': np.random.rand(10).astype(np.float32),
            'scalar_feature': np.random.rand(3).astype(np.float32)
        }

    def step(self, action):
        """执行一步"""
        self.timestep += 1

        # 下一个状态
        next_state = {
            'vector_feature': np.random.rand(10).astype(np.float32),
            'scalar_feature': np.random.rand(3).astype(np.float32)
        }

        # 奖励（示例：动作的负平方和）
        reward = -np.sum(action ** 2)

        # 是否结束
        done = self.timestep >= self.max_timesteps

        # 额外信息
        info = {'timestep': self.timestep}

        return next_state, reward, done, info

    def render(self, mode='human'):
        """渲染（可选）"""
        pass


# ============================================
# 2. 创建并配置PPO Agent
# ============================================
def create_ppo_agent(env):
    """创建PPO agent"""

    # 配置动态学习率
    policy_lr = LinearDecay(
        initial_value=3e-4,
        final_value=1e-5,
        num_episodes=1000
    )

    value_lr = LinearDecay(
        initial_value=1e-3,
        final_value=1e-5,
        num_episodes=1000
    )

    # 配置网络架构
    network_config = {
        'policy': {
            'units': 128,           # 隐藏层单元数
            'num_layers': 2,        # 隐藏层层数
            'activation': 'relu'    # 激活函数
        },
        'value': {
            'units': 128,
            'num_layers': 2,
            'activation': 'relu'
        }
    }

    # 创建agent
    agent = PPOAgent(
        env=env,
        name='simple-ppo',

        # 学习率
        policy_lr=policy_lr,
        value_lr=value_lr,

        # PPO超参数
        gamma=0.99,                     # 折扣因子
        lambda_=0.95,                   # GAE lambda
        clip_ratio=0.2,                 # PPO clip ratio
        entropy_regularization=0.01,    # 熵正则化系数

        # 优化参数
        optimization_steps=(10, 10),    # (policy_steps, value_steps)
        batch_size=256,                 # batch大小
        clip_norm=(1.0, 1.0),          # 梯度裁剪

        # 训练设置
        update_frequency=1,             # 每N个episode更新一次

        # 网络配置
        network=network_config
    )

    return agent


# ============================================
# 3. 训练Agent
# ============================================
def train_agent():
    """训练示例"""
    print("=" * 80)
    print("🚀 开始训练PPO Agent")
    print("=" * 80)

    # 创建环境
    env = SimpleEnv()
    print(f"\n✅ 环境创建成功")
    print(f"   动作空间: {env.action_space}")
    print(f"   观测空间: {env.observation_space}")

    # 创建agent
    agent = create_ppo_agent(env)
    print(f"\n✅ Agent创建成功")
    print(f"   名称: {agent.name}")

    # 训练
    print(f"\n🎯 开始训练...")
    try:
        agent.learn(
            episodes=100,           # 训练100个episode
            timesteps=100,          # 每个episode最多100步
            save_every=20,          # 每20个episode保存一次
            render_every=False,     # 不渲染
            close=False             # 训练后不关闭环境
        )
        print(f"\n✅ 训练完成！")
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被中断")

    # 保存模型
    print(f"\n💾 保存模型...")
    agent.save()
    print(f"✅ 模型已保存到: {agent.base_path}")

    # 关闭环境
    env.close()


# ============================================
# 4. 评估Agent
# ============================================
def evaluate_agent():
    """评估示例"""
    print("=" * 80)
    print("📊 评估PPO Agent")
    print("=" * 80)

    # 创建环境
    env = SimpleEnv()

    # 创建agent（加载已训练的模型）
    agent = create_ppo_agent(env)

    # 加载模型
    print(f"\n📥 加载模型...")
    try:
        agent.load()
        print(f"✅ 模型加载成功")
    except:
        print(f"⚠️  未找到已保存的模型，使用随机初始化")

    # 评估
    print(f"\n🎯 开始评估...")
    agent.collect(
        episodes=10,            # 评估10个episode
        timesteps=100,          # 每个episode最多100步
        render=False,           # 不渲染
        record_threshold=0.0    # 记录所有episode
    )

    print(f"\n✅ 评估完成！")
    env.close()


# ============================================
# 5. 主程序
# ============================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='PPO Agent示例')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='运行模式：train（训练）或 eval（评估）')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent()
    elif args.mode == 'eval':
        evaluate_agent()


if __name__ == '__main__':
    main()
