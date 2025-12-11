"""
NaN问题诊断脚本 - 找出NaN产生的确切位置
"""
import sys
import os
import numpy as np
import tensorflow as tf

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
planners_path = os.path.join(script_dir, 'planners')
sys.path.insert(0, planners_path)

from config import Config
from carla_base.carla_env import CarlaEnv
from rl_agent_only.agents.ppo import PPOAgent
import gym

print("="*70)
print("NaN诊断脚本")
print("="*70)

# 创建配置
config = Config()
config.scenario = "parked_obstacles"
config.render = False  # 关闭渲染加速

# 创建环境
print("\n[1] 创建环境...")
class SimpleEnv(gym.Env):
    def __init__(self, config):
        self.carla_env = CarlaEnv(config, 2000, 8000)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        self.carla_env.reset()
        obs = self.carla_env._get_state_obs()
        print(f"  Reset obs: {obs}")
        print(f"  Obs有NaN: {np.any(np.isnan(obs))}")
        return obs

    def step(self, action):
        obs, reward, done, info = self.carla_env.step(action)
        return obs, reward, done, info

    def close(self):
        self.carla_env.close()

env = SimpleEnv(config)
print("✅ 环境创建成功")

# 创建Agent（不加载权重）
print("\n[2] 创建PPO Agent（随机初始化）...")
agent = PPOAgent(
    env=env,
    policy_lr=5e-5,
    value_lr=1e-4,
    gamma=0.99,
    lambda_=0.95,
    clip_ratio=0.2,
    entropy_regularization=0.05,
    optimization_steps=(10, 10),
    batch_size=256,
    update_frequency=1,
    name='ppo-debug',
    load=False  # 不加载权重
)
print("✅ Agent创建成功")

# 测试第一次预测
print("\n[3] 测试第一次预测...")
try:
    state = env.reset()
    print(f"  State shape: {state.shape}")
    print(f"  State: {state}")
    print(f"  State有NaN: {np.any(np.isnan(state))}")

    # 转换为tensor
    from rl_agent_only import utils
    state_tensor = utils.to_tensor(state)
    print(f"  State tensor: {state_tensor}")

    # 预测
    print("\n  调用agent.predict()...")
    action, mean, std, log_prob, value = agent.predict(state_tensor)

    print(f"  ✅ Action: {action.numpy()}")
    print(f"  Mean: {mean.numpy()}")
    print(f"  Std: {std.numpy()}")
    print(f"  Value: {value.numpy()}")

    # 检查NaN
    if np.any(np.isnan(action.numpy())):
        print("\n  ❌ Action包含NaN!")
        print(f"  Mean有NaN: {np.any(np.isnan(mean.numpy()))}")
        print(f"  Std有NaN: {np.any(np.isnan(std.numpy()))}")

        # 检查网络权重
        print("\n  检查Policy网络权重...")
        for i, w in enumerate(agent.network.policy.trainable_variables):
            has_nan = tf.reduce_any(tf.math.is_nan(w))
            print(f"    Layer {i}: {w.name}, shape={w.shape}, has_NaN={has_nan.numpy()}")
    else:
        print("\n  ✅ Action正常，无NaN")

except Exception as e:
    print(f"\n  ❌ 预测失败: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()

print("\n" + "="*70)
print("诊断完成")
print("="*70)
