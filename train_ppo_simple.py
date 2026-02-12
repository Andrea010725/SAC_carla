"""简化版PPO训练 - 使用agent.learn()避免自定义循环的问题"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

planners_path = os.path.join(script_dir, 'planners')
if planners_path not in sys.path:
    sys.path.insert(0, planners_path)

# 添加 CARLA PythonAPI 路径（用于导入agents模块和carla模块）
carla_api_path = "/home/ajifang/carla/PythonAPI/carla/"
carla_egg_path = "/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
if carla_api_path not in sys.path:
    sys.path.insert(0, carla_api_path)
if carla_egg_path not in sys.path:
    sys.path.insert(0, carla_egg_path)

import gym
import numpy as np

from config import Config
from carla_base.carla_env import CarlaEnv
from training_logger import TrainingLogger


class LoggingWrapper(gym.Wrapper):
    """包装器：在episode开始和结束时记录日志"""

    def __init__(self, env, logger, agent=None):
        super().__init__(env)
        self.logger = logger
        self.agent = agent  # PPO agent引用，用于提取训练指标
        self.episode_num = 0
        self.episode_reward = 0.0
        self.last_extracted_episode = 0  # 记录上次提取训练指标的episode
        self.cached_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

    def reset(self, **kwargs):
        # 如果不是第一次reset，先结束上一个episode
        if self.episode_num > 0:
            # 尝试从agent的statistics中提取训练指标
            print(f"[DEBUG] Episode {self.episode_num} 结束，尝试提取训练指标...")
            print(f"[DEBUG] self.agent is not None: {self.agent is not None}")
            if self.agent is not None:
                print(f"[DEBUG] hasattr(self.agent, 'statistics'): {hasattr(self.agent, 'statistics')}")

            # 使用缓存的训练指标
            print(f"[DEBUG] Episode {self.episode_num} 使用缓存的训练指标: policy_loss={self.cached_metrics['policy_loss']:.4f}, value_loss={self.cached_metrics['value_loss']:.4f}, entropy={self.cached_metrics['entropy']:.4f}")

            self.logger.log_training_metrics(
                policy_loss=self.cached_metrics['policy_loss'],
                value_loss=self.cached_metrics['value_loss'],
                entropy=self.cached_metrics['entropy']
            )

            self.logger.log_episode_reward(self.episode_reward)
            self.logger.end_episode()
            self.logger.print_summary(window=10)

        # 开始新episode
        self.episode_num += 1
        self.episode_reward = 0.0
        self.logger.start_episode(self.episode_num)

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward

        # 尝试在update后立即提取训练指标（在write_summaries清空之前）
        if self.agent is not None and hasattr(self.agent, 'statistics'):
            self._try_extract_metrics()

        return obs, reward, done, info

    def _try_extract_metrics(self):
        """尝试提取训练指标（在write_summaries清空之前）"""
        try:
            stats = self.agent.statistics
            if not hasattr(stats, 'stats'):
                return

            # 检查是否有新的训练数据（list不为空）
            has_data = False
            for key in ['loss_policy', 'loss_value', 'entropy']:
                if key in stats.stats and stats.stats[key]['list']:
                    has_data = True
                    break

            if has_data and self.episode_num > self.last_extracted_episode:
                # 提取训练指标
                policy_loss = self._get_latest_stat(stats, 'loss_policy')
                value_loss = self._get_latest_stat(stats, 'loss_value')
                entropy = self._get_latest_stat(stats, 'entropy')

                # 如果 loss_policy 为0，尝试从 loss_total 提取
                if policy_loss == 0.0:
                    policy_loss = self._get_latest_stat(stats, 'loss_total')

                # 缓存这些指标
                if policy_loss > 0 or value_loss > 0 or entropy > 0:
                    self.cached_metrics = {
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        'entropy': entropy
                    }
                    self.last_extracted_episode = self.episode_num
                    print(f"[DEBUG] 成功提取Episode {self.episode_num}的训练指标: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, entropy={entropy:.4f}")

        except Exception as e:
            pass  # 静默失败，不影响训练

    def _get_latest_stat(self, stats, key):
        """从statistics对象中提取最新的统计值"""
        # 🔍 调试：打印statistics的结构
        if self.episode_num == 2 and key in ['loss_policy', 'entropy', 'loss_total']:  # 只打印关键的keys
            print(f"[DEBUG] 尝试提取 {key}:")
            if hasattr(stats, 'stats') and key in stats.stats:
                v = stats.stats[key]
                if isinstance(v, dict):
                    print(f"[DEBUG]   {key}: step={v.get('step', 'N/A')}, list长度={len(v.get('list', []))}")
                    if 'list' in v and len(v['list']) > 0:
                        sample = v['list'][-1]
                        print(f"[DEBUG]   最后一个值类型={type(sample)}, 值={sample}")
                    else:
                        print(f"[DEBUG]   list为空！")
            else:
                print(f"[DEBUG]   {key} 不在 stats.stats 中")

        # Summary类使用 self.stats 字典存储数据
        # 每个key对应 {'step': X, 'list': [...]}
        if hasattr(stats, 'stats') and key in stats.stats:
            stat_data = stats.stats[key]
            if isinstance(stat_data, dict) and 'list' in stat_data:
                values_list = stat_data['list']
                if len(values_list) > 0:
                    # 获取最后一个值
                    last_value = values_list[-1]

                    # 如果是tuple（例如loss_total返回(loss, kl_div)），取第一个元素
                    if isinstance(last_value, tuple):
                        last_value = last_value[0]

                    # 如果是tensor，转换为float
                    if hasattr(last_value, 'numpy'):
                        return float(last_value.numpy())
                    else:
                        return float(last_value)
        return 0.0


class CarlaGymEnv(gym.Env):
    """CARLA环境的Gym wrapper"""

    def __init__(self, config, logger=None):
        self.config = config
        self.carla_env = CarlaEnv(config, 2000, 8000)
        self.logger = logger  # 训练日志记录器

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.max_episode_steps = config.max_episode_steps
        self.current_episode = 0
        self.action_bias_strength = 0.7
        self.action_bias_decay = 0.999

    def reset(self):
        self.carla_env.reset()
        obs = self.carla_env._get_state_obs()
        self.current_episode += 1
        self._step_count = 0
        self.episode_collision = False  # 重置碰撞标志
        return obs

    def step(self, action):
        # NaN检测
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"⚠️  检测到NaN/Inf action: {action}, 使用安全默认值 [0.3, 0.0]")
            action = np.array([0.3, 0.0], dtype=np.float32)

        # 裁剪
        action = np.clip(action, -1.0, 1.0)

        # Action bias
        if self.current_episode < 500:
            bias = self.action_bias_strength * (self.action_bias_decay ** self.current_episode)
            action = np.array(action, dtype=np.float32)
            action[0] = float(np.clip(action[0] + bias, -1.0, 1.0))

        # 强制最小throttle
        if -0.1 <= action[0] <= 0.1:
            action = np.array(action, dtype=np.float32)
            action[0] = 0.2

        obs, reward, done, info = self.carla_env.step(action)

        # 📊 记录步骤信息到logger
        if self.logger is not None:
            # 获取当前速度 (obs[8]是velocity的标量值)
            speed = float(obs[8]) if len(obs) > 8 else 0.0

            # 🔧 从info中读取collision
            collision = bool(info.get('collision', 0.0)) if info else False

            # 🔍 调试：打印collision信息（前5步或发生碰撞时）
            if self._step_count < 5 or collision:
                collision_value = info.get('collision', 'N/A') if info else 'N/A'
                print(f"[DEBUG] Episode {self.current_episode}, Step {self._step_count}: info['collision']={collision_value}, collision={collision}")

            if collision:
                self.episode_collision = True
                print(f"[DEBUG] 🚨 Collision recorded in CarlaGymEnv, episode {self.current_episode}, step {self._step_count}")

            # 记录到logger
            self.logger.log_step(speed=speed, collision=collision)

            # 🔍 验证logger是否记录了碰撞
            if collision:
                print(f"[DEBUG] logger.log_step called with collision=True, current_episode.collision_count={self.logger.current_episode['collision_count']}")

        self._step_count += 1
        return obs, reward, done, info

    def close(self):
        self.carla_env.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode='human'):
        pass


def train_ppo():
    """训练PPO Agent - 使用原生agent.learn()"""
    print("=" * 70)
    print("PPO简化训练 - 使用agent.learn()")
    print("=" * 70)

    # 创建配置
    config = Config()
    config.scenario = "parked_obstacles"
    config.num_parked_cars = 4
    config.render = True

    print(f"\n🎯 停车障碍场景配置:")
    print(f"  - 停车数量: {config.num_parked_cars}辆")
    print(f"  - 停车间距: {config.parked_car_spacing}m")

    # 加载激进版reward
    print("\n[1] 加载激进版Reward配置...")
    try:
        from reward_config_aggressive import get_aggressive_config
        config.reward_config = get_aggressive_config()
        print("✅ 激进版Reward已加载")
    except Exception as e:
        print(f"⚠️  无法加载激进版reward: {e}")

    # 📊 创建训练日志记录器
    print("\n[2] 创建训练日志记录器...")
    logger = TrainingLogger(log_file="training_log.json", auto_save_interval=5)
    print(f"✅ 日志记录器创建成功")
    print(f"  - 日志文件: training_log.json")
    print(f"  - 自动保存间隔: 每5个episode")

    # 创建环境
    print("\n[3] 创建CARLA Gym环境...")
    base_env = CarlaGymEnv(config, logger=logger)
    env_wrapper = LoggingWrapper(base_env, logger, agent=None)  # 先不传agent
    print(f"✅ 环境创建成功（带日志记录）")

    # 创建PPO Agent
    print("\n[4] 创建PPO Agent...")
    from rl_agent_only.agents.ppo import PPOAgent

    # 检查权重
    weights_dir = './weights/ppo-carla-simple'
    load_existing = False
    if os.path.exists(weights_dir):
        policy_path = os.path.join(weights_dir, 'policy_net.index')
        value_path = os.path.join(weights_dir, 'value_net.index')
        if os.path.exists(policy_path) and os.path.exists(value_path):
            load_existing = True
            print(f"\n✅ 发现已有权重，将加载")
        else:
            print(f"\n⚠️  权重不完整，从头开始")
    else:
        print(f"\n⚠️  未发现权重，从头开始")

    agent = PPOAgent(
        env=env_wrapper,
        policy_lr=5e-5,
        value_lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_ratio=0.2,
        entropy_regularization=0.05,
        optimization_steps=(10, 10),
        batch_size=256,
        update_frequency=1,
        name='ppo-carla-simple',
        load=load_existing
    )

    # 将agent设置到wrapper中，以便提取训练指标
    env_wrapper.agent = agent

    # Hook into agent.statistics.write_summaries to capture metrics before they're cleared
    original_write_summaries = agent.statistics.write_summaries
    def hooked_write_summaries():
        # 在清空之前提取训练指标
        try:
            stats = agent.statistics
            if hasattr(stats, 'stats'):
                policy_loss = 0.0
                value_loss = 0.0
                entropy = 0.0

                # 提取 loss_policy
                if 'loss_policy' in stats.stats and stats.stats['loss_policy']['list']:
                    values = stats.stats['loss_policy']['list']
                    if values:
                        last_val = values[-1]
                        if hasattr(last_val, 'numpy'):
                            policy_loss = float(last_val.numpy())
                        else:
                            policy_loss = float(last_val)

                # 提取 value_loss
                if 'loss_value' in stats.stats and stats.stats['loss_value']['list']:
                    values = stats.stats['loss_value']['list']
                    if values:
                        last_val = values[-1]
                        if hasattr(last_val, 'numpy'):
                            value_loss = float(last_val.numpy())
                        else:
                            value_loss = float(last_val)

                # 提取 entropy
                if 'entropy' in stats.stats and stats.stats['entropy']['list']:
                    values = stats.stats['entropy']['list']
                    if values:
                        last_val = values[-1]
                        if hasattr(last_val, 'numpy'):
                            entropy = float(last_val.numpy())
                        else:
                            entropy = float(last_val)

                # 如果提取到了有效数据，更新缓存
                if policy_loss > 0 or value_loss > 0 or entropy > 0:
                    env_wrapper.cached_metrics = {
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        'entropy': entropy
                    }
                    print(f"[DEBUG] Hook捕获训练指标: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, entropy={entropy:.4f}")
        except Exception as e:
            print(f"[DEBUG] Hook提取失败: {e}")

        # 调用原始的write_summaries
        original_write_summaries()

    agent.statistics.write_summaries = hooked_write_summaries

    print(f"✅ PPO Agent创建成功")
    print(f"  - Policy LR: 5e-5")
    print(f"  - Value LR: 1e-4")
    print(f"  - Entropy: 0.05")
    print(f"  - 模型保存路径: {agent.base_path}")

    # 开始训练
    print("\n[5] 开始训练...")
    print("-" * 70)

    try:
        # 使用原生learn方法
        agent.learn(
            episodes=1000,
            timesteps=512,
            save_every=100,
            render_every=False
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    finally:
        # 保存最后一个episode的日志
        if env_wrapper.episode_num > 0:
            logger.log_episode_reward(env_wrapper.episode_reward)
            logger.end_episode()
        logger.close()
        env_wrapper.close()

    print("\n" + "=" * 70)
    print("✅ PPO训练完成!")
    print("=" * 70)
    print(f"\n模型已保存到: {agent.base_path}")
    print(f"训练日志已保存到: training_log.json")

    return agent.base_path


if __name__ == "__main__":
    model_path = train_ppo()
    print(f"\n下一步: 使用训练好的模型")
    print(f"  模型路径: '{model_path}'")
