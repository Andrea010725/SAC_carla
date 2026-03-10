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
    """统一走 train_ppo_with_wandb 的稳定训练链（simple 档）"""
    from train_ppo_with_wandb import train_ppo as train_ppo_unified

    print("[Route] train_ppo_simple -> train_ppo_with_wandb(profile='simple')")
    return train_ppo_unified(profile="simple")


if __name__ == "__main__":
    model_path = train_ppo()
    print(f"\n下一步: 使用训练好的模型")
    print(f"  模型路径: '{model_path}'")
