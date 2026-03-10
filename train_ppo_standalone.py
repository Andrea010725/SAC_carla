"""单独训练PPO Agent - Phase 1"""
import sys
import os

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 添加主项目路径
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 关键: 添加planners目录,这样rl_agent_only可以作为一个包被导入
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


class CarlaGymEnv(gym.Env):
    """CARLA环境的Gym wrapper供PPO训练"""

    def __init__(self, config, logger=None):
        self.config = config
        self.carla_env = CarlaEnv(config, 2000, 8000)
        self.logger = logger  # 训练日志记录器

        # 定义observation space (9维)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        # 定义action space (2维: [throttle_brake, steer])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.max_episode_steps = config.max_episode_steps

        # 🚗 解决车辆静止问题：添加action bias
        self.current_episode = 0
        self.action_bias_strength = 0.7  # 🔧 大幅提高 (原0.5) - 强制车辆前进
        self.action_bias_decay = 0.999   # 🔧 进一步减慢衰减 (原0.998) - 更长有效期

        # 📊 用于记录当前episode的统计数据
        self.episode_collision = False

    def reset(self):
        """重置环境"""
        self.carla_env.reset()
        obs = self.carla_env._get_state_obs()
        self.current_episode += 1
        self._step_count = 0  # 重置step计数
        self.episode_collision = False  # 重置碰撞标志
        return obs

    def step(self, action):
        """执行一步"""
        # 🔧 NaN检测 - 如果action是NaN，用安全默认值替代
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"⚠️  检测到NaN/Inf action: {action}, 使用安全默认值 [0.3, 0.0]")
            action = np.array([0.3, 0.0], dtype=np.float32)

        # 🔧 强制裁剪action到合法范围，防止NaN和网络发散
        action = np.clip(action, -1.0, 1.0)

        # 🚗 初期添加action bias，鼓励前进（解决车辆静止问题）
        if self.current_episode < 500:  # 🔧 大幅延长 (原300) - 确保长期有bias
            # 计算当前episode的bias强度（随episode衰减）
            bias = self.action_bias_strength * (self.action_bias_decay ** self.current_episode)
            # 给throttle添加正偏置
            # 注意：需要复制action数组，避免修改原始数组
            action = np.array(action, dtype=np.float32)
            action[0] = float(np.clip(action[0] + bias, -1.0, 1.0))

        # 🔧 新增：强制最小throttle（防止完全静止）
        # 如果throttle在[-0.1, 0.1]之间，强制设为0.2
        if -0.1 <= action[0] <= 0.1:
            action = np.array(action, dtype=np.float32)
            action[0] = 0.2  # 最小油门

            # 🔍 调试输出（前10步）
            if self.current_episode <= 2 and hasattr(self, '_step_count'):
                if self._step_count < 10:
                    print(f"[Debug] Episode {self.current_episode}, Step {self._step_count}: "
                          f"action[0]={action[0]:.3f}, bias={bias:.3f}")
                self._step_count += 1
            elif not hasattr(self, '_step_count'):
                self._step_count = 0

        # ✅ 直接使用CarlaEnv.step() - 包含渲染逻辑
        # CarlaEnv.step()接受的action格式正好是 [throttle_brake, steer]
        # 与PPO输出格式完全一致!
        obs, reward, done, info = self.carla_env.step(action)

        # 📊 记录步骤信息到logger
        if self.logger is not None:
            # 获取当前速度 (obs[8]是velocity的标量值)
            speed = float(obs[8]) if len(obs) > 8 else 0.0

            # 🔧 修复：从info中读取collision（因为carla_env.collision在_get_reward后被清零）
            collision = bool(info.get('collision', 0.0)) if info else False

            # 🔍 调试：打印collision信息
            if self._step_count < 5 or collision:
                print(f"[DEBUG] Step {self._step_count}: info={info}, collision={collision}")

            if collision:
                self.episode_collision = True

            self.logger.log_step(speed=speed, collision=collision)

        return obs, reward, done, info

    # 注意: _compute_reward() 和 _check_done() 方法已不需要
    # 因为 CarlaEnv.step() 内部已经实现了 reward 计算和 done 判断

    def close(self):
        """关闭环境"""
        self.carla_env.close()

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)

    def render(self, mode='human'):
        """渲染 (CARLA自动渲染)"""
        pass


def train_with_logging(agent, env, logger, episodes=1000, timesteps=512, save_every=100):
    """
    带日志记录的PPO训练循环

    Args:
        agent: PPO Agent实例
        env: Gym环境实例
        logger: TrainingLogger实例
        episodes: 训练episode数量
        timesteps: 每个episode的最大步数
        save_every: 每多少episode保存一次模型
    """
    import time
    from planners.rl_agent_only import utils

    # 准备训练
    assert episodes % agent.update_frequency == 0
    agent.memory = agent.get_memory()

    # 用于追踪训练指标
    policy_loss_tracker = 0.0
    value_loss_tracker = 0.0
    entropy_tracker = 0.0
    update_count = 0

    for episode in range(1, episodes + 1):
        # 📊 开始episode日志记录
        logger.start_episode(episode)

        # Episode初始化
        agent.seed_regularization()
        agent.on_episode_start()
        agent.reset()

        # 环境重置
        state = env.reset()
        episode_reward = 0.0
        t0 = time.time()

        # Episode主循环
        for t in range(1, timesteps + 1):
            # 预处理状态
            if isinstance(state, dict):
                state = {f'state_{k}': v for k, v in state.items()}
            state = utils.to_tensor(state)

            # Agent预测
            action, mean, std, log_prob, value = agent.predict(state)
            action_env = agent.convert_action(action)

            # 执行action
            next_state, reward, done, _ = env.step(action_env)
            episode_reward += reward

            # 记录到agent内存
            agent.log(actions=action, action_env=action_env, rewards=reward,
                     distribution_mean=mean, distribution_std=std)
            agent.memory.append(state, action, reward, value, log_prob)
            state = next_state

            # Episode终止检查
            if done or (t == timesteps):
                print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                      f'with reward {round(episode_reward, 3)}.')
                agent.log(timestep=t)

                # 处理最终状态
                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}
                state = utils.to_tensor(state)

                last_value = agent.network.predict_last_value(state, timestep=(t + 1) / timesteps,
                                                             is_terminal=done)
                agent.end_episode(last_value, append=agent.update_frequency > 1)
                break

        # 📊 记录episode reward
        logger.log_episode_reward(episode_reward)

        # 网络更新
        if episode % agent.update_frequency == 0:
            # 捕获训练损失
            temp_policy_loss = []
            temp_value_loss = []
            temp_entropy = []

            # 执行更新（这会修改网络）
            value_batches = agent.get_value_batches()
            policy_batches = agent.get_policy_batches()

            # Policy优化
            for opt_step in range(agent.optimization_steps['policy']):
                for data_batch in policy_batches:
                    agent.seed_regularization()

                    # 获取gradients和loss（这会在内部计算entropy并记录）
                    total_loss, policy_grads = agent.get_policy_gradients(data_batch)

                    # 提取loss（total_loss是一个tuple: (loss, kl_div)）
                    if isinstance(total_loss, tuple):
                        policy_loss_val = float(total_loss[0].numpy())
                    else:
                        policy_loss_val = float(total_loss.numpy())

                    # 检查NaN
                    if not (np.isnan(policy_loss_val) or np.isinf(policy_loss_val)):
                        temp_policy_loss.append(policy_loss_val)

                    # 从agent的日志中提取entropy（policy_objective内部已经计算）
                    # agent.log()在policy_objective中被调用，我们从agent._logs中读取
                    if hasattr(agent, '_logs') and 'entropy' in agent._logs:
                        entropy_val = agent._logs['entropy']
                        if isinstance(entropy_val, (list, tuple)):
                            entropy_val = entropy_val[-1] if entropy_val else 0.0
                        if not (np.isnan(entropy_val) or np.isinf(entropy_val)):
                            temp_entropy.append(float(entropy_val))

                    agent.update_policy(policy_grads)

            # Value优化
            for _ in range(agent.optimization_steps['value']):
                for data_batch in value_batches:
                    agent.seed_regularization()
                    value_loss, value_grads = agent.get_value_gradients(data_batch)

                    # 检查NaN
                    value_loss_val = float(value_loss.numpy())
                    if not (np.isnan(value_loss_val) or np.isinf(value_loss_val)):
                        temp_value_loss.append(value_loss_val)

                    agent.update_value(value_grads)

            # 计算平均损失
            if temp_policy_loss:
                policy_loss_tracker = sum(temp_policy_loss) / len(temp_policy_loss)
            if temp_value_loss:
                value_loss_tracker = sum(temp_value_loss) / len(temp_value_loss)
            if temp_entropy:
                entropy_tracker = sum(temp_entropy) / len(temp_entropy)

            # 清理内存
            agent.memory.delete()
            agent.memory = agent.get_memory()
            update_count += 1

        elif agent.update_frequency > 1:
            # 移除最后的reward和value避免shape问题
            agent.memory.rewards = agent.memory.rewards[:-1]
            agent.memory.values = agent.memory.values[:-1]

        # 📊 记录训练指标到logger
        logger.log_training_metrics(
            policy_loss=policy_loss_tracker,
            value_loss=value_loss_tracker,
            entropy=entropy_tracker
        )

        # 📊 结束episode记录
        logger.end_episode()

        # Logging
        agent.log(episode_rewards=episode_reward)
        agent.write_summaries()

        if agent.should_record:
            agent.record(episode)

        agent.on_episode_end()

        # 保存模型
        if episode % save_every == 0:
            agent.save()

        # 打印摘要（每10个episode）
        if episode % 10 == 0:
            logger.print_summary(window=10)


def train_ppo():
    """统一走 train_ppo_with_wandb 的稳定训练链（standalone 档）"""
    from train_ppo_with_wandb import train_ppo as train_ppo_unified

    print("[Route] train_ppo_standalone -> train_ppo_with_wandb(profile='standalone')")
    return train_ppo_unified(profile="standalone")


if __name__ == "__main__":
    model_path = train_ppo()
    print(f"\n下一步: 在router中使用训练好的模型")
    print(f"  ppo_model_path='{model_path}'")
