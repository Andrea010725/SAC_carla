"""
PPO 训练实时可视化监控工具
实时显示：reward、episode 长度、entropy、policy loss、value loss 等指标
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque
import threading
import time
import json
import os


class PPOTrainingMonitor:
    """PPO 训练实时监控和可视化"""

    def __init__(self, log_file="training_log.json", window_size=100, update_interval=1000):
        """
        Args:
            log_file: 训练日志文件路径
            window_size: 滑动窗口大小（显示最近N个episode）
            update_interval: 图表更新间隔（毫秒）
        """
        self.log_file = log_file
        self.window_size = window_size
        self.update_interval = update_interval

        # 数据缓存
        self.episodes = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.entropies = deque(maxlen=window_size)
        self.avg_speeds = deque(maxlen=window_size)
        self.collision_rates = deque(maxlen=window_size)

        # 统计数据
        self.total_episodes = 0
        self.best_reward = -np.inf
        self.avg_reward_100 = 0.0

        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('PPO Training Monitor - Parked Obstacles Scenario',
                          fontsize=16, fontweight='bold')

        # 创建子图布局
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # 第一行：Reward 相关
        self.ax_reward = self.fig.add_subplot(gs[0, :2])
        self.ax_reward_dist = self.fig.add_subplot(gs[0, 2])

        # 第二行：Episode 和 Loss
        self.ax_length = self.fig.add_subplot(gs[1, 0])
        self.ax_policy_loss = self.fig.add_subplot(gs[1, 1])
        self.ax_value_loss = self.fig.add_subplot(gs[1, 2])

        # 第三行：Entropy 和其他
        self.ax_entropy = self.fig.add_subplot(gs[2, 0])
        self.ax_speed = self.fig.add_subplot(gs[2, 1])
        self.ax_collision = self.fig.add_subplot(gs[2, 2])

        # 初始化图表
        self._init_plots()

        # 动画对象
        self.ani = None

    def _init_plots(self):
        """初始化所有子图"""
        # Reward 趋势
        self.ax_reward.set_title('Episode Reward (窗口平均)', fontsize=12)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        self.line_reward, = self.ax_reward.plot([], [], 'b-', linewidth=2, label='Raw')
        self.line_reward_ma, = self.ax_reward.plot([], [], 'r-', linewidth=2, label='MA(10)')
        self.ax_reward.legend(loc='upper left')

        # Reward 分布
        self.ax_reward_dist.set_title('Reward 分布', fontsize=12)
        self.ax_reward_dist.set_xlabel('Reward')
        self.ax_reward_dist.set_ylabel('Frequency')

        # Episode 长度
        self.ax_length.set_title('Episode Length', fontsize=12)
        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.ax_length.grid(True, alpha=0.3)
        self.line_length, = self.ax_length.plot([], [], 'g-', linewidth=2)

        # Policy Loss
        self.ax_policy_loss.set_title('Policy Loss', fontsize=12)
        self.ax_policy_loss.set_xlabel('Episode')
        self.ax_policy_loss.set_ylabel('Loss')
        self.ax_policy_loss.grid(True, alpha=0.3)
        self.line_policy_loss, = self.ax_policy_loss.plot([], [], 'm-', linewidth=2)

        # Value Loss
        self.ax_value_loss.set_title('Value Loss', fontsize=12)
        self.ax_value_loss.set_xlabel('Episode')
        self.ax_value_loss.set_ylabel('Loss')
        self.ax_value_loss.grid(True, alpha=0.3)
        self.line_value_loss, = self.ax_value_loss.plot([], [], 'c-', linewidth=2)

        # Entropy
        self.ax_entropy.set_title('Policy Entropy (探索度)', fontsize=12)
        self.ax_entropy.set_xlabel('Episode')
        self.ax_entropy.set_ylabel('Entropy')
        self.ax_entropy.grid(True, alpha=0.3)
        self.line_entropy, = self.ax_entropy.plot([], [], 'orange', linewidth=2)

        # Average Speed
        self.ax_speed.set_title('Average Speed (m/s)', fontsize=12)
        self.ax_speed.set_xlabel('Episode')
        self.ax_speed.set_ylabel('Speed')
        self.ax_speed.grid(True, alpha=0.3)
        self.line_speed, = self.ax_speed.plot([], [], 'purple', linewidth=2)

        # Collision Rate
        self.ax_collision.set_title('Collision Rate (%)', fontsize=12)
        self.ax_collision.set_xlabel('Episode')
        self.ax_collision.set_ylabel('Rate')
        self.ax_collision.grid(True, alpha=0.3)
        self.line_collision, = self.ax_collision.plot([], [], 'red', linewidth=2)

    def _moving_average(self, data, window=10):
        """计算移动平均"""
        if len(data) < window:
            return data
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    def update_plot(self, frame):
        """更新图表（动画回调）"""
        # 读取最新数据
        if not self._load_latest_data():
            return

        if len(self.episodes) == 0:
            return

        episodes_arr = np.array(self.episodes)
        rewards_arr = np.array(self.rewards)

        # 更新 Reward 趋势
        self.line_reward.set_data(episodes_arr, rewards_arr)
        if len(rewards_arr) >= 10:
            ma_rewards = self._moving_average(rewards_arr, 10)
            ma_episodes = episodes_arr[9:]
            self.line_reward_ma.set_data(ma_episodes, ma_rewards)

        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        # 更新 Reward 分布
        self.ax_reward_dist.clear()
        self.ax_reward_dist.hist(rewards_arr, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax_reward_dist.set_xlabel('Reward')
        self.ax_reward_dist.set_ylabel('Frequency')
        self.ax_reward_dist.set_title('Reward 分布', fontsize=12)

        # 更新 Episode 长度
        if len(self.episode_lengths) > 0:
            self.line_length.set_data(episodes_arr, np.array(self.episode_lengths))
            self.ax_length.relim()
            self.ax_length.autoscale_view()

        # 更新 Policy Loss
        if len(self.policy_losses) > 0:
            self.line_policy_loss.set_data(episodes_arr, np.array(self.policy_losses))
            self.ax_policy_loss.relim()
            self.ax_policy_loss.autoscale_view()

        # 更新 Value Loss
        if len(self.value_losses) > 0:
            self.line_value_loss.set_data(episodes_arr, np.array(self.value_losses))
            self.ax_value_loss.relim()
            self.ax_value_loss.autoscale_view()

        # 更新 Entropy
        if len(self.entropies) > 0:
            self.line_entropy.set_data(episodes_arr, np.array(self.entropies))
            self.ax_entropy.relim()
            self.ax_entropy.autoscale_view()

        # 更新 Speed
        if len(self.avg_speeds) > 0:
            self.line_speed.set_data(episodes_arr, np.array(self.avg_speeds))
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()

        # 更新 Collision Rate
        if len(self.collision_rates) > 0:
            self.line_collision.set_data(episodes_arr, np.array(self.collision_rates))
            self.ax_collision.relim()
            self.ax_collision.autoscale_view()

        # 更新标题（显示统计信息）
        if len(rewards_arr) > 0:
            self.avg_reward_100 = np.mean(rewards_arr[-min(100, len(rewards_arr)):])
            self.best_reward = np.max(rewards_arr)
            title = (f'PPO Training Monitor - Episode {self.total_episodes} | '
                    f'Avg(100): {self.avg_reward_100:.2f} | '
                    f'Best: {self.best_reward:.2f}')
            self.fig.suptitle(title, fontsize=16, fontweight='bold')

    def _load_latest_data(self):
        """从日志文件加载最新数据"""
        if not os.path.exists(self.log_file):
            return False

        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)

            # 提取数据
            episodes_data = data.get('episodes', [])
            if not episodes_data:
                return False

            # 只保留最新的 window_size 个数据
            start_idx = max(0, len(episodes_data) - self.window_size)
            episodes_data = episodes_data[start_idx:]

            # 更新数据
            self.episodes.clear()
            self.rewards.clear()
            self.episode_lengths.clear()
            self.policy_losses.clear()
            self.value_losses.clear()
            self.entropies.clear()
            self.avg_speeds.clear()
            self.collision_rates.clear()

            for ep_data in episodes_data:
                self.episodes.append(ep_data['episode'])
                self.rewards.append(ep_data.get('reward', 0))
                self.episode_lengths.append(ep_data.get('length', 0))
                self.policy_losses.append(ep_data.get('policy_loss', 0))
                self.value_losses.append(ep_data.get('value_loss', 0))
                self.entropies.append(ep_data.get('entropy', 0))
                self.avg_speeds.append(ep_data.get('avg_speed', 0))
                self.collision_rates.append(ep_data.get('collision_rate', 0))

            self.total_episodes = episodes_data[-1]['episode'] if episodes_data else 0
            return True

        except Exception as e:
            print(f"[Monitor] 读取日志文件失败: {e}")
            return False

    def start(self):
        """启动实时监控"""
        print(f"[Monitor] 启动训练监控...")
        print(f"[Monitor] 日志文件: {self.log_file}")
        print(f"[Monitor] 窗口大小: {self.window_size}")
        print(f"[Monitor] 更新间隔: {self.update_interval}ms")

        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=self.update_interval,
            blit=False
        )

        plt.show()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PPO 训练实时监控')
    parser.add_argument('--log-file', type=str, default='training_log.json',
                        help='训练日志文件路径')
    parser.add_argument('--window', type=int, default=100,
                        help='显示窗口大小（最近N个episode）')
    parser.add_argument('--interval', type=int, default=2000,
                        help='更新间隔（毫秒）')

    args = parser.parse_args()

    # 创建监控器
    monitor = PPOTrainingMonitor(
        log_file=args.log_file,
        window_size=args.window,
        update_interval=args.interval
    )

    # 启动监控
    monitor.start()


if __name__ == '__main__':
    main()
