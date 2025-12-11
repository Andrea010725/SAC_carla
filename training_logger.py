"""
训练日志记录器 - 为可视化监控工具提供数据
输出JSON格式的训练日志，包含episode级别的各种指标
"""
import json
import os
import time
from typing import Dict, Any, Optional
import numpy as np


class TrainingLogger:
    """PPO训练日志记录器"""

    def __init__(self, log_file: str = "training_log.json", auto_save_interval: int = 5):
        """
        Args:
            log_file: 日志文件路径
            auto_save_interval: 自动保存间隔（多少个episode保存一次）
        """
        self.log_file = log_file
        self.auto_save_interval = auto_save_interval

        # 训练数据存储
        self.episodes_data = []

        # 当前episode临时数据
        self.current_episode = {
            "episode": 0,
            "reward": 0.0,
            "length": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "avg_speed": 0.0,
            "collision_count": 0,
            "collision_rate": 0.0,
            "start_time": 0.0,
            "duration": 0.0,
        }

        # 统计累积值（用于计算平均值）
        self._speed_accumulator = []
        self._last_save_episode = 0

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

        # 清空或创建新日志文件
        self._initialize_log_file()

    def _initialize_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'w') as f:
            json.dump({"episodes": []}, f, indent=2)
        print(f"[TrainingLogger] 日志文件初始化: {self.log_file}")

    def start_episode(self, episode_num: int):
        """开始新的episode"""
        self.current_episode = {
            "episode": episode_num,
            "reward": 0.0,
            "length": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "avg_speed": 0.0,
            "collision_count": 0,
            "collision_rate": 0.0,
            "start_time": time.time(),
            "duration": 0.0,
        }
        self._speed_accumulator = []

    def log_step(self, speed: Optional[float] = None, collision: bool = False):
        """记录单步数据

        Args:
            speed: 当前速度 (m/s)
            collision: 是否发生碰撞
        """
        self.current_episode["length"] += 1

        if speed is not None:
            self._speed_accumulator.append(speed)

        if collision:
            self.current_episode["collision_count"] += 1

    def log_episode_reward(self, reward: float):
        """记录episode总reward"""
        self.current_episode["reward"] = float(reward)

    def log_training_metrics(self,
                            policy_loss: Optional[float] = None,
                            value_loss: Optional[float] = None,
                            entropy: Optional[float] = None):
        """记录训练指标（在update后调用）

        Args:
            policy_loss: 策略网络损失
            value_loss: 价值网络损失
            entropy: 策略熵
        """
        if policy_loss is not None:
            self.current_episode["policy_loss"] = float(policy_loss)
        if value_loss is not None:
            self.current_episode["value_loss"] = float(value_loss)
        if entropy is not None:
            self.current_episode["entropy"] = float(entropy)

    def end_episode(self):
        """结束当前episode，计算统计量并保存"""
        # 计算持续时间
        self.current_episode["duration"] = time.time() - self.current_episode["start_time"]

        # 计算平均速度
        if self._speed_accumulator:
            self.current_episode["avg_speed"] = float(np.mean(self._speed_accumulator))

        # 计算碰撞率：只要碰撞了就是100%，否则0%
        # 不再按步数比例计算，而是二值化：碰撞=1，不碰撞=0
        self.current_episode["collision_rate"] = 100.0 if self.current_episode["collision_count"] > 0 else 0.0

        # 添加到历史数据
        self.episodes_data.append(self.current_episode.copy())

        # 自动保存
        if len(self.episodes_data) - self._last_save_episode >= self.auto_save_interval:
            self.save()
            self._last_save_episode = len(self.episodes_data)

    def save(self):
        """保存日志到文件"""
        try:
            data = {"episodes": self.episodes_data}
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
            # print(f"[TrainingLogger] 已保存 {len(self.episodes_data)} 个episode的数据")
        except Exception as e:
            print(f"[TrainingLogger] ⚠️  保存日志失败: {e}")

    def get_latest_stats(self, window: int = 10) -> Dict[str, Any]:
        """获取最近N个episode的统计信息

        Args:
            window: 统计窗口大小

        Returns:
            统计字典
        """
        if not self.episodes_data:
            return {}

        recent = self.episodes_data[-window:]
        return {
            "avg_reward": np.mean([ep["reward"] for ep in recent]),
            "avg_length": np.mean([ep["length"] for ep in recent]),
            "avg_policy_loss": np.mean([ep["policy_loss"] for ep in recent]),
            "avg_value_loss": np.mean([ep["value_loss"] for ep in recent]),
            "avg_entropy": np.mean([ep["entropy"] for ep in recent]),
            "avg_speed": np.mean([ep["avg_speed"] for ep in recent]),
            "collision_rate": np.mean([ep["collision_rate"] for ep in recent]),
        }

    def print_summary(self, window: int = 10):
        """打印最近的训练摘要

        Args:
            window: 统计窗口大小
        """
        if not self.episodes_data:
            return

        stats = self.get_latest_stats(window)
        latest = self.episodes_data[-1]

        # 计算最近N个episode中碰撞的数量
        recent = self.episodes_data[-window:]
        collision_count = sum(1 for ep in recent if ep['collision_count'] > 0)
        collision_emoji = "🚨" if latest['collision_count'] > 0 else "✅"

        print(f"\n{'='*70}")
        print(f"Episode {latest['episode']} 完成 {collision_emoji}")
        print(f"{'='*70}")
        print(f"  当前: Reward={latest['reward']:.2f}, Length={latest['length']}, Speed={latest['avg_speed']:.2f} m/s, Collision={'YES' if latest['collision_count'] > 0 else 'NO'}")
        print(f"  最近{len(recent)}个平均:")
        print(f"    - Reward: {stats['avg_reward']:.2f}")
        print(f"    - Length: {stats['avg_length']:.0f}")
        print(f"    - Policy Loss: {stats['avg_policy_loss']:.4f}")
        print(f"    - Value Loss: {stats['avg_value_loss']:.4f}")
        print(f"    - Entropy: {stats['avg_entropy']:.4f}")
        print(f"    - Speed: {stats['avg_speed']:.2f} m/s")
        print(f"    - Collision Rate: {stats['collision_rate']:.2f}% ({collision_count}/{len(recent)} episodes)")
        print(f"{'='*70}\n")

    def close(self):
        """关闭logger，确保数据已保存"""
        self.save()
        print(f"[TrainingLogger] 训练日志已保存: {self.log_file}")
