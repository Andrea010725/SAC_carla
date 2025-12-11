# reward_config_aggressive.py
# 激进版Reward配置 - "尽可能快，适当冒险"

from dataclasses import dataclass

@dataclass
class AggressiveRewardWeights:
    """激进驾驶的Reward权重配置"""

    # ============ 顶层权重 ============
    # 核心思想: Efficiency >> Safety > Comfort
    w_safety: float = 0.5      # 降低 (原1.0) - 适度安全即可
    w_comfort: float = 0.05    # 大幅降低 (原0.2) - 允许激进驾驶
    w_efficiency: float = 0.8  # 🔧 适中 (原1.0) - 防止梯度爆炸

    # ============ Safety 内部权重 ============
    # 只惩罚真正危险的行为
    w_collision: float = 10.0   # 提高 (原5.0) - 碰撞仍需避免
    w_offroad: float = 5.0      # 提高 (原3.0) - 离道也要避免

    # 以下大幅降低，允许偏移和航向误差
    w_deviation: float = 0.2    # 大幅降低 (原1.0) - 允许压线
    w_heading: float = 0.2      # 大幅降低 (原1.0) - 允许斜着开
    w_obs_dist: float = 0.5     # 降低 (原2.0) - 可以跟车近一些
    w_speed_pen: float = 5.0    # 🔧 超级提高 (原3.0) - 极度严厉惩罚低速!

    # ============ Comfort 内部权重 ============
    # 大幅降低所有comfort惩罚，允许激进操作
    w_steer: float = 0.1        # 大幅降低 (原0.5) - 允许急转
    w_jerk: float = 0.2         # 大幅降低 (原1.0) - 允许急加速
    w_brake: float = 0.1        # 大幅降低 (原0.5) - 允许急刹

    # ============ Efficiency 内部权重 ============
    # 大幅提高，鼓励快速前进
    w_speed: float = 2.0        # 🔧 适中 (原3.0) - 防止梯度爆炸
    w_progress: float = 3.0     # 🔧 超级提高 (原2.0) - 极度强烈鼓励前进!


@dataclass
class AggressiveRewardThresholds:
    """激进驾驶的阈值配置"""

    # ============ 距离相关 ============
    time_headway_safe: float = 1.2      # 降低 (原1.8) - 允许跟车更近
    lane_dev_max: float = 2.5           # 提高 (原1.5) - 允许更大偏移
    heading_diff_max_deg: float = 45.0  # 提高 (原30.0) - 允许更大航向差

    # ============ Comfort阈值 ============
    # 提高阈值，允许更激进的操作
    jerk_thr: float = 1.5               # 提高 (原0.5) - 允许更大加速度变化
    brake_thr: float = 5.0              # 提高 (原3.0) - 允许更强制动
    steer_rate_max: float = 1.0         # 提高 (原0.5) - 允许更快转向

    # ============ 速度相关 ============
    min_speed_ratio: float = 0.85       # 🔧 超级提高 (原0.8) - 低于85%限速就惩罚!
    max_speed_multiplier: float = 1.5   # 🚀 提高 (原1.3) - 允许超速50%


# ============ 预设配置 ============

def get_aggressive_config():
    """获取激进版完整配置"""
    from reward_monitor import RewardMonitorConfig

    config = RewardMonitorConfig(
        weights=AggressiveRewardWeights(),
        thresholds=AggressiveRewardThresholds(),
        default_dt=0.05,
        enable_vis=True,
        vis_update_interval=2,
        save_dir="./reward_logs"
    )
    return config


# ============ 对比表格 ============

def print_comparison():
    """打印原版vs激进版的对比"""
    print("=" * 80)
    print("Reward配置对比: 原版 vs 激进版")
    print("=" * 80)

    print("\n【顶层权重】")
    print(f"{'指标':<20} {'原版':>10} {'激进版':>10} {'变化':>15}")
    print("-" * 80)
    print(f"{'Safety':<20} {1.0:>10.2f} {0.5:>10.2f} {'↓50%':>15}")
    print(f"{'Comfort':<20} {0.2:>10.2f} {0.05:>10.2f} {'↓75%':>15}")
    print(f"{'Efficiency':<20} {0.1:>10.2f} {1.0:>10.2f} {'↑900%':>15}")

    print("\n【Safety子项】")
    print(f"{'Collision':<20} {5.0:>10.1f} {10.0:>10.1f} {'↑100%':>15}")
    print(f"{'Offroad':<20} {3.0:>10.1f} {5.0:>10.1f} {'↑67%':>15}")
    print(f"{'Deviation':<20} {1.0:>10.1f} {0.2:>10.1f} {'↓80%':>15}")
    print(f"{'Heading':<20} {1.0:>10.1f} {0.2:>10.1f} {'↓80%':>15}")
    print(f"{'Obs Distance':<20} {2.0:>10.1f} {0.5:>10.1f} {'↓75%':>15}")
    print(f"{'Speed Penalty':<20} {1.0:>10.1f} {0.1:>10.1f} {'↓90%':>15}")

    print("\n【Comfort子项】")
    print(f"{'Steer Cost':<20} {0.5:>10.1f} {0.1:>10.1f} {'↓80%':>15}")
    print(f"{'Jerk Cost':<20} {1.0:>10.1f} {0.2:>10.1f} {'↓80%':>15}")
    print(f"{'Brake Cost':<20} {0.5:>10.1f} {0.1:>10.1f} {'↓80%':>15}")

    print("\n【Efficiency子项】")
    print(f"{'Speed Reward':<20} {0.7:>10.1f} {1.5:>10.1f} {'↑114%':>15}")
    print(f"{'Progress':<20} {0.3:>10.1f} {1.0:>10.1f} {'↑233%':>15}")

    print("\n【阈值变化】")
    print(f"{'安全时距 (秒)':<20} {1.8:>10.1f} {1.2:>10.1f} {'↓33%':>15}")
    print(f"{'最大偏移 (米)':<20} {1.5:>10.1f} {2.5:>10.1f} {'↑67%':>15}")
    print(f"{'最大航向差 (度)':<20} {30.0:>10.1f} {45.0:>10.1f} {'↑50%':>15}")
    print(f"{'Jerk阈值 (m/s³)':<20} {0.5:>10.1f} {1.5:>10.1f} {'↑200%':>15}")
    print(f"{'制动阈值 (m/s²)':<20} {3.0:>10.1f} {5.0:>10.1f} {'↑67%':>15}")

    print("\n" + "=" * 80)
    print("总结: 激进版大幅提升速度奖励，降低舒适性和部分安全性约束")
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()

    print("\n\n【激进版配置】")
    config = get_aggressive_config()
    print(f"Weights: {config.weights}")
    print(f"Thresholds: {config.thresholds}")
