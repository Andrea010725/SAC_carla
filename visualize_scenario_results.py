"""
场景训练结果可视化工具

功能：
1. 读取训练日志 training_log_4scenarios.json
2. 生成各场景的性能对比图表
3. 分析训练过程中的场景表现变化

用法：
    python visualize_scenario_results.py
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from collections import defaultdict
from datetime import datetime
import os


def load_training_log(log_file="training_log_4scenarios.json"):
    """加载训练日志"""
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return None

    with open(log_file, "r") as f:
        data = json.load(f)

    print(f"✅ 加载日志成功: {len(data.get('episodes', []))} episodes")
    return data


def analyze_by_scenario(log_data):
    """按场景分析数据"""
    episodes = log_data.get("episodes", [])

    scenario_data = defaultdict(lambda: {
        "episodes": [],
        "rewards": [],
        "success": [],
        "collision": [],
        "steps": [],
    })

    for ep in episodes:
        scenario = ep.get("scenario", "unknown")

        scenario_data[scenario]["episodes"].append(ep.get("episode", 0))
        scenario_data[scenario]["rewards"].append(ep.get("total_reward", 0))
        scenario_data[scenario]["success"].append(1 if ep.get("success", False) else 0)
        scenario_data[scenario]["collision"].append(1 if ep.get("collision", False) else 0)
        scenario_data[scenario]["steps"].append(ep.get("steps", 0))

    return dict(scenario_data)


def plot_scenario_comparison(scenario_data, output_dir="./scenario_plots"):
    """生成场景对比图表"""
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(scenario_data.keys())
    if not scenarios:
        print("❌ 没有场景数据")
        return

    # 图1: 成功率对比
    fig, ax = plt.subplots(figsize=(10, 6))

    success_rates = []
    for scenario in scenarios:
        success = scenario_data[scenario]["success"]
        if success:
            success_rate = np.mean(success) * 100
            success_rates.append(success_rate)
        else:
            success_rates.append(0)

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(scenarios, success_rates, color=colors[:len(scenarios)])

    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Scenario Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate_comparison.png", dpi=150)
    print(f"✅ 保存: {output_dir}/success_rate_comparison.png")
    plt.close()

    # 图2: 碰撞率对比
    fig, ax = plt.subplots(figsize=(10, 6))

    collision_rates = []
    for scenario in scenarios:
        collision = scenario_data[scenario]["collision"]
        if collision:
            collision_rate = np.mean(collision) * 100
            collision_rates.append(collision_rate)
        else:
            collision_rates.append(0)

    bars = ax.bar(scenarios, collision_rates, color='#e74c3c', alpha=0.7)

    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.set_title('Scenario Collision Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, collision_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/collision_rate_comparison.png", dpi=150)
    print(f"✅ 保存: {output_dir}/collision_rate_comparison.png")
    plt.close()

    # 图3: 平均奖励对比
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_rewards = []
    for scenario in scenarios:
        rewards = scenario_data[scenario]["rewards"]
        if rewards:
            avg_reward = np.mean(rewards)
            avg_rewards.append(avg_reward)
        else:
            avg_rewards.append(0)

    bars = ax.bar(scenarios, avg_rewards, color='#3498db', alpha=0.7)

    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Scenario Average Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, reward in zip(bars, avg_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_reward_comparison.png", dpi=150)
    print(f"✅ 保存: {output_dir}/avg_reward_comparison.png")
    plt.close()

    # 图4: 训练过程中的成功率变化（滑动窗口）
    fig, ax = plt.subplots(figsize=(12, 6))

    window_size = 10
    for scenario in scenarios:
        success = scenario_data[scenario]["success"]
        episodes = scenario_data[scenario]["episodes"]

        if len(success) >= window_size:
            # 计算滑动窗口平均
            success_smooth = []
            for i in range(len(success) - window_size + 1):
                window = success[i:i+window_size]
                success_smooth.append(np.mean(window) * 100)

            ep_smooth = episodes[window_size-1:]
            ax.plot(ep_smooth, success_smooth, marker='o', markersize=4,
                   label=scenario, linewidth=2, alpha=0.8)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%) - 10-episode window', fontsize=12)
    ax.set_title('Training Progress: Success Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate_over_time.png", dpi=150)
    print(f"✅ 保存: {output_dir}/success_rate_over_time.png")
    plt.close()

    # 图5: 综合对比雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 准备数据
    categories = ['Success\nRate', 'Avg\nReward', 'Avg\nSteps', 'Low\nCollision']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)

    for i, scenario in enumerate(scenarios):
        # 归一化指标到0-100
        success_rate = np.mean(scenario_data[scenario]["success"]) * 100
        avg_reward = np.mean(scenario_data[scenario]["rewards"])
        avg_reward_norm = min(100, max(0, (avg_reward + 100) / 4))  # 假设reward范围-100到300
        avg_steps = np.mean(scenario_data[scenario]["steps"])
        avg_steps_norm = min(100, (avg_steps / 512) * 100)  # 归一化到512步
        collision_rate = np.mean(scenario_data[scenario]["collision"]) * 100
        low_collision = 100 - collision_rate

        values = [success_rate, avg_reward_norm, avg_steps_norm, low_collision]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Scenario Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_radar.png", dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {output_dir}/performance_radar.png")
    plt.close()


def generate_text_report(scenario_data, output_file="scenario_analysis_report.txt"):
    """生成文本分析报告"""
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("场景训练分析报告\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for scenario, data in scenario_data.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"场景: {scenario}\n")
            f.write(f"{'='*70}\n")

            num_episodes = len(data["episodes"])
            success_rate = np.mean(data["success"]) * 100 if data["success"] else 0
            collision_rate = np.mean(data["collision"]) * 100 if data["collision"] else 0
            avg_reward = np.mean(data["rewards"]) if data["rewards"] else 0
            avg_steps = np.mean(data["steps"]) if data["steps"] else 0

            f.write(f"\n训练统计:\n")
            f.write(f"  - 训练Episodes: {num_episodes}\n")
            f.write(f"  - 成功率: {success_rate:.2f}%\n")
            f.write(f"  - 碰撞率: {collision_rate:.2f}%\n")
            f.write(f"  - 平均奖励: {avg_reward:.2f}\n")
            f.write(f"  - 平均步数: {avg_steps:.1f}\n")

            # 难度评估
            if success_rate >= 80:
                difficulty = "⭐⭐ 简单"
            elif success_rate >= 60:
                difficulty = "⭐⭐⭐ 中等"
            elif success_rate >= 40:
                difficulty = "⭐⭐⭐⭐ 困难"
            else:
                difficulty = "⭐⭐⭐⭐⭐ 非常困难"

            f.write(f"  - 难度评估: {difficulty}\n")

            # 训练趋势
            if len(data["success"]) >= 20:
                early_success = np.mean(data["success"][:10]) * 100
                late_success = np.mean(data["success"][-10:]) * 100
                improvement = late_success - early_success

                f.write(f"\n训练趋势:\n")
                f.write(f"  - 初期成功率 (前10 episodes): {early_success:.1f}%\n")
                f.write(f"  - 后期成功率 (后10 episodes): {late_success:.1f}%\n")
                f.write(f"  - 提升幅度: {improvement:+.1f}%\n")

                if improvement > 20:
                    f.write(f"  - 评价: ✅ 显著提升\n")
                elif improvement > 10:
                    f.write(f"  - 评价: ✅ 稳定提升\n")
                elif improvement > 0:
                    f.write(f"  - 评价: ⚠️ 轻微提升\n")
                else:
                    f.write(f"  - 评价: ❌ 需要调整\n")

        # 总体统计
        f.write(f"\n\n{'='*70}\n")
        f.write(f"总体统计\n")
        f.write(f"{'='*70}\n")

        total_episodes = sum(len(data["episodes"]) for data in scenario_data.values())
        total_success = sum(sum(data["success"]) for data in scenario_data.values())
        total_collision = sum(sum(data["collision"]) for data in scenario_data.values())

        overall_success_rate = (total_success / total_episodes * 100) if total_episodes > 0 else 0
        overall_collision_rate = (total_collision / total_episodes * 100) if total_episodes > 0 else 0

        f.write(f"\n  - 总Episodes: {total_episodes}\n")
        f.write(f"  - 总体成功率: {overall_success_rate:.2f}%\n")
        f.write(f"  - 总体碰撞率: {overall_collision_rate:.2f}%\n")

        # 场景难度排名
        f.write(f"\n场景难度排名 (按成功率从高到低):\n")
        sorted_scenarios = sorted(scenario_data.items(),
                                 key=lambda x: np.mean(x[1]["success"]) if x[1]["success"] else 0,
                                 reverse=True)

        for i, (scenario, data) in enumerate(sorted_scenarios, 1):
            success_rate = np.mean(data["success"]) * 100 if data["success"] else 0
            f.write(f"  {i}. {scenario:<30} {success_rate:>6.1f}%\n")

    print(f"✅ 保存文本报告: {output_file}")


def main():
    print("="*70)
    print("场景训练结果可视化")
    print("="*70)

    # 加载日志
    log_data = load_training_log("training_log_4scenarios.json")
    if not log_data:
        print("\n⚠️ 请先运行训练生成日志文件")
        return

    # 按场景分析
    print("\n[1] 分析场景数据...")
    scenario_data = analyze_by_scenario(log_data)

    if not scenario_data:
        print("❌ 没有场景数据")
        return

    print(f"✅ 发现 {len(scenario_data)} 个场景:")
    for scenario in scenario_data.keys():
        print(f"  - {scenario}: {len(scenario_data[scenario]['episodes'])} episodes")

    # 生成图表
    print("\n[2] 生成可视化图表...")
    plot_scenario_comparison(scenario_data)

    # 生成文本报告
    print("\n[3] 生成文本报告...")
    generate_text_report(scenario_data)

    print("\n" + "="*70)
    print("✅ 可视化完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - ./scenario_plots/success_rate_comparison.png")
    print("  - ./scenario_plots/collision_rate_comparison.png")
    print("  - ./scenario_plots/avg_reward_comparison.png")
    print("  - ./scenario_plots/success_rate_over_time.png")
    print("  - ./scenario_plots/performance_radar.png")
    print("  - scenario_analysis_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
