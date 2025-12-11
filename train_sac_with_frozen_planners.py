"""训练SAC - 使用Frozen Planners (Phase 3)"""
import sys
sys.path.insert(0, '/home/ajifang/SAC_carla')

from main import TrainPipeline
from config import Config


def train_sac_frozen():
    """训练SAC,所有planner冻结"""
    print("=" * 70)
    print("Phase 3: 训练SAC (Frozen Planners)")
    print("=" * 70)

    # 创建配置
    config = Config()

    # ✅ 关键: 使用训练好的PPO
    config.use_ppo = True
    config.ppo_model_path = "weights/ppo-carla-standalone"
    config.ppo_action_dim = 2

    # SAC训练参数
    config.train_total_steps = 5e5  # 训练50万步
    config.warmup_steps = 1e3       # 预热1000步

    print("\n配置:")
    print(f"  - PPO模型: {config.ppo_model_path}")
    print(f"  - 训练步数: {config.train_total_steps}")
    print(f"  - Planner bins: {config.planner_bins}")

    print("\n三个Planner状态:")
    print("  - Rule Planner: ✅ 规则控制 (不需要训练)")
    print("  - IL Planner: ✅ 模仿学习 (已训练好)")
    print("  - RL Planner (PPO): ✅ 已在Phase 1单独训练")

    # 创建训练pipeline
    pipeline = TrainPipeline(config)

    print("\n开始训练SAC...")
    print("SAC将学习何时使用 Rule/IL/RL(PPO) planner")
    print("-" * 70)

    # 训练
    try:
        pipeline.train()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")

    print("\n✅ SAC训练完成!")
    print("\n完整训练流程:")
    print("  Phase 1: ✅ PPO单独训练完成")
    print("  Phase 2: ✅ 三个planner冻结")
    print("  Phase 3: ✅ SAC学会选择最优planner")


if __name__ == "__main__":
    print("\n⚠️  重要提示:")
    print("  - 确保已完成 Phase 1: python train_ppo_standalone.py")
    print("  - 确保PPO模型存在于: weights/ppo-carla-standalone/")
    print("\n按 Ctrl+C 可随时中断训练\n")

    import os
    if not os.path.exists("weights/ppo-carla-standalone"):
        print("❌ 错误: 未找到PPO模型!")
        print("   请先运行: python train_ppo_standalone.py")
        exit(1)

    train_sac_frozen()
