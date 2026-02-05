#!/bin/bash
# 启动脚本：使用软件渲染运行训练（避免OpenGL错误）

echo "🚀 启动PPO训练（带pygame可视化）"
echo "================================================"

# 设置环境变量使用软件渲染
export LIBGL_ALWAYS_SOFTWARE=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# 显示配置
echo "✅ 环境变量已设置："
echo "   LIBGL_ALWAYS_SOFTWARE=1 (使用软件渲染)"
echo "   SDL_VIDEODRIVER=dummy (虚拟显示驱动)"
echo ""

# 运行训练
python train_ppo_with_wandb.py

echo ""
echo "================================================"
echo "✅ 训练完成"
