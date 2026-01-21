#!/bin/bash
# 快速训练脚本 - 禁用渲染以获得最快速度

echo "=========================================="
echo "🚀 PPO CARLA 快速训练模式"
echo "=========================================="
echo ""

# 1. 检查CARLA是否运行
echo "1️⃣  检查CARLA服务器..."
if pgrep -f "CarlaUE4.*-carla-server" > /dev/null; then
    CARLA_PID=$(pgrep -f "CarlaUE4.*-carla-server" | head -1)
    echo "   ✓ CARLA服务器正在运行 (PID: $CARLA_PID)"
else
    echo "   ⚠️  CARLA服务器未运行！"
    echo ""
    echo "   请先启动CARLA:"
    echo "   cd /home/ajifang/carla"
    echo "   ./CarlaUE4.sh Town05 -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound"
    echo ""
    exit 1
fi

echo ""

# 2. 显示配置
echo "2️⃣  训练配置:"
echo "   - 模式: 快速训练（render=False）"
echo "   - 场景: parked_obstacles (Overtaking)"
echo "   - 地图: Town05"
echo "   - 停车障碍: 4辆车"
echo "   - 最大步数: 500步/episode"
echo "   - 预期速度: ~100倍提升（无渲染）"
echo ""

# 3. 确认开始
echo "3️⃣  准备开始训练..."
echo "   ⚡ 无渲染模式：训练速度快，但看不到画面"
echo "   📊 可通过Wandb实时监控: https://wandb.ai/andrea23/SAC-CARLA-PPO"
echo ""
read -p "   按Enter开始训练，或Ctrl+C取消... " -r

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 4. 运行训练
python train_ppo_with_wandb.py

echo ""
echo "=========================================="
echo "训练结束"
echo "=========================================="
