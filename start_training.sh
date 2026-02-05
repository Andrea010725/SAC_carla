#!/bin/bash
# 启动PPO训练脚本

echo "=========================================="
echo "启动 PPO 训练"
echo "=========================================="

# 检查CARLA服务器是否运行
if ! pgrep -f "CarlaUE4" > /dev/null; then
    echo "❌ CARLA服务器未运行！"
    echo "请先启动CARLA服务器："
    echo "  cd /home/ajifang/carla"
    echo "  ./CarlaUE4.sh"
    exit 1
fi

echo "✅ CARLA服务器正在运行"

# 创建日志目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo ""
echo "训练日志将保存到: $LOG_FILE"
echo ""
echo "使用以下命令查看训练进度："
echo "  tail -f $LOG_FILE"
echo ""
echo "按 Ctrl+C 停止训练"
echo ""

# 启动训练
python train_ppo_with_wandb.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练结束"
echo "=========================================="
