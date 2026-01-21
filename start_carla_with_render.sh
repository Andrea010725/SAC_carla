#!/bin/bash

# CARLA启动脚本 - 带渲染模式
# 用于可视化训练

echo "========================================"
echo "🚀 启动CARLA服务器 (渲染模式)"
echo "========================================"

# 检查CARLA路径
CARLA_PATH="/home/ajifang/carla"
if [ ! -d "$CARLA_PATH" ]; then
    echo "❌ 错误: CARLA路径不存在: $CARLA_PATH"
    exit 1
fi

# 检查CARLA可执行文件
if [ ! -f "$CARLA_PATH/CarlaUE4.sh" ]; then
    echo "❌ 错误: 找不到CarlaUE4.sh"
    exit 1
fi

# 杀死已有的CARLA进程
echo "🔍 检查已有的CARLA进程..."
CARLA_PIDS=$(pgrep -f "CarlaUE4")
if [ ! -z "$CARLA_PIDS" ]; then
    echo "⚠️  发现已有CARLA进程，正在关闭..."
    pkill -9 -f "CarlaUE4"
    sleep 2
    echo "✅ 已关闭旧进程"
fi

# 切换到CARLA目录
cd "$CARLA_PATH"

# 启动CARLA（带渲染，低画质）
echo ""
echo "🎮 启动参数:"
echo "   - 画质: Low"
echo "   - 渲染: 开启"
echo "   - 端口: 2000"
echo ""
echo "⏳ 正在启动CARLA服务器..."
echo "   (预计需要15-30秒)"
echo ""

# 启动CARLA
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600 &

CARLA_PID=$!
echo "✅ CARLA已启动 (PID: $CARLA_PID)"
echo ""
echo "⏳ 等待服务器初始化..."

# 等待CARLA完全启动
sleep 20

# 检查进程是否还在运行
if ps -p $CARLA_PID > /dev/null; then
    echo "✅ CARLA服务器运行正常"
    echo ""
    echo "========================================"
    echo "📝 使用说明:"
    echo "========================================"
    echo "1. CARLA窗口应该已经打开"
    echo "2. 现在可以运行训练脚本:"
    echo "   cd /home/ajifang/SAC_carla"
    echo "   python train_ppo_with_wandb.py"
    echo ""
    echo "3. 停止CARLA:"
    echo "   Ctrl+C 或运行: pkill -9 -f CarlaUE4"
    echo "========================================"
    echo ""
    echo "🎬 CARLA服务器已就绪！"
    echo ""

    # 保持脚本运行，等待用户中断
    wait $CARLA_PID
else
    echo "❌ CARLA启动失败"
    exit 1
fi
