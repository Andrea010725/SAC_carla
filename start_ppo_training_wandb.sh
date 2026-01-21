#!/bin/bash

# PPO训练启动脚本 - 带Wandb监控
# 使用方法: ./start_ppo_training_wandb.sh

echo "======================================================================"
echo "  PPO Training with Wandb Monitoring"
echo "======================================================================"

# 检查CARLA是否运行
echo ""
echo "[1/5] 检查CARLA服务器..."
if ! pgrep -f "CarlaUE4" > /dev/null; then
    echo "❌ CARLA未运行！"
    echo ""
    echo "请先启动CARLA服务器:"
    echo "  cd /home/ajifang/carla"
    echo "  ./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20"
    echo ""
    exit 1
fi
echo "✅ CARLA服务器已运行"

# 检查Python环境
echo ""
echo "[2/5] 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到"
    exit 1
fi

if ! python -c "import tensorflow" 2>/dev/null; then
    echo "❌ TensorFlow未安装"
    echo "   请安装: pip install tensorflow==2.4.0"
    exit 1
fi

echo "✅ Python环境正常"

# 检查/安装Wandb
echo ""
echo "[3/5] 检查Wandb..."
if ! python -c "import wandb" 2>/dev/null; then
    echo "⚠️  Wandb未安装"
    read -p "是否现在安装Wandb? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install wandb
        if [ $? -ne 0 ]; then
            echo "❌ Wandb安装失败"
            echo "   训练将继续，但不会有在线监控"
        else
            echo "✅ Wandb安装成功"
            echo ""
            echo "请登录Wandb:"
            wandb login
        fi
    else
        echo "⚠️  跳过Wandb安装，将以离线模式运行"
    fi
else
    echo "✅ Wandb已安装"

    # 检查是否已登录
    if ! wandb verify >/dev/null 2>&1; then
        echo "⚠️  Wandb未登录"
        read -p "是否现在登录? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            wandb login
        fi
    else
        echo "✅ Wandb已登录"
    fi
fi

# 进入项目目录
cd /home/ajifang/SAC_carla

# 创建必要的目录
echo ""
echo "[4/5] 准备训练环境..."
mkdir -p logs
mkdir -p weights
mkdir -p wandb
echo "✅ 目录创建成功"

# 显示训练配置
echo ""
echo "[5/5] 训练配置:"
echo "  - Episodes: 1000"
echo "  - Max Steps/Episode: 512"
echo "  - Scenario: parked_obstacles (4辆停车)"
echo "  - Policy LR: 5e-5"
echo "  - Value LR: 1e-4"
echo "  - Entropy: 0.05"
echo "  - Render: True"
echo "  - Wandb: $(python -c "import wandb; print('Enabled')" 2>/dev/null || echo 'Disabled')"
echo "  - Model Save Path: weights/ppo-carla-standalone/"
echo "  - Log File: training_log.json"

# 询问是否开始
echo ""
read -p "准备开始训练，是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 训练取消"
    exit 1
fi

# 开始训练
echo ""
echo "======================================================================"
echo "  🚀 开始训练..."
echo "======================================================================"
echo ""

# 设置日志文件名
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# 运行训练并记录日志
python train_ppo_with_wandb.py 2>&1 | tee "$LOG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "  ✅ 训练完成！"
    echo "======================================================================"
    echo ""
    echo "模型保存位置: weights/ppo-carla-standalone/"
    echo "日志文件: $LOG_FILE"
    echo "训练日志: training_log.json"

    if python -c "import wandb" 2>/dev/null; then
        echo ""
        echo "📊 查看Wandb结果:"
        echo "   https://wandb.ai/your-username/SAC-CARLA-PPO"
    fi
else
    echo ""
    echo "======================================================================"
    echo "  ❌ 训练异常终止"
    echo "======================================================================"
    echo ""
    echo "请查看日志: $LOG_FILE"
fi
