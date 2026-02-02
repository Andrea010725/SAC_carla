#!/bin/bash
# 4场景训练 - 一键启动脚本

echo "======================================================================"
echo "4场景PPO训练 - 一键启动"
echo "======================================================================"
echo ""
echo "场景池:"
echo "  1. cones                    - 锥桶避让"
echo "  2. jaywalker                - 鬼探头（行人横穿）"
echo "  3. trimma                   - 包围突围（左右夹击）"
echo "  4. construction_lane_change - 施工变道"
echo ""
echo "======================================================================"
echo ""

# 检查CARLA是否运行
echo "[1/4] 检查CARLA服务器..."
if pgrep -x "CarlaUE4" > /dev/null; then
    echo "✅ CARLA服务器已运行"
else
    echo "❌ CARLA服务器未运行"
    echo ""
    echo "请先启动CARLA服务器："
    echo "  cd /home/ajifang/carla"
    echo "  ./CarlaUE4.sh -RenderOffScreen"
    echo ""
    exit 1
fi

# 检查Python环境
echo ""
echo "[2/4] 检查Python环境..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✅ Python已安装: $PYTHON_VERSION"
else
    echo "❌ Python未安装"
    exit 1
fi

# 检查必要的文件
echo ""
echo "[3/4] 检查必要文件..."
REQUIRED_FILES=(
    "train_ppo_4scenarios.py"
    "config.py"
    "carla_base/carla_env.py"
    "carla_base/scenario_manager.py"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "❌ 缺少必要文件，请检查"
    exit 1
fi

# 创建必要的目录
echo ""
echo "[4/4] 创建输出目录..."
mkdir -p ./weights/ppo-carla-4scenarios
mkdir -p ./scenario_plots
mkdir -p ./logs
echo "✅ 目录创建完成"

# 询问是否开始训练
echo ""
echo "======================================================================"
echo "准备就绪！"
echo "======================================================================"
echo ""
read -p "是否开始训练？(y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "======================================================================"
    echo "开始训练..."
    echo "======================================================================"
    echo ""
    echo "提示："
    echo "  - 按 Ctrl+C 可以随时中断训练"
    echo "  - 训练日志: training_log_4scenarios.json"
    echo "  - 权重保存: ./weights/ppo-carla-4scenarios/"
    echo "  - Wandb监控: 自动打开浏览器"
    echo ""
    echo "======================================================================"
    echo ""

    # 运行训练
    python train_ppo_4scenarios.py

    EXIT_CODE=$?

    echo ""
    echo "======================================================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 训练完成！"
        echo "======================================================================"
        echo ""
        echo "下一步："
        echo "  1. 查看训练日志: cat training_log_4scenarios.json"
        echo "  2. 可视化结果: python visualize_scenario_results.py"
        echo "  3. 测试模型: python test_scenarios_comparison.py"
    else
        echo "❌ 训练异常退出 (退出码: $EXIT_CODE)"
        echo "======================================================================"
        echo ""
        echo "请检查："
        echo "  1. CARLA服务器是否正常运行"
        echo "  2. 查看错误信息"
        echo "  3. 检查日志文件"
    fi
else
    echo ""
    echo "训练已取消"
fi

echo ""
