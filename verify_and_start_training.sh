#!/bin/bash
# PPO训练修复验证和启动脚本
# 版本: v1.0
# 日期: 2026-01-28

echo "======================================================================"
echo "🔧 PPO训练修复验证和启动脚本"
echo "======================================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 错误计数
ERROR_COUNT=0

echo "📋 步骤1: 检查修复文档"
echo "----------------------------------------------------------------------"
if [ -f "TRAINING_FIX_SUMMARY.md" ]; then
    echo -e "${GREEN}✅ 详细修复文档: TRAINING_FIX_SUMMARY.md${NC}"
else
    echo -e "${RED}❌ 缺少详细修复文档${NC}"
    ((ERROR_COUNT++))
fi

if [ -f "QUICK_FIX_GUIDE.md" ]; then
    echo -e "${GREEN}✅ 快速修复指南: QUICK_FIX_GUIDE.md${NC}"
else
    echo -e "${RED}❌ 缺少快速修复指南${NC}"
    ((ERROR_COUNT++))
fi
echo ""

echo "📋 步骤2: 验证Python语法"
echo "----------------------------------------------------------------------"

# 验证train_ppo_with_wandb.py
if python -m py_compile train_ppo_with_wandb.py 2>/dev/null; then
    echo -e "${GREEN}✅ train_ppo_with_wandb.py 语法正确${NC}"
else
    echo -e "${RED}❌ train_ppo_with_wandb.py 语法错误${NC}"
    python -m py_compile train_ppo_with_wandb.py
    ((ERROR_COUNT++))
fi

# 验证carla_env.py
if python -m py_compile carla_base/carla_env.py 2>/dev/null; then
    echo -e "${GREEN}✅ carla_base/carla_env.py 语法正确${NC}"
else
    echo -e "${RED}❌ carla_base/carla_env.py 语法错误${NC}"
    python -m py_compile carla_base/carla_env.py
    ((ERROR_COUNT++))
fi

# 验证config.py
if python -m py_compile config.py 2>/dev/null; then
    echo -e "${GREEN}✅ config.py 语法正确${NC}"
else
    echo -e "${RED}❌ config.py 语法错误${NC}"
    python -m py_compile config.py
    ((ERROR_COUNT++))
fi
echo ""

echo "📋 步骤3: 检查关键修复点"
echo "----------------------------------------------------------------------"

# 检查carla_env.py的修复
if grep -q "K_COLLISION_TERMINAL = 100.0" carla_base/carla_env.py; then
    echo -e "${GREEN}✅ 碰撞惩罚已修复 (100.0)${NC}"
else
    echo -e "${RED}❌ 碰撞惩罚未修复 (应该是100.0)${NC}"
    ((ERROR_COUNT++))
fi

if grep -q "r_success = +50.0" carla_base/carla_env.py; then
    echo -e "${GREEN}✅ 成功奖励已添加 (+50.0)${NC}"
else
    echo -e "${RED}❌ 成功奖励未添加${NC}"
    ((ERROR_COUNT++))
fi

# 检查train_ppo_with_wandb.py的修复
if grep -q "class RunningMeanStd:" train_ppo_with_wandb.py; then
    echo -e "${GREEN}✅ 观测归一化类已添加${NC}"
else
    echo -e "${RED}❌ 观测归一化类未添加${NC}"
    ((ERROR_COUNT++))
fi

if grep -q "entropy_regularization=0.05" train_ppo_with_wandb.py; then
    echo -e "${GREEN}✅ 熵系数已修复 (0.05)${NC}"
else
    echo -e "${RED}❌ 熵系数未修复 (应该是0.05)${NC}"
    ((ERROR_COUNT++))
fi

if grep -q "batch_size=128" train_ppo_with_wandb.py; then
    echo -e "${GREEN}✅ batch_size已修复 (128)${NC}"
else
    echo -e "${RED}❌ batch_size未修复 (应该是128)${NC}"
    ((ERROR_COUNT++))
fi

# 检查config.py的修复
if grep -q "self.yref_steer_gain = 0.15" config.py; then
    echo -e "${GREEN}✅ y_ref增益已修复 (0.15)${NC}"
else
    echo -e "${RED}❌ y_ref增益未修复 (应该是0.15)${NC}"
    ((ERROR_COUNT++))
fi
echo ""

echo "📋 步骤4: 检查CARLA服务器"
echo "----------------------------------------------------------------------"
if pgrep -x "CarlaUE4" > /dev/null; then
    echo -e "${GREEN}✅ CARLA服务器正在运行${NC}"
    CARLA_PID=$(pgrep -x "CarlaUE4")
    echo "   进程ID: $CARLA_PID"
else
    echo -e "${YELLOW}⚠️  CARLA服务器未运行${NC}"
    echo ""
    echo "请在另一个终端启动CARLA服务器："
    echo -e "${BLUE}  cd /home/ajifang/carla${NC}"
    echo -e "${BLUE}  ./CarlaUE4.sh -RenderOffScreen${NC}"
    echo ""
    ((ERROR_COUNT++))
fi
echo ""

echo "📋 步骤5: 检查Python环境"
echo "----------------------------------------------------------------------"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✅ Python已安装: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python未安装${NC}"
    ((ERROR_COUNT++))
fi

# 检查必要的Python包
echo ""
echo "检查必要的Python包..."
REQUIRED_PACKAGES=("numpy" "gym" "tensorflow" "carla")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}  ✅ $pkg${NC}"
    else
        echo -e "${RED}  ❌ $pkg (未安装)${NC}"
        ((ERROR_COUNT++))
    fi
done

# 检查wandb（可选）
if python -c "import wandb" 2>/dev/null; then
    echo -e "${GREEN}  ✅ wandb (可选)${NC}"
else
    echo -e "${YELLOW}  ⚠️  wandb (可选，未安装)${NC}"
    echo "     安装命令: pip install wandb"
fi
echo ""

echo "📋 步骤6: 检查磁盘空间"
echo "----------------------------------------------------------------------"
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "可用磁盘空间: $DISK_SPACE"

DISK_SPACE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_SPACE_GB" -gt 10 ]; then
    echo -e "${GREEN}✅ 磁盘空间充足 (>10GB)${NC}"
else
    echo -e "${YELLOW}⚠️  磁盘空间不足 (<10GB)${NC}"
    echo "   建议清理磁盘空间"
fi
echo ""

echo "📋 步骤7: 检查GPU"
echo "----------------------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA驱动已安装${NC}"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1)
    echo "   GPU: $GPU_INFO"
else
    echo -e "${YELLOW}⚠️  nvidia-smi未找到 (可能使用CPU训练)${NC}"
fi
echo ""

echo "======================================================================"
echo "📊 验证结果汇总"
echo "======================================================================"
echo ""

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有检查通过！可以开始训练${NC}"
    echo ""
    echo "======================================================================"
    echo "🚀 准备启动训练"
    echo "======================================================================"
    echo ""
    echo "训练配置："
    echo "  - 算法: PPO"
    echo "  - Episodes: 300"
    echo "  - 观测维度: 30"
    echo "  - 动作维度: 3"
    echo "  - 观测归一化: ✅ 启用"
    echo "  - 熵系数: 0.05"
    echo "  - batch_size: 128"
    echo "  - y_ref增益: 0.15"
    echo ""
    echo "预期效果："
    echo "  - 成功率: 80-90% (当前20-30%)"
    echo "  - 收敛时间: 150-200 episodes"
    echo "  - 碰撞率: 5-10% (当前60-70%)"
    echo ""
    echo "======================================================================"
    echo ""

    read -p "是否立即开始训练？(y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "======================================================================"
        echo "🎯 开始训练..."
        echo "======================================================================"
        echo ""
        echo "提示："
        echo "  - 按 Ctrl+C 可以随时中断训练"
        echo "  - 训练日志: training_log.json"
        echo "  - 权重保存: ./weights/ppo-carla-obs30/"
        echo "  - Wandb监控: 自动打开浏览器"
        echo ""
        echo "======================================================================"
        echo ""

        # 运行训练
        python train_ppo_with_wandb.py

        EXIT_CODE=$?

        echo ""
        echo "======================================================================"
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✅ 训练完成！${NC}"
            echo "======================================================================"
            echo ""
            echo "下一步："
            echo "  1. 查看训练日志: cat training_log.json | jq '.episodes[-10:]'"
            echo "  2. 查看Wandb面板: https://wandb.ai"
            echo "  3. 测试模型: python test_ppo.py"
        else
            echo -e "${RED}❌ 训练异常退出 (退出码: $EXIT_CODE)${NC}"
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
        echo ""
        echo "手动启动训练："
        echo -e "${BLUE}  python train_ppo_with_wandb.py${NC}"
    fi
else
    echo -e "${RED}❌ 发现 $ERROR_COUNT 个问题，请先修复${NC}"
    echo ""
    echo "常见问题解决方案："
    echo ""
    echo "1. CARLA服务器未运行："
    echo -e "   ${BLUE}cd /home/ajifang/carla${NC}"
    echo -e "   ${BLUE}./CarlaUE4.sh -RenderOffScreen${NC}"
    echo ""
    echo "2. Python包缺失："
    echo -e "   ${BLUE}pip install numpy gym tensorflow-gpu carla${NC}"
    echo ""
    echo "3. 代码修复未完成："
    echo "   请查看 QUICK_FIX_GUIDE.md 完成所有修改"
    echo ""
    exit 1
fi

echo ""
