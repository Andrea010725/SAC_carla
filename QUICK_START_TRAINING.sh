#!/bin/bash
# Quick Start Script for PPO Training with Pygame & Wandb
# All issues have been fixed - this script should work without errors

echo "======================================================================"
echo "🚀 PPO Training Quick Start"
echo "======================================================================"
echo ""

# Check if CARLA is running
echo "[1/5] Checking CARLA server..."
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "✅ CARLA server is running"
else
    echo "⚠️  CARLA server not detected"
    echo "    Start CARLA with: cd /home/ajifang/carla && ./CarlaUE4.sh -RenderOffScreen"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check conda environment
echo ""
echo "[2/5] Checking conda environment..."
if [[ "$CONDA_DEFAULT_ENV" == "py37" ]]; then
    echo "✅ py37 environment is active"
else
    echo "⚠️  py37 environment not active"
    echo "    Activating py37..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate py37
fi

# Verify imports
echo ""
echo "[3/5] Verifying Python packages..."
python -c "
import sys
try:
    import wandb
    print('✅ Wandb: OK (version ' + wandb.__version__ + ')')
except Exception as e:
    print('❌ Wandb: FAILED - ' + str(e))
    sys.exit(1)

try:
    import pygame
    pygame.init()
    pygame.quit()
    print('✅ Pygame: OK (version ' + pygame.version.ver + ')')
except Exception as e:
    print('❌ Pygame: FAILED - ' + str(e))
    sys.exit(1)

try:
    import tensorflow as tf
    print('✅ TensorFlow: OK (version ' + tf.__version__ + ')')
except Exception as e:
    print('❌ TensorFlow: FAILED - ' + str(e))
    sys.exit(1)
" || exit 1

# Check GPU
echo ""
echo "[4/5] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1 | \
    awk -F', ' '{printf "✅ GPU: %s (Total: %s, Free: %s)\n", $1, $2, $3}'

# Start training
echo ""
echo "[5/5] Starting training..."
echo "======================================================================"
echo ""
echo "📝 Training logs will be saved to:"
echo "   - Console output (real-time)"
echo "   - training_log.json (detailed logs)"
echo "   - reward_plots/ (visualization)"
echo "   - wandb/ (wandb logs)"
echo ""
echo "🛑 To stop training: Press Ctrl+C"
echo ""
echo "======================================================================"
echo ""

cd /home/ajifang/SAC_carla
python train_ppo_with_wandb.py
