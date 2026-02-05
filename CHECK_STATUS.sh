#!/bin/bash
# Status Check Script - Verify all components are working

echo "======================================================================"
echo "🔍 System Status Check"
echo "======================================================================"
echo ""

# Check CARLA
echo "1️⃣  CARLA Server:"
if pgrep -f "CarlaUE4" > /dev/null; then
    CARLA_PID=$(pgrep -f "CarlaUE4" | head -1)
    CARLA_MEM=$(ps -p $CARLA_PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "   ✅ Running (PID: $CARLA_PID, Memory: $CARLA_MEM)"
else
    echo "   ❌ Not running"
fi

# Check Training
echo ""
echo "2️⃣  Training Script:"
if pgrep -f "train_ppo_with_wandb.py" > /dev/null; then
    TRAIN_PID=$(pgrep -f "train_ppo_with_wandb.py" | head -1)
    TRAIN_MEM=$(ps -p $TRAIN_PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "   ✅ Running (PID: $TRAIN_PID, Memory: $TRAIN_MEM)"
else
    echo "   ❌ Not running"
fi

# Check GPU
echo ""
echo "3️⃣  GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   GPU Usage: %s | Memory: %s / %s | Temp: %s\n", $1, $2, $3, $4}'

# Check Python packages
echo ""
echo "4️⃣  Python Packages:"
python -c "
import wandb, pygame, tensorflow as tf
print(f'   ✅ Wandb: {wandb.__version__}')
print(f'   ✅ Pygame: {pygame.version.ver}')
print(f'   ✅ TensorFlow: {tf.__version__}')
" 2>/dev/null || echo "   ❌ Package import failed"

# Check recent logs
echo ""
echo "5️⃣  Recent Training Activity:"
if [ -f "/home/ajifang/SAC_carla/training_log.json" ]; then
    LAST_MOD=$(stat -c %y /home/ajifang/SAC_carla/training_log.json | cut -d'.' -f1)
    echo "   📝 Last log update: $LAST_MOD"
    
    # Count episodes
    if command -v jq &> /dev/null; then
        EPISODES=$(jq -s 'length' /home/ajifang/SAC_carla/training_log.json 2>/dev/null || echo "?")
        echo "   📊 Total episodes logged: $EPISODES"
    fi
else
    echo "   ⚠️  No training log found"
fi

# Check wandb
echo ""
echo "6️⃣  Wandb Status:"
if [ -d "/home/ajifang/SAC_carla/wandb" ]; then
    RUNS=$(ls -1 /home/ajifang/SAC_carla/wandb/ | grep -c "run-" 2>/dev/null || echo "0")
    echo "   📁 Wandb directory exists ($RUNS runs)"
else
    echo "   ⚠️  No wandb directory"
fi

# Check reward plots
echo ""
echo "7️⃣  Reward Plots:"
if [ -d "/home/ajifang/SAC_carla/reward_plots" ]; then
    PLOTS=$(ls -1 /home/ajifang/SAC_carla/reward_plots/*.png 2>/dev/null | wc -l)
    echo "   📊 $PLOTS plot files generated"
else
    echo "   ⚠️  No reward plots directory"
fi

echo ""
echo "======================================================================"
echo "✅ Status check complete"
echo "======================================================================"
