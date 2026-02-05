#!/bin/bash
# Comprehensive verification script for all fixes

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              🔍 VERIFYING ALL FIXES 🔍                               ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0

# Test 1: Wandb Import
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Wandb Import"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -c "import wandb; assert hasattr(wandb, 'init'), 'No init attribute'; print('✅ PASS: Wandb imports correctly')" 2>/dev/null; then
    ((PASS++))
else
    echo "❌ FAIL: Wandb import failed"
    ((FAIL++))
fi
echo ""

# Test 2: Pygame Display
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Pygame Display Initialization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -c "import pygame; pygame.init(); screen = pygame.display.set_mode((400, 300)); pygame.quit(); print('✅ PASS: Pygame display works')" 2>/dev/null; then
    ((PASS++))
else
    echo "❌ FAIL: Pygame display failed"
    ((FAIL++))
fi
echo ""

# Test 3: TensorFlow GPU
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: TensorFlow GPU Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); assert len(gpus) > 0, 'No GPU found'; print('✅ PASS: GPU detected -', gpus[0].name)" 2>/dev/null; then
    ((PASS++))
else
    echo "❌ FAIL: GPU not detected"
    ((FAIL++))
fi
echo ""

# Test 4: CARLA Environment Import
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: CARLA Environment Import"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -c "import sys; sys.path.insert(0, '/home/ajifang/SAC_carla'); from carla_base.carla_env import CarlaEnv; print('✅ PASS: CarlaEnv imports correctly')" 2>/dev/null; then
    ((PASS++))
else
    echo "❌ FAIL: CarlaEnv import failed"
    ((FAIL++))
fi
echo ""

# Test 5: Training Script Syntax
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 5: Training Script Syntax Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -m py_compile /home/ajifang/SAC_carla/train_ppo_with_wandb.py 2>/dev/null; then
    echo "✅ PASS: Training script has no syntax errors"
    ((PASS++))
else
    echo "❌ FAIL: Training script has syntax errors"
    ((FAIL++))
fi
echo ""

# Test 6: File Modifications
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 6: Verify Code Modifications"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if grep -q "Try to create real display first" /home/ajifang/SAC_carla/carla_base/carla_env.py; then
    echo "✅ PASS: Pygame fix is present in carla_env.py"
    ((PASS++))
else
    echo "❌ FAIL: Pygame fix not found in carla_env.py"
    ((FAIL++))
fi
echo ""

# Test 7: Documentation Files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 7: Documentation Files Created"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DOCS_FOUND=0
for doc in START_HERE.md README_FIXES.md FINAL_SETUP_INSTRUCTIONS.md PYGAME_DISPLAY_FIX.md; do
    if [ -f "/home/ajifang/SAC_carla/$doc" ]; then
        ((DOCS_FOUND++))
    fi
done
if [ $DOCS_FOUND -eq 4 ]; then
    echo "✅ PASS: All documentation files created ($DOCS_FOUND/4)"
    ((PASS++))
else
    echo "❌ FAIL: Missing documentation files ($DOCS_FOUND/4)"
    ((FAIL++))
fi
echo ""

# Test 8: Helper Scripts
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 8: Helper Scripts Created"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SCRIPTS_FOUND=0
for script in QUICK_START_TRAINING.sh CHECK_STATUS.sh; do
    if [ -x "/home/ajifang/SAC_carla/$script" ]; then
        ((SCRIPTS_FOUND++))
    fi
done
if [ $SCRIPTS_FOUND -eq 2 ]; then
    echo "✅ PASS: All helper scripts created and executable ($SCRIPTS_FOUND/2)"
    ((PASS++))
else
    echo "❌ FAIL: Missing or non-executable scripts ($SCRIPTS_FOUND/2)"
    ((FAIL++))
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                        📊 TEST RESULTS 📊                            ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ✅ Passed: $PASS/8"
echo "  ❌ Failed: $FAIL/8"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║              🎉 ALL TESTS PASSED! 🎉                                 ║"
    echo "║                                                                      ║"
    echo "║         Your training environment is fully operational!              ║"
    echo "║                                                                      ║"
    echo "║                  Ready to start training!                            ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🚀 To start training, run:"
    echo "   cd /home/ajifang/SAC_carla"
    echo "   ./QUICK_START_TRAINING.sh"
    echo ""
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║              ⚠️  SOME TESTS FAILED ⚠️                                ║"
    echo "║                                                                      ║"
    echo "║         Please review the failed tests above.                        ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi
