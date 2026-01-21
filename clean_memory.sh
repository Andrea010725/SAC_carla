#!/bin/bash
# 内存清理脚本 - Memory Cleanup Script

echo "=========================================="
echo "清理内存 - Cleaning Memory"
echo "=========================================="

echo ""
echo "1. 清理前内存状态："
free -h

echo ""
echo "2. 停止所有CARLA和训练进程..."
# 查找并显示相关进程
echo "   当前进程："
ps aux | grep -E "CarlaUE4|train_ppo|python.*carla" | grep -v grep | awk '{printf "   PID %s: %s (CPU: %s%%, MEM: %s%%)\n", $2, $11, $3, $4}'

# 清理进程
killall -9 CarlaUE4-Linux-Shipping 2>/dev/null
killall -9 CarlaUE4 2>/dev/null
pkill -9 -f "train_ppo" 2>/dev/null
pkill -9 -f "carla_env" 2>/dev/null

sleep 2

echo ""
echo "3. 验证进程已清理..."
REMAINING=$(ps aux | grep -E "CarlaUE4|train_ppo" | grep -v grep | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo "   ✓ 所有相关进程已清理"
else
    echo "   ⚠ 仍有 $REMAINING 个进程残留："
    ps aux | grep -E "CarlaUE4|train_ppo" | grep -v grep | awk '{printf "   PID %s: %s\n", $2, $11}'
fi

echo ""
echo "4. 清理后内存状态："
free -h

echo ""
echo "5. 清理系统缓存（需要sudo权限）..."
if [ "$EUID" -eq 0 ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "   ✓ 系统缓存已清理"
    echo ""
    echo "6. 最终内存状态："
    free -h
else
    echo "   ⚠ 非root用户，跳过缓存清理"
    echo "   如需清理缓存，运行: sudo ./clean_memory.sh"
fi

echo ""
echo "=========================================="
echo "内存清理完成"
echo "=========================================="

# 显示可用内存
AVAILABLE=$(free -m | awk 'NR==2{print $7}')
echo "可用内存: ${AVAILABLE}MB ($(echo "scale=1; $AVAILABLE/1024" | bc)GB)"

if [ $AVAILABLE -gt 10240 ]; then
    echo "✓ 内存充足，可以开始训练"
else
    echo "⚠ 可用内存较低，建议先关闭其他程序"
fi
