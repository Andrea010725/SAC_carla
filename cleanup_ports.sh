#!/bin/bash
# CARLA端口清理脚本

echo "=========================================="
echo "CARLA端口清理工具"
echo "=========================================="

# 定义端口列表
PORTS=(2000 2001 2002 3000 3001 3002 8000 8001 8002)

echo ""
echo "1. 检查占用的端口..."
for port in "${PORTS[@]}"; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        process=$(ps -p $pid -o comm= 2>/dev/null)
        echo "  端口 $port 被占用 (PID: $pid, 进程: $process)"
    fi
done

echo ""
echo "2. 查找所有CARLA相关进程..."
ps aux | grep -E "(carla|Carla|CARLA)" | grep -v grep | grep -v cleanup

echo ""
read -p "是否要清理所有CARLA进程和端口? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "取消清理"
    exit 0
fi

echo ""
echo "3. 停止CARLA进程..."
pkill -9 -f CarlaUE4
pkill -9 -f carla
sleep 2

echo ""
echo "4. 清理占用的端口..."
for port in "${PORTS[@]}"; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "  杀死占用端口 $port 的进程 (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

echo ""
echo "5. 清理Python进程（如果有卡住的）..."
pkill -9 -f "python.*main.py"
pkill -9 -f "python.*carla"

sleep 1

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "现在可以重新启动CARLA服务器："
echo "  cd /home/ajifang/carla"
echo "  ./CarlaUE4.sh -carla-rpc-port=2000"
echo ""
