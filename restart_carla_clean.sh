#!/bin/bash
# 完整重启脚本 - 确保CARLA和配置完全匹配

echo "======================================================================"
echo "🔧 CARLA完整重启 - 确保Town05配置正确"
echo "======================================================================"
echo ""

# 1. 停止所有CARLA和训练进程
echo "1️⃣  停止现有进程..."
pkill -9 -f train_ppo 2>/dev/null
pkill -9 CarlaUE4 2>/dev/null
sleep 5
echo "   ✅ 所有进程已停止"
echo ""

# 2. 启动CARLA（Town05）
echo "2️⃣  启动CARLA服务器（Town05）..."
cd /home/ajifang/carla
./CarlaUE4.sh Town05 -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound &
CARLA_PID=$!
echo "   ⏳ CARLA启动中（PID: $CARLA_PID）..."
echo ""

# 3. 等待端口就绪
echo "3️⃣  等待CARLA端口2000就绪..."
for i in {1..20}; do
    if nc -z 127.0.0.1 2000 2>/dev/null; then
        echo "   ✅ 端口2000已就绪（尝试 $i/20）"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "   ❌ 端口2000超时未就绪"
        echo "   请检查CARLA是否启动成功"
        exit 1
    fi
    sleep 3
done
echo ""

# 4. 等待Town05完全加载
echo "4️⃣  等待Town05地图完全加载（60秒）..."
sleep 60
echo "   ✅ 等待完成"
echo ""

# 5. 验证地图配置
echo "5️⃣  验证地图配置..."
cd /home/ajifang/SAC_carla

python3 - <<EOF
import sys, os
sys.path.insert(0, '/home/ajifang/carla/PythonAPI/carla')
sys.path.insert(0, '/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
import carla

try:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f'   当前CARLA地图: {map_name}')

    # 检查Config
    sys.path.insert(0, '/home/ajifang/SAC_carla')
    from config import Config
    config = Config()
    print(f'   Config期望地图: {config.map_name}')
    print()

    if config.map_name in map_name:
        print('   ✅ 地图配置匹配！')
        print('   ✅ 可以安全开始训练')
        sys.exit(0)
    else:
        print('   ❌ 地图配置不匹配！')
        print('   ❌ 请重新运行此脚本')
        sys.exit(1)
except Exception as e:
    print(f'   ❌ 验证失败: {e}')
    sys.exit(1)
EOF

VERIFY_EXIT=$?
echo ""

# 6. 结果
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "======================================================================"
    echo "✅ CARLA已正确配置！"
    echo "======================================================================"
    echo ""
    echo "现在可以开始训练:"
    echo "   cd /home/ajifang/SAC_carla"
    echo "   ./start_fast_training.sh"
    echo ""
else
    echo "======================================================================"
    echo "❌ 验证失败，请检查错误信息"
    echo "======================================================================"
    echo ""
    exit 1
fi
