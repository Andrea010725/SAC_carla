#!/bin/bash

# Overtaking能力专项测试脚本
# 基于分批测试，每批5条路线，自动重启CARLA

set -e

echo "========================================================================"
echo "🚗 Overtaking 能力专项测试 (45条路线)"
echo "========================================================================"
echo ""

# ========== 环境变量设置 ==========
export CARLA_ROOT=/home/ajifang/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/home/ajifang/b2drive/leaderboard
export PYTHONPATH=$PYTHONPATH:/home/ajifang/b2drive/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:/home/ajifang/b2drive/scenario_runner
export SCENARIO_RUNNER_ROOT=/home/ajifang/b2drive/scenario_runner

# ========== 配置参数 ==========
CARLA_PORT=2000
CARLA_TM_PORT=8000
OVERTAKING_DIR="/home/ajifang/b2drive/overtaking"
RESULTS_DIR="$OVERTAKING_DIR/results"
ROUTES_XML="$OVERTAKING_DIR/data/overtaking.xml"
ROUTE_IDS_FILE="$OVERTAKING_DIR/data/overtaking_route_ids.txt"
AGENT_CONFIG="/home/ajifang/b2drive/leaderboard/team_code/il_agent.py"
BATCH_SIZE=5  # 每批5条路线
TOTAL_ROUTES=45
RESUME_FILE="$RESULTS_DIR/.resume_state"

# 切换到B2D目录
cd /home/ajifang/b2drive

echo "📋 测试配置:"
echo "   - 能力类型: Overtaking (超车能力)"
echo "   - Agent: IL Agent"
echo "   - 总路线数: $TOTAL_ROUTES"
echo "   - 批次大小: $BATCH_SIZE 条/批"
echo "   - 批次数: $((TOTAL_ROUTES / BATCH_SIZE)) 批"
echo "   - CARLA端口: $CARLA_PORT"
echo "   - 路线文件: $ROUTES_XML"
echo "   - 结果目录: $RESULTS_DIR"
echo ""

# ========== 创建结果目录 ==========
mkdir -p $RESULTS_DIR/batches

# ========== 检查IL模型 ==========
if [ ! -f "rl_ppo_model/checkpoints/best_model.pth" ]; then
    echo "❌ IL模型权重不存在: rl_ppo_model/checkpoints/best_model.pth"
    exit 1
fi
echo "✅ IL模型权重存在"
echo ""

# ========== 检查路线文件 ==========
if [ ! -f "$ROUTES_XML" ]; then
    echo "❌ 路线文件不存在: $ROUTES_XML"
    echo "   请先运行: cd $OVERTAKING_DIR && python3 scripts/extract_overtaking_routes.py"
    exit 1
fi
echo "✅ 路线文件存在"
echo ""

# ========== 提取路线ID ==========
echo "🔍 提取路线ID..."
if [ -f "$ROUTE_IDS_FILE" ]; then
    ROUTE_IDS=$(cat $ROUTE_IDS_FILE)
    ACTUAL_TOTAL_ROUTES=$(echo $ROUTE_IDS | tr ',' '\n' | wc -l)
    echo "✅ 从文件读取 $ACTUAL_TOTAL_ROUTES 条路线"
else
    grep -oP 'route id="\K[0-9]+' $ROUTES_XML > $ROUTE_IDS_FILE
    ROUTE_IDS=$(cat $ROUTE_IDS_FILE | paste -sd,)
    ACTUAL_TOTAL_ROUTES=$(wc -l < $ROUTE_IDS_FILE)
    echo "✅ 从XML提取 $ACTUAL_TOTAL_ROUTES 条路线"
fi

# 更新总路线数
TOTAL_ROUTES=$ACTUAL_TOTAL_ROUTES
echo ""

# ========== 检查是否需要恢复 ==========
START_BATCH=0
if [ -f "$RESUME_FILE" ]; then
    START_BATCH=$(cat $RESUME_FILE)
    echo "🔄 检测到中断，从批次 $START_BATCH 恢复"
    echo ""
fi

# 计算批次数
NUM_BATCHES=$(( (TOTAL_ROUTES + BATCH_SIZE - 1) / BATCH_SIZE ))

# ========== 辅助函数 ==========

# 函数：检查CARLA健康状态
check_carla_health() {
    if ! pgrep -f "CarlaUE4" > /dev/null; then
        echo "❌ CARLA进程已终止"
        return 1
    fi

    if ! nc -z localhost $CARLA_PORT 2>/dev/null; then
        echo "❌ CARLA端口 $CARLA_PORT 无响应"
        return 1
    fi

    return 0
}

# 函数：启动CARLA
start_carla() {
    echo "🚀 启动CARLA服务器..."
    cd $CARLA_ROOT
    nohup ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -fps=20 -benchmark -nosound > /tmp/carla_overtaking.log 2>&1 &
    CARLA_PID=$!
    echo "   CARLA PID: $CARLA_PID"

    # 等待CARLA就绪
    echo "   等待CARLA启动..."
    MAX_WAIT=60
    WAIT_COUNT=0
    while ! nc -z localhost $CARLA_PORT 2>/dev/null; do
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "❌ CARLA启动超时"
            kill -9 $CARLA_PID 2>/dev/null || true
            return 1
        fi
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
    done

    # 额外等待确保完全就绪
    sleep 10
    echo "✅ CARLA已就绪"
    cd /home/ajifang/b2drive
    return 0
}

# 函数：停止CARLA
stop_carla() {
    echo "🛑 停止CARLA服务器..."

    # 停止CARLA和leaderboard进程
    pkill -9 -f "CarlaUE4" 2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true

    # 清理占用端口的进程
    for port in $CARLA_PORT $CARLA_TM_PORT; do
        for pid in $(lsof -t -i:$port 2>/dev/null); do
            kill -9 $pid 2>/dev/null || true
        done
    done

    # 等待端口释放
    echo "   等待端口释放..."
    local wait_count=0
    while [ $wait_count -lt 10 ]; do
        if ! nc -z localhost $CARLA_PORT 2>/dev/null && ! nc -z localhost $CARLA_TM_PORT 2>/dev/null; then
            break
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    sleep 3
    echo "✅ CARLA已停止"
}

# 函数：运行一个批次
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * BATCH_SIZE))
    local end_idx=$((start_idx + BATCH_SIZE - 1))

    # 确保不超过总路线数
    if [ $end_idx -ge $TOTAL_ROUTES ]; then
        end_idx=$((TOTAL_ROUTES - 1))
    fi

    local batch_file="$RESULTS_DIR/batches/batch_${batch_num}.json"
    local batch_debug="$RESULTS_DIR/batches/batch_${batch_num}_debug.json"

    # 从路线ID列表中提取该批次的ID
    local route_ids_array=(${ROUTE_IDS//,/ })
    local batch_route_ids=""

    for i in $(seq $start_idx $end_idx); do
        if [ -n "$batch_route_ids" ]; then
            batch_route_ids="$batch_route_ids,${route_ids_array[$i]}"
        else
            batch_route_ids="${route_ids_array[$i]}"
        fi
    done

    local num_routes_in_batch=$(echo $batch_route_ids | tr ',' '\n' | wc -l)

    echo ""
    echo "========================================================================"
    echo "📦 批次 $((batch_num + 1))/$NUM_BATCHES: $num_routes_in_batch 条路线"
    echo "========================================================================"
    echo ""

    echo "🎯 路线ID: $batch_route_ids"
    echo ""

    # 运行评估
    python3 leaderboard/leaderboard/leaderboard_evaluator.py \
        --routes=$ROUTES_XML \
        --routes-subset=$batch_route_ids \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint=$batch_file \
        --debug-checkpoint=$batch_debug \
        --agent=$AGENT_CONFIG \
        --agent-config="" \
        --port=$CARLA_PORT \
        --traffic-manager-port=$CARLA_TM_PORT \
        --timeout=600.0 \
        --debug=0

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ 批次 $((batch_num + 1)) 完成"

        # 检查CARLA健康状态
        echo "   检查CARLA健康状态..."
        if ! check_carla_health; then
            echo "   ⚠️  CARLA已不健康，下一批次将重启"
        else
            echo "   ✅ CARLA运行正常"
        fi

        # 记录进度
        echo $((batch_num + 1)) > $RESUME_FILE
        return 0
    else
        echo "❌ 批次 $((batch_num + 1)) 失败 (退出码: $exit_code)"

        # 检查是否因为CARLA崩溃
        if ! check_carla_health; then
            echo "   ⚠️  检测到CARLA崩溃，将在下一批次重启"
        fi

        return 1
    fi
}

# ========== 主循环 ==========
echo "========================================================================"
echo "⏰ 开始Overtaking能力测试..."
echo "========================================================================"
echo ""

FAILED_BATCHES=()

for batch in $(seq $START_BATCH $((NUM_BATCHES - 1))); do
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "🔄 准备批次 $((batch + 1))/$NUM_BATCHES"
    echo "════════════════════════════════════════════════════════════════════"

    # 停止旧的CARLA
    stop_carla

    # 启动新的CARLA
    if ! start_carla; then
        echo "❌ CARLA启动失败，跳过批次 $((batch + 1))"
        FAILED_BATCHES+=($batch)
        continue
    fi

    # 运行批次
    if ! run_batch $batch; then
        echo "⚠️  批次 $((batch + 1)) 失败，但继续下一批次"
        FAILED_BATCHES+=($batch)
    fi

    echo ""
    echo "⏸️  批次间休息5秒..."
    sleep 5
done

# 最终清理
stop_carla

echo ""
echo "========================================================================"
echo "📊 合并批次结果..."
echo "========================================================================"
echo ""

# 使用Python合并结果
python3 << 'PYTHON_SCRIPT'
import json
import glob
import sys
from pathlib import Path

results_dir = "/home/ajifang/b2drive/overtaking/results"
batch_files = sorted(glob.glob(f"{results_dir}/batches/batch_*.json"))

if not batch_files:
    print("❌ 没有找到批次结果文件")
    sys.exit(1)

print(f"📦 找到 {len(batch_files)} 个批次文件")

all_records = []

for batch_file in batch_files:
    if "_debug" in batch_file:
        continue

    print(f"   - 读取: {Path(batch_file).name}")

    try:
        with open(batch_file, 'r') as f:
            data = json.load(f)

        if 'records' in data:
            all_records.extend(data['records'])
        elif isinstance(data, list):
            all_records.extend(data)
    except Exception as e:
        print(f"   ⚠️  读取 {batch_file} 失败: {e}")
        continue

# 构建最终结果
result = {
    'total_routes': len(all_records),
    'records': all_records,
    'ability': 'Overtaking'
}

# 计算统计信息
if all_records:
    completed = sum(1 for r in all_records if r.get('status') == 'Completed')
    result['summary'] = {
        'completed': completed,
        'failed': len(all_records) - completed,
        'completion_rate': completed / len(all_records) * 100 if all_records else 0
    }

# 保存合并结果
output_file = f"{results_dir}/overtaking_results.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ 合并完成: {output_file}")
print(f"   总路线数: {result['total_routes']}")
if 'summary' in result:
    print(f"   完成: {result['summary']['completed']}")
    print(f"   失败: {result['summary']['failed']}")
    print(f"   完成率: {result['summary']['completion_rate']:.1f}%")
PYTHON_SCRIPT

echo ""
echo "========================================================================"
echo "✅ Overtaking能力测试完成!"
echo "========================================================================"
echo ""

# 显示失败的批次
if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
    echo "⚠️  失败的批次: ${FAILED_BATCHES[@]}"
    echo ""
fi

echo "📊 结果文件:"
echo "   - 合并结果: $RESULTS_DIR/overtaking_results.json"
echo "   - 批次结果: $RESULTS_DIR/batches/batch_*.json"
echo ""

# 清理临时文件
rm -f $RESUME_FILE

echo "📈 查看结果统计:"
echo "   cat $RESULTS_DIR/overtaking_results.json | python3 -m json.tool"
echo ""

echo "🎉 Overtaking能力测试完成!"
