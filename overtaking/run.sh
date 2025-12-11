#!/bin/bash

# Overtaking能力测试 - 快速启动脚本

echo "========================================================================="
echo "🚗 Overtaking 能力测试 - 快速启动"
echo "========================================================================="
echo ""

OVERTAKING_DIR="/home/ajifang/b2drive/overtaking"

# 检查是否在正确的目录
if [ ! -d "$OVERTAKING_DIR" ]; then
    echo "❌ Overtaking目录不存在: $OVERTAKING_DIR"
    exit 1
fi

cd $OVERTAKING_DIR

echo "📋 选择操作:"
echo ""
echo "  1) 运行Overtaking测试 (45条路线，预计6-8小时)"
echo "  2) 查看测试进度"
echo "  3) 分析测试结果"
echo "  4) 重新提取路线"
echo "  5) 清理测试结果"
echo "  6) 查看帮助文档"
echo ""
read -p "请输入选项 [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动Overtaking能力测试..."
        echo ""
        echo "建议使用screen后台运行:"
        echo "  screen -S overtaking_test"
        echo "  ./scripts/test_overtaking.sh"
        echo "  按 Ctrl+A, D 分离screen"
        echo ""
        read -p "是否直接运行测试？(y/n): " confirm
        if [ "$confirm" = "y" ]; then
            ./scripts/test_overtaking.sh
        else
            echo "已取消。你可以手动运行:"
            echo "  cd $OVERTAKING_DIR && ./scripts/test_overtaking.sh"
        fi
        ;;

    2)
        echo ""
        echo "📊 测试进度:"
        echo ""

        if [ -d "results/batches" ]; then
            batch_count=$(ls results/batches/batch_*.json 2>/dev/null | wc -l)
            total_batches=9

            echo "  已完成批次: $batch_count / $total_batches"

            if [ $batch_count -eq 0 ]; then
                echo "  状态: 尚未开始测试"
            elif [ $batch_count -lt $total_batches ]; then
                progress=$((batch_count * 100 / total_batches))
                echo "  进度: $progress%"
                echo "  状态: 测试进行中..."
            else
                echo "  状态: ✅ 测试已完成"
            fi

            echo ""
            echo "实时监控命令:"
            echo "  watch -n 10 'ls $OVERTAKING_DIR/results/batches/*.json 2>/dev/null | wc -l'"
        else
            echo "  尚未开始测试"
        fi
        ;;

    3)
        echo ""
        echo "📈 分析测试结果..."
        echo ""

        if [ -f "results/overtaking_results.json" ]; then
            python3 scripts/analyze_results.py

            echo ""
            echo "分析报告已生成: results/overtaking_analysis.json"
        else
            echo "❌ 测试结果不存在"
            echo "   请先运行测试: 选项 1"
        fi
        ;;

    4)
        echo ""
        echo "🔄 重新提取Overtaking路线..."
        echo ""
        python3 scripts/extract_overtaking_routes.py
        ;;

    5)
        echo ""
        echo "⚠️  警告: 这将删除所有测试结果!"
        read -p "确认清理？(y/n): " confirm

        if [ "$confirm" = "y" ]; then
            rm -rf results/batches/*
            rm -f results/overtaking_results.json
            rm -f results/overtaking_analysis.json
            rm -f results/.resume_state
            echo "✅ 测试结果已清理"
        else
            echo "已取消"
        fi
        ;;

    6)
        echo ""
        echo "📖 查看帮助文档..."
        echo ""

        if command -v less &> /dev/null; then
            less README.md
        else
            cat README.md
        fi
        ;;

    *)
        echo ""
        echo "❌ 无效选项"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "========================================================================="
