#!/bin/bash

# Pi0.5 异步推理测试启动脚本

echo "🚀 Pi0.5 异步推理测试启动脚本"
echo "=================================="

# 检查参数
MODE=${1:-"quick"}
HOST=${2:-"localhost"}
PORT=${3:-8765}

echo "📋 测试配置:"
echo "   模式: $MODE"
echo "   服务器: $HOST:$PORT"
echo ""

# 检查服务器是否运行
echo "🔍 检查服务器状态..."
if curl -s "http://$HOST:$PORT/healthz" > /dev/null 2>&1; then
    echo "✅ 服务器正在运行"
else
    echo "❌ 服务器未运行，请先启动服务器"
    echo "💡 启动命令: python async_pi05_websocket_server.py"
    exit 1
fi

echo ""

# 运行测试
case $MODE in
    "quick")
        echo "⚡ 运行快速测试..."
        python quick_test.py
        ;;
    "refresh")
        echo "🔄 运行子任务刷新测试..."
        python test_subtask_refresh.py --test-mode refresh --host $HOST --port $PORT
        ;;
    "evolution")
        echo "🧬 运行子任务演化测试..."
        python test_subtask_refresh.py --test-mode evolution --host $HOST --port $PORT
        ;;
    "consistency")
        echo "🎯 运行一致性测试..."
        python test_subtask_refresh.py --test-mode consistency --host $HOST --port $PORT
        ;;
    "all")
        echo "🎯 运行完整测试套件..."
        python test_subtask_refresh.py --test-mode all --host $HOST --port $PORT
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "💡 可用模式: quick, refresh, evolution, consistency, all"
        echo ""
        echo "📖 使用说明:"
        echo "   quick        - 快速测试子任务刷新功能"
        echo "   refresh      - 测试定期刷新循环"
        echo "   evolution    - 测试子任务演化过程"
        echo "   consistency  - 测试子任务一致性"
        echo "   all          - 运行所有测试"
        exit 1
        ;;
esac

echo ""
echo "✅ 测试完成！"
