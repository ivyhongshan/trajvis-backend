#!/bin/bash
set -e

echo ">>> Starting Flask app..."
python3 main.py > /tmp/flask.log 2>&1 &
PID=$!

# 等待 Flask 启动
sleep 15

# 打印健康检查
echo ">>> Checking health..."
curl -s http://127.0.0.1:8080/health || echo "health failed"

# 打印 routes
echo -e "\n>>> Checking routes..."
curl -s http://127.0.0.1:8080/__routes__ || echo "routes failed"

# 查看日志
echo -e "\n>>> Tail logs (press Ctrl+C to stop)..."
tail -f /tmp/flask.log

# 保持 Flask 进程
wait $PID
