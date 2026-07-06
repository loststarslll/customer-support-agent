#!/bin/bash

# 一键启动智能客服 Agent 系统
# 启动内容：
# 1. FastAPI 后端
# 2. Streamlit 用户端
# 3. Streamlit 商家端

PROJECT_DIR="/Users/lvlvlv/study_materials/rag/customer-support-agent"

cd "$PROJECT_DIR" || exit 1

echo "进入项目目录：$PROJECT_DIR"

if [ ! -d "venv" ]; then
  echo "未找到 venv 虚拟环境，请先创建并安装依赖。"
  exit 1
fi

source venv/bin/activate

export PYTHONPATH=.

echo "正在启动 FastAPI 后端..."
python -m uvicorn src.api.app:app --reload --port 8000 > logs_api.txt 2>&1 &
API_PID=$!

sleep 3

echo "正在启动 Streamlit 用户端..."
python -m streamlit run src/ui/customer_app.py --server.port 8501 > logs_customer_ui.txt 2>&1 &
CUSTOMER_PID=$!

sleep 2

echo "正在启动 Streamlit 商家端..."
python -m streamlit run src/ui/admin_app.py --server.port 8502 > logs_admin_ui.txt 2>&1 &
ADMIN_PID=$!

echo ""
echo "======================================"
echo "智能客服 Agent 系统已启动"
echo "======================================"
echo "FastAPI 后端：http://127.0.0.1:8000"
echo "API 文档：http://127.0.0.1:8000/docs"
echo "用户端页面：http://localhost:8501"
echo "商家端后台：http://localhost:8502"
echo ""
echo "进程 PID："
echo "FastAPI: $API_PID"
echo "用户端: $CUSTOMER_PID"
echo "商家端: $ADMIN_PID"
echo ""
echo "日志文件："
echo "FastAPI: logs_api.txt"
echo "用户端: logs_customer_ui.txt"
echo "商家端: logs_admin_ui.txt"
echo ""
echo "停止服务请运行：bash scripts/stop_all.sh"
echo "======================================"

echo "$API_PID" > .api.pid
echo "$CUSTOMER_PID" > .customer_ui.pid
echo "$ADMIN_PID" > .admin_ui.pid
