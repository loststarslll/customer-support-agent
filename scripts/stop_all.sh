#!/bin/bash

# 停止智能客服 Agent 系统

echo "正在停止智能客服 Agent 系统..."

if [ -f ".api.pid" ]; then
  kill $(cat .api.pid) 2>/dev/null
  rm .api.pid
  echo "已停止 FastAPI 后端"
fi

if [ -f ".customer_ui.pid" ]; then
  kill $(cat .customer_ui.pid) 2>/dev/null
  rm .customer_ui.pid
  echo "已停止 Streamlit 用户端"
fi

if [ -f ".admin_ui.pid" ]; then
  kill $(cat .admin_ui.pid) 2>/dev/null
  rm .admin_ui.pid
  echo "已停止 Streamlit 商家端"
fi

echo "全部服务已停止。"
