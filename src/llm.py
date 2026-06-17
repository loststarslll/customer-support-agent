"""
文件作用：创建大模型对象，以后所有模块都从这里获取模型，不要每个文件重复写 API 配置。
"""
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL_ID,
    validate_config,
)


def create_llm() -> ChatOpenAI:
    """创建并返回统一的大模型客户端。"""
    validate_config()

    return ChatOpenAI(
        api_key=LLM_API_KEY,
        model=LLM_MODEL_ID,
        base_url=LLM_BASE_URL,
        temperature=0,
    )