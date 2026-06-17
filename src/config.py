"""
文件作用：读取配置，检查配置，不调用大模型。
"""

import os

from dotenv import load_dotenv


load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")


def validate_config() -> None:
    """检查必要的环境变量是否已经配置。"""
    if not LLM_API_KEY:
        raise ValueError(
            "没有读取到 LLM_API_KEY，请检查项目根目录下的 .env 文件。"
        )