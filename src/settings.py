import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_str(
    name: str,
    default: str,
) -> str:
    """读取字符串配置。"""

    return os.getenv(name, default).strip()


def get_int(
    name: str,
    default: int,
) -> int:
    """读取整数配置。"""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def get_float(
    name: str,
    default: float,
) -> float:
    """读取浮点数配置。"""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


API_BASE_URL = get_str(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

DEFAULT_USER_ID = get_str(
    "DEFAULT_USER_ID",
    "user_001",
)

RETRIEVAL_TOP_K = get_int(
    "RETRIEVAL_TOP_K",
    3,
)

RETRIEVAL_THRESHOLD = get_float(
    "RETRIEVAL_THRESHOLD",
    0.9,
)

FAQ_MAX_RETRIES = get_int(
    "FAQ_MAX_RETRIES",
    1,
)

MOCK_ORDERS_PATH = get_str(
    "MOCK_ORDERS_PATH",
    "data/mock/orders.json",
)
