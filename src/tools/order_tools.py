import json
from pathlib import Path
from typing import Any

from src.settings import MOCK_ORDERS_PATH, PROJECT_ROOT


def get_orders_path() -> Path:
    """获取模拟订单文件路径。"""

    path = Path(MOCK_ORDERS_PATH)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_orders() -> list[dict[str, Any]]:
    """读取模拟订单数据。"""

    orders_path = get_orders_path()

    if not orders_path.exists():
        return []

    try:
        content = orders_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [
        order
        for order in data
        if isinstance(order, dict)
    ]


def query_order(
    order_id: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    查询订单。

    返回结构：
    {
        "success": bool,
        "error": str | None,
        "message": str,
        "order": dict | None
    }
    """

    cleaned_order_id = str(order_id).strip()

    if not cleaned_order_id:
        return {
            "success": False,
            "error": "missing_order_id",
            "message": "缺少订单号，请提供订单号。",
            "order": None,
        }

    orders = load_orders()

    matched_order = None

    for order in orders:
        if str(order.get("order_id")) == cleaned_order_id:
            matched_order = order
            break

    if matched_order is None:
        return {
            "success": False,
            "error": "order_not_found",
            "message": f"没有找到订单 {cleaned_order_id}，请检查订单号是否正确。",
            "order": None,
        }

    if user_id:
        owner_user_id = str(
            matched_order.get("user_id", "")
        )

        if owner_user_id != str(user_id):
            return {
                "success": False,
                "error": "permission_denied",
                "message": "该订单不属于当前用户，无法查看订单详情。",
                "order": None,
            }

    return {
        "success": True,
        "error": None,
        "message": "订单查询成功。",
        "order": matched_order,
    }


if __name__ == "__main__":
    print("订单文件路径：", get_orders_path())

    tests = [
        {"order_id": "10001", "user_id": "user_001"},
        {"order_id": "10001", "user_id": "user_002"},
        {"order_id": "99999", "user_id": "user_001"},
    ]

    for test in tests:
        print("=" * 60)
        print(test)
        print(query_order(**test))
