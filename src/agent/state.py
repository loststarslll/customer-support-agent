from typing import Any


# 当前先使用进程内存保存会话状态。
# 程序退出后会清空，后续再改成持久化存储。
_SESSION_STATES: dict[str, dict[str, Any]] = {}


def get_session_state(session_id: str) -> dict[str, Any]:
    """读取指定会话状态。"""

    return _SESSION_STATES.setdefault(
        session_id,
        {
            "pending_action": None,
            "pending_parameters": [],
            "last_question": None,
            "last_observation": None,
        },
    )


def update_session_state(
    session_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """更新指定会话状态。"""

    state = get_session_state(session_id)
    state.update(updates)
    return state


def clear_pending_action(session_id: str) -> None:
    """清除已经完成或取消的待执行动作。"""

    update_session_state(
        session_id,
        pending_action=None,
        pending_parameters=[],
    )


def reset_session_state(session_id: str) -> None:
    """清空整个会话。"""

    _SESSION_STATES.pop(session_id, None)


if __name__ == "__main__":
    print(get_session_state("demo"))
    update_session_state(
        "demo",
        pending_action="query_order",
        pending_parameters=["order_id"],
    )
    print(get_session_state("demo"))
    clear_pending_action("demo")
    print(get_session_state("demo"))
