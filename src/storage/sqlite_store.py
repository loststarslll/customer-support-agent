import json
import sqlite3
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "agent_logs.sqlite3"


def get_connection() -> sqlite3.Connection:
    """创建SQLite连接。"""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def init_db() -> None:
    """初始化数据库表。"""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT,
                question TEXT NOT NULL,
                action TEXT,
                skill TEXT,
                answer TEXT,
                waiting_for_input INTEGER DEFAULT 0,
                latency_ms REAL,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                sources_json TEXT,
                trace_json TEXT,
                session_state_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id
            ON chat_logs(session_id)
            """
        )


def safe_json_dumps(value: Any) -> str:
    """安全转换为JSON字符串。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except TypeError:
        return "null"


def insert_chat_log(
    session_id: str,
    user_id: str | None,
    question: str,
    result: dict[str, Any] | None,
    latency_ms: float,
    success: bool,
    error_message: str | None = None,
) -> int:
    """插入一条聊天日志。"""

    init_db()

    result = result or {}

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO chat_logs (
                session_id,
                user_id,
                question,
                action,
                skill,
                answer,
                waiting_for_input,
                latency_ms,
                success,
                error_message,
                sources_json,
                trace_json,
                session_state_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                question,
                result.get("action"),
                result.get("skill"),
                result.get("answer"),
                1 if result.get("waiting_for_input") else 0,
                latency_ms,
                1 if success else 0,
                error_message,
                safe_json_dumps(result.get("sources", [])),
                safe_json_dumps(result.get("trace", [])),
                safe_json_dumps(result.get("session_state", {})),
                int(time.time()),
            ),
        )

        return int(cursor.lastrowid)


def list_chat_logs(
    session_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """读取某个会话的聊天日志。"""

    init_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                session_id,
                user_id,
                question,
                action,
                skill,
                answer,
                waiting_for_input,
                latency_ms,
                success,
                error_message,
                sources_json,
                trace_json,
                session_state_json,
                created_at
            FROM chat_logs
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                session_id,
                limit,
            ),
        ).fetchall()

    logs: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)

        for json_field in [
            "sources_json",
            "trace_json",
            "session_state_json",
        ]:
            raw_value = item.get(json_field)

            try:
                item[json_field.replace("_json", "")] = json.loads(
                    raw_value
                    if raw_value
                    else "null"
                )
            except json.JSONDecodeError:
                item[json_field.replace("_json", "")] = None

            item.pop(json_field, None)

        item["waiting_for_input"] = bool(
            item.get("waiting_for_input")
        )
        item["success"] = bool(
            item.get("success")
        )

        logs.append(item)

    return logs


def get_recent_logs(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """读取最近的聊天日志。"""

    init_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                session_id,
                user_id,
                question,
                action,
                skill,
                answer,
                waiting_for_input,
                latency_ms,
                success,
                error_message,
                created_at
            FROM chat_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    logs = []

    for row in rows:
        item = dict(row)
        item["waiting_for_input"] = bool(
            item.get("waiting_for_input")
        )
        item["success"] = bool(
            item.get("success")
        )
        logs.append(item)

    return logs


if __name__ == "__main__":
    init_db()

    log_id = insert_chat_log(
        session_id="demo-session",
        user_id="user_001",
        question="帮我查订单10001",
        result={
            "action": "query_order",
            "skill": "order_query",
            "answer": "订单查询成功。",
            "waiting_for_input": False,
            "sources": [],
            "trace": [],
            "session_state": {},
        },
        latency_ms=123.4,
        success=True,
    )

    print("inserted:", log_id)
    print(list_chat_logs("demo-session"))
