import requests
import streamlit as st

from src.settings import API_BASE_URL


HEALTH_URL = f"{API_BASE_URL}/health"
SKILLS_URL = f"{API_BASE_URL}/skills"
RECENT_LOGS_URL = f"{API_BASE_URL}/logs"


st.set_page_config(
    page_title="客服Agent管理后台",
    page_icon="📊",
    layout="wide",
)


def check_api_health() -> bool:
    """检查后端服务是否在线。"""

    try:
        response = requests.get(
            HEALTH_URL,
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def fetch_skills() -> list[dict]:
    """读取 Skill 列表。"""

    try:
        response = requests.get(
            SKILLS_URL,
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("skills", [])
    except requests.RequestException:
        return []


def fetch_recent_logs(limit: int = 20) -> list[dict]:
    """读取最近全部日志。"""

    try:
        response = requests.get(
            RECENT_LOGS_URL,
            params={"limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.RequestException:
        return []


def fetch_session_logs(
    session_id: str,
    limit: int = 20,
) -> list[dict]:
    """读取指定会话日志。"""

    if not session_id.strip():
        return []

    url = f"{RECENT_LOGS_URL}/{session_id.strip()}"

    try:
        response = requests.get(
            url,
            params={"limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.RequestException:
        return []


def render_overview(logs: list[dict]) -> None:
    """显示概览指标。"""

    total = len(logs)
    success_count = sum(
        1
        for log in logs
        if log.get("success")
    )
    failure_count = total - success_count

    latencies = [
        float(log.get("latency_ms"))
        for log in logs
        if log.get("latency_ms") is not None
    ]

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("日志条数", total)
    col2.metric("成功数", success_count)
    col3.metric("失败数", failure_count)
    col4.metric(
        "平均耗时",
        f"{avg_latency:.2f} ms",
    )


def render_logs_table(logs: list[dict]) -> None:
    """显示日志表格。"""

    if not logs:
        st.info("暂无日志。")
        return

    compact_logs = []

    for log in logs:
        compact_logs.append(
            {
                "id": log.get("id"),
                "session_id": log.get("session_id"),
                "user_id": log.get("user_id"),
                "question": log.get("question"),
                "action": log.get("action"),
                "skill": log.get("skill"),
                "latency_ms": log.get("latency_ms"),
                "success": log.get("success"),
                "waiting": log.get("waiting_for_input"),
                "error": log.get("error_message"),
                "created_at": log.get("created_at"),
            }
        )

    st.dataframe(
        compact_logs,
        use_container_width=True,
        hide_index=True,
    )


def render_log_details(logs: list[dict]) -> None:
    """显示日志详情。"""

    if not logs:
        return

    st.subheader("日志详情")

    log_options = {
        f"#{log.get('id')} | {log.get('action')} | {log.get('question')}": log
        for log in logs
    }

    selected_label = st.selectbox(
        "选择一条日志查看详情",
        options=list(log_options.keys()),
    )

    selected_log = log_options[selected_label]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 用户问题")
        st.write(selected_log.get("question"))

        st.markdown("#### Agent回答")
        st.write(selected_log.get("answer"))

        st.markdown("#### 基础信息")
        st.json(
            {
                "id": selected_log.get("id"),
                "session_id": selected_log.get("session_id"),
                "user_id": selected_log.get("user_id"),
                "action": selected_log.get("action"),
                "skill": selected_log.get("skill"),
                "latency_ms": selected_log.get("latency_ms"),
                "success": selected_log.get("success"),
                "waiting_for_input": selected_log.get(
                    "waiting_for_input"
                ),
                "error_message": selected_log.get(
                    "error_message"
                ),
                "created_at": selected_log.get("created_at"),
            }
        )

    with col2:
        st.markdown("#### Trace")
        trace = selected_log.get("trace")

        if trace:
            st.json(trace)
        else:
            st.caption("该日志没有 Trace。")

        st.markdown("#### Sources")
        sources = selected_log.get("sources")

        if sources:
            st.dataframe(
                sources,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("该日志没有 FAQ 检索来源。")

        st.markdown("#### Session State")
        session_state = selected_log.get("session_state")

        if session_state:
            st.json(session_state)
        else:
            st.caption("该日志没有会话状态。")


def render_sidebar() -> None:
    """管理端侧边栏。"""

    with st.sidebar:
        st.title("管理后台")

        if check_api_health():
            st.success("FastAPI 后端在线")
        else:
            st.error("FastAPI 后端未连接")

        st.caption("API 地址")
        st.code(API_BASE_URL)

        st.divider()

        st.subheader("已注册 Skill")
        skills = fetch_skills()

        if skills:
            for skill in skills:
                st.markdown(
                    f"- **{skill.get('name')}**："
                    f"{skill.get('description')}"
                )
        else:
            st.caption("暂无 Skill 信息。")

        st.divider()

        st.caption(
            "该页面面向商家、运营和开发者，"
            "用于查看日志、Trace 和系统状态。"
        )


def main() -> None:
    """商家/管理员后台页面。"""

    render_sidebar()

    st.title("📊 客服 Agent 管理后台")
    st.caption(
        "用于查看客服请求日志、耗时、Skill 调用和 Agent Trace。"
    )

    tab_recent, tab_session = st.tabs(
        [
            "最近日志",
            "按会话查询",
        ]
    )

    with tab_recent:
        st.subheader("最近日志")

        limit = st.slider(
            "最近日志条数",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )

        logs = fetch_recent_logs(limit=limit)

        render_overview(logs)
        render_logs_table(logs)
        render_log_details(logs)

    with tab_session:
        st.subheader("按 session_id 查询")

        session_id = st.text_input(
            "请输入 session_id",
            placeholder="例如：demo-session",
        )

        session_limit = st.slider(
            "会话日志条数",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="session_limit",
        )

        if session_id:
            session_logs = fetch_session_logs(
                session_id=session_id,
                limit=session_limit,
            )

            render_overview(session_logs)
            render_logs_table(session_logs)
            render_log_details(session_logs)
        else:
            st.info("请输入 session_id 后查询。")


if __name__ == "__main__":
    main()
