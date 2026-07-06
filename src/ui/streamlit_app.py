from uuid import uuid4

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE_URL}/chat"
RESET_URL = f"{API_BASE_URL}/reset"
HEALTH_URL = f"{API_BASE_URL}/health"
RECENT_LOGS_URL = f"{API_BASE_URL}/logs"


st.set_page_config(
    page_title="客服 Agent",
    page_icon="🤖",
    layout="wide",
)


def initialize_state() -> None:
    """初始化前端会话状态。"""

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def check_api_health() -> bool:
    """检查 FastAPI 后端是否正常运行。"""

    try:
        response = requests.get(
            HEALTH_URL,
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_agent(question: str) -> dict:
    """调用 FastAPI 聊天接口。"""

    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "user_id": "user_001",
    }

    response = requests.post(
        CHAT_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def fetch_recent_logs(limit: int = 10) -> list[dict]:
    """读取最近日志。"""

    try:
        response = requests.get(
            RECENT_LOGS_URL,
            params={"limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("logs", [])
    except requests.RequestException:
        return []


def fetch_session_logs(limit: int = 10) -> list[dict]:
    """读取当前会话日志。"""

    url = f"{RECENT_LOGS_URL}/{st.session_state.session_id}"

    try:
        response = requests.get(
            url,
            params={"limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("logs", [])
    except requests.RequestException:
        return []


def reset_conversation() -> None:
    """同时清空前端记录和后端会话状态。"""

    old_session_id = st.session_state.session_id

    try:
        requests.post(
            RESET_URL,
            json={
                "session_id": old_session_id,
            },
            timeout=10,
        )
    except requests.RequestException:
        pass

    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.last_result = None


def render_logs_panel() -> None:
    """显示日志面板。"""

    st.subheader("日志面板")

    tab1, tab2 = st.tabs(
        [
            "当前会话日志",
            "最近全部日志",
        ]
    )

    with tab1:
        session_logs = fetch_session_logs(limit=10)

        if not session_logs:
            st.caption("当前会话还没有日志。")
        else:
            for log in session_logs:
                with st.expander(
                    f"#{log.get('id')} | "
                    f"{log.get('action')} / "
                    f"{log.get('skill')} | "
                    f"{log.get('latency_ms')} ms",
                    expanded=False,
                ):
                    st.markdown("**用户问题**")
                    st.write(log.get("question"))

                    st.markdown("**Agent回答**")
                    st.write(log.get("answer"))

                    st.markdown("**关键字段**")
                    st.json(
                        {
                            "session_id": log.get("session_id"),
                            "user_id": log.get("user_id"),
                            "action": log.get("action"),
                            "skill": log.get("skill"),
                            "waiting_for_input": log.get(
                                "waiting_for_input"
                            ),
                            "latency_ms": log.get("latency_ms"),
                            "success": log.get("success"),
                            "error_message": log.get(
                                "error_message"
                            ),
                        }
                    )

                    if log.get("trace"):
                        st.markdown("**Trace**")
                        st.json(log.get("trace"))

                    if log.get("sources"):
                        st.markdown("**Sources**")
                        st.dataframe(
                            log.get("sources"),
                            use_container_width=True,
                            hide_index=True,
                        )

    with tab2:
        recent_logs = fetch_recent_logs(limit=10)

        if not recent_logs:
            st.caption("暂无最近日志，或后端未连接。")
        else:
            st.dataframe(
                recent_logs,
                use_container_width=True,
                hide_index=True,
            )


def render_sidebar() -> None:
    """显示侧边栏。"""

    with st.sidebar:
        st.title("系统信息")

        api_ok = check_api_health()

        if api_ok:
            st.success("FastAPI 后端已连接")
        else:
            st.error("FastAPI 后端未连接")

        st.caption("当前会话 ID")
        st.code(st.session_state.session_id)

        if st.button(
            "清空当前对话",
            use_container_width=True,
        ):
            reset_conversation()
            st.rerun()

        st.divider()

        if st.session_state.last_result:
            result = st.session_state.last_result

            st.subheader("上次请求")
            st.metric(
                "耗时",
                f"{result.get('latency_ms', '未知')} ms",
            )
            st.metric(
                "动作",
                result.get("action") or "无",
            )
            st.metric(
                "Skill",
                result.get("skill") or "无",
            )

        st.divider()

        st.subheader("当前能力")
        st.markdown(
            """
- FAQ 知识库问答
- 订单状态查询
- 多轮参数补全
- 权限校验
- Agent Trace
- SQLite 日志记录
            """
        )

        st.divider()

        st.caption(
            "提示：FastAPI 服务和 Streamlit 页面"
            "需要分别在两个终端中运行。"
        )


def render_debug_panel(result: dict) -> None:
    """显示 Agent 内部执行信息。"""

    with st.expander(
        "查看 Agent 执行详情",
        expanded=False,
    ):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Planner 动作",
                result.get("action") or "无",
            )

        with col2:
            st.metric(
                "执行 Skill",
                result.get("skill") or "无",
            )

        with col3:
            waiting = result.get(
                "waiting_for_input",
                False,
            )
            st.metric(
                "等待补充",
                "是" if waiting else "否",
            )

        with col4:
            st.metric(
                "耗时",
                f"{result.get('latency_ms', '未知')} ms",
            )

        st.markdown("#### 动作原因")
        st.write(
            result.get("reason")
            or "没有返回动作原因。"
        )

        sources = result.get("sources", [])

        st.markdown("#### 检索来源")

        if sources:
            st.dataframe(
                sources,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("本次没有使用 FAQ 检索来源。")

        trace = result.get("trace", [])

        st.markdown("#### Agent Trace")

        if trace:
            st.json(trace)
        else:
            st.caption("本次没有返回执行轨迹。")

        st.markdown("#### 会话状态")
        st.json(
            result.get("session_state", {})
        )


def render_chat_page() -> None:
    """显示聊天页面。"""

    st.title("🤖 客服 Agent")
    st.caption(
        "基于 DeepSeek、FAISS、RAG、Skill、Tool Calling 和日志系统"
    )

    api_ok = check_api_health()

    if not api_ok:
        st.warning(
            "尚未连接 FastAPI 后端。请先在另一个终端运行："
        )
        st.code(
            "python -m uvicorn src.api.app:app --reload"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "请输入订单、物流、退款、支付或账户问题"
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        if not api_ok:
            error_message = (
                "FastAPI 后端尚未启动，无法发送问题。"
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

            with st.chat_message("assistant"):
                st.error(error_message)

            return

        with st.chat_message("assistant"):
            with st.spinner("Agent 正在处理..."):
                try:
                    result = call_agent(question)
                    answer = result.get(
                        "answer",
                        "没有生成答案。",
                    )

                    st.markdown(answer)
                    render_debug_panel(result)

                    st.session_state.last_result = result

                except requests.Timeout:
                    answer = (
                        "请求超时，请稍后重新尝试。"
                    )
                    st.error(answer)

                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get(
                            "detail",
                            str(exc),
                        )
                    except Exception:
                        detail = str(exc)

                    answer = (
                        f"后端接口返回错误：{detail}"
                    )
                    st.error(answer)

                except requests.RequestException as exc:
                    answer = (
                        f"无法连接后端服务：{exc}"
                    )
                    st.error(answer)

                except Exception as exc:
                    answer = (
                        f"页面处理失败：{exc}"
                    )
                    st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


def main() -> None:
    initialize_state()
    render_sidebar()

    tab_chat, tab_logs = st.tabs(
        [
            "客服对话",
            "日志与监控",
        ]
    )

    with tab_chat:
        render_chat_page()

    with tab_logs:
        render_logs_panel()


if __name__ == "__main__":
    main()
