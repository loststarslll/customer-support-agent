from uuid import uuid4

import requests
import streamlit as st

from src.settings import API_BASE_URL, DEFAULT_USER_ID


CHAT_URL = f"{API_BASE_URL}/chat"
RESET_URL = f"{API_BASE_URL}/reset"
HEALTH_URL = f"{API_BASE_URL}/health"


st.set_page_config(
    page_title="在线客服",
    page_icon="💬",
    layout="centered",
)


def initialize_state() -> None:
    """初始化用户端会话状态。"""

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID


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


def call_agent(question: str) -> dict:
    """调用客服 Agent 接口。"""

    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
    }

    response = requests.post(
        CHAT_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    return response.json()


def reset_conversation() -> None:
    """清空当前用户会话。"""

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


def render_sidebar() -> None:
    """用户端侧边栏。"""

    with st.sidebar:
        st.title("客服中心")

        if check_api_health():
            st.success("客服在线")
        else:
            st.error("客服暂时不可用")

        st.caption("当前模拟用户")
        st.session_state.user_id = st.selectbox(
            "选择用户ID",
            options=[
                "user_001",
                "user_002",
                "user_003",
            ],
            index=[
                "user_001",
                "user_002",
                "user_003",
            ].index(st.session_state.user_id)
            if st.session_state.user_id in [
                "user_001",
                "user_002",
                "user_003",
            ]
            else 0,
        )

        st.markdown(
            """
你可以咨询：

- 订单状态
- 物流进度
- 退款问题
- 支付问题
- 账户问题
- 商品售后
            """
        )

        if st.button(
            "重新开始对话",
            use_container_width=True,
        ):
            reset_conversation()
            st.rerun()

        st.caption(
            "提示：请不要在对话中输入银行卡、密码等敏感信息。"
        )


def main() -> None:
    """用户端客服聊天页面。"""

    initialize_state()
    render_sidebar()

    st.title("💬 在线客服")
    st.caption("请描述你的问题，我会尽力帮你处理。")

    api_ok = check_api_health()

    if not api_ok:
        st.warning(
            "客服服务暂时不可用，请稍后再试。"
        )

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "你好，我是在线客服助手。"
                "你可以问我订单、物流、退款、支付或账户相关问题。"
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "请输入你的问题"
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_ok:
            answer = "客服服务暂时不可用，请稍后再试。"
            st.error(answer)
        else:
            with st.spinner("正在处理，请稍候..."):
                try:
                    result = call_agent(question)
                    answer = result.get(
                        "answer",
                        "暂时没有生成回答。",
                    )
                    st.markdown(answer)

                    if result.get("waiting_for_input"):
                        st.info(
                            "请继续补充上面需要的信息。"
                        )

                except requests.Timeout:
                    answer = (
                        "处理时间较长，请稍后重试。"
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
                        f"请求处理失败：{detail}"
                    )
                    st.error(answer)

                except requests.RequestException:
                    answer = (
                        "暂时无法连接客服服务，请稍后再试。"
                    )
                    st.error(answer)

                except Exception as exc:
                    answer = (
                        f"页面发生错误：{exc}"
                    )
                    st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


if __name__ == "__main__":
    main()
