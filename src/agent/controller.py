import re
from typing import Any

from src.agent.planner import plan_action
from src.agent.state import (
    clear_pending_action,
    get_session_state,
    update_session_state,
)
from src.skills.registry import get_skill


ORDER_ID_PATTERN = re.compile(r"\b\d{5,20}\b")


def extract_order_id(text: str) -> str:
    """从文本中提取5到20位数字订单号。"""

    match = ORDER_ID_PATTERN.search(text)
    return match.group(0) if match else ""


def build_result(
    question: str,
    action: str,
    reason: str,
    skill_result: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """将Skill结果整理成Controller统一输出。"""

    return {
        "question": question,
        "action": action,
        "reason": reason,
        "skill": skill_result.get("skill"),
        "answer": skill_result.get(
            "answer",
            "没有生成答案。",
        ),
        "sources": skill_result.get("sources", []),
        "waiting_for_input": skill_result.get(
            "waiting_for_input",
            False,
        ),
        "observation": skill_result.get("observation"),
        "trace": skill_result.get("trace", []),
        "session_state": get_session_state(session_id),
    }


def execute_order_skill(
    question: str,
    order_id: str,
    user_id: str | None,
    session_id: str,
    reason: str,
) -> dict[str, Any]:
    """执行订单查询Skill。"""

    skill = get_skill("order_query")

    skill_result = skill.execute(
        question=question,
        order_id=order_id,
        user_id=user_id,
    )

    update_session_state(
        session_id,
        last_observation=skill_result.get("observation"),
    )

    if skill_result.get("waiting_for_input"):
        update_session_state(
            session_id,
            pending_action="query_order",
            pending_parameters=["order_id"],
        )
    else:
        clear_pending_action(session_id)

    return build_result(
        question=question,
        action="query_order",
        reason=reason,
        skill_result=skill_result,
        session_id=session_id,
    )


def execute_faq_skill(
    question: str,
    session_id: str,
    reason: str,
) -> dict[str, Any]:
    """执行FAQ问答Skill。"""

    skill = get_skill("faq_search")

    skill_result = skill.execute(
        question=question,
    )

    update_session_state(
        session_id,
        last_observation=skill_result.get("observation"),
    )

    return build_result(
        question=question,
        action="search_faq",
        reason=reason,
        skill_result=skill_result,
        session_id=session_id,
    )


def run_agent(
    question: str,
    user_id: str | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """运行一次Skill化客服Agent。"""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("用户问题不能为空。")

    state = get_session_state(session_id)

    update_session_state(
        session_id,
        last_question=cleaned_question,
    )

    # 上一轮正在等待订单号。
    if state.get("pending_action") == "query_order":
        order_id = extract_order_id(cleaned_question)

        if not order_id:
            return {
                "question": cleaned_question,
                "action": "ask_order_id",
                "reason": "上一轮订单查询仍缺少订单号。",
                "skill": None,
                "answer": (
                    "请提供5位及以上数字组成的订单号，"
                    "例如10001。"
                ),
                "sources": [],
                "waiting_for_input": True,
                "observation": None,
                "trace": [],
                "session_state": get_session_state(session_id),
            }

        return execute_order_skill(
            question=cleaned_question,
            order_id=order_id,
            user_id=user_id,
            session_id=session_id,
            reason="用户补充了上一轮缺少的订单号。",
        )

    plan = plan_action(cleaned_question)
    action = plan["action"]

    if action == "reject":
        return {
            "question": cleaned_question,
            "action": "reject",
            "reason": plan["reason"],
            "skill": None,
            "answer": (
                "我是客服助手，目前只能处理订单、物流、退款、"
                "支付、账户和商品售后问题。"
            ),
            "sources": [],
            "waiting_for_input": False,
            "observation": None,
            "trace": [
                {
                    "step": 0,
                    "stage": "plan",
                    "content": plan,
                }
            ],
            "session_state": get_session_state(session_id),
        }

    if action == "ask_order_id":
        update_session_state(
            session_id,
            pending_action="query_order",
            pending_parameters=["order_id"],
        )

        return {
            "question": cleaned_question,
            "action": "ask_order_id",
            "reason": plan["reason"],
            "skill": None,
            "answer": "请提供需要查询的订单号，例如10001。",
            "sources": [],
            "waiting_for_input": True,
            "observation": None,
            "trace": [
                {
                    "step": 0,
                    "stage": "plan",
                    "content": plan,
                }
            ],
            "session_state": get_session_state(session_id),
        }

    if action == "query_order":
        order_id = (
            str(plan.get("order_id", "")).strip()
            or extract_order_id(cleaned_question)
        )

        if not order_id:
            update_session_state(
                session_id,
                pending_action="query_order",
                pending_parameters=["order_id"],
            )

            return {
                "question": cleaned_question,
                "action": "ask_order_id",
                "reason": "订单查询缺少订单号。",
                "skill": None,
                "answer": "请提供需要查询的订单号，例如10001。",
                "sources": [],
                "waiting_for_input": True,
                "observation": None,
                "trace": [
                    {
                        "step": 0,
                        "stage": "plan",
                        "content": plan,
                    }
                ],
                "session_state": get_session_state(session_id),
            }

        return execute_order_skill(
            question=cleaned_question,
            order_id=order_id,
            user_id=user_id,
            session_id=session_id,
            reason=plan["reason"],
        )

    if action == "search_faq":
        return execute_faq_skill(
            question=cleaned_question,
            session_id=session_id,
            reason=plan["reason"],
        )

    return {
        "question": cleaned_question,
        "action": "reject",
        "reason": "Controller未识别Planner返回的动作。",
        "skill": None,
        "answer": "暂时无法处理该请求。",
        "sources": [],
        "waiting_for_input": False,
        "observation": None,
        "trace": [],
        "session_state": get_session_state(session_id),
    }
