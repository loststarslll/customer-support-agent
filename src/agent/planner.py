import json
from typing import Any

from src.llm import create_llm


PLANNER_SYSTEM_PROMPT = """
You are the planner of a customer support agent.

Analyze the user's message and choose exactly one action.

Available actions:

1. query_order
Use this when the user wants to check a specific order, shipment,
delivery status, tracking information, or refund status and provides
an order number.

2. ask_order_id
Use this when the user wants to check an order but has not provided
an order number.

3. search_faq
Use this for general customer support policy questions, such as
refund rules, payment problems, account issues, exchanges, damaged
products, and general delivery questions.

4. reject
Use this for questions unrelated to customer support, such as poetry,
coding, general knowledge, or casual conversation.

Return valid JSON only:

{
  "action": "query_order | ask_order_id | search_faq | reject",
  "order_id": "order number or empty string",
  "reason": "brief reason"
}

Do not return Markdown.
Do not call tools yourself.
""".strip()


ALLOWED_ACTIONS = {
    "query_order",
    "ask_order_id",
    "search_faq",
    "reject",
}


def clean_json_text(text: str) -> str:
    """清理模型偶尔返回的 Markdown 代码块。"""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def plan_action(question: str) -> dict[str, Any]:
    """让 LLM 为当前问题选择候选动作。"""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("用户问题不能为空。")

    llm = create_llm()

    messages = [
        {
            "role": "system",
            "content": PLANNER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": cleaned_question,
        },
    ]

    response = llm.invoke(messages)
    raw_content = clean_json_text(str(response.content))

    try:
        plan = json.loads(raw_content)
    except json.JSONDecodeError:
        return {
            "action": "reject",
            "order_id": "",
            "reason": "模型没有返回有效JSON。",
        }

    action = str(plan.get("action", "")).strip()
    order_id = str(plan.get("order_id", "")).strip()
    reason = str(plan.get("reason", "")).strip()

    if action not in ALLOWED_ACTIONS:
        return {
            "action": "reject",
            "order_id": "",
            "reason": "模型返回了不允许的动作。",
        }

    return {
        "action": action,
        "order_id": order_id,
        "reason": reason,
    }


if __name__ == "__main__":
    test_questions = [
        "帮我查一下订单10001到哪里了",
        "我的订单怎么还没到？",
        "退款一般多久到账？",
        "帮我写一首诗",
    ]

    for test_question in test_questions:
        print("\n问题：", test_question)
        print("计划：", plan_action(test_question))
