import json
from typing import Any

from src.llm import create_llm


REVIEW_SYSTEM_PROMPT = """
You are the reviewer in a customer support agent.

You receive:
1. The original user question
2. The action that was executed
3. The observation returned by the tool or retrieval system

Choose exactly one next decision:

- finish
  The observation is sufficient to answer the user.

- retry_search
  The FAQ retrieval result is insufficient, but rewriting the search query
  may help.

- ask_user
  Important information is missing and the user must provide it.

- reject
  The request cannot be safely or correctly handled.

Return valid JSON only:

{
  "decision": "finish | retry_search | ask_user | reject",
  "rewritten_query": "rewritten search query or empty string",
  "message": "brief explanation or question for the user"
}

Rules:
- Never invent order information.
- Never invent customer support policies.
- Use retry_search only for FAQ retrieval.
- If an order tool returns success=true, choose finish.
- If an order number is missing, choose ask_user.
- Do not return Markdown.
""".strip()


ALLOWED_DECISIONS = {
    "finish",
    "retry_search",
    "ask_user",
    "reject",
}


def clean_json_text(text: str) -> str:
    """移除模型偶尔返回的 Markdown 代码块。"""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def review_observation(
    question: str,
    action: str,
    observation: dict[str, Any],
) -> dict[str, str]:
    """让 LLM 根据执行结果建议下一步。"""

    llm = create_llm()

    payload = {
        "question": question,
        "action": action,
        "observation": observation,
    }

    messages = [
        {
            "role": "system",
            "content": REVIEW_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
        },
    ]

    response = llm.invoke(messages)
    raw_content = clean_json_text(str(response.content))

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        return {
            "decision": "reject",
            "rewritten_query": "",
            "message": "Reviewer没有返回有效JSON。",
        }

    decision = str(result.get("decision", "")).strip()
    rewritten_query = str(
        result.get("rewritten_query", "")
    ).strip()
    message = str(result.get("message", "")).strip()

    if decision not in ALLOWED_DECISIONS:
        return {
            "decision": "reject",
            "rewritten_query": "",
            "message": "Reviewer返回了不允许的决策。",
        }

    if decision != "retry_search":
        rewritten_query = ""

    return {
        "decision": decision,
        "rewritten_query": rewritten_query,
        "message": message,
    }


if __name__ == "__main__":
    examples = [
        {
            "question": "帮我查订单10001",
            "action": "query_order",
            "observation": {
                "success": True,
                "order": {
                    "order_id": "10001",
                    "status": "shipped",
                },
            },
        },
        {
            "question": "我的商品好像有问题",
            "action": "search_faq",
            "observation": {
                "document_count": 0,
                "documents": [],
            },
        },
    ]

    for example in examples:
        print(
            review_observation(
                question=example["question"],
                action=example["action"],
                observation=example["observation"],
            )
        )
