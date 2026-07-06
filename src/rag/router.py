from src.llm import create_llm


SYSTEM_PROMPT = """
You are a router for a customer support system.

Classify the user question into exactly ONE category:

- support_question: refund, delivery, order, payment, account, product issues
- non_support_question: poetry, coding, general knowledge, casual chat

Return ONLY one label:
support_question OR non_support_question
"""


def route_question(question: str) -> str:
    llm = create_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    res = llm.invoke(messages)

    return res.content.strip()
