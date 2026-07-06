from src.llm import create_llm
import json


SYSTEM_PROMPT = """
You are a memory extraction system for a customer support agent.

Your task:
Decide whether the user's message contains information worth storing.

You must extract ONLY important long-term or task-related memory.

Return JSON format:

{
  "should_save": true/false,
  "memory_type": "user_profile | task | preference | none",
  "content": "clean memory content or empty"
}

Rules:
- Ignore greetings and irrelevant chat
- Extract user identity, preferences, orders, tasks
- Be concise
"""


def extract_memory(question: str) -> dict:
    llm = create_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    res = llm.invoke(messages)

    try:
        return json.loads(res.content)
    except:
        return {
            "should_save": False,
            "memory_type": "none",
            "content": ""
        }
