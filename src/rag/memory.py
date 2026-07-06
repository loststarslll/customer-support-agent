import json
from pathlib import Path


MEMORY_PATH = Path("data/memory.json")


def load_memory():
    if not MEMORY_PATH.exists():
        return {}
    return json.loads(MEMORY_PATH.read_text("utf-8"))


def save_memory(data):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def store_memory(user_id: str, memory_item: dict):
    data = load_memory()

    if user_id not in data:
        data[user_id] = []

    data[user_id].append(memory_item)

    save_memory(data)


def get_memory(user_id: str):
    data = load_memory()
    return data.get(user_id, [])
