import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FAQ_PATH = PROJECT_ROOT / "data" / "public" / "processed" / "public_faq_corpus_zh.csv"
OUTPUT_CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases_public_zh.json"
CACHE_PATH = PROJECT_ROOT / "data" / "public" / "processed" / "oral_query_cache_zh.json"

MAX_PER_CATEGORY = int(os.getenv("PUBLIC_EVAL_MAX_PER_CATEGORY", "8"))


def load_dotenv_simple() -> None:
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}

    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON 对象：{text[:200]}")

    return json.loads(text[start:end + 1])


def generate_oral_query(question_zh: str, answer_zh: str, category: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请先配置 .env")

    url = f"{base_url}/chat/completions"

    prompt = f"""
请把下面这个标准客服 FAQ 问题改写成一个真实用户会说的中文口语化问题。

要求：
1. 只改写问题，不要回答。
2. 要像真实用户输入客服系统的话，可以不完整、带口语。
3. 不要直接照抄原问题。
4. 不要改变问题意图。
5. 不要出现订单号、手机号、身份证号等真实隐私信息。
6. 只输出 JSON。

客服类别：{category}

标准 FAQ 问题：
{question_zh}

FAQ 答案：
{answer_zh}

输出格式：
{{
  "oral_query": "..."
}}
""".strip()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是客服评测集构造助手，只输出合法 JSON。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek API HTTPError: {exc.code} {body}") from exc

    content = response_data["choices"][0]["message"]["content"]
    obj = extract_json_object(content)
    oral_query = clean_text(obj.get("oral_query", ""))

    if not oral_query:
        raise ValueError(f"口语化结果为空：{obj}")

    return oral_query


def main() -> None:
    load_dotenv_simple()

    if not PUBLIC_FAQ_PATH.exists():
        raise FileNotFoundError(
            f"未找到中文公开 FAQ：{PUBLIC_FAQ_PATH}，请先运行 scripts/prepare_public_faq_zh.py"
        )

    rows = list(csv.DictReader(PUBLIC_FAQ_PATH.open("r", encoding="utf-8")))

    grouped = defaultdict(list)

    for row in rows:
        category = row.get("category", "general")
        question_zh = row.get("question_zh") or row.get("question")
        answer_zh = row.get("answer_zh") or row.get("answer")
        faq_id = row.get("id", "").strip()

        if not question_zh or not answer_zh or not faq_id:
            continue

        grouped[category].append(row)

    random.seed(42)
    cache = load_cache()
    cases = []

    for category, items in grouped.items():
        random.shuffle(items)

        for row in items[:MAX_PER_CATEGORY]:
            question_zh = row.get("question_zh") or row.get("question")
            answer_zh = row.get("answer_zh") or row.get("answer")
            cache_key = f"{row['id']}|||{question_zh}"

            if cache_key in cache:
                oral_query = cache[cache_key]
            else:
                print(f"正在生成口语问法：{row['id']} | {question_zh[:60]}")
                oral_query = generate_oral_query(
                    question_zh=question_zh,
                    answer_zh=answer_zh,
                    category=category,
                )
                cache[cache_key] = oral_query
                save_cache(cache)
                time.sleep(0.2)

            cases.append(
                {
                    "id": f"public_zh_retrieval_{len(cases) + 1:04d}",
                    "question": oral_query,
                    "standard_question": question_zh,
                    "expected_faq_id": row["id"],
                    "expected_category": row["category"],
                    "source_title": row.get("source_title", ""),
                    "dataset_source": row.get("dataset_source", ""),
                    "language": "zh",
                    "query_style": "oral",
                }
            )

    random.shuffle(cases)

    OUTPUT_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CASES_PATH.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    category_count = defaultdict(int)

    for case in cases:
        category_count[case["expected_category"]] += 1

    print("=" * 80)
    print(f"中文口语化检索评测集已生成：{OUTPUT_CASES_PATH}")
    print(f"评测用例数：{len(cases)}")
    print("类别分布：")
    print(json.dumps(dict(category_count), ensure_ascii=False, indent=2))

    print("\n样例：")
    for case in cases[:5]:
        print("-" * 80)
        print("口语问题：", case["question"])
        print("标准问题：", case["standard_question"])
        print("expected_faq_id：", case["expected_faq_id"])


if __name__ == "__main__":
    main()
