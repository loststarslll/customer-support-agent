import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DIR = PROJECT_ROOT / "data" / "public" / "processed"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ZH_FAQ = PUBLIC_DIR / "public_faq_corpus_zh.csv"
OUTPUT_MAIN_FAQ = PROJECT_ROOT / "data" / "raw" / "customer_support_faq.csv"
TRANSLATION_CACHE_PATH = PUBLIC_DIR / "translation_cache_zh.json"

MAX_ROWS = int(os.getenv("PUBLIC_FAQ_MAX_ROWS", "160"))

DATASETS = [
    {
        "name": "MakTek/Customer_support_faqs_dataset",
        "source_title": "MakTek Customer Support FAQs Dataset",
        "source_type": "public_huggingface_dataset",
        "dataset_source": "MakTek/Customer_support_faqs_dataset",
    },
    {
        "name": "Andyrasika/Ecommerce_FAQ",
        "source_title": "Ecommerce FAQ Dataset",
        "source_type": "public_huggingface_dataset",
        "dataset_source": "Andyrasika/Ecommerce_FAQ",
    },
]


CATEGORY_RULES = [
    ("refund", ["refund", "money back", "reimbursement", "退款", "退钱"]),
    ("return", ["return", "returns", "退货"]),
    ("exchange", ["exchange", "replace", "replacement", "换货", "更换"]),
    ("delivery", ["shipping", "delivery", "shipment", "track", "tracking", "package", "物流", "快递", "包裹", "配送"]),
    ("payment", ["payment", "pay", "paid", "card", "billing", "charge", "付款", "支付", "扣款", "银行卡"]),
    ("account", ["account", "password", "login", "sign in", "profile", "账户", "账号", "密码", "登录"]),
    ("invoice", ["invoice", "receipt", "tax", "发票", "票据", "税号"]),
    ("coupon", ["coupon", "voucher", "discount", "promo", "promotion", "优惠券", "折扣", "满减"]),
    ("membership", ["membership", "member", "points", "rewards", "会员", "积分", "权益"]),
    ("product", ["product", "item", "size", "color", "specification", "商品", "产品", "规格", "尺寸", "颜色"]),
    ("privacy", ["privacy", "personal information", "security", "verification code", "隐私", "个人信息", "验证码", "安全"]),
]


SUBCATEGORY_MAPPING = {
    "refund": "refund_policy",
    "return": "return_policy",
    "exchange": "exchange_policy",
    "delivery": "delivery_tracking",
    "payment": "payment_issue",
    "account": "account_security",
    "invoice": "invoice_service",
    "coupon": "coupon_rule",
    "membership": "membership_benefit",
    "product": "product_info",
    "privacy": "privacy_security",
    "general": "general_support",
}


def load_dotenv_simple() -> None:
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def flatten_dataset(dataset_obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if hasattr(dataset_obj, "keys"):
        for split_name in dataset_obj.keys():
            split = dataset_obj[split_name]
            for row in split:
                item = dict(row)
                item["_split"] = split_name
                rows.append(item)
    else:
        for row in dataset_obj:
            rows.append(dict(row))

    return rows


def pick_field(row: dict[str, Any], candidates: list[str]) -> str:
    lowered = {str(key).lower(): key for key in row.keys()}

    for candidate in candidates:
        key = lowered.get(candidate.lower())

        if key is None:
            continue

        value = row.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def extract_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    question_candidates = [
        "question",
        "questions",
        "prompt",
        "instruction",
        "user",
        "query",
        "input",
        "utterance",
        "customer_question",
    ]

    answer_candidates = [
        "answer",
        "answers",
        "response",
        "completion",
        "assistant",
        "output",
        "customer_answer",
    ]

    question = pick_field(row, question_candidates)
    answer = pick_field(row, answer_candidates)

    if not question:
        for key, value in row.items():
            if "question" in str(key).lower() and value:
                question = str(value).strip()
                break

    if not answer:
        for key, value in row.items():
            key_lower = str(key).lower()
            if ("answer" in key_lower or "response" in key_lower) and value:
                answer = str(value).strip()
                break

    return question, answer


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_category(question: str, answer: str) -> str:
    text = f"{question} {answer}".lower()

    for category, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if keyword.lower() in text:
                return category

    return "general"


def load_translation_cache() -> dict[str, dict[str, str]]:
    if not TRANSLATION_CACHE_PATH.exists():
        return {}

    return json.loads(TRANSLATION_CACHE_PATH.read_text(encoding="utf-8"))


def save_translation_cache(cache: dict[str, dict[str, str]]) -> None:
    TRANSLATION_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_json_object(text: str) -> dict[str, str]:
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


def translate_pair_to_zh(question_en: str, answer_en: str) -> dict[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请先在 .env 中配置 DEEPSEEK_API_KEY。"
        )

    url = f"{base_url}/chat/completions"

    prompt = f"""
请把下面的英文客服 FAQ 翻译成自然、简洁、适合中文电商客服场景的中文。

要求：
1. 不要改变事实含义。
2. question_zh 要像真实中文用户会问的问题。
3. answer_zh 要像客服知识库答案，语气正式但不生硬。
4. 只输出 JSON，不要输出解释。

英文问题：
{question_en}

英文答案：
{answer_en}

输出格式：
{{
  "question_zh": "...",
  "answer_zh": "..."
}}
""".strip()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是专业的中英客服知识库翻译助手，只输出合法 JSON。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
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

    question_zh = clean_text(obj.get("question_zh", ""))
    answer_zh = clean_text(obj.get("answer_zh", ""))

    if not question_zh or not answer_zh:
        raise ValueError(f"翻译结果缺字段：{obj}")

    return {
        "question_zh": question_zh,
        "answer_zh": answer_zh,
    }


def collect_public_faq_rows() -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    seen_pairs = set()

    for dataset_config in DATASETS:
        dataset_name = dataset_config["name"]
        print(f"正在读取公开数据集：{dataset_name}")

        dataset_obj = load_dataset(dataset_name)
        rows = flatten_dataset(dataset_obj)

        print(f"原始样本数：{len(rows)}")

        for raw_row in rows:
            question_en, answer_en = extract_question_answer(raw_row)

            question_en = clean_text(question_en)
            answer_en = clean_text(answer_en)

            if not question_en or not answer_en:
                continue

            pair_key = (question_en.lower(), answer_en.lower())

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            category = infer_category(question_en, answer_en)

            collected.append(
                {
                    "category": category,
                    "subcategory": SUBCATEGORY_MAPPING.get(category, "general_support"),
                    "question_en": question_en,
                    "answer_en": answer_en,
                    "source_title": dataset_config["source_title"],
                    "source_type": dataset_config["source_type"],
                    "dataset_source": dataset_config["dataset_source"],
                }
            )

    random.seed(42)
    random.shuffle(collected)

    if len(collected) > MAX_ROWS:
        collected = collected[:MAX_ROWS]

    return collected


def main() -> None:
    load_dotenv_simple()

    source_rows = collect_public_faq_rows()
    cache = load_translation_cache()

    final_rows: list[dict[str, str]] = []

    for index, row in enumerate(source_rows, start=1):
        cache_key = f"{row['question_en']}|||{row['answer_en']}"

        if cache_key in cache:
            translated = cache[cache_key]
        else:
            print(f"[{index}/{len(source_rows)}] 正在翻译：{row['question_en'][:80]}")

            translated = translate_pair_to_zh(
                question_en=row["question_en"],
                answer_en=row["answer_en"],
            )

            cache[cache_key] = translated
            save_translation_cache(cache)
            time.sleep(0.3)

        faq_id = f"public_zh_faq_{len(final_rows) + 1:04d}"

        final_rows.append(
            {
                "id": faq_id,
                "category": row["category"],
                "subcategory": row["subcategory"],

                # 兼容原有检索代码：question / answer 使用中文
                "question": translated["question_zh"],
                "answer": translated["answer_zh"],

                # 新增中英文字段
                "question_zh": translated["question_zh"],
                "answer_zh": translated["answer_zh"],
                "question_en": row["question_en"],
                "answer_en": row["answer_en"],

                "source_title": row["source_title"],
                "source_type": row["source_type"],
                "dataset_source": row["dataset_source"],
                "updated_at": "2026-07-01",
                "language": "zh",
            }
        )

    fieldnames = [
        "id",
        "category",
        "subcategory",
        "question",
        "answer",
        "question_zh",
        "answer_zh",
        "question_en",
        "answer_en",
        "source_title",
        "source_type",
        "dataset_source",
        "updated_at",
        "language",
    ]

    with OUTPUT_ZH_FAQ.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    OUTPUT_MAIN_FAQ.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_MAIN_FAQ.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    category_count: dict[str, int] = {}

    for row in final_rows:
        category_count[row["category"]] = category_count.get(row["category"], 0) + 1

    print("=" * 80)
    print(f"中文公开 FAQ 已生成：{OUTPUT_ZH_FAQ}")
    print(f"主知识库已替换为中文公开 FAQ：{OUTPUT_MAIN_FAQ}")
    print(f"最终 FAQ 数量：{len(final_rows)}")
    print("类别分布：")
    print(json.dumps(category_count, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
