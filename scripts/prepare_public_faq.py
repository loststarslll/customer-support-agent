import csv
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_PROCESSED_DIR = PROJECT_ROOT / "data" / "public" / "processed"
PUBLIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PUBLIC_FAQ = PUBLIC_PROCESSED_DIR / "public_faq_corpus.csv"
OUTPUT_MAIN_FAQ = PROJECT_ROOT / "data" / "raw" / "customer_support_faq.csv"


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
    ("refund", ["refund", "money back", "reimbursement", "退款", "退钱", "退回"]),
    ("return", ["return", "returns", "退货", "寄回"]),
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


def flatten_dataset(dataset_obj: Any) -> list[dict[str, Any]]:
    """把 Hugging Face DatasetDict / Dataset 统一转成 list[dict]。"""

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
    """从可能字段名里提取文本。"""

    lowered = {
        str(key).lower(): key
        for key in row.keys()
    }

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
    """兼容不同公开数据集字段名，提取 question / answer。"""

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

    # 兜底：如果字段名不标准，尝试找包含 question/answer 的字段
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


def infer_category(question: str, answer: str) -> str:
    """用规则从公开 FAQ 中推断客服类别。"""

    text = f"{question} {answer}".lower()

    for category, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if keyword.lower() in text:
                return category

    return "general"


def infer_subcategory(category: str) -> str:
    mapping = {
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

    return mapping.get(category, "general_support")


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    public_rows: list[dict[str, str]] = []
    seen_pairs = set()

    for dataset_config in DATASETS:
        dataset_name = dataset_config["name"]

        print(f"正在下载/读取公开数据集：{dataset_name}")

        dataset_obj = load_dataset(dataset_name)
        rows = flatten_dataset(dataset_obj)

        print(f"原始样本数：{len(rows)}")

        for row in rows:
            question, answer = extract_question_answer(row)

            question = clean_text(question)
            answer = clean_text(answer)

            if not question or not answer:
                continue

            pair_key = (question.lower(), answer.lower())

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            category = infer_category(question, answer)

            public_rows.append(
                {
                    "id": f"public_faq_{len(public_rows) + 1:04d}",
                    "category": category,
                    "subcategory": infer_subcategory(category),
                    "question": question,
                    "answer": answer,
                    "source_title": dataset_config["source_title"],
                    "source_type": dataset_config["source_type"],
                    "updated_at": "2026-07-01",
                    "dataset_source": dataset_config["dataset_source"],
                }
            )

    fieldnames = [
        "id",
        "category",
        "subcategory",
        "question",
        "answer",
        "source_title",
        "source_type",
        "updated_at",
        "dataset_source",
    ]

    with OUTPUT_PUBLIC_FAQ.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)

    # 全量替换项目主知识库
    OUTPUT_MAIN_FAQ.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_MAIN_FAQ.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)

    category_count = {}

    for row in public_rows:
        category = row["category"]
        category_count[category] = category_count.get(category, 0) + 1

    print("=" * 80)
    print(f"公开 FAQ 已生成：{OUTPUT_PUBLIC_FAQ}")
    print(f"项目主 FAQ 已替换：{OUTPUT_MAIN_FAQ}")
    print(f"最终 FAQ 数量：{len(public_rows)}")
    print("类别分布：")
    print(json.dumps(category_count, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
