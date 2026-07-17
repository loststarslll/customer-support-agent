import csv
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FAQ_PATH = PROJECT_ROOT / "data" / "public" / "processed" / "public_faq_corpus.csv"
OUTPUT_CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases_public.json"


def main() -> None:
    if not PUBLIC_FAQ_PATH.exists():
        raise FileNotFoundError(
            f"未找到公开 FAQ：{PUBLIC_FAQ_PATH}，请先运行 scripts/prepare_public_faq.py"
        )

    rows = list(csv.DictReader(PUBLIC_FAQ_PATH.open("r", encoding="utf-8")))

    grouped = defaultdict(list)

    for row in rows:
        category = row.get("category", "general")
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        faq_id = row.get("id", "").strip()

        if not question or not answer or not faq_id:
            continue

        grouped[category].append(row)

    random.seed(42)

    cases = []

    # 每类最多抽 8 条，避免某一类过多
    for category, items in grouped.items():
        random.shuffle(items)

        for row in items[:8]:
            cases.append(
                {
                    "id": f"public_retrieval_{len(cases) + 1:04d}",
                    "question": row["question"],
                    "expected_faq_id": row["id"],
                    "expected_category": row["category"],
                    "source_title": row.get("source_title", ""),
                    "dataset_source": row.get("dataset_source", ""),
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

    print(f"公开检索评测集已生成：{OUTPUT_CASES_PATH}")
    print(f"评测用例数：{len(cases)}")
    print("类别分布：")
    print(json.dumps(dict(category_count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
