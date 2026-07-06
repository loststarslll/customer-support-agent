import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hybrid_retriever import retrieve_hybrid
from src.rag.query_rewriter import rewrite_query
from src.rag.reranker import rerank_results
from src.settings import (
    BM25_TOP_K,
    HYBRID_FINAL_TOP_K,
    RERANKER_TOP_K,
    RETRIEVAL_THRESHOLD,
    RRF_K,
    VECTOR_TOP_K,
)


CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases_public_zh.json"


def load_cases():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"未找到中文公开评测集：{CASES_PATH}，请先运行 scripts/build_public_retrieval_cases_zh.py"
        )

    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def reciprocal_rank(items, expected_value):
    for rank, value in enumerate(items, start=1):
        if value == expected_value:
            return 1 / rank

    return 0.0


def precision_at_k(items, expected_value, k=3):
    top_k = items[:k]

    if not top_k:
        return 0.0

    return sum(1 for item in top_k if item == expected_value) / len(top_k)


def run_retrieval(question: str):
    rewrite_info = rewrite_query(question)
    search_query = rewrite_info["rewritten_query"]

    candidates = retrieve_hybrid(
        search_query,
        vector_k=VECTOR_TOP_K,
        bm25_k=BM25_TOP_K,
        final_k=max(HYBRID_FINAL_TOP_K, RERANKER_TOP_K, 8),
        threshold=RETRIEVAL_THRESHOLD,
        rrf_k=RRF_K,
    )

    results = rerank_results(
        query=question,
        results=candidates,
        top_k=RERANKER_TOP_K,
    )

    return results, rewrite_info


def main():
    cases = load_cases()

    id_top1_hits = 0
    id_recall3_hits = 0
    id_mrr_sum = 0.0

    category_top1_hits = 0
    category_recall3_hits = 0
    category_precision3_sum = 0.0
    category_mrr_sum = 0.0

    total = 0

    print("=" * 80)
    print("中文公开 FAQ 检索评测")
    print("=" * 80)
    print(f"评测集：{CASES_PATH}")
    print(f"用例数：{len(cases)}")
    print("=" * 80)

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected_faq_id = case["expected_faq_id"]
        expected_category = case["expected_category"]

        try:
            results, rewrite_info = run_retrieval(question)

            result_ids = [
                document.metadata.get("id")
                for document, _ in results
            ]

            result_categories = [
                document.metadata.get("category")
                for document, _ in results
            ]

            id_top1 = bool(result_ids and result_ids[0] == expected_faq_id)
            id_recall3 = expected_faq_id in result_ids[:3]
            id_rr = reciprocal_rank(result_ids[:3], expected_faq_id)

            category_top1 = bool(result_categories and result_categories[0] == expected_category)
            category_recall3 = expected_category in result_categories[:3]
            category_precision3 = precision_at_k(result_categories, expected_category, 3)
            category_rr = reciprocal_rank(result_categories[:3], expected_category)

            id_top1_hits += int(id_top1)
            id_recall3_hits += int(id_recall3)
            id_mrr_sum += id_rr

            category_top1_hits += int(category_top1)
            category_recall3_hits += int(category_recall3)
            category_precision3_sum += category_precision3
            category_mrr_sum += category_rr

            total += 1

            status = "PASS" if id_recall3 else "FAIL"

            print(f"[{index:03d}/{len(cases):03d}] {status} | {case['id']}")
            print(f"  中文问题：{question}")
            print(f"  改写查询：{rewrite_info['rewritten_query']}")
            print(f"  预期 FAQ：{expected_faq_id}")
            print(f"  实际 Top3 FAQ：{result_ids[:3]}")
            print(f"  预期类别：{expected_category}")
            print(f"  实际 Top3 类别：{result_categories[:3]}")
            print(f"  ID Recall@3={id_recall3} | Category Recall@3={category_recall3}")

        except Exception as exc:
            print(f"[{index:03d}/{len(cases):03d}] ERROR | {case.get('id')} | {type(exc).__name__}: {exc}")

    print("\n" + "=" * 80)
    print("中文公开 FAQ 检索评测汇总")
    print("=" * 80)

    if not total:
        print("没有有效用例。")
        return

    print("\n严格指标：按 expected_faq_id 评估")
    print(f"FAQ ID Top-1 Accuracy：{id_top1_hits / total:.1%}")
    print(f"FAQ ID Recall@3：{id_recall3_hits / total:.1%}")
    print(f"FAQ ID MRR：{id_mrr_sum / total:.3f}")

    print("\n弱标注指标：按 expected_category 评估")
    print(f"Category Top-1 Accuracy：{category_top1_hits / total:.1%}")
    print(f"Category Recall@3：{category_recall3_hits / total:.1%}")
    print(f"Category Precision@3：{category_precision3_sum / total:.1%}")
    print(f"Category MRR：{category_mrr_sum / total:.3f}")

    print("\n说明：")
    print("- FAQ ID 指标更严格，要求检索回对应的中文公开 FAQ 条目。")
    print("- Category 指标是弱标注，只判断是否检索到同类 FAQ。")
    print("- 当前评测对应真实使用场景：中文用户问题检索中文 FAQ 知识库。")


if __name__ == "__main__":
    main()
