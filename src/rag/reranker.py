from functools import lru_cache
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.settings import (
    RERANKER_ENABLED,
    RERANKER_LOCAL_ONLY,
    RERANKER_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_reranker_model():
    """加载 Cross-Encoder Reranker 模型。"""

    if not RERANKER_ENABLED:
        return None

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=512,
            trust_remote_code=True,
            local_files_only=RERANKER_LOCAL_ONLY,
        )

    except Exception as exc:
        print(
            "[WARN] Reranker 模型加载失败，"
            f"将退回到 RRF 排序：{type(exc).__name__}: {exc}"
        )
        return None


def build_rerank_text(document: Document) -> str:
    """构造用于重排的文本。"""

    question = document.metadata.get("question", "")
    answer = document.metadata.get("answer", "")
    category = document.metadata.get("category", "")

    return (
        f"Category: {category}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Content: {document.page_content}"
    )


def rerank_results(
    query: str,
    results: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    使用 Cross-Encoder 对混合检索候选结果重排。

    如果模型加载失败，则返回原始 RRF 结果。
    """

    if not results:
        return []

    model = get_reranker_model()

    if model is None:
        return results[:top_k]

    pairs = [
        [
            query,
            build_rerank_text(document),
        ]
        for document, _ in results
    ]

    scores = model.predict(pairs)

    reranked_items: list[tuple[Document, float]] = []

    for index, ((document, hybrid_score), rerank_score) in enumerate(
        zip(results, scores),
        start=1,
    ):
        metadata = dict(document.metadata)
        metadata.update(
            {
                "reranker_enabled": True,
                "reranker_model": RERANKER_MODEL_NAME,
                "rerank_score": float(rerank_score),
                "pre_rerank_rank": index,
                "pre_rerank_hybrid_score": float(hybrid_score),
            }
        )

        reranked_document = Document(
            page_content=document.page_content,
            metadata=metadata,
        )

        reranked_items.append(
            (
                reranked_document,
                float(rerank_score),
            )
        )

    reranked_items.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    final_results: list[tuple[Document, float]] = []

    for final_rank, (document, score) in enumerate(
        reranked_items[:top_k],
        start=1,
    ):
        metadata = dict(document.metadata)
        metadata["final_rank"] = final_rank
        metadata["retrieval_mode"] = "hybrid_rrf_rerank"

        final_document = Document(
            page_content=document.page_content,
            metadata=metadata,
        )

        final_results.append(
            (
                final_document,
                score,
            )
        )

    return final_results


if __name__ == "__main__":
    from src.rag.hybrid_retriever import retrieve_hybrid

    query = "优惠券为什么用不了？"

    candidates = retrieve_hybrid(
        query=query,
        final_k=8,
    )

    results = rerank_results(
        query=query,
        results=candidates,
        top_k=3,
    )

    print("Query:", query)

    for index, (document, score) in enumerate(results, start=1):
        print("=" * 60)
        print("rank:", index)
        print("id:", document.metadata.get("id"))
        print("category:", document.metadata.get("category"))
        print("question:", document.metadata.get("question"))
        print("rerank_score:", score)
        print("pre_rerank_rank:", document.metadata.get("pre_rerank_rank"))
        print("vector_rank:", document.metadata.get("vector_rank"))
        print("bm25_rank:", document.metadata.get("bm25_rank"))
