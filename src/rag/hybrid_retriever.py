import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.rag.retriever import retrieve_filtered
from src.settings import (
    BM25_TOP_K,
    FAQ_CSV_PATH,
    HYBRID_FINAL_TOP_K,
    PROJECT_ROOT,
    RETRIEVAL_THRESHOLD,
    RRF_K,
    VECTOR_TOP_K,
)


def get_faq_csv_path() -> Path:
    """获取 FAQ CSV 文件路径。"""

    path = Path(FAQ_CSV_PATH)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def tokenize(text: str) -> list[str]:
    """
    简单中英文分词。

    英文按单词切分；
    中文按单字切分；
    这个实现不依赖额外分词库，适合作为轻量 BM25 baseline。
    """

    text = text.lower()

    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]", text)

    return english_tokens + chinese_tokens


def load_faq_documents() -> list[Document]:
    """从 CSV 加载 FAQ 文档。"""

    path = get_faq_csv_path()

    if not path.exists():
        return []

    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            faq_id = row.get("id", "").strip()
            category = row.get("category", "").strip()
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()

            if not question and not answer:
                continue

            page_content = (
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "id": faq_id,
                        "category": category,
                        "question": question,
                        "answer": answer,
                        "source": "faq_csv",
                    },
                )
            )

    return documents


class SimpleBM25:
    """轻量 BM25 实现。"""

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            tokenize(document.page_content)
            for document in documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        """计算 IDF。"""

        document_frequency: dict[str, int] = defaultdict(int)
        total_documents = len(self.tokenized_documents)

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                document_frequency[token] += 1

        idf: dict[str, float] = {}

        for token, frequency in document_frequency.items():
            idf[token] = math.log(
                1
                + (
                    total_documents
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )

        return idf

    def score(
        self,
        query: str,
    ) -> list[float]:
        """计算每篇文档的 BM25 分数。"""

        query_tokens = tokenize(query)
        scores: list[float] = []

        for tokens, doc_length in zip(
            self.tokenized_documents,
            self.document_lengths,
        ):
            token_counts = Counter(tokens)
            score = 0.0

            for token in query_tokens:
                if token not in token_counts:
                    continue

                term_frequency = token_counts[token]
                idf = self.idf.get(token, 0.0)

                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * doc_length
                        / (
                            self.average_document_length
                            or 1.0
                        )
                    )
                )

                score += (
                    idf
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                )

            scores.append(score)

        return scores


def retrieve_bm25(
    query: str,
    k: int = BM25_TOP_K,
) -> list[tuple[Document, float]]:
    """BM25 关键词召回。"""

    documents = load_faq_documents()

    if not documents:
        return []

    bm25 = SimpleBM25(documents)
    scores = bm25.score(query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (document, score)
        for document, score in ranked[:k]
        if score > 0
    ]


def get_document_id(
    document: Document,
    fallback: str,
) -> str:
    """获取文档 ID。"""

    doc_id = document.metadata.get("id")

    if doc_id:
        return str(doc_id)

    return fallback


def clone_document_with_metadata(
    document: Document,
    extra_metadata: dict[str, Any],
) -> Document:
    """复制 Document，并追加元数据。"""

    metadata = dict(document.metadata)
    metadata.update(extra_metadata)

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


def retrieve_hybrid(
    query: str,
    vector_k: int = VECTOR_TOP_K,
    bm25_k: int = BM25_TOP_K,
    final_k: int = HYBRID_FINAL_TOP_K,
    threshold: float = RETRIEVAL_THRESHOLD,
    rrf_k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """
    混合检索：
    1. FAISS 向量召回；
    2. BM25 关键词召回；
    3. RRF 融合排序。
    """

    vector_results = retrieve_filtered(
        query,
        k=vector_k,
        threshold=threshold,
    )

    bm25_results = retrieve_bm25(
        query,
        k=bm25_k,
    )

    candidates: dict[str, dict[str, Any]] = {}

    for rank, (document, distance) in enumerate(
        vector_results,
        start=1,
    ):
        doc_id = get_document_id(
            document,
            fallback=f"vector_{rank}",
        )

        if doc_id not in candidates:
            candidates[doc_id] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "vector_distance": None,
                "bm25_score": None,
            }

        candidates[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        candidates[doc_id]["vector_rank"] = rank
        candidates[doc_id]["vector_distance"] = float(distance)

    for rank, (document, score) in enumerate(
        bm25_results,
        start=1,
    ):
        doc_id = get_document_id(
            document,
            fallback=f"bm25_{rank}",
        )

        if doc_id not in candidates:
            candidates[doc_id] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "vector_distance": None,
                "bm25_score": None,
            }

        candidates[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        candidates[doc_id]["bm25_rank"] = rank
        candidates[doc_id]["bm25_score"] = float(score)

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    results: list[tuple[Document, float]] = []

    for item in ranked_candidates[:final_k]:
        document = clone_document_with_metadata(
            item["document"],
            {
                "retrieval_mode": "hybrid_rrf",
                "vector_rank": item["vector_rank"],
                "bm25_rank": item["bm25_rank"],
                "vector_distance": item["vector_distance"],
                "bm25_score": item["bm25_score"],
                "hybrid_score": item["rrf_score"],
            },
        )

        results.append(
            (
                document,
                float(item["rrf_score"]),
            )
        )

    return results


if __name__ == "__main__":
    test_queries = [
        "退款一般多久到账？",
        "优惠券为什么用不了？",
        "我的包裹没有送到怎么办？",
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        results = retrieve_hybrid(query)

        for index, (document, score) in enumerate(results, start=1):
            print(
                index,
                document.metadata.get("id"),
                document.metadata.get("category"),
                round(score, 6),
                "vector_rank=",
                document.metadata.get("vector_rank"),
                "bm25_rank=",
                document.metadata.get("bm25_rank"),
            )
            print(document.metadata.get("question"))
