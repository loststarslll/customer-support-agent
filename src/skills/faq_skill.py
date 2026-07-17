from typing import Any

from src.agent.reviewer import review_observation
from src.rag.generator import generate_answer
from src.rag.hybrid_retriever import retrieve_hybrid
from src.rag.query_rewriter import rewrite_query
from src.rag.reranker import rerank_results
from src.settings import (
    BM25_TOP_K,
    FAQ_MAX_RETRIES,
    HYBRID_FINAL_TOP_K,
    QUERY_REWRITE_ENABLED,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVAL_THRESHOLD,
    RRF_K,
    SHOW_REFERENCE_SOURCES,
    VECTOR_TOP_K,
)
from src.skills.base import BaseSkill


class FAQSkill(BaseSkill):
    """客服知识库问答 Skill。"""

    name = "faq_search"
    description = "使用查询改写、混合检索和重排查询客服FAQ，并根据资料生成回答。"

    def _build_sources(
        self,
        results: list,
    ) -> list[dict[str, Any]]:
        """整理检索来源。"""

        sources = []

        for document, score in results:
            sources.append(
                {
                    "id": document.metadata.get("id"),
                    "category": document.metadata.get("category"),
                    "question": document.metadata.get("question"),
                    "retrieval_mode": document.metadata.get("retrieval_mode"),
                    "final_score": round(float(score), 6),
                    "rerank_score": document.metadata.get("rerank_score"),
                    "pre_rerank_rank": document.metadata.get("pre_rerank_rank"),
                    "pre_rerank_hybrid_score": document.metadata.get(
                        "pre_rerank_hybrid_score"
                    ),
                    "vector_rank": document.metadata.get("vector_rank"),
                    "bm25_rank": document.metadata.get("bm25_rank"),
                    "vector_distance": document.metadata.get("vector_distance"),
                    "bm25_score": document.metadata.get("bm25_score"),
                    "final_rank": document.metadata.get("final_rank"),
                }
            )

        return sources

    def _build_observation(
        self,
        results: list,
        rewrite_info: dict[str, Any],
    ) -> dict[str, Any]:
        """构造 Reviewer 使用的观察结果。"""

        return {
            "document_count": len(results),
            "retrieval_mode": (
                "query_rewrite_hybrid_rrf_rerank"
                if RERANKER_ENABLED
                else "query_rewrite_hybrid_rrf"
            ),
            "query_rewrite": rewrite_info,
            "documents": self._build_sources(results),
        }

    def _format_reference_sources(
        self,
        results: list,
    ) -> str:
        """将检索来源格式化为用户可见的参考来源。"""

        if not SHOW_REFERENCE_SOURCES or not results:
            return ""

        lines = [
            "",
            "",
            "参考来源：",
        ]

        used_ids = set()

        for document, _ in results[:3]:
            source_id = document.metadata.get("id", "unknown")
            question = document.metadata.get("question", "")
            category = document.metadata.get("category", "")

            if source_id in used_ids:
                continue

            used_ids.add(source_id)

            if question:
                lines.append(
                    f"- [{source_id}] {question}（{category}）"
                )
            else:
                lines.append(
                    f"- [{source_id}] {category}"
                )

        return "\n".join(lines)

    def execute(
        self,
        question: str,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行查询改写、混合检索、重排、结果检查和必要重试。"""

        current_query = question
        trace: list[dict[str, Any]] = []

        actual_max_retries = (
            FAQ_MAX_RETRIES
            if max_retries is None
            else max_retries
        )

        for attempt in range(actual_max_retries + 1):
            rewrite_info = rewrite_query(current_query)
            search_query = rewrite_info["rewritten_query"]

            trace.append(
                {
                    "step": attempt + 1,
                    "stage": "query_rewrite",
                    "content": rewrite_info,
                }
            )

            trace.append(
                {
                    "step": attempt + 1,
                    "stage": "action",
                    "content": {
                        "skill": self.name,
                        "retrieval_mode": (
                            "query_rewrite_hybrid_rrf_rerank"
                            if RERANKER_ENABLED
                            else "query_rewrite_hybrid_rrf"
                        ),
                        "original_query": current_query,
                        "search_query": search_query,
                        "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
                        "vector_top_k": VECTOR_TOP_K,
                        "bm25_top_k": BM25_TOP_K,
                        "candidate_top_k": max(
                            HYBRID_FINAL_TOP_K,
                            RERANKER_TOP_K,
                            8,
                        ),
                        "final_top_k": RERANKER_TOP_K,
                        "threshold": RETRIEVAL_THRESHOLD,
                        "rrf_k": RRF_K,
                        "reranker_enabled": RERANKER_ENABLED,
                    },
                }
            )

            candidate_results = retrieve_hybrid(
                search_query,
                vector_k=VECTOR_TOP_K,
                bm25_k=BM25_TOP_K,
                final_k=max(
                    HYBRID_FINAL_TOP_K,
                    RERANKER_TOP_K,
                    8,
                ),
                threshold=RETRIEVAL_THRESHOLD,
                rrf_k=RRF_K,
            )

            results = rerank_results(
                query=current_query,
                results=candidate_results,
                top_k=RERANKER_TOP_K,
            )

            observation = self._build_observation(
                results=results,
                rewrite_info=rewrite_info,
            )

            trace.append(
                {
                    "step": attempt + 1,
                    "stage": "observation",
                    "content": observation,
                }
            )

            review = review_observation(
                question=question,
                action="search_faq",
                observation=observation,
            )

            trace.append(
                {
                    "step": attempt + 1,
                    "stage": "review",
                    "content": review,
                }
            )

            if review["decision"] == "retry_search":
                rewritten_query = review["rewritten_query"].strip()

                if (
                    attempt < actual_max_retries
                    and rewritten_query
                    and rewritten_query != current_query
                ):
                    current_query = rewritten_query
                    continue

            if review["decision"] == "ask_user":
                return {
                    "success": False,
                    "skill": self.name,
                    "answer": (
                        review["message"]
                        or "请补充更具体的问题信息。"
                    ),
                    "sources": self._build_sources(results),
                    "waiting_for_input": True,
                    "observation": observation,
                    "trace": trace,
                }

            if review["decision"] == "reject":
                return {
                    "success": False,
                    "skill": self.name,
                    "answer": (
                        review["message"]
                        or "当前知识库无法处理该问题。"
                    ),
                    "sources": [],
                    "waiting_for_input": False,
                    "observation": observation,
                    "trace": trace,
                }

            documents = [
                document
                for document, _ in results
            ]

            answer = generate_answer(
                question=question,
                documents=documents,
            )

            answer += self._format_reference_sources(results)

            return {
                "success": bool(documents),
                "skill": self.name,
                "answer": answer,
                "sources": self._build_sources(results),
                "waiting_for_input": False,
                "observation": observation,
                "trace": trace,
            }

        return {
            "success": False,
            "skill": self.name,
            "answer": (
                "多次检索后仍未找到足够可靠的资料，"
                "建议联系人工客服。"
            ),
            "sources": [],
            "waiting_for_input": False,
            "observation": {
                "document_count": 0,
                "retrieval_mode": (
                    "query_rewrite_hybrid_rrf_rerank"
                    if RERANKER_ENABLED
                    else "query_rewrite_hybrid_rrf"
                ),
                "documents": [],
            },
            "trace": trace,
        }


if __name__ == "__main__":
    skill = FAQSkill()

    print(
        skill.execute(
            question="钱什么时候回来？",
        )
    )
