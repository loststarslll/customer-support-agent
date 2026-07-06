from typing import Any

from src.agent.reviewer import review_observation
from src.rag.generator import generate_answer
from src.rag.retriever import retrieve_filtered
from src.settings import (
    FAQ_MAX_RETRIES,
    RETRIEVAL_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from src.skills.base import BaseSkill


class FAQSkill(BaseSkill):
    """客服知识库问答 Skill。"""

    name = "faq_search"
    description = "检索客服FAQ，并根据资料生成回答。"

    def _build_sources(
        self,
        results: list,
    ) -> list[dict[str, Any]]:
        """整理检索来源。"""

        return [
            {
                "id": document.metadata.get("id"),
                "category": document.metadata.get("category"),
                "question": document.metadata.get("question"),
                "distance": round(float(score), 4),
            }
            for document, score in results
        ]

    def _build_observation(
        self,
        results: list,
    ) -> dict[str, Any]:
        """构造 Reviewer 使用的观察结果。"""

        return {
            "document_count": len(results),
            "documents": self._build_sources(results),
        }

    def execute(
        self,
        question: str,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行 FAQ 检索、结果检查和必要重试。"""

        current_query = question
        trace: list[dict[str, Any]] = []

        actual_max_retries = (
            FAQ_MAX_RETRIES
            if max_retries is None
            else max_retries
        )

        for attempt in range(actual_max_retries + 1):
            trace.append(
                {
                    "step": attempt + 1,
                    "stage": "action",
                    "content": {
                        "skill": self.name,
                        "query": current_query,
                        "top_k": RETRIEVAL_TOP_K,
                        "threshold": RETRIEVAL_THRESHOLD,
                    },
                }
            )

            results = retrieve_filtered(
                current_query,
                k=RETRIEVAL_TOP_K,
                threshold=RETRIEVAL_THRESHOLD,
            )

            observation = self._build_observation(results)

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
                "documents": [],
            },
            "trace": trace,
        }


if __name__ == "__main__":
    skill = FAQSkill()

    print(
        skill.execute(
            question="退款一般多久到账？",
        )
    )
