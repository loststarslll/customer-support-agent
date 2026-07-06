from typing import Any

from src.agent.controller import run_agent


def run_rag_pipeline(
    question: str,
    user_id: str | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """兼容旧入口，执行受控Agent Loop。"""

    return run_agent(
        question=question,
        user_id=user_id,
        session_id=session_id,
    )


if __name__ == "__main__":
    print(
        run_rag_pipeline(
            question="退款一般多久到账？",
            session_id="pipeline_demo",
        )
    )
