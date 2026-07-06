import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.state import reset_session_state
from src.rag.pipeline import run_rag_pipeline
from src.skills.registry import list_skills
from src.storage.sqlite_store import (
    get_recent_logs,
    insert_chat_log,
    list_chat_logs,
)


app = FastAPI(
    title="Customer Support Agent API",
    description=(
        "基于DeepSeek、FAISS、RAG、Skill和Tool Calling的"
        "客服Agent接口。"
    ),
    version="1.1.0",
)


class ChatRequest(BaseModel):
    """聊天请求。"""

    question: str = Field(
        ...,
        min_length=1,
        description="用户问题",
        examples=["帮我查询订单10001"],
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "会话ID。同一段多轮对话必须使用同一个session_id。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description="用户ID，用于订单权限校验。",
    )


class ChatResponse(BaseModel):
    """聊天响应。"""

    session_id: str
    question: str
    action: str | None = None
    skill: str | None = None
    reason: str | None = None
    answer: str
    waiting_for_input: bool = False
    latency_ms: float | None = None

    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )
    trace: list[dict[str, Any]] = Field(
        default_factory=list
    )
    session_state: dict[str, Any] = Field(
        default_factory=dict
    )


class ResetRequest(BaseModel):
    """清空会话请求。"""

    session_id: str = Field(
        ...,
        min_length=1,
        description="需要清空的会话ID",
    )


@app.get("/")
def root() -> dict[str, str]:
    """API首页。"""

    return {
        "name": "Customer Support Agent API",
        "version": "1.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。"""

    return {
        "status": "ok",
    }


@app.get("/skills")
def skills() -> dict[str, Any]:
    """查看当前已经注册的Skill。"""

    return {
        "skills": list_skills(),
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    """执行一次客服Agent对话，并记录日志。"""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="用户问题不能为空。",
        )

    session_id = (
        request.session_id.strip()
        if request.session_id
        else str(uuid4())
    )

    start_time = time.perf_counter()
    result: dict[str, Any] | None = None

    try:
        result = run_rag_pipeline(
            question=question,
            user_id=request.user_id,
            session_id=session_id,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        insert_chat_log(
            session_id=session_id,
            user_id=request.user_id,
            question=question,
            result=result,
            latency_ms=latency_ms,
            success=True,
        )

    except ValueError as exc:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        insert_chat_log(
            session_id=session_id,
            user_id=request.user_id,
            question=question,
            result=result,
            latency_ms=latency_ms,
            success=False,
            error_message=str(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        insert_chat_log(
            session_id=session_id,
            user_id=request.user_id,
            question=question,
            result=result,
            latency_ms=latency_ms,
            success=False,
            error_message=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Agent执行失败：{exc}",
        ) from exc

    return ChatResponse(
        session_id=session_id,
        question=str(
            result.get("question", question)
        ),
        action=result.get("action"),
        skill=result.get("skill"),
        reason=result.get("reason"),
        answer=str(
            result.get(
                "answer",
                "没有生成答案。",
            )
        ),
        waiting_for_input=bool(
            result.get(
                "waiting_for_input",
                False,
            )
        ),
        latency_ms=round(
            float(latency_ms),
            2,
        ),
        sources=result.get("sources", []),
        trace=result.get("trace", []),
        session_state=result.get(
            "session_state",
            {},
        ),
    )


@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, str]:
    """清空指定会话的状态。"""

    reset_session_state(request.session_id)

    return {
        "message": "会话状态已清空。",
        "session_id": request.session_id,
    }


@app.get("/logs/{session_id}")
def logs(
    session_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """查看某个会话的历史日志。"""

    return {
        "session_id": session_id,
        "logs": list_chat_logs(
            session_id=session_id,
            limit=limit,
        ),
    }


@app.get("/logs")
def recent_logs(
    limit: int = 20,
) -> dict[str, Any]:
    """查看最近的聊天日志。"""

    return {
        "logs": get_recent_logs(limit=limit),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
