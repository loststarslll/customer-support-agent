from typing import Any
from uuid import uuid4

from src.agent.state import reset_session_state
from src.rag.pipeline import run_rag_pipeline


def print_sources(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    print("\n参考资料：")

    for position, source in enumerate(sources, start=1):
        print(
            f"{position}. "
            f"{source.get('id')} | "
            f"{source.get('category')} | "
            f"距离：{source.get('distance')} | "
            f"{source.get('question')}"
        )


def print_trace(
    trace: list[dict[str, Any]],
) -> None:
    if not trace:
        return

    print("\n执行轨迹：")

    for item in trace:
        print(
            f"- Step {item.get('step')} | "
            f"{item.get('stage')}: "
            f"{item.get('content')}"
        )


def main() -> None:
    session_id = str(uuid4())

    print("=" * 60)
    print("客服 Skill Agent")
    print("输入 exit 或 quit 结束程序")
    print("输入 reset 清空会话状态")
    print("=" * 60)

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit"}:
            reset_session_state(session_id)
            print("程序已结束。")
            break

        if question.lower() == "reset":
            reset_session_state(session_id)
            print("当前会话状态已清空。")
            continue

        if not question:
            print("问题不能为空。")
            continue

        try:
            result = run_rag_pipeline(
                question=question,
                session_id=session_id,
            )

            print(f"\nPlanner动作：{result.get('action')}")
            print(f"执行Skill：{result.get('skill')}")
            print(f"动作原因：{result.get('reason')}")

            print("\n客服回答：")
            print(result.get("answer", "没有生成答案。"))

            print_sources(result.get("sources", []))
            print_trace(result.get("trace", []))

            if result.get("waiting_for_input"):
                print("\n当前任务等待补充信息。")

        except Exception as exc:
            print("\n本次处理失败：")
            print(exc)


if __name__ == "__main__":
    main()
