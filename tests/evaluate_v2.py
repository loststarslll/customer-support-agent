import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.agent.controller import run_agent
from src.agent.state import reset_session_state


RETRIEVAL_CASES_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_cases.json"
)

MULTITURN_CASES_PATH = (
    PROJECT_ROOT / "tests" / "multiturn_cases.json"
)


def load_json(path: Path) -> list[dict[str, Any]]:
    """读取JSON测试集。"""

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)

    if not isinstance(data, list):
        raise ValueError(f"{path.name}最外层必须是列表。")

    return data


def get_source_categories(
    result: dict[str, Any],
) -> list[str]:
    """按照检索排名提取来源类别。"""

    categories: list[str] = []

    for source in result.get("sources", []):
        category = source.get("category")

        if category:
            categories.append(str(category))

    return categories


def get_observation(
    result: dict[str, Any],
) -> dict[str, Any]:
    """获取Skill或工具返回的Observation。"""

    observation = result.get("observation")

    if isinstance(observation, dict):
        return observation

    return {}


def get_order_id(
    result: dict[str, Any],
) -> str:
    """从订单工具结果中提取订单号。"""

    observation = get_observation(result)
    order = observation.get("order")

    if not isinstance(order, dict):
        return ""

    return str(order.get("order_id", ""))


def reciprocal_rank(
    categories: list[str],
    expected_category: str,
) -> float:
    """
    计算单个问题的倒数排名。

    正确类别排第1：1.0
    正确类别排第2：0.5
    正确类别排第3：0.333...
    没出现：0
    """

    for rank, category in enumerate(categories, start=1):
        if category == expected_category:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval() -> dict[str, float]:
    """评测FAQ检索排名质量。"""

    cases = load_json(RETRIEVAL_CASES_PATH)

    top1_hits = 0
    top3_hits = 0
    reciprocal_rank_sum = 0.0
    valid_cases = 0

    print("\n" + "=" * 72)
    print("一、FAQ检索评测")
    print("=" * 72)

    for index, case in enumerate(cases, start=1):
        session_id = f"retrieval-{uuid4()}"
        result: dict[str, Any] = {}

        try:
            result = run_agent(
                question=str(case["question"]),
                session_id=session_id,
            )

            expected_category = str(
                case["expected_category"]
            )

            categories = get_source_categories(result)

            top1_hit = (
                len(categories) >= 1
                and categories[0] == expected_category
            )

            top3_hit = (
                expected_category in categories[:3]
            )

            rr = reciprocal_rank(
                categories=categories[:3],
                expected_category=expected_category,
            )

            valid_cases += 1
            top1_hits += int(top1_hit)
            top3_hits += int(top3_hit)
            reciprocal_rank_sum += rr

            status = (
                "PASS"
                if top3_hit
                else "FAIL"
            )

            print(
                f"[{index:02d}/{len(cases):02d}] "
                f"{status} | "
                f"{case['id']}"
            )
            print(f"    问题：{case['question']}")
            print(
                f"    预期类别：{expected_category}"
            )
            print(
                f"    实际排名：{categories[:3]}"
            )
            print(
                f"    Top-1={top1_hit} | "
                f"Top-3={top3_hit} | "
                f"RR={rr:.3f}"
            )

        except Exception as exc:
            print(
                f"[{index:02d}/{len(cases):02d}] "
                f"ERROR | {case.get('id')}"
            )
            print(
                f"    {type(exc).__name__}: {exc}"
            )

        finally:
            reset_session_state(session_id)

    top1_accuracy = (
        top1_hits / valid_cases
        if valid_cases
        else 0.0
    )

    top3_recall = (
        top3_hits / valid_cases
        if valid_cases
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / valid_cases
        if valid_cases
        else 0.0
    )

    print("\nFAQ检索汇总：")
    print(
        f"Top-1 Accuracy："
        f"{top1_hits}/{valid_cases} "
        f"({top1_accuracy:.1%})"
    )
    print(
        f"Top-3 Recall："
        f"{top3_hits}/{valid_cases} "
        f"({top3_recall:.1%})"
    )
    print(f"MRR：{mrr:.3f}")

    return {
        "top1_accuracy": top1_accuracy,
        "top3_recall": top3_recall,
        "mrr": mrr,
    }


def check_multiturn_step(
    step: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    """检查多轮对话中的单个步骤。"""

    errors: list[str] = []

    expected_action = step.get("expected_action")

    if (
        expected_action is not None
        and result.get("action") != expected_action
    ):
        errors.append(
            f"action预期={expected_action}，"
            f"实际={result.get('action')}"
        )

    expected_skill = step.get("expected_skill")

    if (
        expected_skill is not None
        and result.get("skill") != expected_skill
    ):
        errors.append(
            f"skill预期={expected_skill}，"
            f"实际={result.get('skill')}"
        )

    expected_waiting = step.get(
        "expected_waiting_for_input"
    )

    if (
        expected_waiting is not None
        and result.get("waiting_for_input")
        != expected_waiting
    ):
        errors.append(
            f"waiting预期={expected_waiting}，"
            f"实际={result.get('waiting_for_input')}"
        )

    expected_pending = step.get(
        "expected_pending_action",
        "__not_checked__",
    )

    if expected_pending != "__not_checked__":
        session_state = result.get(
            "session_state",
            {},
        )

        actual_pending = session_state.get(
            "pending_action"
        )

        if actual_pending != expected_pending:
            errors.append(
                f"pending_action预期="
                f"{expected_pending}，"
                f"实际={actual_pending}"
            )

    expected_tool_success = step.get(
        "expected_tool_success"
    )

    if expected_tool_success is not None:
        observation = get_observation(result)
        actual_success = observation.get("success")

        if actual_success != expected_tool_success:
            errors.append(
                f"工具success预期="
                f"{expected_tool_success}，"
                f"实际={actual_success}"
            )

    expected_order_id = step.get(
        "expected_order_id"
    )

    if expected_order_id is not None:
        actual_order_id = get_order_id(result)

        if actual_order_id != expected_order_id:
            errors.append(
                f"订单号预期={expected_order_id}，"
                f"实际={actual_order_id}"
            )

    return errors


def evaluate_multiturn() -> dict[str, float]:
    """评测多轮参数补全和状态延续。"""

    conversations = load_json(
        MULTITURN_CASES_PATH
    )

    conversation_passed = 0
    step_passed = 0
    total_steps = 0

    print("\n" + "=" * 72)
    print("二、多轮任务评测")
    print("=" * 72)

    for index, conversation in enumerate(
        conversations,
        start=1,
    ):
        session_id = f"multiturn-{uuid4()}"
        conversation_errors: list[str] = []

        print(
            f"\n[{index:02d}/{len(conversations):02d}] "
            f"{conversation['id']}"
        )

        try:
            for step_index, step in enumerate(
                conversation["steps"],
                start=1,
            ):
                total_steps += 1

                result = run_agent(
                    question=str(step["question"]),
                    session_id=session_id,
                )

                errors = check_multiturn_step(
                    step=step,
                    result=result,
                )

                if errors:
                    conversation_errors.extend(
                        [
                            f"第{step_index}轮：{error}"
                            for error in errors
                        ]
                    )
                    status = "FAIL"
                else:
                    step_passed += 1
                    status = "PASS"

                print(
                    f"    第{step_index}轮 "
                    f"{status} | "
                    f"用户：{step['question']}"
                )
                print(
                    f"        action="
                    f"{result.get('action')}"
                )
                print(
                    f"        skill="
                    f"{result.get('skill')}"
                )
                print(
                    f"        pending="
                    f"{result.get('session_state', {}).get('pending_action')}"
                )

                for error in errors:
                    print(f"        - {error}")

            if not conversation_errors:
                conversation_passed += 1
                print("    对话整体：PASS")
            else:
                print("    对话整体：FAIL")

        except Exception as exc:
            conversation_errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"    运行异常："
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            reset_session_state(session_id)

    conversation_accuracy = (
        conversation_passed / len(conversations)
        if conversations
        else 0.0
    )

    step_accuracy = (
        step_passed / total_steps
        if total_steps
        else 0.0
    )

    print("\n多轮任务汇总：")
    print(
        f"对话整体通过率："
        f"{conversation_passed}/{len(conversations)} "
        f"({conversation_accuracy:.1%})"
    )
    print(
        f"单步通过率："
        f"{step_passed}/{total_steps} "
        f"({step_accuracy:.1%})"
    )

    return {
        "conversation_accuracy": (
            conversation_accuracy
        ),
        "step_accuracy": step_accuracy,
    }


def main() -> None:
    retrieval_metrics = evaluate_retrieval()
    multiturn_metrics = evaluate_multiturn()

    print("\n" + "=" * 72)
    print("Evaluation 2.0 总结")
    print("=" * 72)

    print(
        f"FAQ Top-1 Accuracy："
        f"{retrieval_metrics['top1_accuracy']:.1%}"
    )
    print(
        f"FAQ Top-3 Recall："
        f"{retrieval_metrics['top3_recall']:.1%}"
    )
    print(
        f"FAQ MRR："
        f"{retrieval_metrics['mrr']:.3f}"
    )
    print(
        f"多轮对话通过率："
        f"{multiturn_metrics['conversation_accuracy']:.1%}"
    )
    print(
        f"多轮步骤通过率："
        f"{multiturn_metrics['step_accuracy']:.1%}"
    )


if __name__ == "__main__":
    main()
