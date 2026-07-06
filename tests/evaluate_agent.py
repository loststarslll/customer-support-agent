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


CASES_PATH = PROJECT_ROOT / "tests" / "evaluation_cases.json"


def load_cases() -> list[dict[str, Any]]:
    """读取评测案例。"""

    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"没有找到评测文件：{CASES_PATH}"
        )

    content = CASES_PATH.read_text(encoding="utf-8")
    cases = json.loads(content)

    if not isinstance(cases, list):
        raise ValueError("评测文件最外层必须是列表。")

    return cases


def get_source_categories(
    result: dict[str, Any],
) -> list[str]:
    """获取检索来源类别。"""

    return [
        str(source.get("category"))
        for source in result.get("sources", [])
        if source.get("category")
    ]


def get_tool_order_id(
    result: dict[str, Any],
) -> str:
    """从Observation中提取订单号。"""

    observation = result.get("observation")

    if not isinstance(observation, dict):
        return ""

    order = observation.get("order")

    if not isinstance(order, dict):
        return ""

    return str(order.get("order_id", ""))


def get_tool_success(
    result: dict[str, Any],
) -> bool | None:
    """从Observation中提取工具执行状态。"""

    observation = result.get("observation")

    if not isinstance(observation, dict):
        return None

    success = observation.get("success")

    return success if isinstance(success, bool) else None


def check_case(
    case: dict[str, Any],
    result: dict[str, Any],
) -> tuple[bool, list[str]]:
    """比较实际结果和预期结果。"""

    errors: list[str] = []

    expected_action = case.get("expected_action")

    if (
        expected_action is not None
        and result.get("action") != expected_action
    ):
        errors.append(
            f"action预期={expected_action}，"
            f"实际={result.get('action')}"
        )

    expected_skill = case.get("expected_skill")

    if (
        expected_skill is not None
        and result.get("skill") != expected_skill
    ):
        errors.append(
            f"skill预期={expected_skill}，"
            f"实际={result.get('skill')}"
        )

    expected_category = case.get("expected_category")

    if expected_category is not None:
        categories = get_source_categories(result)

        if expected_category not in categories:
            errors.append(
                f"来源类别应包含={expected_category}，"
                f"实际={categories}"
            )

    expected_order_id = case.get("expected_order_id")

    if expected_order_id is not None:
        actual_order_id = get_tool_order_id(result)

        if actual_order_id != expected_order_id:
            errors.append(
                f"订单号预期={expected_order_id}，"
                f"实际={actual_order_id}"
            )

    expected_tool_success = case.get(
        "expected_tool_success"
    )

    if expected_tool_success is not None:
        actual_tool_success = get_tool_success(result)

        if actual_tool_success != expected_tool_success:
            errors.append(
                f"工具success预期={expected_tool_success}，"
                f"实际={actual_tool_success}"
            )

    expected_waiting = case.get(
        "expected_waiting_for_input"
    )

    if (
        expected_waiting is not None
        and result.get("waiting_for_input")
        != expected_waiting
    ):
        errors.append(
            f"waiting_for_input预期={expected_waiting}，"
            f"实际={result.get('waiting_for_input')}"
        )

    return len(errors) == 0, errors


def evaluate() -> None:
    """执行全部评测案例。"""

    cases = load_cases()

    total = len(cases)
    passed = 0
    failed_cases: list[dict[str, Any]] = []

    print("=" * 70)
    print("客服Agent自动评测")
    print(f"共 {total} 条测试案例")
    print("=" * 70)

    for index, case in enumerate(cases, start=1):
        session_id = f"eval-{uuid4()}"
        result: dict[str, Any] = {}

        try:
            result = run_agent(
                question=str(case["question"]),
                session_id=session_id,
            )

            is_passed, errors = check_case(
                case,
                result,
            )

        except Exception as exc:
            is_passed = False
            errors = [
                f"运行异常：{type(exc).__name__}: {exc}"
            ]

        finally:
            reset_session_state(session_id)

        status = "PASS" if is_passed else "FAIL"

        print(
            f"[{index:02d}/{total:02d}] "
            f"{status} | "
            f"{case.get('id')} | "
            f"{case.get('question')}"
        )

        if is_passed:
            passed += 1
        else:
            failed_cases.append(
                {
                    "case": case,
                    "result": result,
                    "errors": errors,
                }
            )

            for error in errors:
                print(f"    - {error}")

    print("\n" + "=" * 70)
    print("评测汇总")
    print("=" * 70)

    accuracy = passed / total if total else 0.0

    print(
        f"整体通过率："
        f"{passed}/{total} "
        f"({accuracy:.1%})"
    )

    if failed_cases:
        print("\n失败案例详情：")

        for item in failed_cases:
            print("\n" + "-" * 70)
            print(f"案例：{item['case'].get('id')}")
            print(f"问题：{item['case'].get('question')}")
            print(f"错误：{item['errors']}")
            print(f"实际结果：{item['result']}")


if __name__ == "__main__":
    evaluate()
