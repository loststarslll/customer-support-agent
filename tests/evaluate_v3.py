import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.controller import run_agent
from src.agent.state import reset_session_state


RETRIEVAL_CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases.json"
TOOL_CASES_PATH = PROJECT_ROOT / "tests" / "tool_cases.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_categories(result):
    sources = result.get("sources", [])

    return [
        source.get("category")
        for source in sources
        if isinstance(source, dict) and source.get("category")
    ]


def precision_at_k(categories, expected_category, k=3):
    top_k = categories[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1
        for category in top_k
        if category == expected_category
    )

    return relevant_count / len(top_k)


def reciprocal_rank(categories, expected_category):
    for index, category in enumerate(categories, start=1):
        if category == expected_category:
            return 1 / index

    return 0.0


def evaluate_retrieval():
    cases = load_json(RETRIEVAL_CASES_PATH)

    top1_hits = 0
    recall3_hits = 0
    precision3_sum = 0.0
    mrr_sum = 0.0
    total = 0

    print("\n" + "=" * 70)
    print("一、RAG 检索评测")
    print("=" * 70)

    for case in cases:
        session_id = f"eval-retrieval-{uuid4()}"
        expected_category = case["expected_category"]

        try:
            result = run_agent(
                question=case["question"],
                session_id=session_id,
                user_id=case.get("user_id", "user_001"),
            )

            categories = get_categories(result)

            top1 = bool(categories and categories[0] == expected_category)
            recall3 = expected_category in categories[:3]
            precision3 = precision_at_k(categories, expected_category, 3)
            rr = reciprocal_rank(categories[:3], expected_category)

            top1_hits += int(top1)
            recall3_hits += int(recall3)
            precision3_sum += precision3
            mrr_sum += rr
            total += 1

            status = "PASS" if recall3 else "FAIL"

            print(f"{status} | {case.get('id')}")
            print(f"  问题：{case['question']}")
            print(f"  预期类别：{expected_category}")
            print(f"  实际Top3：{categories[:3]}")
            print(f"  Top1={top1}, Recall@3={recall3}, Precision@3={precision3:.3f}, RR={rr:.3f}")

        except Exception as exc:
            print(f"ERROR | {case.get('id')} | {type(exc).__name__}: {exc}")

        finally:
            reset_session_state(session_id)

    print("\nRAG 检索汇总：")
    print(f"Top-1 Accuracy：{top1_hits / total:.1%}" if total else "Top-1 Accuracy：0.0%")
    print(f"Recall@3：{recall3_hits / total:.1%}" if total else "Recall@3：0.0%")
    print(f"Precision@3：{precision3_sum / total:.1%}" if total else "Precision@3：0.0%")
    print(f"MRR：{mrr_sum / total:.3f}" if total else "MRR：0.000")


def get_observation(result):
    observation = result.get("observation")

    if isinstance(observation, dict):
        return observation

    return {}


def get_tool_success(result):
    return get_observation(result).get("success")


def get_tool_error(result):
    return get_observation(result).get("error")


def get_order_id(result):
    order = get_observation(result).get("order")

    if isinstance(order, dict):
        return str(order.get("order_id", ""))

    return ""


def evaluate_tool():
    cases = load_json(TOOL_CASES_PATH)

    total = 0
    action_hits = 0
    skill_hits = 0
    success_hits = 0
    error_hits = 0
    permission_total = 0
    permission_hits = 0

    print("\n" + "=" * 70)
    print("二、Tool Calling 与权限评测")
    print("=" * 70)

    for case in cases:
        session_id = f"eval-tool-{uuid4()}"

        try:
            result = run_agent(
                question=case["question"],
                session_id=session_id,
                user_id=case.get("user_id", "user_001"),
            )

            actual_action = result.get("action")
            actual_skill = result.get("skill")
            actual_success = get_tool_success(result)
            actual_error = get_tool_error(result)
            actual_order_id = get_order_id(result)

            expected_action = case.get("expected_action")
            expected_skill = case.get("expected_skill")
            expected_success = case.get("expected_tool_success")
            expected_error = case.get("expected_error")
            expected_order_id = case.get("expected_order_id", "")

            action_ok = actual_action == expected_action
            skill_ok = actual_skill == expected_skill
            success_ok = actual_success == expected_success
            error_ok = actual_error == expected_error
            order_ok = actual_order_id == expected_order_id

            total += 1
            action_hits += int(action_ok)
            skill_hits += int(skill_ok)
            success_hits += int(success_ok)
            error_hits += int(error_ok)

            if expected_error == "permission_denied":
                permission_total += 1
                permission_hits += int(actual_error == "permission_denied")

            status = "PASS" if all([action_ok, skill_ok, success_ok, error_ok, order_ok]) else "FAIL"

            print(f"{status} | {case.get('id')}")
            print(f"  问题：{case['question']} | user_id={case.get('user_id')}")
            print(f"  action：{actual_action} / expected={expected_action}")
            print(f"  skill：{actual_skill} / expected={expected_skill}")
            print(f"  success：{actual_success} / expected={expected_success}")
            print(f"  order_id：{actual_order_id} / expected={expected_order_id}")
            print(f"  error：{actual_error} / expected={expected_error}")

        except Exception as exc:
            print(f"ERROR | {case.get('id')} | {type(exc).__name__}: {exc}")

        finally:
            reset_session_state(session_id)

    print("\nTool / 权限汇总：")
    print(f"Action Accuracy：{action_hits / total:.1%}" if total else "Action Accuracy：0.0%")
    print(f"Skill Accuracy：{skill_hits / total:.1%}" if total else "Skill Accuracy：0.0%")
    print(f"Tool Success Accuracy：{success_hits / total:.1%}" if total else "Tool Success Accuracy：0.0%")
    print(f"Error Type Accuracy：{error_hits / total:.1%}" if total else "Error Type Accuracy：0.0%")
    print(
        f"Permission Accuracy：{permission_hits / permission_total:.1%}"
        if permission_total
        else "Permission Accuracy：0.0%"
    )


def main():
    evaluate_retrieval()
    evaluate_tool()


if __name__ == "__main__":
    main()
