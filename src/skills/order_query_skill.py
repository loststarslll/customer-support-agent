from typing import Any

from src.agent.reviewer import review_observation
from src.skills.base import BaseSkill
from src.tools.order_tools import query_order


class OrderQuerySkill(BaseSkill):
    """订单及物流状态查询 Skill。"""

    name = "order_query"
    description = "根据订单号查询订单、物流及退款状态。"

    def _format_success_answer(
        self,
        tool_result: dict[str, Any],
    ) -> str:
        """格式化成功查询到的订单信息。"""

        order = tool_result["order"]

        status_map = {
            "shipped": "已发货",
            "delivered": "已送达",
            "refund_processing": "退款处理中",
        }

        refund_map = {
            "not_requested": "未申请退款",
            "processing": "退款处理中",
            "completed": "退款已完成",
        }

        status = status_map.get(
            str(order.get("status")),
            str(order.get("status", "未知")),
        )

        refund_status = refund_map.get(
            str(order.get("refund_status")),
            str(order.get("refund_status", "未知")),
        )

        return (
            f"订单号：{order.get('order_id')}\n"
            f"商品：{order.get('product')}\n"
            f"订单状态：{status}\n"
            f"承运商：{order.get('carrier')}\n"
            f"物流单号：{order.get('tracking_number')}\n"
            f"预计送达：{order.get('estimated_delivery')}\n"
            f"退款状态：{refund_status}"
        )

    def _format_error_answer(
        self,
        tool_result: dict[str, Any],
    ) -> str:
        """根据工具错误类型生成客服回答。"""

        error = tool_result.get("error")
        message = str(
            tool_result.get(
                "message",
                "订单查询失败。",
            )
        )

        if error == "missing_order_id":
            return "请提供需要查询的订单号，例如10001。"

        if error == "order_not_found":
            return (
                f"{message}\n"
                "如果订单号无误，建议联系人工客服进一步核实。"
            )

        if error == "permission_denied":
            return (
                f"{message}\n"
                "请确认你登录的是下单账号，或联系人工客服处理。"
            )

        return (
            f"{message}\n"
            "当前暂时无法完成订单查询，请稍后重试或联系人工客服。"
        )

    def execute(
        self,
        question: str,
        order_id: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """查询订单并处理权限、缺失、查无结果等异常。"""

        trace: list[dict[str, Any]] = [
            {
                "step": 1,
                "stage": "action",
                "content": {
                    "skill": self.name,
                    "tool": "query_order",
                    "order_id": order_id,
                    "user_id": user_id,
                },
            }
        ]

        tool_result = query_order(
            order_id=order_id,
            user_id=user_id,
        )

        trace.append(
            {
                "step": 1,
                "stage": "observation",
                "content": tool_result,
            }
        )

        # 成功结果可以交给Reviewer确认；
        # 明确错误不完全依赖LLM，先用规则保证稳定性。
        if not tool_result.get("success"):
            error = tool_result.get("error")

            waiting_for_input = error == "missing_order_id"

            trace.append(
                {
                    "step": 1,
                    "stage": "rule_review",
                    "content": {
                        "decision": (
                            "ask_user"
                            if waiting_for_input
                            else "finish"
                        ),
                        "error": error,
                    },
                }
            )

            return {
                "success": False,
                "skill": self.name,
                "answer": self._format_error_answer(
                    tool_result
                ),
                "sources": [],
                "waiting_for_input": waiting_for_input,
                "observation": tool_result,
                "trace": trace,
            }

        review = review_observation(
            question=question,
            action="query_order",
            observation=tool_result,
        )

        trace.append(
            {
                "step": 1,
                "stage": "review",
                "content": review,
            }
        )

        return {
            "success": True,
            "skill": self.name,
            "answer": self._format_success_answer(
                tool_result
            ),
            "sources": [],
            "waiting_for_input": False,
            "observation": tool_result,
            "trace": trace,
        }


if __name__ == "__main__":
    skill = OrderQuerySkill()

    tests = [
        {
            "question": "帮我查询订单10001",
            "order_id": "10001",
            "user_id": "user_001",
        },
        {
            "question": "帮我查询订单10001",
            "order_id": "10001",
            "user_id": "user_002",
        },
        {
            "question": "帮我查询订单99999",
            "order_id": "99999",
            "user_id": "user_001",
        },
    ]

    for test in tests:
        print("=" * 60)
        print(skill.execute(**test))
