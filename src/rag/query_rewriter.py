from typing import Any

from src.settings import QUERY_REWRITE_ENABLED


REWRITE_RULES = [
    {
        "keywords": ["钱", "退钱", "退回来", "到账", "多久到账", "什么时候回来"],
        "append_terms": ["退款", "到账时间"],
        "category_hint": "refund",
    },
    {
        "keywords": ["包裹", "快递", "物流", "没收到", "还没到", "送到"],
        "append_terms": ["物流", "配送状态", "包裹未收到"],
        "category_hint": "delivery",
    },
    {
        "keywords": ["发票", "开票", "抬头", "税号", "电子票"],
        "append_terms": ["发票", "开票规则"],
        "category_hint": "invoice",
    },
    {
        "keywords": ["优惠券", "券", "用不了", "满减", "折扣码"],
        "append_terms": ["优惠券", "不可用原因"],
        "category_hint": "coupon",
    },
    {
        "keywords": ["密码", "忘记密码", "登录不了", "账号锁定"],
        "append_terms": ["账户", "密码重置", "账号安全"],
        "category_hint": "account",
    },
    {
        "keywords": ["验证码", "银行卡", "敏感信息", "个人信息", "隐私"],
        "append_terms": ["隐私安全", "验证码", "个人信息保护"],
        "category_hint": "privacy",
    },
    {
        "keywords": ["退货", "寄回", "运费谁出", "退回商品"],
        "append_terms": ["退货", "退货流程", "退货运费"],
        "category_hint": "return",
    },
    {
        "keywords": ["换货", "换一个", "尺码", "颜色"],
        "append_terms": ["换货", "换货规则"],
        "category_hint": "exchange",
    },
    {
        "keywords": ["支付失败", "扣款", "付款", "银行卡", "重复扣款"],
        "append_terms": ["支付", "付款失败", "扣款异常"],
        "category_hint": "payment",
    },
    {
        "keywords": ["会员", "积分", "等级", "权益"],
        "append_terms": ["会员", "积分", "会员权益"],
        "category_hint": "membership",
    },
]


def normalize_text(text: str) -> str:
    """基础文本标准化。"""

    return text.strip()


def rewrite_query(question: str) -> dict[str, Any]:
    """
    将用户口语化问题改写为更适合检索的 query。

    当前采用轻量规则改写：
    - 不改变原始问题；
    - 根据关键词追加检索提示词；
    - 返回 category_hint，方便后续扩展元数据过滤。
    """

    original_query = normalize_text(question)

    if not QUERY_REWRITE_ENABLED:
        return {
            "enabled": False,
            "original_query": original_query,
            "rewritten_query": original_query,
            "applied_rules": [],
            "category_hints": [],
        }

    appended_terms: list[str] = []
    applied_rules: list[dict[str, Any]] = []
    category_hints: list[str] = []

    for rule in REWRITE_RULES:
        matched_keywords = [
            keyword
            for keyword in rule["keywords"]
            if keyword.lower() in original_query.lower()
        ]

        if not matched_keywords:
            continue

        for term in rule["append_terms"]:
            if term not in appended_terms and term not in original_query:
                appended_terms.append(term)

        category_hint = rule.get("category_hint")

        if category_hint and category_hint not in category_hints:
            category_hints.append(category_hint)

        applied_rules.append(
            {
                "matched_keywords": matched_keywords,
                "append_terms": rule["append_terms"],
                "category_hint": category_hint,
            }
        )

    if appended_terms:
        rewritten_query = original_query + " " + " ".join(appended_terms)
    else:
        rewritten_query = original_query

    return {
        "enabled": True,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "applied_rules": applied_rules,
        "category_hints": category_hints,
    }


if __name__ == "__main__":
    tests = [
        "钱什么时候回来？",
        "我的包裹怎么还没到？",
        "优惠券为什么用不了？",
        "客服会向我要验证码吗？",
        "怎么开发票？",
    ]

    for test in tests:
        print("=" * 80)
        print(rewrite_query(test))
