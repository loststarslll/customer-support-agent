import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORAL_CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases_public_zh.json"
OUTPUT_CASES_PATH = PROJECT_ROOT / "tests" / "retrieval_cases_hard_negative_zh.json"


CONFUSING_PREFIXES = {
    "refund": [
        "我不是问退货流程，也不是换货，我就想确认钱这块：",
        "商品怎么寄回先不说，我主要想问退款相关的：",
        "不是优惠券问题，也不是重新下单，就是钱什么时候处理：",
    ],
    "return": [
        "我不是问钱多久到账，我是想问东西怎么退：",
        "不是换货，也不是取消订单，我就是不想要了：",
        "退款先不管，我主要想知道商品怎么退回去：",
    ],
    "exchange": [
        "我不是想直接退款，也不是单纯退货，我想换一个：",
        "这个不是取消订单，我主要想换货：",
        "商品有问题但我不想退钱，想换一下：",
    ],
    "delivery": [
        "不是退款问题，我现在主要想知道包裹情况：",
        "钱已经付了，主要是物流这块不太清楚：",
        "不是退货也不是换货，我就是想知道东西到哪了：",
    ],
    "payment": [
        "不是优惠券没生效，我是付款这块有问题：",
        "不是发票问题，我想确认支付和扣款：",
        "不是物流问题，订单付款这里有点不对：",
    ],
    "coupon": [
        "不是支付失败，我能付款，但优惠这块有问题：",
        "不是会员积分，我主要想问优惠券：",
        "订单本身没问题，就是折扣好像没用上：",
    ],
    "invoice": [
        "不是问支付方式，我是想要票据：",
        "钱已经付了，现在主要是发票问题：",
        "不是查物流，我想问收据或者发票：",
    ],
    "account": [
        "不是订单问题，我账号这边有点麻烦：",
        "不是隐私政策，我现在主要是登录不上：",
        "下单前我得先解决账号问题：",
    ],
    "privacy": [
        "不是普通登录问题，我担心信息安全：",
        "不是查订单，我主要想确认个人信息会不会泄露：",
        "客服会不会要敏感信息，我有点不放心：",
    ],
    "membership": [
        "不是普通优惠券，我想问会员这块：",
        "不是支付问题，我主要想知道会员权益：",
        "这个和普通折扣不太一样，我想问会员相关的：",
    ],
    "product": [
        "不是物流也不是退款，我是想了解商品本身：",
        "下单前我想先确认商品信息：",
        "这个和售后无关，主要是商品规格问题：",
    ],
    "general": [
        "我不太确定该找哪类客服，想问一下：",
        "这个问题可能有点像订单或售后，但我主要想确认：",
    ],
}


def load_oral_cases():
    if not ORAL_CASES_PATH.exists():
        raise FileNotFoundError(
            f"未找到口语化评测集：{ORAL_CASES_PATH}，请先运行 scripts/build_public_retrieval_cases_zh.py"
        )

    cases = json.loads(ORAL_CASES_PATH.read_text(encoding="utf-8"))

    valid_cases = []

    for case in cases:
        question = case.get("question", "").strip()
        expected_faq_id = case.get("expected_faq_id", "").strip()
        expected_category = case.get("expected_category", "general").strip()

        if not question or not expected_faq_id:
            continue

        valid_cases.append(case)

    return valid_cases


def build_cases():
    oral_cases = load_oral_cases()

    random.seed(42)
    random.shuffle(oral_cases)

    hard_cases = []

    for case in oral_cases:
        category = case.get("expected_category", "general")
        prefixes = CONFUSING_PREFIXES.get(category, CONFUSING_PREFIXES["general"])
        prefix = random.choice(prefixes)

        oral_question = case["question"].strip()

        hard_query = f"{prefix}{oral_question}"

        hard_cases.append(
            {
                "id": f"hard_zh_retrieval_{len(hard_cases) + 1:04d}",
                "question": hard_query,
                "oral_question": oral_question,
                "standard_question": case.get("standard_question", ""),
                "expected_faq_id": case["expected_faq_id"],
                "expected_category": category,
                "source_title": case.get("source_title", ""),
                "dataset_source": case.get("dataset_source", ""),
                "language": "zh",
                "query_style": "hard_negative",
                "hard_negative_type": f"{category}_confusing_prefix_without_standard_question",
            }
        )

    OUTPUT_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CASES_PATH.write_text(
        json.dumps(hard_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return hard_cases


def main():
    cases = build_cases()

    category_count = Counter(case["expected_category"] for case in cases)

    print("=" * 80)
    print(f"Hard Negative 中文检索评测集已重新生成：{OUTPUT_CASES_PATH}")
    print(f"评测用例数：{len(cases)}")
    print("类别分布：")
    print(json.dumps(dict(category_count), ensure_ascii=False, indent=2))

    print("\n样例：")
    for case in cases[:8]:
        print("-" * 80)
        print("Hard Query：", case["question"])
        print("口语问题：", case["oral_question"])
        print("标准问题：", case.get("standard_question"))
        print("expected_faq_id：", case["expected_faq_id"])
        print("expected_category：", case["expected_category"])


if __name__ == "__main__":
    main()
