from src.llm import create_llm


def main() -> None:
    """运行最小的大模型对话测试。"""
    llm = create_llm()

    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    try:
        response = llm.invoke(question)
        print("\n模型回答：")
        print(response.content)
    except Exception as exc:
        print("\n模型调用失败：")
        print(exc)


if __name__ == "__main__":
    main()