from langchain_core.documents import Document

from src.llm import create_llm
from src.rag.prompts import CUSTOMER_SUPPORT_PROMPT


def format_documents(documents: list[Document]) -> str:
    """将多条 Document 整理成适合放入 Prompt 的上下文。"""

    if not documents:
        return "No relevant customer support information was found."

    formatted_parts: list[str] = []

    for position, document in enumerate(documents, start=1):
        faq_id = document.metadata.get("id", "unknown")
        category = document.metadata.get("category", "unknown")

        formatted_part = (
            f"[Document {position}]\n"
            f"FAQ ID: {faq_id}\n"
            f"Category: {category}\n"
            f"{document.page_content}"
        )

        formatted_parts.append(formatted_part)

    return "\n\n".join(formatted_parts)


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """让大模型根据检索文档生成客服回答。"""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("用户问题不能为空。")

    context = format_documents(documents)

    prompt_messages = CUSTOMER_SUPPORT_PROMPT.format_messages(
        context=context,
        question=cleaned_question,
    )

    llm = create_llm()
    response = llm.invoke(prompt_messages)

    answer = str(response.content).strip()

    if not answer:
        raise ValueError("大模型返回了空答案。")

    return answer


if __name__ == "__main__":
    example_documents = [
        Document(
            page_content=(
                "Category: refund\n"
                "Question: When will I receive my refund?\n"
                "Answer: Approved refunds are usually returned to the "
                "original payment method within 3 to 7 business days."
            ),
            metadata={
                "id": "faq_002",
                "category": "refund",
            },
        )
    ]

    test_answer = generate_answer(
        question="退款一般多久到账？",
        documents=example_documents,
    )

    print(test_answer)
