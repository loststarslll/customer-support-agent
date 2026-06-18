from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_support_faq.csv"

REQUIRED_COLUMNS = {"id", "category", "question", "answer"}


def load_faq_documents(
    file_path: Path = DEFAULT_DATA_PATH,
) -> list[Document]:
    """读取客服 FAQ CSV，并将每行转换成一个 LangChain Document。"""

    if not file_path.exists():
        raise FileNotFoundError(f"找不到客服数据文件：{file_path}")

    dataframe = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"CSV 缺少必要字段：{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("客服 FAQ 数据为空。")

    documents: list[Document] = []

    for _, row in dataframe.iterrows():
        faq_id = str(row["id"]).strip()
        category = str(row["category"]).strip()
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        if not all([faq_id, category, question, answer]):
            continue

        content = (
            f"Category: {category}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        document = Document(
            page_content=content,
            metadata={
                "id": faq_id,
                "category": category,
                "question": question,
            },
        )

        documents.append(document)

    if not documents:
        raise ValueError("没有成功生成任何 Document。")

    return documents


if __name__ == "__main__":
    loaded_documents = load_faq_documents()

    print(f"成功读取 {len(loaded_documents)} 条客服 FAQ。\n")

    print("第一条 Document：")
    print(loaded_documents[0].page_content)

    print("\n第一条 Metadata：")
    print(loaded_documents[0].metadata)

