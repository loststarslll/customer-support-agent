from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import create_embeddings
from src.indexing.loader import load_faq_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "faiss"


def build_faiss_index() -> None:
    """读取客服资料，建立 FAISS 向量索引并保存到本地。"""

    print("1. 正在读取客服 FAQ...")
    documents = load_faq_documents()
    print(f"   已读取 {len(documents)} 条资料。")

    print("2. 正在加载 Embedding 模型...")
    embeddings = create_embeddings()

    print("3. 正在将资料转换为向量并建立 FAISS 索引...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    print("4. 正在保存索引...")
    vector_store.save_local(str(INDEX_PATH))

    print(f"\n索引建立完成，保存位置：{INDEX_PATH}")


if __name__ == "__main__":
    build_faiss_index()