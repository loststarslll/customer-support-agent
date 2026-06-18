from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import create_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "faiss"


def load_vector_store() -> FAISS:
    """加载本地保存的 FAISS 向量库。"""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "没有找到 FAISS 索引，请先运行："
            "python -m src.indexing.build_index"
        )

    embeddings = create_embeddings()

    return FAISS.load_local(
        folder_path=str(INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def search_faq(query: str, top_k: int = 3) -> None:
    """检索并打印最相似的客服 FAQ。"""

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    print(f"\n查询问题：{query}")
    print(f"返回前 {len(results)} 条结果：")

    for position, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"第 {position} 条")
        print(f"FAQ ID：{document.metadata.get('id')}")
        print(f"类别：{document.metadata.get('category')}")
        print(f"距离分数：{score:.4f}")
        print("内容：")
        print(document.page_content)


def main() -> None:
    query = input("请输入客服问题：").strip()

    if not query:
        print("问题不能为空。")
        return

    search_faq(query=query, top_k=3)


if __name__ == "__main__":
    main()