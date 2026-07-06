from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.indexing.embeddings import create_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "faiss"


def load_vector_store() -> FAISS:
    embeddings = create_embeddings()

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(question: str, k: int = 3):
    db = load_vector_store()
    return db.similarity_search_with_score(question, k=k)


def filter_docs(results, threshold: float = 0.9):
    return [
        (doc, score)
        for doc, score in results
        if score <= threshold
    ]


def retrieve_filtered(question: str, k: int = 3, threshold: float = 0.9):
    results = retrieve(question, k)
    return filter_docs(results, threshold)


if __name__ == "__main__":
    q = "我的包裹一直没有到"
    res = retrieve(q)

    print("\nRAW RESULTS:")
    for d, s in res:
        print(d.metadata.get("id"), s)

    print("\nFILTERED:")
    for d, s in filter_docs(res):
        print(d.metadata.get("id"), s)
