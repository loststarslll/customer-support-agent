import os

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def create_embeddings() -> HuggingFaceEmbeddings:
    """创建支持中英文检索的 Embedding 模型。"""

    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        DEFAULT_EMBEDDING_MODEL_NAME,
    )

    local_only = (
        os.getenv("EMBEDDING_LOCAL_ONLY", "false").lower()
        == "true"
    )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
            "local_files_only": local_only,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


if __name__ == "__main__":
    embeddings = create_embeddings()

    test_text = "我的包裹一直没有到。"
    vector = embeddings.embed_query(test_text)

    print(
        "Embedding 模型："
        f"{os.getenv('EMBEDDING_MODEL_NAME', DEFAULT_EMBEDDING_MODEL_NAME)}"
    )
    print(
        "仅使用本地模型："
        f"{os.getenv('EMBEDDING_LOCAL_ONLY', 'false')}"
    )
    print(f"测试文本：{test_text}")
    print(f"向量维度：{len(vector)}")
    print(f"前 10 个数值：{vector[:10]}")
