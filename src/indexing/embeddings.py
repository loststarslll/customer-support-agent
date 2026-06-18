from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    """创建本地文本向量模型。"""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


if __name__ == "__main__":
    embeddings = create_embeddings()

    test_text = "My package has not arrived."
    vector = embeddings.embed_query(test_text)

    print(f"测试文本：{test_text}")
    print(f"向量维度：{len(vector)}")
    print(f"前 10 个数值：{vector[:10]}")