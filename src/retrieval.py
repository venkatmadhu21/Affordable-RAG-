from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

query = "What is 50059 certification?"

results = vector_store.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result #{i}")
    print("Source:", doc.metadata["source"])
    print()
    print(doc.page_content[:500])
    print()