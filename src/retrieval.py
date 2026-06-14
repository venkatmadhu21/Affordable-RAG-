from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path("data")

documents = []

# =========================
# Load TXT Files
# =========================

for file in DATA_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    documents.append(
        Document(
            page_content=content,
            metadata={
                "source": file.name,
                "type": "txt"
            }
        )
    )

# =========================
# Load PDF Files
# =========================

for file in DATA_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(file))

    pdf_docs = loader.load()

    for doc in pdf_docs:
        doc.metadata["source"] = file.name
        doc.metadata["type"] = "pdf"

    documents.extend(pdf_docs)

print(f"\nLoaded Documents: {len(documents)}")

# =========================
# Chunking
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# =========================
# Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Create Vector DB
# =========================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\nVector Database Created Successfully!")