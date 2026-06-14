from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------------------
# Embeddings + Vector Store
# ---------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# ---------------------------
# Gemini
# ---------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ---------------------------
# Question
# ---------------------------
print("\nAffordable Housing RAG Assistant Started!")
print("Press Ctrl+C to exit.\n")

while True:
    try:
        question = input("\nAsk a question: ").strip()

        if not question:
            continue

        # ---------------------------
        # Retrieve
        # ---------------------------

        docs = vector_store.similarity_search(
            question,
            k=3
        )

        # ---------------------------
        # Build Context
        # ---------------------------

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        sources = list(
            set(doc.metadata["source"] for doc in docs)
        )

        # ---------------------------
        # Prompt
        # ---------------------------

        prompt = f"""
You are an Affordable Housing expert.

Answer ONLY from the provided context.

If the answer is not found in the context,
say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}
"""

        # ---------------------------
        # Generate Answer
        # ---------------------------

        response = llm.invoke(prompt)
        print("\n" + "=" * 60)
        print("ANSWER\n")
        print(response.content)

        print("\nSOURCES\n")

        for source in sources:
            print("-", source)
#prints all sources
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nExiting Affordable Housing RAG Assistant...")
        break