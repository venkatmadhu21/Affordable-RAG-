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
# Function for Streamlit
# ---------------------------

def ask_question(question):

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

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

    response = llm.invoke(prompt)

    return response.content