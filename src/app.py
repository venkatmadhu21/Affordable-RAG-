import streamlit as st
from rag import ask_question

st.title("Affordable Housing Assistant")

question = st.text_input(
    "Ask a question"
)

if question:
    with st.spinner("Searching..."):
        answer = ask_question(question)

    st.write(answer)