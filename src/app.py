import streamlit as st

st.title("Affordable Housing Assistant")

question = st.text_input("Ask a question")

if question:
    # your existing retrieval code
    answer = ask_question(question)
    st.write(answer)