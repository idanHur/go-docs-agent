import streamlit as st
from agent import ask_full

st.title("Go Docs Agent")
st.write("Ask a question about Go. The agent retrieves from the docs, checks the results, and answers.")

question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = ask_full(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Examples from the web")
    st.write(result["web_examples"])