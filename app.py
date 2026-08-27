"""
Streamlit chat UI for MedLabel-RAG.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.config import settings, validate_provider_config
from src.rag_chain import answer_question
from src.vectorstore import load_vectorstore

st.set_page_config(page_title="MedLabel-RAG", page_icon="💊", layout="wide")

st.title("💊 MedLabel-RAG")
st.caption(
    "Ask questions about drug labels. Every answer cites the exact source "
    "excerpt it came from — this is a portfolio project, not medical advice."
)

with st.sidebar:
    st.subheader("Index")
    persist_dir = st.text_input("Chroma index path", value="./chroma_db")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=settings.top_k)
    st.subheader("LLM backend")
    st.write(f"Provider: `{settings.llm_provider}`")
    if settings.llm_provider == "groq":
        st.write(f"Model: `{settings.groq_model}`")
    elif settings.llm_provider == "ollama":
        st.write(f"Model: `{settings.ollama_model}` (local)")
    elif settings.llm_provider == "openai":
        st.write(f"Model: `{settings.openai_model}`")
    st.caption("Change LLM_PROVIDER in your .env file to switch backends.")


@st.cache_resource
def _load_vectorstore(path: str):
    return load_vectorstore(path)


try:
    validate_provider_config()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

try:
    vectorstore = _load_vectorstore(persist_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"Sources ({len(turn['citations'])})"):
            for c in turn["citations"]:
                st.markdown(f"**[Source {c.index}] {c.source} — {c.section}**")
                st.text(c.text)

question = st.chat_input("e.g. What is the maximum daily dose of ibuprofen for adults?")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving label excerpts and generating answer..."):
            result = answer_question(vectorstore, question, k=top_k)
        st.write(result.answer)
        with st.expander(f"Sources ({len(result.citations)})"):
            for c in result.citations:
                st.markdown(f"**[Source {c.index}] {c.source} — {c.section}**")
                st.text(c.text)

    st.session_state.history.append(
        {"question": question, "answer": result.answer, "citations": result.citations}
    )
