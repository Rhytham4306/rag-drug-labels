"""
Build and query a Chroma vector store over the chunked drug labels.

Embeddings run locally via sentence-transformers (all-MiniLM-L6-v2 by
default) — no API key or internet call needed for this step, which keeps
the retrieval half of the pipeline fully offline-capable.
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import settings
from src.ingest import Chunk, load_and_chunk_directory


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    return [Document(page_content=c.text, metadata=c.to_metadata()) for c in chunks]


def build_vectorstore(docs_dir: str, persist_dir: str) -> Chroma:
    chunks = load_and_chunk_directory(docs_dir)
    documents = chunks_to_documents(chunks)

    embedding_fn = get_embedding_function()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name="drug_labels",
    )
    print(f"Indexed {len(documents)} chunks from '{docs_dir}' -> '{persist_dir}'")
    return vectorstore


def load_vectorstore(persist_dir: str) -> Chroma:
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No index found at {persist_dir}. Build one first with:\n"
            f"  python -m src.vectorstore --docs data/sample_docs --persist {persist_dir}"
        )
    embedding_fn = get_embedding_function()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        collection_name="drug_labels",
    )


def retrieve(vectorstore: Chroma, query: str, k: int | None = None) -> list[Document]:
    k = k or settings.top_k
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Chroma index")
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--persist", default="./chroma_db")
    args = parser.parse_args()

    build_vectorstore(args.docs, args.persist)
