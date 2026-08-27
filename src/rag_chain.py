"""
The core RAG chain: retrieve relevant chunks, then ask the LLM to answer
*strictly* from those chunks, returning both the answer and structured
citations pointing back to (source document, section, chunk text).
"""
from dataclasses import dataclass

from src.config import settings, validate_provider_config
from src.llm import get_llm_provider
from src.vectorstore import load_vectorstore, retrieve

SYSTEM_PROMPT = """You are a clinical information assistant that answers questions ONLY using the provided drug label excerpts.

Rules:
1. Base your answer strictly on the excerpts given. Do not use outside knowledge, even if you know it.
2. If the excerpts do not contain enough information to answer, say exactly: "The provided labels don't contain enough information to answer this." Do not guess.
3. After the answer, list which excerpt number(s) support each claim, like: [Source 1], [Source 2].
4. Be concise and precise with dosages, numbers, and contraindications — these are safety-critical. Do not round or approximate numbers from the source.
5. Never present it as personal medical advice; you are summarizing label content.
"""


@dataclass
class Citation:
    index: int
    source: str
    section: str
    text: str


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[Citation]


def format_context(documents) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        src = doc.metadata.get("source", "unknown")
        sec = doc.metadata.get("section", "unknown")
        blocks.append(f"[Source {i}] (Drug label: {src} | Section: {sec})\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(vectorstore, question: str, k: int | None = None) -> RAGResult:
    docs = retrieve(vectorstore, question, k=k)
    context = format_context(docs)

    user_prompt = f"""Drug label excerpts:

{context}

Question: {question}

Answer the question using only the excerpts above, citing [Source N] for each claim."""

    provider = get_llm_provider()
    answer = provider.complete(SYSTEM_PROMPT, user_prompt)

    citations = [
        Citation(
            index=i + 1,
            source=doc.metadata.get("source", "unknown"),
            section=doc.metadata.get("section", "unknown"),
            text=doc.page_content,
        )
        for i, doc in enumerate(docs)
    ]

    return RAGResult(question=question, answer=answer, citations=citations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask a question against the drug label index")
    parser.add_argument("--persist", default="./chroma_db")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    validate_provider_config()
    vs = load_vectorstore(args.persist)
    result = answer_question(vs, args.question, k=args.k)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Citations:")
    for c in result.citations:
        preview = c.text[:150].replace("\n", " ")
        print(f"  [Source {c.index}] {c.source} / {c.section}: {preview}...")
