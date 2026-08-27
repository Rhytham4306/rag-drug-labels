"""
Evaluation harness for the RAG system.

Two things are measured, deliberately kept separate:

1. RETRIEVAL QUALITY — did the vector search find the right document at all?
   - Hit Rate@k: fraction of questions where the gold source document
     appears among the top-k retrieved chunks.
   - MRR (Mean Reciprocal Rank): rewards the gold document appearing
     higher in the ranking, not just anywhere in top-k.

2. FAITHFULNESS — given what was retrieved, did the LLM's answer stick to
   it, or did it hallucinate? This uses an "LLM-as-judge" prompt: a
   separate call to the same LLM backend is shown ONLY the retrieved
   chunk text and the generated answer (never outside knowledge) and
   asked to score whether every claim is supported. This is the same
   basic idea used by evaluation frameworks like RAGAS.

Usage:
    python -m eval.evaluate --persist ./chroma_db --eval-set eval/eval_dataset.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from src.config import validate_provider_config
from src.llm import get_llm_provider
from src.rag_chain import answer_question, format_context
from src.vectorstore import load_vectorstore, retrieve

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. You will be given source excerpts and a generated answer.
Score, from 1 to 5, whether EVERY factual claim in the answer is directly supported by the source excerpts:
5 = fully supported, no invented facts
3 = partially supported, some unsupported or vague claims
1 = mostly or entirely unsupported / hallucinated
Respond with ONLY a single integer from 1 to 5, nothing else."""

ABSTENTION_PHRASE = "don't contain enough information"


def hit_rate_and_mrr(vectorstore, eval_items: list[dict], k: int) -> tuple[float, float, list[dict]]:
    hits = 0
    reciprocal_ranks = []
    per_item_results = []

    for item in eval_items:
        docs = retrieve(vectorstore, item["question"], k=k)
        sources = [d.metadata.get("source", "") for d in docs]

        rank = None
        for i, src in enumerate(sources, start=1):
            if src == item["gold_source"]:
                rank = i
                break

        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)

        per_item_results.append(
            {
                "question": item["question"],
                "gold_source": item["gold_source"],
                "retrieved_sources": sources,
                "hit": hit,
                "rank": rank,
            }
        )

    hit_rate = hits / len(eval_items)
    mrr = statistics.mean(reciprocal_ranks)
    return hit_rate, mrr, per_item_results


def judge_faithfulness(provider, question: str, context: str, answer: str) -> int:
    user_prompt = f"""Source excerpts:
{context}

Generated answer:
{answer}

Score (1-5):"""
    raw = provider.complete(JUDGE_SYSTEM_PROMPT, user_prompt).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return int(digits[0]) if digits else None


def run_full_evaluation(persist_dir: str, eval_set_path: str, k: int, skip_generation: bool):
    eval_items = json.loads(Path(eval_set_path).read_text())
    vectorstore = load_vectorstore(persist_dir)

    print(f"Evaluating retrieval on {len(eval_items)} questions (k={k})...\n")
    hit_rate, mrr, retrieval_results = hit_rate_and_mrr(vectorstore, eval_items, k)

    print(f"Hit Rate@{k}: {hit_rate:.2%}")
    print(f"MRR:          {mrr:.3f}\n")

    for r in retrieval_results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} [{r['gold_source']:>12}] {r['question']}")
        print(f"     retrieved: {r['retrieved_sources']}")

    if skip_generation:
        print("\n--skip-generation set: not calling the LLM for faithfulness scoring.")
        return

    print("\nRunning generation + faithfulness scoring (calls the LLM)...\n")
    validate_provider_config()
    provider = get_llm_provider()

    faithfulness_scores = []
    abstained_scores = []
    latencies = []
    abstention_count = 0

    for item in eval_items:
        start = time.time()
        result = answer_question(vectorstore, item["question"], k=k)
        latencies.append(time.time() - start)

        context = format_context(
            [type("D", (), {"page_content": c.text, "metadata": {}}) for c in result.citations]
        )
        is_abstention = ABSTENTION_PHRASE in result.answer
        score = judge_faithfulness(provider, item["question"], context, result.answer)

        if is_abstention:
            abstention_count += 1
            if score is not None:
                abstained_scores.append(score)
        else:
            if score is not None:
                faithfulness_scores.append(score)

        tag = "🚫 ABSTAINED" if is_abstention else f"Faithfulness: {score}/5"
        print(f"Q: {item['question']}")
        print(f"A: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")
        print(f"{tag}\n")

    n = len(eval_items)
    abstention_rate = abstention_count / n

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Hit Rate@{k}:                     {hit_rate:.2%}")
    print(f"MRR:                              {mrr:.3f}")
    print(f"Abstention rate:                  {abstention_rate:.2%}  ({abstention_count}/{n})")
    if faithfulness_scores:
        print(f"Faithfulness (answered only):     {statistics.mean(faithfulness_scores):.2f} / 5  (n={len(faithfulness_scores)})")
        hallucination_rate = sum(1 for s in faithfulness_scores if s <= 2) / len(faithfulness_scores)
        print(f"Hallucination rate (answered):     {hallucination_rate:.2%}")
    else:
        print("Faithfulness (answered only):     n/a — every answer was an abstention")
    print(f"Avg. latency / query:             {statistics.mean(latencies):.2f}s")
    print()
    print("Note: abstentions ('labels don't contain enough information') are tracked")
    print("separately from faithfulness. A high abstention rate on questions the docs")
    print("DO cover suggests retrieval is missing the right chunk (see Hit Rate@k vs.")
    print("actual chunk coverage) rather than the model hallucinating — these are")
    print("different failure modes and should not be averaged into one number.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("--persist", default="./chroma_db")
    parser.add_argument("--eval-set", default="eval/eval_dataset.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only run retrieval metrics (no LLM calls, no API key needed).",
    )
    args = parser.parse_args()

    run_full_evaluation(args.persist, args.eval_set, args.k, args.skip_generation)