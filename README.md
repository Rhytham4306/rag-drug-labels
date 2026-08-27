# 💊 MedLabel-RAG

A chatbot that answers questions about medications using real FDA drug label
data — and only answers from what it actually finds, citing its sources. If
it can't find a good answer, it says so instead of guessing.

![alt text](<Screenshot 2026-08-27 at 14.19.27.png>) 
![alt text](<Screenshot 2026-08-27 at 14.19.33.png>)
![alt text](<Screenshot 2026-08-27 at 14.23.53.png>)
---

## What it does

1. You ask a question, e.g. *"What's the max daily dose of ibuprofen?"*
2. It searches real drug label documents for the most relevant passages
3. It answers **using only those passages** — never from memory
4. It shows you the exact source text behind the answer
5. If nothing relevant is found, it says *"I don't know"* instead of guessing

This is a **RAG** (Retrieval-Augmented Generation) system — the same pattern
used in real production AI tools when accuracy and traceability matter more
than a confident-sounding guess.

---

## Demo

**Question it can answer:**
> Q: *What is the maximum daily dose of ibuprofen for adults?*
> A: *6 tablets (1200 mg) in 24 hours, unless a doctor directs otherwise.* `[Source 2]`

**Question it correctly refuses:**
> Q: *What is the maximum dose of paracetamol?*
> A: *The provided labels don't contain enough information to answer this.*

(Paracetamol was never loaded into the system — instead of guessing based on
a similar drug, it says so. That's the point of this project.)

*Screenshots:*
`![answered example](assets/demo-answered.png)`
`![refused example](assets/demo-refused.png)`

---

## Results

Tested on 3 real drug labels (420 text chunks) against 15 known questions:

| Metric | Result |
|---|---|
| Found the right document | 100% of the time |
| Made up a fact when it answered | 0% (never) |
| Said "I don't know" instead of guessing | ~15% of the time |
| Average answer time | ~4 seconds |

Run it yourself:
```bash
python -m eval.evaluate --persist ./chroma_db --k 3
```

---

## How it's built

```
Documents → split into paragraphs → turned into searchable "meaning" data
        → question comes in → most relevant paragraphs found
        → AI answers using only those paragraphs → cites its source
```

**Tech used:** Python, LangChain, ChromaDB (search), Groq (free AI model),
Streamlit (chat interface)

---

## Setup

```bash
git clone <your-repo-url>
cd rag-drug-labels
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Get a free API key at [console.groq.com](https://console.groq.com) and add it
to `.env` as `GROQ_API_KEY=...`

**Download real drug data (free, no key needed):**
```bash
python scripts/fetch_dailymed.py --drugs ibuprofen metformin lisinopril --out data/dailymed
```

**Build the search index:**
```bash
python -m src.vectorstore --docs data/dailymed --persist ./chroma_db
```

**Run the chat app:**
```bash
python -m streamlit run app.py
```

**Ask a question from the terminal instead:**
```bash
python -m src.rag_chain --persist ./chroma_db --question "What is the maximum daily dose of ibuprofen?"
```

---

## Project files

```
├── src/ingest.py        → splits documents into searchable paragraphs
├── src/vectorstore.py   → builds the search index
├── src/llm.py           → talks to the AI model (swappable: Groq/Ollama/OpenAI)
├── src/rag_chain.py     → the main search → answer → cite logic
├── app.py               → the chat interface
├── eval/evaluate.py     → tests accuracy automatically
└── scripts/fetch_dailymed.py → downloads real drug label data
```

---

## What could be improved

- Add keyword-based search alongside meaning-based search, for exact drug
  names and numbers
- Add a re-ranking step to reduce how often it says "I don't know"
- Test on more drugs and a bigger question set

## Disclaimer

Educational/portfolio project — not a medical device, not for clinical use.

## License

MIT — see [LICENSE](LICENSE).
