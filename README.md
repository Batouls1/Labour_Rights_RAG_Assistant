# 📚 Legal RAG Assistant — Lebanese Labor Law

A bilingual (Arabic/English) Legal Question-Answering assistant for Employment Rights and Self-Employment law in Lebanon. Built on a production-shaped RAG pipeline with hybrid retrieval, cross-encoder reranking, and a full bilingual evaluation suite.

---

## Highlights

- **Bilingual support** — answers questions in both Arabic and English, with language-aware retrieval and generation.

- **Hybrid retrieval** — combines dense semantic search (multilingual E5 embeddings + FAISS) with sparse keyword search (BM25) for more robust retrieval.

- **Cross-encoder reranking** — re-scores retrieved chunks jointly with the query for higher-quality context before generation.

- **Grounded generation with refusal** — GPT-4.1-mini answers strictly from retrieved context; explicitly refuses to answer when information is not in the guides (validated at 100% refusal accuracy).

- **Full evaluation suite** — Precision@K, Recall@K, Hit@K, MRR, behavioral accuracy, and LLM-as-judge scoring across both languages.

- **Production-shaped architecture** — separated pipeline, FastAPI backend, Gradio interface, test suite with metrics thresholds, structured logging throughout.

---

## Evaluation Results

| Metric | English | Arabic |
|---|---|---|
| Answer accuracy | 0.96 | 0.88 |
| Refusal accuracy | 1.00 | 1.00 |
| Recall@3 | 1.00 | 0.62 |
| MRR (K=10) | 0.822 | 0.416 |
| LLM judge avg | 3.00 / 3 | 2.75 / 3 |

---

## Folder Structure:
```
├─ development_notebook.ipynb # Full pipeline development + evaluation
├─ rag_pipeline.py            # Core bilingual RAG pipeline (hybrid retrieval, reranking, generation)
├─ test_pipeline.py           # Test suite: language detection, retrieval filter, refusal, metrics
├─ gradio_app.py              # Bilingual Gradio chat interface
├─ api.py                     # FastAPI backend with input validation, error handling, logging
├─ data/                      # Lebanese labor law PDF guides (Arabic + English)
├─ assets/                    # Demo screenshot                         
├─ requirements.txt 
└─ README.md
```
---

## Pipeline Overview

**1. Document loading & chunking**
PDFs are loaded, blank pages removed, and language-tagged by filename. Text is split into token-aware overlapping chunks (500 tokens, 100 overlap) using tiktoken.

**2. Embedding generation**
Chunks are encoded with `intfloat/multilingual-e5-base` using E5's required `passage:` prefix. Embeddings are L2-normalized for cosine similarity via inner product.

**3. Hybrid retrieval (FAISS + BM25)**
Dense vectors indexed with `IndexFlatIP`. BM25 provides keyword matching with language-aware tokenization. Scores combined with language-specific weights (EN: 0.7/0.3, AR: 0.5/0.5).

**4. Language-aware filtering**
Query language detected via Unicode range analysis. Retrieval strictly filtered to same-language chunks to prevent cross-language mixing.

**5. Cross-encoder reranking**
Retrieved candidates reranked with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Final context: top 3 chunks (English), top 5 (Arabic).

**6. Grounded generation**
Language-specific prompts injected into GPT-4.1-mini with an explicit refusal mechanism when the answer is not found in context.

---

## Setup and Installation:

1. **Clone the repository:**
```bash
git clone https://github.com/Batouls1/Labour_Rights_RAG_Assistant
cd Labour_Rights_RAG_Assistant
```

2. **Create a virtual environment and activate it:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create a `.env` file in the project root with your OpenAI API key:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
API_URL=http://127.0.0.1:8000/ask
```
---

## Running the Project:

**Step 1 - Start the FastAPI server**
```bash
uvicorn api:app --reload
```
- The server runs at: http://127.0.0.1:8000/ask
- This endpoint handles legal questions and returns JSON responses.

**Step 2 - Start the Gradio interface** (new terminal, same venv):
```bash
python gradio_app.py
```
- Open your browser at `http://127.0.0.1:7860`
- Type questions in English or Arabic.

---

## Running the Test Suite
```bash
python test_pipeline.py
```
Runs 7 tests covering pipeline initialization, language detection, retrieval language filtering, refusal on unanswerable questions, source citation, empty query handling, and retrieval metric thresholds (Hit@3, MRR) for both languages. All tests must pass before committing changes.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector search | FAISS `IndexFlatIP` |
| Sparse retrieval | BM25Okapi (rank-bm25) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Generation | GPT-4.1-mini (OpenAI) |
| API | FastAPI + Uvicorn |
| Interface | Gradio |
| Chunking | LangChain + tiktoken |

---

## Limitations

- Evaluation conducted on a manually curated dataset of 30 questions — results are directional rather than statistically definitive
- Arabic retrieval underperforms English due to BM25 morphology limitations and multilingual embedding tradeoffs (Recall@3: 0.62 vs 1.00)
- Currently runs locally; cloud deployment, containerization, and concurrent user handling are not yet implemented

---

## Demo

![Demo Screenshot](assets/demo_screenshot.png)

*Example question:* "Is maternity leave paid and for how long?"
