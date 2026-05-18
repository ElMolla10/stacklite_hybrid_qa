# StackLite Hybrid QA

> A retrieval-augmented question-answering system for technical AI and data-science questions, built over the StackLite Stack Exchange corpus.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

Ask a technical question — the system retrieves the most relevant Stack Exchange posts using **hybrid search** (BM25 + semantic embeddings fused with RRF), then generates a grounded answer with cited sources using a language model.

```
User question
      │
      ▼
┌─────────────────────────────────┐
│         Hybrid Retrieval        │
│  BM25 ──┐                       │
│          ├─► RRF Fusion ─► Top-K│
│  MiniLM ─┘                      │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│         RAG Generation          │
│  FLAN-T5 / OpenAI / Extractive  │
│  + inline citations [doc_id]    │
└─────────────────────────────────┘
      │
      ▼
  Answer + source table (Gradio UI)
```

---

## Features

| Feature | Details |
|---|---|
| **BM25 retrieval** | Keyword-based lexical search via `rank-bm25` |
| **Dense retrieval** | Semantic search using `all-MiniLM-L6-v2` embeddings |
| **Hybrid fusion** | Reciprocal Rank Fusion (RRF, k=60) combines both signals |
| **RAG generation** | HuggingFace FLAN-T5, OpenAI, or extractive fallback |
| **Citations** | Every answer cites source IDs and Stack Exchange URLs |
| **Evaluation** | MAP@10, MRR@10, nDCG@10, HitRate@10 over 10 eval questions |
| **Gradio UI** | Live Q&A with retrieval mode toggle and source table |
| **Colab-ready** | Demo notebook works on free T4/V100 GPU runtime |

---

## Retrieval Results

Evaluated on 10 paraphrased technical questions against the StackLite corpus:

| Method | MAP@10 | MRR@10 | nDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.800 | 0.800 | 0.849 | **1.000** |
| MiniLM Dense | **0.933** | **0.933** | **0.950** | **1.000** |
| Hybrid RRF | 0.825 | 0.825 | 0.869 | **1.000** |

All three methods retrieve every relevant document in the top 10. MiniLM leads on paraphrase-style queries; Hybrid RRF is the recommended default for robustness across query types.

---

## Project Structure

```
stacklite_hybrid_qa/
├── app.py                          # Gradio UI entry point
├── requirements.txt
├── stacklite_qa/
│   ├── core.py                     # BM25, dense, RRF, RAG, evaluation logic
│   └── __init__.py
├── data/
│   ├── top_ai_questions.json       # 750 AI Stack Exchange records
│   ├── top_datascience_questions.json  # 750 Data Science records
│   └── README.md
├── notebooks/
│   └── StackLite_Hybrid_QA_Demo.ipynb  # Colab walkthrough
├── evaluation/
│   └── eval_questions.json         # 10 evaluation questions with ground truth
├── reports/
│   └── evaluation_results.json     # Saved metric outputs
├── docs/
│   └── report.pdf                  # Full project report
└── video/
    └── stacklite_hybrid_qa_demo.mp4
```

---

## Quick Start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/ElMolla10/stacklite_hybrid_qa.git
cd stacklite_hybrid_qa
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Then open the local Gradio URL printed in the terminal.

> **First run in Hybrid mode** will download the MiniLM model (~90 MB). BM25 mode is instant.

### Colab

Open `notebooks/StackLite_Hybrid_QA_Demo.ipynb` in Google Colab, select a **T4 or V100 GPU runtime**, and run all cells. The final cell launches a public Gradio link.

---

## Generation Backends

The system supports three answer-generation modes, selectable without code changes:

| Backend | How to enable | Requires |
|---|---|---|
| **Extractive** (default) | Always active as fallback | Nothing |
| **HuggingFace FLAN-T5** | Check "Use HuggingFace generator" in UI | GPU recommended |
| **OpenAI** | Set `OPENAI_API_KEY` env var | API key |

```bash
# OpenAI example
export OPENAI_API_KEY=sk-...
.venv/bin/python app.py
```

---

## Dataset

1,500 Stack Exchange records across two domains:

- `data/top_ai_questions.json` — Artificial Intelligence SE
- `data/top_datascience_questions.json` — Data Science SE

Each record contains a title, HTML body, tags, score, question ID, and source URL. Drop a larger StackLite-6K export into `data/` and rerun — the loader picks up any `top_*_questions.json` file automatically.

---

## Team

| Name | Contribution |
|---|---|
| **Mohamed Ehab** | Corpus loading, preprocessing, BM25 retrieval |
| **Yahia Abdelmoneam** | Dense retrieval, RRF fusion, evaluation metrics |
| **Mohamed Atef** | RAG generation, citations, Gradio UI, report & demo packaging |

---

## Deliverables

| Item | Location |
|---|---|
| Demo notebook | `notebooks/StackLite_Hybrid_QA_Demo.ipynb` |
| Core library | `stacklite_qa/core.py` |
| Gradio app | `app.py` |
| Evaluation set | `evaluation/eval_questions.json` |
| Citation quality examples | `docs/citation_quality.md` |
| Report | `docs/report.pdf` |
| Video walkthrough | [`video/stacklite_hybrid_qa_demo.mp4`](https://github.com/ElMolla10/stacklite_hybrid_qa/blob/main/video/stacklite_hybrid_qa_demo.mp4) |
