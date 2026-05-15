# StackLite Hybrid QA

Question-answering assistant for technical AI/data-science questions over the provided StackLite corpus archive.

## What It Includes

- BM25 lexical retrieval with top-10 inspection.
- Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- Reciprocal Rank Fusion (RRF) across BM25 and dense results.
- Retrieval evaluation with MAP@10, MRR@10, nDCG@10, and HitRate@10.
- RAG answer generation with citations using either HuggingFace FLAN-T5, OpenAI, or a local extractive fallback.
- Gradio UI for live Q&A.
- Colab-ready demo notebook in `notebooks/StackLite_Hybrid_QA_Demo.ipynb`.

## Dataset

The project was built and evaluated on the provided dataset archive at `/Users/mohamedehabelmolla/Downloads/DataSet.zip`. The extracted files are committed under `data/`:

- `data/top_ai_questions.json`
- `data/top_datascience_questions.json`

These two files contain 1,500 Stack Exchange records from the supplied StackLite dataset. See `data/README.md` for file sizes and replacement instructions for a larger StackLite-6K export.

## Quick Start

```bash
cd /Users/mohamedehabelmolla/stacklite_hybrid_qa
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/evaluate.py --skip-dense
.venv/bin/python app.py
```

For full dense retrieval and hybrid evaluation:

```bash
.venv/bin/python scripts/evaluate.py
```

The first dense run downloads MiniLM. In Colab, select a T4/V100 GPU runtime before running the notebook cells.

## Optional Generation Backends

- HuggingFace: enable the UI checkbox or call `generate_answer(..., backend="hf")`.
- OpenAI: install requirements, set `OPENAI_API_KEY`, optionally set `OPENAI_MODEL`, and call `backend="openai"`.
- Extractive fallback: default mode, no API key required.

## Team Log

| Role | Work |
| --- | --- |
| Member 1 | Corpus loading, preprocessing, BM25 retrieval. |
| Member 2 | Dense retrieval, RRF fusion, evaluation metrics. |
| Member 3 | RAG generation, citations, Gradio UI, report/demo packaging. |

## Submission Checklist

- Demo notebook: `notebooks/StackLite_Hybrid_QA_Demo.ipynb`
- Code: `stacklite_qa/core.py`, `app.py`, `scripts/evaluate.py`
- Evaluation questions: `evaluation/eval_questions.json`
- Citation quality examples: `docs/citation_quality.md`
- Short report: `docs/report.md`
- Video walkthrough: `video/stacklite_hybrid_qa_demo.mp4`
- Public GitHub video link: https://github.com/ElMolla10/stacklite_hybrid_qa/blob/main/video/stacklite_hybrid_qa_demo.mp4
- Video walkthrough guide: `docs/video_demo_script.md`
