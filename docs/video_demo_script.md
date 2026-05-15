# 3-5 Minute Video Demo Script

## 0:00-0:30 Project Overview

Show the repository. Explain that the system loads the provided StackLite JSON corpus, builds lexical and semantic indexes, fuses results with RRF, and generates cited answers through a Gradio interface.

## 0:30-1:20 Notebook Walkthrough

Open `notebooks/StackLite_Hybrid_QA_Demo.ipynb`.
Run the setup and corpus-loading cells.
Point out that each Stack Exchange question becomes one searchable passage with title, tags, body, URL, and document ID.

## 1:20-2:05 BM25 and Evaluation

Run the BM25 sample query cell.
Show top-10 results for a query such as:

```text
What is the difference between fit and fit_transform in scikit-learn?
```

Run the BM25 evaluation cell and show MAP@10 and MRR@10.

## 2:05-2:55 Dense Search and Hybrid Fusion

Run the MiniLM embedding cell.
Compare BM25, dense, and hybrid metrics, including nDCG@10.
Explain that RRF rewards documents that rank well in either retrieval system.

## 2:55-3:45 RAG Answer With Citations

Run a RAG example question:

```text
What is the positional encoding in the transformer model?
```

Show that the answer includes citations like `[datascience:51065]` plus source URLs.

## 3:45-4:30 Gradio UI

Run `python3 app.py` or the notebook UI cell.
Ask two live questions, one data-science query and one AI query.
Show the answer panel and retrieved source table.

## 4:30-5:00 Wrap-Up

Summarize the final deliverables: code, notebook, evaluation questions, citation-quality document, report, and UI. Mention that HuggingFace generation works without an OpenAI key, while OpenAI can be enabled with `OPENAI_API_KEY`.
