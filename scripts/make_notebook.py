import json
import itertools
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "StackLite_Hybrid_QA_Demo.ipynb"
CORE_SOURCE = (ROOT / "stacklite_qa" / "core.py").read_text(encoding="utf-8")
INIT_SOURCE = (ROOT / "stacklite_qa" / "__init__.py").read_text(encoding="utf-8")
EVAL_SOURCE = (ROOT / "evaluation" / "eval_questions.json").read_text(encoding="utf-8")
CELL_IDS = itertools.count(1)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-{next(CELL_IDS):03d}",
        "metadata": {},
        "source": source.strip().splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"cell-{next(CELL_IDS):03d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        """
        # StackLite Hybrid QA Demo

        This Colab-ready notebook builds a technical question-answering assistant over the provided StackLite JSON corpus.

        It demonstrates BM25 retrieval, MiniLM dense retrieval, RRF hybrid fusion, retrieval metrics, RAG answer generation with citations, and a Gradio UI.
        """
    ),
    md(
        """
        ## 1. Setup

        In Google Colab, upload `DataSet.zip` to the session or mount Drive and update `DATA_ZIP`.
        Use a free GPU runtime for faster MiniLM embeddings and HuggingFace generation.
        """
    ),
    code(
        """
        !pip -q install beautifulsoup4 gradio numpy openai pandas rank-bm25 scikit-learn sentence-transformers torch transformers tqdm
        """
    ),
    code(
        """
        from pathlib import Path
        import shutil
        import zipfile

        PROJECT_DIR = Path.cwd() / "stacklite_hybrid_qa"
        DATA_ZIP = Path("/content/DataSet.zip")

        PROJECT_DIR.mkdir(exist_ok=True)
        (PROJECT_DIR / "data").mkdir(exist_ok=True)

        if DATA_ZIP.exists():
            with zipfile.ZipFile(DATA_ZIP) as zf:
                zf.extractall(PROJECT_DIR / "data")
        else:
            print("Upload DataSet.zip to /content/DataSet.zip, or run this notebook from the submitted repo where data/ already exists.")
        """
    ),
    md("## 2. Core Code\n\nThis cell writes the same project module used by the repository so the notebook can run standalone in Colab."),
    code(
        f"""
        import sys
        from pathlib import Path

        Path("stacklite_qa").mkdir(exist_ok=True)
        Path("evaluation").mkdir(exist_ok=True)
        Path("stacklite_qa/core.py").write_text({CORE_SOURCE!r}, encoding="utf-8")
        Path("stacklite_qa/__init__.py").write_text({INIT_SOURCE!r}, encoding="utf-8")
        Path("evaluation/eval_questions.json").write_text({EVAL_SOURCE!r}, encoding="utf-8")
        sys.path.insert(0, str(Path.cwd()))
        """
    ),
    md("## 3. Load Corpus"),
    code(
        """
        import sys
        sys.path.insert(0, str(Path.cwd()))

        from stacklite_qa import StackLiteHybridQA, evaluate_retriever, load_evaluation_questions, load_stacklite_documents

        data_dir = Path("data") if Path("data/top_ai_questions.json").exists() else PROJECT_DIR / "data"
        docs = load_stacklite_documents(data_dir)
        engine = StackLiteHybridQA(docs)
        print(f"Loaded {len(docs)} StackLite documents")
        print(docs[0])
        """
    ),
    md("## 4. BM25 Retrieval: Top-10 Results"),
    code(
        """
        sample_question = "What is the difference between fit and fit_transform in scikit-learn?"
        bm25_results = engine.bm25_search(sample_question, top_k=10)
        for result in bm25_results:
            print(f"{result.rank:02d} | {result.doc.doc_id} | {result.score:.2f} | {result.doc.title}")
            print(f"     {result.doc.link}")
        """
    ),
    md("## 5. Retrieval Evaluation: MAP@10 and MRR@10"),
    code(
        """
        eval_file = Path("evaluation/eval_questions.json")
        questions = load_evaluation_questions(eval_file)
        bm25_metrics = evaluate_retriever(questions, lambda q, k: engine.bm25_search(q, top_k=k), k=10)
        bm25_metrics
        """
    ),
    md("## 6. Dense Retrieval with MiniLM and nDCG@10"),
    code(
        """
        engine.build_dense()
        dense_metrics = evaluate_retriever(questions, lambda q, k: engine.dense_search(q, top_k=k), k=10)
        dense_metrics
        """
    ),
    md("## 7. Hybrid RRF Retrieval"),
    code(
        """
        hybrid_metrics = evaluate_retriever(questions, lambda q, k: engine.hybrid_search(q, top_k=k), k=10)
        print("BM25:", bm25_metrics)
        print("Dense MiniLM:", dense_metrics)
        print("Hybrid RRF:", hybrid_metrics)
        """
    ),
    md("## 8. RAG Answer with Citations"),
    code(
        """
        rag_question = "What is the positional encoding in the transformer model?"
        contexts = engine.hybrid_search(rag_question, top_k=5)

        # Use backend='hf' for an open-source LLM answer. The extractive fallback is faster for class demos.
        answer = engine.generate_answer(rag_question, contexts, backend="extractive")
        print(answer)
        """
    ),
    md("## 9. Optional HuggingFace Generation"),
    code(
        """
        # This downloads google/flan-t5-base on first run.
        # answer = engine.generate_answer(rag_question, contexts, backend="hf", model_name="google/flan-t5-base")
        # print(answer)
        """
    ),
    md("## 10. Gradio UI"),
    code(
        """
        import gradio as gr

        def ask(query, mode, top_k, use_hf):
            if mode == "BM25":
                results = engine.bm25_search(query, top_k=int(top_k))
            else:
                results = engine.hybrid_search(query, top_k=int(top_k))
            backend = "hf" if use_hf else "extractive"
            try:
                answer = engine.generate_answer(query, results, backend=backend)
            except Exception as exc:
                answer = engine.generate_answer(query, results, backend="extractive") + f"\\n\\nFallback: {exc}"
            rows = [[r.rank, r.doc.doc_id, r.doc.title, round(r.score, 4), r.doc.link] for r in results]
            return answer, rows

        with gr.Blocks(title="StackLite Hybrid QA") as demo:
            gr.Markdown("# StackLite Hybrid QA")
            query = gr.Textbox(label="Question", lines=2)
            mode = gr.Radio(["Hybrid RRF", "BM25"], value="Hybrid RRF", label="Retrieval")
            top_k = gr.Slider(3, 10, value=5, step=1, label="Citations")
            use_hf = gr.Checkbox(value=False, label="Use HuggingFace FLAN-T5")
            button = gr.Button("Ask")
            answer = gr.Textbox(label="Answer with citations", lines=8)
            sources = gr.Dataframe(headers=["rank", "doc_id", "title", "score", "url"], label="Retrieved sources")
            button.click(ask, [query, mode, top_k, use_hf], [answer, sources])

        demo.launch(share=True)
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK)
