from pathlib import Path

import gradio as gr

from stacklite_qa import StackLiteHybridQA, load_stacklite_documents


ROOT = Path(__file__).resolve().parent
DOCS = load_stacklite_documents(ROOT / "data")
ENGINE = StackLiteHybridQA(DOCS)
ENGINE.build_bm25()


def answer_question(query: str, mode: str, top_k: int, use_generator: bool) -> tuple[str, str]:
    if not query.strip():
        return "Ask a technical question about AI, ML, or data science.", ""

    if mode == "BM25":
        results = ENGINE.bm25_search(query, top_k=top_k)
    else:
        try:
            results = ENGINE.hybrid_search(query, top_k=top_k)
        except ImportError:
            results = ENGINE.bm25_search(query, top_k=top_k)

    backend = "hf" if use_generator else "extractive"
    try:
        answer = ENGINE.generate_answer(query, results, backend=backend)
    except Exception as exc:
        answer = ENGINE.generate_answer(query, results, backend="extractive")
        answer += f"\n\nGenerator fallback: {exc}"

    rows = [
        {
            "rank": result.rank,
            "doc_id": result.doc.doc_id,
            "title": result.doc.title,
            "score": round(result.score, 4),
            "url": result.doc.link,
        }
        for result in results
    ]
    return answer, rows


with gr.Blocks(title="StackLite Hybrid QA") as demo:
    gr.Markdown("# StackLite Hybrid QA")
    with gr.Row():
        query = gr.Textbox(
            label="Question",
            placeholder="Example: What is the difference between fit and fit_transform in scikit-learn?",
            lines=2,
        )
    with gr.Row():
        mode = gr.Radio(["Hybrid RRF", "BM25"], value="Hybrid RRF", label="Retrieval")
        top_k = gr.Slider(3, 10, value=5, step=1, label="Citations")
        use_generator = gr.Checkbox(value=False, label="Use HuggingFace FLAN-T5 generator")
    submit = gr.Button("Ask")
    answer = gr.Textbox(label="Answer with citations", lines=10)
    sources = gr.Dataframe(label="Retrieved sources", wrap=True)
    submit.click(answer_question, inputs=[query, mode, top_k, use_generator], outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
