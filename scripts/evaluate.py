import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stacklite_qa import StackLiteHybridQA, evaluate_retriever, load_evaluation_questions, load_stacklite_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StackLite BM25, dense, and hybrid retrieval.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--eval-file", default="evaluation/eval_questions.json")
    parser.add_argument("--output", default="reports/evaluation_results.json")
    parser.add_argument("--skip-dense", action="store_true", help="Run only BM25 metrics.")
    args = parser.parse_args()

    docs = load_stacklite_documents(args.data_dir)
    questions = load_evaluation_questions(args.eval_file)
    engine = StackLiteHybridQA(docs)

    results = {
        "corpus_size": len(docs),
        "num_eval_questions": len(questions),
        "bm25": evaluate_retriever(questions, lambda q, k: engine.bm25_search(q, top_k=k), k=10),
    }

    if not args.skip_dense:
        engine.build_dense()
        results["dense_minilm"] = evaluate_retriever(questions, lambda q, k: engine.dense_search(q, top_k=k), k=10)
        results["hybrid_rrf"] = evaluate_retriever(questions, lambda q, k: engine.hybrid_search(q, top_k=k), k=10)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
