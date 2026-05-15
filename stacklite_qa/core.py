from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - used only in minimal environments.
    BeautifulSoup = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.\-]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    question_id: int
    source: str
    title: str
    body: str
    tags: list[str]
    link: str
    score: int

    @property
    def text(self) -> str:
        tags = " ".join(self.tags)
        return f"{self.title}\nTags: {tags}\n{self.body}"

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}] {self.title} ({self.link})"


@dataclass
class SearchResult:
    doc: Document
    score: float
    rank: int
    bm25_score: float | None = None
    dense_score: float | None = None


def clean_html(raw_html: str) -> str:
    if BeautifulSoup is not None:
        text = BeautifulSoup(raw_html or "", "html.parser").get_text(" ")
    else:
        text = re.sub(r"<[^>]+>", " ", raw_html or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def load_stacklite_documents(data_dir: str | Path = "data") -> list[Document]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("top_*_questions.json"))
    if not files:
        raise FileNotFoundError(f"No StackLite JSON files found in {data_path.resolve()}")

    docs: list[Document] = []
    for file_path in files:
        source = "ai" if "ai" in file_path.stem else "datascience"
        with file_path.open(encoding="utf-8") as f:
            records = json.load(f)

        for row in records:
            qid = int(row["question_id"])
            title = unescape(str(row.get("title", ""))).strip()
            tags = [str(tag) for tag in row.get("tags", [])]
            docs.append(
                Document(
                    doc_id=f"{source}:{qid}",
                    question_id=qid,
                    source=source,
                    title=title,
                    body=clean_html(str(row.get("body", ""))),
                    tags=tags,
                    link=str(row.get("link", "")),
                    score=int(row.get("score", 0) or 0),
                )
            )
    return docs


class StackLiteHybridQA:
    def __init__(
        self,
        documents: Iterable[Document],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document is required.")
        self.embedding_model_name = embedding_model_name
        self._doc_by_id = {doc.doc_id: doc for doc in self.documents}
        self._bm25 = None
        self._tokenized_corpus: list[list[str]] | None = None
        self._embedding_model = None
        self._doc_embeddings: np.ndarray | None = None

    def build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25 to use BM25 retrieval.") from exc
        self._tokenized_corpus = [tokenize(doc.text) for doc in self.documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def build_dense(self, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use dense retrieval.") from exc
        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        embeddings = self._embedding_model.encode(
            [doc.text for doc in self.documents],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._doc_embeddings = np.asarray(embeddings, dtype=np.float32)

    def bm25_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._bm25 is None:
            self.build_bm25()
        assert self._bm25 is not None
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(doc=self.documents[i], score=float(scores[i]), rank=rank, bm25_score=float(scores[i]))
            for rank, i in enumerate(order, start=1)
        ]

    def dense_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._doc_embeddings is None or self._embedding_model is None:
            self.build_dense()
        assert self._embedding_model is not None and self._doc_embeddings is not None
        query_vec = self._embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(self._doc_embeddings, query_vec.astype(np.float32))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(doc=self.documents[i], score=float(scores[i]), rank=rank, dense_score=float(scores[i]))
            for rank, i in enumerate(order, start=1)
        ]

    def rrf_fuse(
        self,
        ranked_lists: list[list[SearchResult]],
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        combined: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}
        for results in ranked_lists:
            for result in results:
                doc_id = result.doc.doc_id
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + result.rank)
                if doc_id not in combined:
                    combined[doc_id] = SearchResult(doc=result.doc, score=0.0, rank=0)
                if result.bm25_score is not None:
                    combined[doc_id].bm25_score = result.bm25_score
                if result.dense_score is not None:
                    combined[doc_id].dense_score = result.dense_score

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        fused: list[SearchResult] = []
        for rank, (doc_id, score) in enumerate(ordered, start=1):
            result = combined[doc_id]
            result.score = float(score)
            result.rank = rank
            fused.append(result)
        return fused

    def hybrid_search(self, query: str, top_k: int = 10, candidate_k: int = 50) -> list[SearchResult]:
        bm25 = self.bm25_search(query, top_k=candidate_k)
        dense = self.dense_search(query, top_k=candidate_k)
        return self.rrf_fuse([bm25, dense], top_k=top_k)

    def format_contexts(self, results: list[SearchResult], max_chars_per_doc: int = 900) -> str:
        chunks = []
        for result in results:
            doc = result.doc
            text = doc.body[:max_chars_per_doc].strip()
            chunks.append(
                f"Source {result.rank}: {doc.doc_id}\n"
                f"Title: {doc.title}\n"
                f"Tags: {', '.join(doc.tags)}\n"
                f"URL: {doc.link}\n"
                f"Passage: {text}"
            )
        return "\n\n".join(chunks)

    def generate_answer(
        self,
        query: str,
        results: list[SearchResult],
        backend: str = "extractive",
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 220,
    ) -> str:
        if backend == "openai":
            return self._generate_openai(query, results, max_new_tokens=max_new_tokens)
        if backend == "hf":
            return self._generate_hf(query, results, model_name=model_name, max_new_tokens=max_new_tokens)
        return self._generate_extractive(query, results)

    def _generate_hf(
        self,
        query: str,
        results: list[SearchResult],
        model_name: str,
        max_new_tokens: int,
    ) -> str:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install transformers and torch to use the HuggingFace RAG backend.") from exc
        context = self.format_contexts(results[:4], max_chars_per_doc=700)
        prompt = (
            "Answer the technical question using only the sources below. "
            "Cite source ids like [datascience:15989] after each factual claim.\n\n"
            f"Question: {query}\n\nSources:\n{context}\n\nAnswer:"
        )
        generator = pipeline("text2text-generation", model=model_name)
        answer = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"].strip()
        citations = "\n".join(f"- {result.doc.citation}" for result in results[:4])
        return f"{answer}\n\nCitations:\n{citations}"

    def _generate_openai(self, query: str, results: list[SearchResult], max_new_tokens: int) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai and set OPENAI_API_KEY to use the OpenAI backend.") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI()
        context = self.format_contexts(results[:5], max_chars_per_doc=900)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful technical QA assistant. Use only the provided StackLite sources. "
                        "Cite source ids in square brackets for claims."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
            ],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        citations = "\n".join(f"- {result.doc.citation}" for result in results[:5])
        return f"{answer}\n\nCitations:\n{citations}"

    def _generate_extractive(self, query: str, results: list[SearchResult]) -> str:
        query_terms = set(tokenize(query))
        selected = []
        for result in results[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", result.doc.body)
            best = max(
                sentences[:16] or [result.doc.body],
                key=lambda sent: len(query_terms.intersection(tokenize(sent))) / math.sqrt(len(tokenize(sent)) + 1),
            )
            selected.append(f"{best.strip()} [{result.doc.doc_id}]")
        answer = " ".join(part for part in selected if part)
        citations = "\n".join(f"- {result.doc.citation}" for result in results[:3])
        return f"{answer}\n\nCitations:\n{citations}"


def load_evaluation_questions(path: str | Path = "evaluation/eval_questions.json") -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / idx
    return total / min(len(relevant), k)


def _reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def _dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum((1.0 / math.log2(idx + 1)) for idx, doc_id in enumerate(retrieved[:k], start=1) if doc_id in relevant)


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return _dcg_at_k(retrieved, relevant, k) / ideal


def evaluate_retriever(
    questions: list[dict],
    search_fn: Callable[[str, int], list[SearchResult]],
    k: int = 10,
) -> dict[str, float]:
    ap_scores = []
    rr_scores = []
    ndcg_scores = []
    hit_scores = []
    for item in questions:
        relevant = set(item["relevant_doc_ids"])
        results = search_fn(item["question"], k)
        retrieved = [result.doc.doc_id for result in results]
        ap_scores.append(_average_precision_at_k(retrieved, relevant, k))
        rr_scores.append(_reciprocal_rank_at_k(retrieved, relevant, k))
        ndcg_scores.append(_ndcg_at_k(retrieved, relevant, k))
        hit_scores.append(float(bool(relevant.intersection(retrieved[:k]))))
    return {
        f"MAP@{k}": float(np.mean(ap_scores)),
        f"MRR@{k}": float(np.mean(rr_scores)),
        f"nDCG@{k}": float(np.mean(ndcg_scores)),
        f"HitRate@{k}": float(np.mean(hit_scores)),
    }
