import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "video"
SLIDE_DIR = OUT_DIR / "slides"
VIDEO = OUT_DIR / "stacklite_hybrid_qa_demo.mp4"
VOICE = OUT_DIR / "voiceover.aiff"
METRICS = json.loads((ROOT / "reports" / "evaluation_results.json").read_text(encoding="utf-8"))

SLIDES = [
    (
        "StackLite Hybrid QA",
        "A retrieval-augmented question-answering assistant for technical AI and data-science questions.",
        "This demo walks through the StackLite Hybrid QA project. The system loads the provided Stack Exchange JSON files, indexes the questions, retrieves relevant passages, and generates cited answers.",
    ),
    (
        "Corpus and Indexing",
        "1,500 StackLite question records are converted into passages with title, tags, body, URL, and stable document ID.",
        "Each question becomes one searchable passage. The stable document IDs, such as datascience colon twelve three twenty one, are used in citations and evaluation.",
    ),
    (
        "BM25 Retrieval",
        "Lexical search uses rank BM25 and returns top-10 results for sample technical questions.",
        "BM25 is the keyword baseline. It works well for exact terms such as scikit learn, Keras, transformer, and reinforcement learning.",
    ),
    (
        "MiniLM Dense Search",
        "Semantic search uses sentence-transformers all MiniLM L6 v2 with normalized cosine similarity.",
        "The dense retriever helps when the user asks a paraphrase rather than repeating the Stack Exchange title exactly.",
    ),
    (
        "Hybrid RRF Fusion",
        "Reciprocal Rank Fusion combines BM25 and dense rankings without score normalization.",
        "The hybrid retriever gives credit to documents that rank highly in either retrieval list. This keeps the live demo robust across keyword-heavy and semantic questions.",
    ),
    (
        "Evaluation Results",
        f"BM25 nDCG@10: {METRICS['bm25']['nDCG@10']:.3f} | MiniLM nDCG@10: {METRICS['dense_minilm']['nDCG@10']:.3f} | Hybrid nDCG@10: {METRICS['hybrid_rrf']['nDCG@10']:.3f}",
        "The evaluation file contains ten paraphrased questions with relevant document IDs. In the local run, all three methods retrieved every target document in the top ten.",
    ),
    (
        "RAG With Citations",
        "Top passages are formatted as sources and the answer includes inline IDs such as [datascience:51065].",
        "The RAG layer can use HuggingFace FLAN T5, OpenAI when an API key is configured, or a local extractive fallback for quick no key demos.",
    ),
    (
        "Gradio UI",
        "The app lets users ask a question, choose BM25 or Hybrid RRF, select citation count, and inspect retrieved sources.",
        "The Gradio interface is the live testing surface. It returns an answer panel and a table containing rank, document ID, title, score, and URL.",
    ),
    (
        "Deliverables",
        "Notebook, code, evaluation set, citation-quality notes, report, Gradio app, README, and this video walkthrough.",
        "The submitted repository contains the Colab notebook, reusable Python package, app, metrics script, report, citation quality document, team log, and this demo video.",
    ),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, font_obj, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines():
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if draw.textbbox((0, 0), trial, font=font_obj)[2] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_slide(index: int, title: str, body: str) -> Path:
    image = Image.new("RGB", (1280, 720), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = font(54)
    body_font = font(32)
    small_font = font(22)

    draw.rectangle((0, 0, 1280, 96), fill="#0f172a")
    draw.text((56, 28), "StackLite Hybrid QA", fill="#ffffff", font=small_font)
    draw.text((1130, 28), f"{index + 1}/{len(SLIDES)}", fill="#cbd5e1", font=small_font)

    draw.text((72, 150), title, fill="#111827", font=title_font)
    y = 260
    for line in wrap(draw, body, body_font, 1040):
        draw.text((72, y), line, fill="#1f2937", font=body_font)
        y += 48

    draw.rectangle((72, 610, 1208, 614), fill="#2563eb")
    draw.text((72, 636), "BM25 + MiniLM + RRF + cited RAG + Gradio", fill="#475569", font=small_font)

    path = SLIDE_DIR / f"slide_{index:02d}.png"
    image.save(path)
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SLIDE_DIR.mkdir(exist_ok=True)

    narration = " ".join(textwrap.dedent(script).strip() for _, _, script in SLIDES)
    subprocess.run(["say", "-o", str(VOICE), narration], check=True)

    per_slide = 24.0

    concat_file = OUT_DIR / "segments.txt"
    entries = []
    for idx, (title, body, _) in enumerate(SLIDES):
        path = draw_slide(idx, title, body)
        segment = OUT_DIR / f"segment_{idx:02d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(path),
                "-t",
                f"{per_slide:.3f}",
                "-vf",
                "fps=25,format=yuv420p",
                "-c:v",
                "libx264",
                str(segment),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        entries.append(f"file '{segment}'\n")
    concat_file.write_text("".join(entries), encoding="utf-8")

    silent_video = OUT_DIR / "stacklite_hybrid_qa_demo_silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(silent_video),
        ],
        check=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(VOICE),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(VIDEO),
        ],
        check=True,
    )
    print(VIDEO)


if __name__ == "__main__":
    main()
