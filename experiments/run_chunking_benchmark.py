"""
Aegis Chunking Benchmark — Poster Experiment
=============================================
Compares Fixed-Token chunking vs Recursive Text Character Splitting (RCTS)
on your actual Aegis PDF documents.

Run from project root:
    pip install pymupdf sentence-transformers matplotlib rank-bm25 numpy
    python experiments/run_chunking_benchmark.py

Outputs:
    experiments/results/chunking_comparison.png  — accuracy bar chart
    experiments/results/chunking_metrics.json    — raw numbers
"""

import json
import time
import sys
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "app" / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. GROUND TRUTH — Questions answerable from the PDF documents
# ═══════════════════════════════════════════════════════════════

PDF_QUERIES = [
    {
        "query": "What are the risk categories defined in the EU AI Act?",
        "answer_keywords": ["unacceptable", "high-risk", "limited", "minimal"],
        "source_pdf": "EU AI Act",
    },
    {
        "query": "What penalties does the EU AI Act impose for violations?",
        "answer_keywords": ["fine", "million", "turnover", "35"],
        "source_pdf": "EU AI Act",
    },
    {
        "query": "What does the Nigeria AI bill say about data protection?",
        "answer_keywords": ["data", "protection", "privacy", "personal"],
        "source_pdf": "Nigeria",
    },
    {
        "query": "How does China regulate generative AI content?",
        "answer_keywords": ["content", "generat", "label", "regulat"],
        "source_pdf": "China",
    },
    {
        "query": "What is the US approach to AI regulation compared to the EU?",
        "answer_keywords": ["innovat", "sector", "voluntary", "executive"],
        "source_pdf": "Americas",
    },
    {
        "query": "What are the transparency requirements for AI systems?",
        "answer_keywords": ["transparen", "disclos", "inform", "user"],
        "source_pdf": "EU AI Act",
    },
    {
        "query": "How are AI regulatory sandboxes implemented?",
        "answer_keywords": ["sandbox", "test", "controlled", "innovat"],
        "source_pdf": "EU AI Act",
    },
    {
        "query": "What obligations do AI providers have for high-risk systems?",
        "answer_keywords": ["provider", "risk", "assessment", "conformity"],
        "source_pdf": "EU AI Act",
    },
    {
        "query": "How does China's AI law address algorithmic recommendation?",
        "answer_keywords": ["algorithm", "recommend", "user", "internet"],
        "source_pdf": "China",
    },
    {
        "query": "What global jurisdictions have comprehensive AI regulation?",
        "answer_keywords": ["EU", "China", "Brazil", "Canada"],
        "source_pdf": "Global",
    },
]


# ═══════════════════════════════════════════════════════════════
# 2. PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print("ERROR: pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


# ═══════════════════════════════════════════════════════════════
# 3. CHUNKING STRATEGIES
# ═══════════════════════════════════════════════════════════════

def fixed_token_chunk(text: str, chunk_size: int = 512, overlap: int = 0) -> List[Dict]:
    """Naive fixed-token chunking — the baseline."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i:i + chunk_size])
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "method": "fixed_token",
                "size": len(chunk_text.split()),
            })
    return chunks


def rcts_chunk(text: str, chunk_size: int = 512, overlap: int = 76) -> List[Dict]:
    """
    Recursive Text Character Splitter (RCTS) — structure-aware.
    Splits on: sections → paragraphs → sentences → words.
    Preserves document structure by preferring natural boundaries.
    """
    separators = [
        r"\n#{1,3}\s",         # Markdown headers
        r"\nArticle\s+\d+",    # Legal articles
        r"\nSection\s+\d+",    # Legal sections
        r"\nChapter\s+\d+",    # Chapters
        r"\n\n",               # Double newline (paragraphs)
        r"\n",                 # Single newline
        r"\.\s",               # Sentence boundary
        r"\s",                 # Word boundary (fallback)
    ]

    def _split_recursive(text: str, separators: List[str], chunk_size: int) -> List[str]:
        if len(text.split()) <= chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for sep in separators:
            parts = re.split(sep, text)
            if len(parts) > 1:
                chunks = []
                current = ""
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len((current + " " + part).split()) <= chunk_size:
                        current = (current + " " + part).strip()
                    else:
                        if current:
                            chunks.append(current)
                        if len(part.split()) > chunk_size:
                            # Recurse with next separator
                            remaining_seps = separators[separators.index(sep) + 1:]
                            if remaining_seps:
                                chunks.extend(_split_recursive(part, remaining_seps, chunk_size))
                            else:
                                # Last resort: force split
                                words = part.split()
                                for i in range(0, len(words), chunk_size):
                                    chunks.append(" ".join(words[i:i + chunk_size]))
                        else:
                            current = part
                if current:
                    chunks.append(current)
                if chunks:
                    return chunks
        # Fallback
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    raw_chunks = _split_recursive(text, separators, chunk_size)

    # Add overlap
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if overlap > 0 and i > 0:
            prev_words = raw_chunks[i - 1].split()[-overlap:]
            chunk_text = " ".join(prev_words) + " " + chunk_text

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "index": len(chunks),
                "method": "rcts",
                "size": len(chunk_text.split()),
            })

    return chunks


# ═══════════════════════════════════════════════════════════════
# 4. RETRIEVAL + EVALUATION
# ═══════════════════════════════════════════════════════════════

def keyword_hit_score(retrieved_text: str, keywords: List[str]) -> float:
    """Score how many answer keywords appear in retrieved text."""
    text_lower = retrieved_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords) if keywords else 0.0


def evaluate_chunking(chunks: List[Dict], queries: List[Dict], model=None, top_k: int = 3):
    """Evaluate retrieval quality on chunks using BM25 + optional dense."""
    from rank_bm25 import BM25Okapi

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    results = []
    for qdata in queries:
        query = qdata["query"]
        keywords = qdata["answer_keywords"]

        # BM25 retrieval
        scores = bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_texts = [chunks[i]["text"] for i in top_indices if scores[i] > 0]
        combined = " ".join(retrieved_texts)

        score = keyword_hit_score(combined, keywords)
        results.append({
            "query": query[:60],
            "keyword_coverage": score,
            "n_chunks_retrieved": len(retrieved_texts),
            "avg_chunk_size": np.mean([chunks[i]["size"] for i in top_indices]) if len(top_indices) > 0 else 0,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def run_benchmark():
    print("=" * 60)
    print("AEGIS CHUNKING BENCHMARK")
    print("=" * 60)

    # Extract all PDFs
    print("\n[1/3] Extracting PDF documents...")
    all_text = ""
    pdf_count = 0
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        text = extract_pdf_text(str(pdf_file))
        all_text += text + "\n\n"
        word_count = len(text.split())
        print(f"  ✓ {pdf_file.name[:60]:60s} — {word_count:,} words")
        pdf_count += 1

    total_words = len(all_text.split())
    print(f"\n  Total: {pdf_count} PDFs, {total_words:,} words")

    # Chunk with both methods
    print("\n[2/3] Chunking documents...")

    strategies = {}
    for chunk_size in [256, 512]:
        fixed = fixed_token_chunk(all_text, chunk_size=chunk_size, overlap=0)
        rcts = rcts_chunk(all_text, chunk_size=chunk_size, overlap=int(chunk_size * 0.15))

        strategies[f"Fixed-{chunk_size}"] = fixed
        strategies[f"RCTS-{chunk_size}"] = rcts

        print(f"  Fixed-{chunk_size}: {len(fixed):4d} chunks (avg {np.mean([c['size'] for c in fixed]):.0f} words)")
        print(f"  RCTS-{chunk_size}:  {len(rcts):4d} chunks (avg {np.mean([c['size'] for c in rcts]):.0f} words)")

    # Evaluate
    print("\n[3/3] Evaluating retrieval quality...")
    all_metrics = {}

    for name, chunks in strategies.items():
        t0 = time.perf_counter()
        results = evaluate_chunking(chunks, PDF_QUERIES, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000

        avg_coverage = np.mean([r["keyword_coverage"] for r in results])
        perfect_hits = sum(1 for r in results if r["keyword_coverage"] >= 0.75)

        all_metrics[name] = {
            "n_chunks": len(chunks),
            "avg_keyword_coverage": avg_coverage,
            "perfect_retrieval_rate": perfect_hits / len(results),
            "latency_ms": elapsed,
            "per_query": results,
        }

        print(f"  {name:12s} | Coverage={avg_coverage:.3f} | Perfect@75%={perfect_hits}/{len(results)} | {elapsed:.1f}ms")

    # Save
    with open(RESULTS_DIR / "chunking_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  ✓ Saved to {RESULTS_DIR / 'chunking_metrics.json'}")

    # Charts
    print("\n  Generating charts...")
    generate_charts(all_metrics)

    print("\n" + "=" * 60)
    print("DONE — Charts saved to experiments/results/")
    print("=" * 60)


def generate_charts(all_metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  ✗ matplotlib not installed")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ═══ CHART 1: Chunking comparison — coverage ═══
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by chunk size
    for idx, size in enumerate([256, 512]):
        ax = axes[idx]
        fixed_key = f"Fixed-{size}"
        rcts_key = f"RCTS-{size}"

        if fixed_key in all_metrics and rcts_key in all_metrics:
            methods = ["Fixed-Token", "RCTS (Aegis)"]
            coverages = [
                all_metrics[fixed_key]["avg_keyword_coverage"],
                all_metrics[rcts_key]["avg_keyword_coverage"],
            ]
            colors = ["#EF4444", "#10B981"]

            bars = ax.bar(methods, coverages, color=colors, width=0.5,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, coverages):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=13, fontweight="bold")

            ax.set_title(f"Chunk Size = {size} tokens", fontsize=12, fontweight="bold")
            ax.set_ylabel("Keyword Coverage" if idx == 0 else "")
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            # Add improvement annotation
            if coverages[0] > 0:
                improvement = ((coverages[1] - coverages[0]) / coverages[0]) * 100
                if improvement > 0:
                    ax.annotate(
                        f"+{improvement:.0f}%",
                        xy=(1, coverages[1]),
                        xytext=(1.3, coverages[1] - 0.05),
                        fontsize=12, fontweight="bold", color="#10B981",
                        arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.5),
                    )

    fig.suptitle("Chunking Strategy: Keyword Coverage on Policy PDFs",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "chunking_comparison.png", dpi=200, bbox_inches="tight")
    print(f"  ✓ chunking_comparison.png")
    plt.close(fig)

    # ═══ CHART 2: Per-query breakdown ═══
    fig, ax = plt.subplots(figsize=(10, 6))

    queries_short = [r["query"][:40] + "..." for r in all_metrics.get("RCTS-512", all_metrics[list(all_metrics.keys())[0]])["per_query"]]
    fixed_scores = [r["keyword_coverage"] for r in all_metrics.get("Fixed-512", {}).get("per_query", [])]
    rcts_scores = [r["keyword_coverage"] for r in all_metrics.get("RCTS-512", {}).get("per_query", [])]

    if fixed_scores and rcts_scores:
        x = np.arange(len(queries_short))
        width = 0.35

        ax.barh(x - width / 2, fixed_scores, width, label="Fixed-Token", color="#EF4444", alpha=0.85)
        ax.barh(x + width / 2, rcts_scores, width, label="RCTS (Aegis)", color="#10B981", alpha=0.85)

        ax.set_yticks(x)
        ax.set_yticklabels(queries_short, fontsize=9)
        ax.set_xlabel("Keyword Coverage", fontsize=11)
        ax.set_title("Per-Query Chunking Performance (512 tokens)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim(0, 1.1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "chunking_per_query.png", dpi=200, bbox_inches="tight")
    print(f"  ✓ chunking_per_query.png")
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()
