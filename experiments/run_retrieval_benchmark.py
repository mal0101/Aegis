"""
Aegis Retrieval Benchmark — BM25 Evaluation
============================================
Tests BM25 retrieval on your actual Aegis corpus (10 concepts + 8 case studies + PDF chunks).
No heavy ML dependencies needed — just rank-bm25, numpy, matplotlib.

Run from project root:
    pip install rank-bm25 numpy matplotlib pymupdf
    python experiments/run_retrieval_benchmark.py

Outputs → experiments/results/
"""

import json
import re
import time
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# ─── Paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "app" / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. GROUND TRUTH — 20 queries with expected relevant doc IDs
# ═══════════════════════════════════════════════════════════════

CONCEPT_QUERIES = [
    {"query": "What is algorithmic bias in AI systems?",
     "relevant": ["algorithmic-bias"], "type": "exact"},
    {"query": "How can AI explain its decisions to users?",
     "relevant": ["explainable-ai", "transparency"], "type": "paraphrase"},
    {"query": "Which AI applications are considered dangerous or high-risk?",
     "relevant": ["high-risk-ai"], "type": "paraphrase"},
    {"query": "How to audit AI systems for compliance and fairness",
     "relevant": ["ai-auditing", "fairness-in-ai"], "type": "cross-cutting"},
    {"query": "Training datasets and their quality for machine learning",
     "relevant": ["training-data"], "type": "exact"},
    {"query": "Government frameworks to oversee artificial intelligence",
     "relevant": ["ai-governance", "transparency"], "type": "paraphrase"},
    {"query": "Safe testing environments for new AI regulations",
     "relevant": ["regulatory-sandbox"], "type": "paraphrase"},
    {"query": "Managing and protecting data used by AI in Morocco",
     "relevant": ["data-governance", "training-data"], "type": "morocco-specific"},
    {"query": "Ensuring equal treatment of all groups by AI",
     "relevant": ["fairness-in-ai", "algorithmic-bias"], "type": "cross-cutting"},
    {"query": "Making AI systems open and publicly accountable",
     "relevant": ["transparency", "explainable-ai"], "type": "cross-cutting"},
]

CASE_STUDY_QUERIES = [
    {"query": "Comprehensive AI regulation with risk-based classification tiers",
     "relevant": ["eu_ai_act_2024"], "type": "exact"},
    {"query": "African country national AI policy for economic development",
     "relevant": ["rwanda_ai_policy_2023", "tunisia_ai_strategy_2023"], "type": "regional"},
    {"query": "Flexible voluntary AI governance framework not binding law",
     "relevant": ["singapore_ai_framework_2023", "uk_ai_framework_2024"], "type": "paraphrase"},
    {"query": "Data protection and artificial intelligence law in developing country",
     "relevant": ["brazil_ai_bill_2024", "rwanda_ai_policy_2023"], "type": "cross-cutting"},
    {"query": "Innovation-friendly approach to regulating AI technology",
     "relevant": ["uk_ai_framework_2024", "singapore_ai_framework_2023"], "type": "paraphrase"},
    {"query": "Recent AI legislation passed in East Asia",
     "relevant": ["south_korea_ai_act_2024"], "type": "regional"},
    {"query": "Impact of AI regulation on small businesses and startups",
     "relevant": ["eu_ai_act_2024", "brazil_ai_bill_2024"], "type": "cross-cutting"},
    {"query": "National AI strategy for small economy with limited resources",
     "relevant": ["rwanda_ai_policy_2023", "tunisia_ai_strategy_2023"], "type": "paraphrase"},
    {"query": "Consumer rights protection through artificial intelligence rules",
     "relevant": ["canada_aia_2023", "south_korea_ai_act_2024"], "type": "cross-cutting"},
    {"query": "Risk assessment based approach to AI governance similar to EU",
     "relevant": ["eu_ai_act_2024", "brazil_ai_bill_2024"], "type": "paraphrase"},
]

PDF_QUERIES = [
    {"query": "What are the risk categories defined in the EU AI Act?",
     "answer_keywords": ["unacceptable", "high-risk", "limited", "minimal"],
     "type": "exact"},
    {"query": "What penalties does the EU AI Act impose for violations?",
     "answer_keywords": ["fine", "million", "turnover", "35"],
     "type": "exact"},
    {"query": "What does the Nigeria AI bill say about data protection?",
     "answer_keywords": ["data", "protection", "privacy", "personal"],
     "type": "cross-cutting"},
    {"query": "How does China regulate generative AI content?",
     "answer_keywords": ["content", "generat", "label", "regulat"],
     "type": "paraphrase"},
    {"query": "What is the US approach to AI regulation compared to the EU?",
     "answer_keywords": ["innovat", "sector", "voluntary", "executive"],
     "type": "cross-cutting"},
    {"query": "What are the transparency requirements for AI systems?",
     "answer_keywords": ["transparen", "disclos", "inform", "user"],
     "type": "paraphrase"},
    {"query": "How are AI regulatory sandboxes implemented?",
     "answer_keywords": ["sandbox", "test", "controlled", "innovat"],
     "type": "paraphrase"},
    {"query": "What obligations do AI providers have for high-risk systems?",
     "answer_keywords": ["provider", "risk", "assessment", "conformity"],
     "type": "exact"},
    {"query": "How does China's AI law address algorithmic recommendation?",
     "answer_keywords": ["algorithm", "recommend", "user", "internet"],
     "type": "paraphrase"},
    {"query": "What global jurisdictions have comprehensive AI regulation?",
     "answer_keywords": ["EU", "China", "Brazil", "Canada"],
     "type": "cross-cutting"},
]


# ═══════════════════════════════════════════════════════════════
# 2. DATA LOADING — mirrors your search_service.py exactly
# ═══════════════════════════════════════════════════════════════

from rank_bm25 import BM25Okapi


def load_concepts() -> List[Dict]:
    with open(DATA_DIR / "concepts.json") as f:
        concepts = json.load(f)
    docs = []
    for c in concepts:
        searchable = c.get(
            "searchable_text",
            f"{c['term']} {c['definition']} {c.get('simple_explanation', '')} "
            f"{' '.join(c.get('examples', []))} {c.get('morocco_context', '')}"
        )
        docs.append({"id": c["id"], "text": searchable})
    return docs


def load_case_studies() -> List[Dict]:
    with open(DATA_DIR / "case_studies.json") as f:
        case_studies = json.load(f)
    docs = []
    for cs in case_studies:
        policy = cs["policy"]
        outcomes = cs.get("outcomes", {})
        searchable = (
            f"{policy['name']} -- {cs['country']}\n"
            f"{policy['description']}\n"
            f"Key provisions: {'; '.join(policy.get('key_provisions', []))}\n"
            f"Insights: {'; '.join(outcomes.get('qualitative_insights', []))}"
        )
        docs.append({"id": cs["id"], "text": searchable})
    return docs


def load_pdf_chunks(chunk_size: int = 512) -> List[Dict]:
    """Extract and chunk PDFs using RCTS (structure-aware splitting)."""
    try:
        import fitz
    except ImportError:
        print("  ⚠ pymupdf not installed — skipping PDF corpus")
        return []

    separators = [
        r"\nArticle\s+\d+", r"\nSection\s+\d+", r"\nChapter\s+\d+",
        r"\n\n", r"\n", r"\.\s", r"\s",
    ]

    def rcts_split(text, seps, size):
        if len(text.split()) <= size:
            return [text] if text.strip() else []
        for sep in seps:
            parts = re.split(sep, text)
            if len(parts) <= 1:
                continue
            chunks, current = [], ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len((current + " " + part).split()) <= size:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        chunks.append(current)
                    if len(part.split()) > size:
                        remaining = seps[seps.index(sep) + 1:] if sep in seps else []
                        chunks.extend(rcts_split(part, remaining, size) if remaining
                                      else [" ".join(part.split()[i:i+size])
                                            for i in range(0, len(part.split()), size)])
                    else:
                        current = part
            if current:
                chunks.append(current)
            if chunks:
                return chunks
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

    manifest = json.load(open(DATA_DIR / "pdf_manifest.json"))
    all_chunks = []

    for doc_meta in manifest["documents"]:
        pdf_path = PDF_DIR / doc_meta["filename"]
        if not pdf_path.exists():
            continue
        doc = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        raw_chunks = rcts_split(text, separators, chunk_size)
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                all_chunks.append({
                    "id": f"{doc_meta['country']}_{i:04d}",
                    "text": chunk_text.strip(),
                    "source": doc_meta["policy_name"],
                    "country": doc_meta["country"],
                })

    return all_chunks


# ═══════════════════════════════════════════════════════════════
# 3. BM25 SEARCH — identical to your production search_service
# ═══════════════════════════════════════════════════════════════

def bm25_search(corpus: List[Dict], query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i]["id"], float(scores[i])) for i in top_indices if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / k if top else 0.0

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / len(relevant) if relevant else 0.0

def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def keyword_coverage(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower) / len(keywords) if keywords else 0.0


# ═══════════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════

def evaluate_collection(name: str, corpus: List[Dict], queries: List[Dict], top_k: int = 3, is_pdf: bool = False):
    """Run BM25 evaluation on one collection."""
    per_query = []

    for qdata in queries:
        query = qdata["query"]
        t0 = time.perf_counter()
        results = bm25_search(corpus, query, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        retrieved_ids = [r[0] for r in results]
        retrieved_scores = [r[1] for r in results]

        if is_pdf:
            tokenized = [doc["text"].lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            top_idx = np.argsort(scores)[::-1][:top_k]
            combined_text = " ".join(corpus[i]["text"] for i in top_idx if scores[i] > 0)

            kw_cov = keyword_coverage(combined_text, qdata["answer_keywords"])
            per_query.append({
                "query": query,
                "type": qdata["type"],
                "keyword_coverage": kw_cov,
                "hit": int(kw_cov >= 0.5),
                "latency_ms": latency_ms,
                "top_scores": retrieved_scores[:3],
            })
        else:
            relevant = qdata["relevant"]
            p = precision_at_k(retrieved_ids, relevant, top_k)
            r = recall_at_k(retrieved_ids, relevant, top_k)
            m = mrr(retrieved_ids, relevant)
            n = ndcg_at_k(retrieved_ids, relevant, top_k)
            hit = int(any(rid in relevant for rid in retrieved_ids))

            per_query.append({
                "query": query,
                "type": qdata["type"],
                "relevant": relevant,
                "retrieved": retrieved_ids,
                "precision": p,
                "recall": r,
                "mrr": m,
                "ndcg": n,
                "hit": hit,
                "latency_ms": latency_ms,
                "top_scores": retrieved_scores[:3],
            })

    if is_pdf:
        return {
            "collection": name,
            "n_docs": len(corpus),
            "n_queries": len(queries),
            "avg_keyword_coverage": np.mean([pq["keyword_coverage"] for pq in per_query]),
            "hit_rate": np.mean([pq["hit"] for pq in per_query]),
            "avg_latency_ms": np.mean([pq["latency_ms"] for pq in per_query]),
            "per_query": per_query,
        }
    else:
        return {
            "collection": name,
            "n_docs": len(corpus),
            "n_queries": len(queries),
            "precision_at_3": np.mean([pq["precision"] for pq in per_query]),
            "recall_at_3": np.mean([pq["recall"] for pq in per_query]),
            "mrr": np.mean([pq["mrr"] for pq in per_query]),
            "ndcg_at_3": np.mean([pq["ndcg"] for pq in per_query]),
            "hit_rate": np.mean([pq["hit"] for pq in per_query]),
            "avg_latency_ms": np.mean([pq["latency_ms"] for pq in per_query]),
            "per_query": per_query,
        }


def run_benchmark():
    TOP_K = 3

    print("=" * 60)
    print("  AEGIS RETRIEVAL BENCHMARK — BM25 Evaluation")
    print("=" * 60)

    # Load corpora
    print("\n[1/3] Loading Aegis data...")
    concepts = load_concepts()
    case_studies = load_case_studies()
    pdf_chunks = load_pdf_chunks(chunk_size=512)
    print(f"  Concepts:     {len(concepts)} documents")
    print(f"  Case Studies: {len(case_studies)} documents")
    print(f"  PDF Chunks:   {len(pdf_chunks)} chunks from {len(set(c['country'] for c in pdf_chunks))} sources")

    # Evaluate
    print(f"\n[2/3] Running BM25 evaluation (top_k={TOP_K})...")

    results = {}

    r = evaluate_collection("concepts", concepts, CONCEPT_QUERIES, TOP_K)
    results["concepts"] = r
    print(f"\n  CONCEPTS ({r['n_docs']} docs, {r['n_queries']} queries)")
    print(f"    Precision@3 = {r['precision_at_3']:.3f}")
    print(f"    Recall@3    = {r['recall_at_3']:.3f}")
    print(f"    MRR         = {r['mrr']:.3f}")
    print(f"    NDCG@3      = {r['ndcg_at_3']:.3f}")
    print(f"    Hit Rate    = {r['hit_rate']:.1%}")
    print(f"    Avg Latency = {r['avg_latency_ms']:.2f} ms")

    r = evaluate_collection("case_studies", case_studies, CASE_STUDY_QUERIES, TOP_K)
    results["case_studies"] = r
    print(f"\n  CASE STUDIES ({r['n_docs']} docs, {r['n_queries']} queries)")
    print(f"    Precision@3 = {r['precision_at_3']:.3f}")
    print(f"    Recall@3    = {r['recall_at_3']:.3f}")
    print(f"    MRR         = {r['mrr']:.3f}")
    print(f"    NDCG@3      = {r['ndcg_at_3']:.3f}")
    print(f"    Hit Rate    = {r['hit_rate']:.1%}")
    print(f"    Avg Latency = {r['avg_latency_ms']:.2f} ms")

    if pdf_chunks:
        r = evaluate_collection("pdf_chunks", pdf_chunks, PDF_QUERIES, TOP_K, is_pdf=True)
        results["pdf_chunks"] = r
        print(f"\n  PDF CHUNKS ({r['n_docs']} chunks, {r['n_queries']} queries)")
        print(f"    Keyword Cov = {r['avg_keyword_coverage']:.3f}")
        print(f"    Hit Rate    = {r['hit_rate']:.1%}")
        print(f"    Avg Latency = {r['avg_latency_ms']:.2f} ms")

    # Aggregate
    agg = {}
    struct = [results["concepts"], results["case_studies"]]
    agg["Precision@3"] = np.mean([r["precision_at_3"] for r in struct])
    agg["Recall@3"] = np.mean([r["recall_at_3"] for r in struct])
    agg["MRR"] = np.mean([r["mrr"] for r in struct])
    agg["NDCG@3"] = np.mean([r["ndcg_at_3"] for r in struct])
    agg["Hit Rate"] = np.mean([r["hit_rate"] for r in struct])
    agg["Avg Latency (ms)"] = np.mean([r["avg_latency_ms"] for r in struct])

    if "pdf_chunks" in results:
        agg["PDF Keyword Coverage"] = results["pdf_chunks"]["avg_keyword_coverage"]
        agg["PDF Hit Rate"] = results["pdf_chunks"]["hit_rate"]

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE (structured collections)")
    print(f"{'=' * 60}")
    for k, v in agg.items():
        if "ms" in k:
            print(f"    {k:25s}: {v:.2f}")
        else:
            print(f"    {k:25s}: {v:.1%}")

    # By query type
    type_metrics = {}
    for coll_name in ["concepts", "case_studies"]:
        for pq in results[coll_name]["per_query"]:
            qt = pq["type"]
            if qt not in type_metrics:
                type_metrics[qt] = {"mrr": [], "hit": [], "precision": [], "recall": []}
            type_metrics[qt]["mrr"].append(pq["mrr"])
            type_metrics[qt]["hit"].append(pq["hit"])
            type_metrics[qt]["precision"].append(pq["precision"])
            type_metrics[qt]["recall"].append(pq["recall"])

    print(f"\n  BY QUERY TYPE:")
    for qt, m in sorted(type_metrics.items()):
        n = len(m["mrr"])
        print(f"    {qt:20s} (n={n:2d}) | MRR={np.mean(m['mrr']):.3f} | "
              f"P@3={np.mean(m['precision']):.3f} | R@3={np.mean(m['recall']):.3f} | "
              f"Hit={np.mean(m['hit']):.1%}")

    # Save
    output = {
        "method": "BM25 (rank_bm25.BM25Okapi)",
        "top_k": TOP_K,
        "corpus": {
            "concepts": len(concepts),
            "case_studies": len(case_studies),
            "pdf_chunks": len(pdf_chunks),
        },
        "aggregate": {k: round(v, 4) for k, v in agg.items()},
        "by_query_type": {qt: {k: round(np.mean(v), 4) for k, v in m.items()}
                         for qt, m in type_metrics.items()},
        "collections": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"}
                        for k, v in results.items()},
        "per_query": {k: v["per_query"] for k, v in results.items()},
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: round(float(x), 4) if isinstance(x, (np.floating, float)) else x)
    print(f"\n  ✓ Raw metrics → {RESULTS_DIR / 'metrics.json'}")

    # Charts
    print(f"\n[3/3] Generating poster charts...")
    generate_charts(results, agg, type_metrics)
    print(f"\n{'=' * 60}")
    print(f"  DONE — All outputs in experiments/results/")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════
# 6. CHARTS
# ═══════════════════════════════════════════════════════════════

def generate_charts(results, aggregate, type_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    C_PRIMARY = "#0EA5E9"
    C_SECONDARY = "#8B5CF6"
    C_ACCENT = "#10B981"
    C_GRAY = "#94A3B8"

    # ═══ CHART 1: Collection comparison ═══
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metrics = ["Precision@3", "Recall@3", "MRR", "NDCG@3", "Hit Rate"]
    collections = ["concepts", "case_studies"]
    colors = [C_PRIMARY, C_SECONDARY]
    labels = ["Concepts (10 docs)", "Case Studies (8 docs)"]
    bar_w = 0.3
    x = np.arange(len(metrics))

    for i, (coll, color, label) in enumerate(zip(collections, colors, labels)):
        r = results[coll]
        vals = [r["precision_at_3"], r["recall_at_3"], r["mrr"], r["ndcg_at_3"], r["hit_rate"]]
        offset = (i - 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=label, color=color,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    agg_vals = [aggregate[m] for m in metrics]
    ax.plot(x, agg_vals, "D--", color=C_ACCENT, markersize=7, linewidth=1.5,
            label="Aggregate", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_title("Aegis BM25 Retrieval — Performance by Collection",
                 fontsize=14, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "retrieval_by_collection.png", dpi=200, bbox_inches="tight")
    print(f"  ✓ retrieval_by_collection.png")
    plt.close(fig)

    # ═══ CHART 2: Query type breakdown ═══
    fig, ax = plt.subplots(figsize=(9, 5))

    types_ordered = ["exact", "paraphrase", "cross-cutting", "morocco-specific", "regional"]
    valid = [t for t in types_ordered if t in type_metrics]

    x = np.arange(len(valid))
    metric_names = ["MRR", "Precision", "Recall"]
    metric_keys = ["mrr", "precision", "recall"]
    bar_w = 0.22

    for j, (mname, mkey) in enumerate(zip(metric_names, metric_keys)):
        vals = [np.mean(type_metrics[t][mkey]) for t in valid]
        offset = (j - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=mname, alpha=0.85,
                      color=[C_PRIMARY, C_SECONDARY, C_ACCENT][j],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    type_labels = [t.replace("-", " ").title() for t in valid]
    counts = [len(type_metrics[t]["mrr"]) for t in valid]
    type_labels = [f"{l}\n(n={c})" for l, c in zip(type_labels, counts)]

    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_title("BM25 Retrieval — Performance by Query Type",
                 fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "retrieval_by_query_type.png", dpi=200, bbox_inches="tight")
    print(f"  ✓ retrieval_by_query_type.png")
    plt.close(fig)

    # ═══ CHART 3: Per-query hit/miss ═══
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1]})

    type_colors_map = {"exact": "#10B981", "paraphrase": "#F59E0B",
                       "cross-cutting": "#EF4444", "morocco-specific": "#8B5CF6",
                       "regional": "#0EA5E9"}

    for idx, coll in enumerate(["concepts", "case_studies"]):
        ax = axes[idx]
        pqs = results[coll]["per_query"]
        queries = [pq["query"][:45] + "..." for pq in pqs]
        mrrs = [pq["mrr"] for pq in pqs]
        types = [pq["type"] for pq in pqs]
        colors = [type_colors_map.get(t, C_GRAY) for t in types]

        y = np.arange(len(queries))
        bars = ax.barh(y, mrrs, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

        for bar, val, typ in zip(bars, mrrs, types):
            label = f"{val:.0%}" if val > 0 else "MISS"
            color = "white" if val >= 0.5 else "#EF4444"
            x_pos = max(val - 0.02, 0.02) if val >= 0.5 else val + 0.02
            ha = "right" if val >= 0.5 else "left"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha=ha, fontsize=8.5, fontweight="bold", color=color)

        ax.set_yticks(y)
        ax.set_yticklabels(queries, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("MRR", fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"{coll.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.2, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace("-", " ").title())
                       for t, c in type_colors_map.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Query BM25 Retrieval Performance", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "retrieval_per_query.png", dpi=200, bbox_inches="tight")
    print(f"  ✓ retrieval_per_query.png")
    plt.close(fig)

    # ═══ CHART 4: PDF chunk retrieval ═══
    if "pdf_chunks" in results:
        fig, ax = plt.subplots(figsize=(9, 5))

        pqs = results["pdf_chunks"]["per_query"]
        queries = [pq["query"][:45] + "..." for pq in pqs]
        coverages = [pq["keyword_coverage"] for pq in pqs]

        y = np.arange(len(queries))
        colors = [C_ACCENT if c >= 0.75 else C_PRIMARY if c >= 0.5 else "#EF4444" for c in coverages]

        bars = ax.barh(y, coverages, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
        for bar, val in zip(bars, coverages):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(queries, fontsize=8.5)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Keyword Coverage", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.set_title(f"PDF Retrieval — BM25 on {results['pdf_chunks']['n_docs']} RCTS Chunks",
                     fontsize=13, fontweight="bold", pad=12)

        avg = np.mean(coverages)
        ax.axvline(avg, color=C_GRAY, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(avg + 0.02, len(queries) - 0.5, f"avg = {avg:.0%}",
                fontsize=9, color=C_GRAY, fontstyle="italic")

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "pdf_retrieval.png", dpi=200, bbox_inches="tight")
        print(f"  ✓ pdf_retrieval.png")
        plt.close(fig)


if __name__ == "__main__":
    run_benchmark()
