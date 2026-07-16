"""
evaluate.py — Evaluation, Comparative Study & Experiment Logging.

Course: Deep Learning for Search
Lecture refs: L01 (Search Quality), L08 (LambdaMART evaluation), L10 (Benchmarking)

PDF Requirements:
  1. Search quality metrics: NDCG@k, MRR, Recall@k, Precision@k
  2. Validation set with (query + relevance)
  3. Comparative study of all approaches (Iteration 1 vs Iteration 2)
  4. Hardware requirements profiling
  5. Experiment logging to experiments_log.jsonl

Iteration 1 (Baseline): Individual retrieval systems (BM25, SPLADE, Dense Text, Dense Curve)
Iteration 2 (Full Alliance): RRF fusion + LambdaMART + Agentic CRAG
See pipeline.py for the full Alliance architecture.
"""

import json
import time
import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

_SCRIPT_DIR = Path(__file__).parent
LOG_PATH = _SCRIPT_DIR / "experiments_log.jsonl"
RESULTS_PATH = _SCRIPT_DIR / "evaluation_results.json"


# ================================================================== #
# Experiment Logger
# ================================================================== #

class ExperimentLogger:
    """Logs all experiment steps to experiments_log.jsonl."""

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)

    def log(self, step: str, metrics: dict, notes: str = ""):
        """
        Append a structured JSON line: {timestamp, step, metrics, notes}.

        Args:
            step: name of the experiment step (e.g., "BM25 baseline")
            metrics: dict of metric name -> value
            notes: optional free-text annotation
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics,
            "notes": notes,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> pd.DataFrame:
        """Return all logged experiments as a DataFrame."""
        if not self.log_path.exists():
            return pd.DataFrame()
        rows = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)


# ================================================================== #
# Evaluation Dataset Generation
# ================================================================== #

def generate_eval_dataset(
    df: pd.DataFrame,
    bm25_index=None,
    n_queries: int = 200,
    relevant_k: int = 10,
    random_state: int = 42,
) -> List[dict]:
    """
    Generate evaluation queries with ground-truth relevance.

    For each query:
      1. Select a random document as the "query document" and use its text
         as the search query.
      2. Retrieve top candidates via BM25 (or random sample if no index).
      3. Label top-10 BM25 results as relevant (rel=1) and sample additional
         random non-relevant docs (rel=0).

    This gives us a validation set for NDCG / MRR / Recall computation.

    Args:
        df: clean strategies DataFrame
        bm25_index: BM25Index instance (optional; falls back to random)
        n_queries: number of evaluation queries to generate
        relevant_k: number of docs to label as relevant per query
        random_state: random seed

    Returns:
        List of dicts: {"query_text": str, "relevant_ids": set[int], "query_id": int}
    """
    rng = np.random.RandomState(random_state)
    n_docs = len(df)
    eval_queries = []

    for qi in range(n_queries):
        # Pick a random doc as query
        q_idx = rng.randint(0, n_docs)
        query_text = df.iloc[q_idx]["text"]

        relevant_ids: Set[int] = set()

        if bm25_index is not None:
            # Use BM25 top-K as "relevant" (weak supervision)
            try:
                ids, _ = bm25_index.search(query_text, top_k=relevant_k)
                relevant_ids = set(int(x) for x in ids if x < n_docs)
            except Exception:
                pass

        # Fallback: if BM25 didn't return enough, use nearest neighbors by Sharpe
        if len(relevant_ids) < relevant_k:
            q_sharpe = df.iloc[q_idx]["sharpe"]
            sharpe_diffs = (df["sharpe"] - q_sharpe).abs().values
            nearest = np.argsort(sharpe_diffs)[:relevant_k]
            relevant_ids.update(int(x) for x in nearest if x < n_docs)

        eval_queries.append({
            "query_id": qi,
            "query_text": query_text,
            "relevant_ids": relevant_ids,
        })

    return eval_queries


# ================================================================== #
# Search Quality Metrics (L01)
# ================================================================== #

def ndcg_at_k(ranked_doc_ids: np.ndarray, relevant_ids: set, k: int = 10) -> float:
    """
    NDCG@k (Normalized Discounted Cumulative Gain at k).

    Uses binary relevance: rel(d) in {0, 1}.
    DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)
    IDCG@k = DCG of the ideal ranking (all relevant docs at the top).
    NDCG@k = DCG@k / IDCG@k
    """
    if k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if int(doc_id) in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Ideal DCG: all relevant at top positions
    n_rel = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(n_rel):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(ranked_doc_ids: np.ndarray, relevant_ids: set) -> float:
    """
    Mean Reciprocal Rank (MRR).

    Returns 1/rank of the first relevant document, or 0 if none found.
    """
    for i, doc_id in enumerate(ranked_doc_ids):
        if int(doc_id) in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_doc_ids: np.ndarray, relevant_ids: set, k: int = 10) -> float:
    """
    Recall@k: fraction of relevant docs found in top-k.

    recall@k = |relevant ∩ top-k| / |relevant|
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = set(int(x) for x in ranked_doc_ids[:k])
    found = len(top_k & relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(ranked_doc_ids: np.ndarray, relevant_ids: set, k: int = 10) -> float:
    """
    Precision@k: fraction of top-k that are relevant.

    precision@k = |relevant ∩ top-k| / k
    """
    if k <= 0:
        return 0.0
    top_k = set(int(x) for x in ranked_doc_ids[:k])
    found = len(top_k & relevant_ids)
    return found / k


# ================================================================== #
# Hardware Profiling (L10)
# ================================================================== #

def profile_hardware() -> dict:
    """
    Log hardware requirements.

    Returns dict with:
      - CPU info, GPU (if available)
      - RAM usage (current process)
      - GPU memory usage (if CUDA)
      - Disk usage for project artifacts
    """
    import platform
    import psutil

    info = {}

    # CPU
    info["cpu_model"] = platform.processor() or platform.machine()
    info["cpu_count_physical"] = psutil.cpu_count(logical=False) or "N/A"
    info["cpu_count_logical"] = psutil.cpu_count(logical=True) or "N/A"

    # GPU
    info["gpu"] = "N/A"
    info["gpu_memory_used_mb"] = None
    info["gpu_memory_total_mb"] = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            info["gpu"] = gpu_name
            mem_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
            mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
            info["gpu_memory_used_mb"] = round(mem_used, 1)
            info["gpu_memory_total_mb"] = round(mem_total, 1)
    except ImportError:
        pass

    # RAM
    process = psutil.Process(os.getpid())
    info["ram_used_mb"] = round(process.memory_info().rss / (1024 ** 2), 1)
    info["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 ** 2), 1)

    # Disk usage for key artifacts
    artifacts = [
        "clean_strategies.parquet",
        "bm25_index.pkl",
        "splade_sparse_matrix.npz",
        "faiss_text.index",
        "faiss_curve.index",
        "captain_lgbm.txt",
    ]
    disk = {}
    for artifact in artifacts:
        fpath = _SCRIPT_DIR / artifact
        if fpath.exists():
            disk[artifact] = round(fpath.stat().st_size / (1024 ** 2), 2)
        else:
            disk[artifact] = None
    info["artifacts_disk_mb"] = disk

    return info


# ================================================================== #
# Per-Approach Retrieval Helpers
# ================================================================== #

def _run_bm25_only(query: str, bm25_index, top_k: int = 100) -> np.ndarray:
    """Retrieve using BM25 only."""
    ids, _ = bm25_index.search(query, top_k=top_k)
    return ids


def _run_splade_only(query: str, splade_index, top_k: int = 100) -> np.ndarray:
    """Retrieve using SPLADE only (if available)."""
    if splade_index is None:
        return np.array([], dtype=int)
    ids, _ = splade_index.search(query, top_k=top_k)
    return ids


def _run_dense_text_only(query: str, dense_searcher, top_k: int = 100) -> np.ndarray:
    """Retrieve using Dense Text Bi-encoder only."""
    ids, _ = dense_searcher.search_text(query, top_k=top_k)
    return ids


def _run_dense_curve_only(dense_searcher, top_k: int = 100) -> np.ndarray:
    """Retrieve using Dense Curve encoder only (reference upward trend)."""
    ref_curve = np.linspace(100, 120, 252).astype(np.float32)
    ids, _ = dense_searcher.search_curve(ref_curve, top_k=top_k)
    return ids


def _run_rrf_text(result_lists, k: int = 60, top_k: int = 100) -> np.ndarray:
    """RRF fusion of text-only systems (BM25 + Dense Text, optionally SPLADE)."""
    from pipeline import reciprocal_rank_fusion
    fused_ids, _ = reciprocal_rank_fusion(result_lists, k=k, top_k=top_k)
    return fused_ids


def _run_rrf_all(result_lists, k: int = 60, top_k: int = 100) -> np.ndarray:
    """RRF fusion of all 4 systems (BM25 + SPLADE + Dense Text + Dense Curve)."""
    from pipeline import reciprocal_rank_fusion
    fused_ids, _ = reciprocal_rank_fusion(result_lists, k=k, top_k=top_k)
    return fused_ids


# ================================================================== #
# Comparative Study — THE KEY PDF DELIVERABLE
# ================================================================== #

def comparative_study(
    alliance,
    eval_queries: List[dict],
    logger: ExperimentLogger,
    top_k: int = 100,
    eval_k: int = 10,
) -> pd.DataFrame:
    """
    Run ALL retrieval approaches on the same evaluation queries and compare.

    Iteration 1 (Baseline approaches — individual systems):
      - BM25 only
      - SPLADE only (if available)
      - Dense Text (Bi-encoder) only
      - Dense Curve only
      - BM25 + Dense Text (early fusion via RRF)

    Iteration 2 (Full Alliance — our approach):
      - RRF of all 4 systems (no Captain)
      - RRF + Captain (LambdaMART) reranking
      - RRF + Captain + Agentic CRAG

    For each approach, compute: NDCG@5, NDCG@10, MRR, Recall@10, Precision@5.
    Also measure: avg latency per query.

    Args:
        alliance: AllianceRetriever instance (loaded)
        eval_queries: list of eval query dicts from generate_eval_dataset()
        logger: ExperimentLogger instance
        top_k: number of candidates to retrieve per approach
        eval_k: k for metric computation (e.g., 10)

    Returns:
        DataFrame with all results for the comparison table
    """
    from ltr_captain import extract_features

    # Define approaches
    approaches = {
        "Iter1_BM25": lambda q: _run_bm25_only(q, alliance.bm25, top_k),
    }

    if alliance.splade is not None:
        approaches["Iter1_SPLADE"] = lambda q: _run_splade_only(q, alliance.splade, top_k)

    approaches["Iter1_DenseText"] = lambda q: _run_dense_text_only(q, alliance.dense_searcher, top_k)
    approaches["Iter1_DenseCurve"] = lambda q: _run_dense_curve_only(alliance.dense_searcher, top_k)
    approaches["Iter1_BM25+DenseText_RRF"] = lambda q: _run_rrf_text([
        _run_bm25_only(q, alliance.bm25, top_k),
        _run_dense_text_only(q, alliance.dense_searcher, top_k),
    ], top_k=top_k)

    # Iteration 2 approaches
    def _rrf_all_systems(query):
        rrf_inputs = [
            _run_bm25_only(query, alliance.bm25, top_k),
        ]
        if alliance.splade is not None:
            rrf_inputs.append(_run_splade_only(query, alliance.splade, top_k))
        try:
            rrf_inputs.append(_run_dense_text_only(query, alliance.dense_searcher, top_k))
            rrf_inputs.append(_run_dense_curve_only(alliance.dense_searcher, top_k))
        except Exception:
            pass
        return _run_rrf_all(rrf_inputs, top_k=top_k)

    approaches["Iter2_RRF_All"] = _rrf_all_systems

    def _rrf_captain(query):
        fused_ids = _rrf_all_systems(query)
        if len(fused_ids) == 0:
            return fused_ids
        feats = extract_features(
            query, fused_ids, alliance.df,
            alliance.bm25, alliance.splade, alliance.dense_searcher,
        )
        try:
            reranked = alliance.captain.rerank(feats)
            return reranked["doc_id"].values.astype(int)
        except Exception:
            return fused_ids

    approaches["Iter2_RRF+Captain"] = _rrf_captain

    def _rrf_captain_crag(query):
        fused_ids = _rrf_all_systems(query)
        if len(fused_ids) == 0:
            return fused_ids
        feats = extract_features(
            query, fused_ids, alliance.df,
            alliance.bm25, alliance.splade, alliance.dense_searcher,
        )
        try:
            reranked = alliance.captain.rerank(feats)
            refined = alliance.crag_agent.refine_query(reranked)
            return refined["doc_id"].values.astype(int)
        except Exception:
            return fused_ids

    approaches["Iter2_RRF+Captain+CRAG"] = _rrf_captain_crag

    # ---- Run evaluation ----
    results_rows = []

    for approach_name, retriever_fn in approaches.items():
        print(f"\n[Eval] Running: {approach_name} ...")

        ndcg5_list, ndcg10_list, mrr_list = [], [], []
        recall10_list, prec5_list = [], []
        latencies = []

        for eq in eval_queries:
            query_text = eq["query_text"]
            relevant_ids = eq["relevant_ids"]

            if len(relevant_ids) == 0:
                continue

            t0 = time.time()
            try:
                ranked_ids = retriever_fn(query_text)
            except Exception as e:
                print(f"  [Eval] Error in {approach_name}: {e}")
                continue
            latency = time.time() - t0
            latencies.append(latency)

            ranked_arr = np.array(ranked_ids, dtype=int)

            ndcg5_list.append(ndcg_at_k(ranked_arr, relevant_ids, k=5))
            ndcg10_list.append(ndcg_at_k(ranked_arr, relevant_ids, k=10))
            mrr_list.append(mrr(ranked_arr, relevant_ids))
            recall10_list.append(recall_at_k(ranked_arr, relevant_ids, k=10))
            prec5_list.append(precision_at_k(ranked_arr, relevant_ids, k=5))

        if not ndcg5_list:
            print(f"  [Eval] No valid queries for {approach_name}, skipping.")
            continue

        row = {
            "Approach": approach_name,
            "NDCG@5": round(np.mean(ndcg5_list), 4),
            "NDCG@10": round(np.mean(ndcg10_list), 4),
            "MRR": round(np.mean(mrr_list), 4),
            "Recall@10": round(np.mean(recall10_list), 4),
            "Precision@5": round(np.mean(prec5_list), 4),
            "Avg_Latency_ms": round(np.mean(latencies) * 1000, 1),
            "P95_Latency_ms": round(np.percentile(latencies, 95) * 1000, 1),
            "P99_Latency_ms": round(np.percentile(latencies, 99) * 1000, 1),
            "N_Queries": len(ndcg5_list),
        }
        results_rows.append(row)

        # Log to experiments_log.jsonl
        logger.log(
            step=approach_name,
            metrics={k: v for k, v in row.items() if k != "Approach"},
            notes=f"Comparative study — {len(ndcg5_list)} evaluation queries",
        )

        print(f"  NDCG@5={row['NDCG@5']:.4f}  NDCG@10={row['NDCG@10']:.4f}  "
              f"MRR={row['MRR']:.4f}  Recall@10={row['Recall@10']:.4f}  "
              f"P@5={row['Precision@5']:.4f}  Lat={row['Avg_Latency_ms']:.1f}ms")

    results_df = pd.DataFrame(results_rows)

    # Sort by NDCG@10 descending for easy reading
    if not results_df.empty:
        results_df = results_df.sort_values("NDCG@10", ascending=False).reset_index(drop=True)

    return results_df


# ================================================================== #
# MAIN
# ================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("[evaluate.py] Evaluation, Comparative Study & Hardware Profiling")
    print("=" * 70)

    logger = ExperimentLogger(LOG_PATH)

    # ---- 1. Hardware profiling ----
    print("\n[Eval] Step 1: Hardware profiling...")
    hw = profile_hardware()
    logger.log(step="hardware_profile", metrics=hw, notes="System hardware before index loading")
    print(f"  CPU: {hw['cpu_model']}")
    print(f"  GPU: {hw['gpu']}")
    print(f"  RAM used: {hw['ram_used_mb']} MB / {hw['ram_total_mb']} MB")
    print(f"  Artifacts: {hw['artifacts_disk_mb']}")

    # ---- 2. Load Alliance ----
    print("\n[Eval] Step 2: Loading AllianceRetriever...")
    from pipeline import AllianceRetriever
    alliance = AllianceRetriever()

    # Re-profile after index loading
    hw_after = profile_hardware()
    logger.log(step="hardware_after_loading", metrics=hw_after, notes="System hardware after all indices loaded")
    print(f"  RAM used after loading: {hw_after['ram_used_mb']} MB")

    # ---- 3. Generate evaluation dataset ----
    print("\n[Eval] Step 3: Generating evaluation dataset...")
    eval_queries = generate_eval_dataset(
        alliance.df,
        bm25_index=alliance.bm25,
        n_queries=200,
    )
    logger.log(
        step="eval_dataset_generated",
        metrics={"n_queries": len(eval_queries), "avg_relevant": np.mean([len(q["relevant_ids"]) for q in eval_queries])},
        notes="Validation set with query + ground-truth relevance via BM25 weak supervision",
    )
    print(f"  Generated {len(eval_queries)} evaluation queries.")

    # ---- 4. Run comparative study ----
    print("\n[Eval] Step 4: Running comparative study (8 approaches)...")
    results_df = comparative_study(alliance, eval_queries, logger)

    # ---- 5. Print results table ----
    print("\n" + "=" * 70)
    print("[Eval] COMPARATIVE STUDY RESULTS")
    print("=" * 70)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("  No results collected.")
    print("=" * 70)

    # ---- 6. Save results ----
    results_dict = results_df.to_dict(orient="records") if not results_df.empty else []
    output = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw,
        "hardware_after_loading": hw_after,
        "evaluation": {
            "n_queries": len(eval_queries),
            "approaches_tested": len(results_dict),
        },
        "comparative_results": results_dict,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results saved to {RESULTS_PATH}")
    print(f"[Eval] Experiment log at {LOG_PATH}")
    print("[evaluate.py] All done.")