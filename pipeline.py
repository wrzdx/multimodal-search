"""
pipeline.py — Шаг 6 & 7: Сборка Альянса, RRF и Агент обратной связи (Agentic CRAG).

Course: Deep Learning for Search
Lecture refs:
  L02 (RRF — Reciprocal Rank Fusion),
  L08 (The Captain / LambdaMART),
  L11 (Agentic CRAG, RAGAS / Faithfulness)

AllianceRetriever:
  1. Recall phase: Retrieve top-100 from 4 systems via RRF (k=60).
  2. Feature Extraction: Collect LTR features for the 100 candidates.
  3. Precision phase: The Captain (LambdaMART) reranks.
  4. Agentic CRAG: refine_query() adjusts results based on user feedback
     (e.g., slider "Drawdown < 15%" triggers post-retrieval filtering and
     re-scoring, per L11 CRAG logic).

Iteration 1 (Baseline): Individual retrieval systems (BM25, SPLADE, Dense Text, Dense Curve)
Iteration 2 (Full Alliance): RRF fusion + LambdaMART + Agentic CRAG
See evaluate.py for comparative study.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import sparse as sp
import pickle
from datetime import datetime

from sparse_index import BM25Index, SPLADEIndex
from dense_index import DenseIndexSearcher
from ltr_captain import TheCaptain, extract_features, FEATURE_NAMES
from models import TextBiEncoder

# Paths
_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
BM25_PATH = _SCRIPT_DIR / "bm25_index.pkl"
SPLADE_MATRIX_PATH = _SCRIPT_DIR / "splade_sparse_matrix.npz"
TEXT_INDEX_PATH = _SCRIPT_DIR / "faiss_text.index"
CURVE_INDEX_PATH = _SCRIPT_DIR / "faiss_curve.index"
CAPTAIN_MODEL_PATH = _SCRIPT_DIR / "captain_lgbm.txt"
EXPERIMENTS_LOG_PATH = _SCRIPT_DIR / "experiments_log.jsonl"

# RRF constant (L02)
RRF_K = 60


# ------------------------------------------------------------------ #
# Reciprocal Rank Fusion (RRF) — L02
# ------------------------------------------------------------------ #

def reciprocal_rank_fusion(
    result_lists: List[Tuple[np.ndarray, np.ndarray]],
    k: int = RRF_K,
    top_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reciprocal Rank Fusion (RRF, L02).

    Combines multiple ranked result lists into a single ranking:
      RRF_score(d) = Σ_i 1 / (k + rank_i(d))

    where rank_i(d) is the rank of document d in the i-th result list.

    Args:
        result_lists: list of (doc_ids, scores) tuples from different retrievers
        k: RRF constant (default 60, per Cormack et al. 2009)
        top_k: number of results to return

    Returns:
        (fused_doc_ids, fused_scores) — sorted by fused score (descending)
    """
    rrf_scores: Dict[int, float] = {}

    for doc_ids, _ in result_lists:
        for rank, doc_id in enumerate(doc_ids):
            doc_id = int(doc_id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    if not rrf_scores:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Sort by RRF score (descending)
    sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])
    top_items = sorted_items[:top_k]

    doc_ids = np.array([item[0] for item in top_items])
    scores = np.array([item[1] for item in top_items])

    return doc_ids, scores


# ------------------------------------------------------------------ #
# Agentic CRAG — L11
# ------------------------------------------------------------------ #

class AgenticCRAG:
    """
    Agentic Corrective RAG (L11).

    When the user provides feedback (e.g., adjusts a slider for Max Drawdown),
    the agent:

    1. THOUGHT (L11): Understands that this is a hard constraint filter.
    2. ACTION (CRAG logic):
       a. Scans the top-100 RRF candidates.
       b. Marks candidates violating the constraint as "wrong".
       c. Re-scores: penalizes "wrong" candidates by shifting their
          Captain score down (post-retrieval correction).
    3. Re-reranks the adjusted list.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def refine_query(
        self,
        candidate_results: pd.DataFrame,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Agentic CRAG: refine results based on user feedback constraints.

        Thought (L11): The system understands that slider adjustments
        are hard constraints that must be enforced.

        Action (CRAG logic): Post-retrieval correction on the top-K
        from RRF. Candidates that violate constraints are marked as "wrong"
        and their scores are penalized.

        Args:
            candidate_results: DataFrame from The Captain with 'doc_id', 'captain_score', etc.
            min_sharpe: user's desired minimum Sharpe ratio
            max_drawdown: user's desired maximum drawdown (e.g., -15)

        Returns:
            Adjusted DataFrame, re-sorted by adjusted score
        """
        results = candidate_results.copy()

        # --- THOUGHT: Analyze constraints ---
        has_sharpe_constraint = min_sharpe is not None
        has_dd_constraint = max_drawdown is not None

        if not has_sharpe_constraint and not has_dd_constraint:
            # No constraints — return as-is
            results["crag_action"] = "none"
            return results

        print(f"[CRAG] Thought: User constraint detected — "
              f"Min Sharpe: {min_sharpe}, Max Drawdown: {max_drawdown}")

        # --- ACTION: Evaluate and penalize ---
        adjusted_scores = results["captain_score"].values.copy()
        crag_actions = []

        for i, row in results.iterrows():
            doc_id = int(row["doc_id"])
            doc_row = self.df.iloc[doc_id]
            actions = []

            # Check Sharpe constraint
            if has_sharpe_constraint and doc_row["sharpe"] < min_sharpe:
                # Penalize: shift score down significantly
                adjusted_scores[i] -= 10.0
                actions.append("sharpe_violation")

            # Check Drawdown constraint
            if has_dd_constraint and doc_row["max_drawdown"] < max_drawdown:
                # Note: max_drawdown is negative, e.g., -15 means "no worse than -15%"
                # So "worse" means more negative (e.g., -25 < -15)
                adjusted_scores[i] -= 10.0
                actions.append("drawdown_violation")

            if actions:
                crag_actions.append(f"wrong({'+'.join(actions)})")
            else:
                crag_actions.append("ok")

        results["captain_score"] = adjusted_scores
        results["crag_action"] = crag_actions

        # Re-sort by adjusted score
        results = results.sort_values("captain_score", ascending=False).reset_index(drop=True)

        n_corrected = sum(1 for a in crag_actions if a != "ok")
        print(f"[CRAG] Action: {n_corrected} candidates corrected, "
              f"{len(crag_actions) - n_corrected} passed.")

        return results


# ------------------------------------------------------------------ #
# AllianceRetriever — Full Pipeline
# ------------------------------------------------------------------ #

class AllianceRetriever:
    """
    The Alliance: Multimodal Retrieval Pipeline.

    Combines 4 retrieval systems via RRF (L02), extracts features (L09),
    reranks with LambdaMART / The Captain (L08), and applies Agentic
    CRAG corrections (L11).

    Architecture:
      User Query
        |
        v
    +---------------------------+
    |   Phase 1: RECALL (RRF)   |
    |  BM25 + SPLADE + Dense    |
    |  Text + Dense Curve       |
    |  -> Top-100 candidates    |
    +---------------------------+
        |
        v
    +---------------------------+
    | Phase 2: FEATURE EXTRACT  |
    |  LTR features for each    |
    |  candidate (L09)          |
    +---------------------------+
        |
        v
    +---------------------------+
    | Phase 3: PRECISION        |
    |  The Captain (LambdaMART) |
    |  reranks top-100 -> top-5 |
    +---------------------------+
        |
        v
    +---------------------------+
    | Phase 4: AGENTIC CRAG     |
    |  Post-retrieval correction|
    |  based on user feedback   |
    +---------------------------+
        |
        v
      Final Results
    """

    def __init__(self, build_if_missing: bool = True):
        """
        Initialize the full Alliance pipeline.

        Args:
            build_if_missing: If True, run all build steps if artifacts are missing.
        """
        print("=" * 60)
        print("[Alliance] Initializing Multimodal Search Alliance")
        print("=" * 60)

        # Load clean data
        if not CLEAN_PATH.exists():
            raise FileNotFoundError(
                f"Clean data not found at {CLEAN_PATH}. "
                f"Run: python data_parser.py"
            )
        self.df = pd.read_parquet(CLEAN_PATH)
        print(f"[Alliance] Loaded {len(self.df)} documents.")

        # Load BM25 Index (L02: Inverted Index)
        if not BM25_PATH.exists():
            raise FileNotFoundError(f"BM25 index not found. Run: python sparse_index.py")
        self.bm25 = BM25Index()
        self.bm25.load(BM25_PATH)

        # Load SPLADE Index (L05: Learned Sparse) — OPTIONAL
        # Model is gated on HuggingFace; pipeline works without it.
        self.splade = None
        if SPLADE_MATRIX_PATH.exists():
            try:
                self.splade = SPLADEIndex()
                self.splade.load(SPLADE_MATRIX_PATH, doc_ids=self.df["id"].values)
                print(f"[Alliance] SPLADE loaded: {self.splade.sparse_matrix.shape[0]} docs.")
            except Exception as e:
                print(f"[Alliance] SPLADE unavailable ({e}). Using BM25 + Dense only.")
                self.splade = None
        else:
            print("[Alliance] SPLADE matrix not found. Using BM25 + Dense only.")
            print("[Alliance] For SPLADE: huggingface-cli login && python sparse_index.py")

        # Load Dense Indexes (L05: Dense Retrieval via FAISS)
        if not TEXT_INDEX_PATH.exists() or not CURVE_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS indices not found. Run: python dense_index.py")
        self.dense_searcher = DenseIndexSearcher()

        # Load The Captain (L08: LambdaMART) — optional
        self.captain = TheCaptain()
        self._captain_ready = False
        if CAPTAIN_MODEL_PATH.exists():
            try:
                self.captain.load(CAPTAIN_MODEL_PATH)
                self._captain_ready = True
                print("[Alliance] The Captain (LambdaMART) loaded.")
            except Exception:
                print("[Alliance] Captain файл повреждён — поиск работает через RRF.")
        else:
            print("[Alliance] Captain не найден — поиск работает через RRF.")
            print("[Alliance] Для переранжирования LambdaMART запустите: python train_captain.py")

        # Initialize CRAG Agent (L11)
        self.crag_agent = AgenticCRAG(self.df)

        print("[Alliance] All systems online.\n")

    def _log_search_phase(self, query: str, metrics: dict):
        """Append per-search timing/candidate-count to experiments_log.jsonl."""
        import json
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "alliance_search",
            "metrics": metrics,
            "notes": f"Query: {query[:80]}",
        }
        try:
            with open(EXPERIMENTS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # logging is best-effort

    def _train_captain(self):
        """Train The Captain if no pre-trained model exists."""
        import time as _t
        from ltr_captain import generate_training_data
        print("\n[Alliance] ╔══════════════════════════════════════════════════════════╗")
        print("[Alliance] ║  The Captain (LambdaMART) не найден — обучаем...         ║")
        print("[Alliance] ║  Это произойдёт ОДИН раз, модель сохранится на диск.     ║")
        print("[Alliance] ╚══════════════════════════════════════════════════════════╝\n")
        t0 = _t.time()
        train_df, groups = generate_training_data(
            self.df, self.bm25, self.splade, self.dense_searcher,
            n_queries=100, top_k=50,
        )
        self.captain.train(train_df, groups)
        elapsed = _t.time() - t0
        print(f"\n[Alliance] Captain обучен за {elapsed:.1f}s. Модель сохранена — "
              f"повторный запуск будет мгновенным.\n")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        rrf_top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Full Alliance search pipeline.

        Args:
            query: text query from user
            top_k: number of final results to return
            min_sharpe: CRAG constraint — minimum Sharpe ratio
            max_drawdown: CRAG constraint — maximum drawdown (negative, e.g., -15)
            rrf_top_k: number of candidates from RRF recall phase

        Returns:
            DataFrame with top_k results, including text, metrics, curve, scores
        """
        print(f"[Alliance] Query: '{query}'")
        print(f"[Alliance] Constraints: min_sharpe={min_sharpe}, max_drawdown={max_drawdown}")

        # ---- Phase 1: RECALL — RRF from 4 systems ----
        print("[Alliance] Phase 1: Recall (RRF)...")
        t_recall_start = time.time()
        bm25_ids, bm25_scores = self.bm25.search(query, top_k=rrf_top_k)

        # Collect result lists for RRF (2-4 systems depending on availability)
        rrf_inputs = [(bm25_ids, bm25_scores)]

        if self.splade is not None:
            splade_ids, splade_scores = self.splade.search(query, top_k=rrf_top_k)
            rrf_inputs.append((splade_ids, splade_scores))

        try:
            dense_text_ids, dense_text_scores = self.dense_searcher.search_text(query, top_k=rrf_top_k)
            rrf_inputs.append((dense_text_ids, dense_text_scores))

            # For curve search, use a reference curve (upward trending)
            ref_curve = np.linspace(100, 120, 252).astype(np.float32)
            dense_curve_ids, dense_curve_scores = self.dense_searcher.search_curve(ref_curve, top_k=rrf_top_k)
            rrf_inputs.append((dense_curve_ids, dense_curve_scores))
        except Exception as e:
            print(f"[Alliance] Dense search error ({e}), using sparse only.")

        # RRF Fusion (L02)
        fused_ids, fused_scores = reciprocal_rank_fusion(rrf_inputs,
            k=RRF_K,
            top_k=rrf_top_k,
        )
        t_recall_elapsed = time.time() - t_recall_start
        print(f"[Alliance]  RRF returned {len(fused_ids)} unique candidates.")

        # ---- Phase 2: FEATURE EXTRACTION (L09) ----
        print("[Alliance] Phase 2: Feature extraction...")
        t_feat_start = time.time()
        features_df = extract_features(
            query, fused_ids, self.df,
            self.bm25, self.splade, self.dense_searcher,
            query_sharpe=min_sharpe,
            query_max_drawdown=max_drawdown,
        )

        # If SPLADE scores are missing, fill with 0
        if "splade_cos" not in features_df.columns:
            features_df["splade_cos"] = 0.0
        if "dense_text_cos" not in features_df.columns:
            features_df["dense_text_cos"] = 0.0
        if "dense_curve_cos" not in features_df.columns:
            features_df["dense_curve_cos"] = 0.0
        t_feat_elapsed = time.time() - t_feat_start

        # ---- Phase 3: PRECISION — The Captain (L08) ----
        t_captain_start = time.time()
        if self._captain_ready:
            print("[Alliance] Phase 3: The Captain reranking...")
            try:
                reranked = self.captain.rerank(features_df)
            except Exception:
                reranked = features_df.copy()
                reranked["captain_score"] = fused_scores[np.searchsorted(fused_ids, reranked["doc_id"].values.astype(int).tolist(), sorter=None)]
        else:
            # Captain not trained — use RRF scores as captain_score
            print("[Alliance] Phase 3: skipped (Captain not trained, using RRF scores)")
            reranked = features_df.copy()
            # Map RRF scores to candidates
            rrf_map = dict(zip(fused_ids.astype(int), fused_scores))
            reranked["captain_score"] = reranked["doc_id"].map(rrf_map).fillna(0).values
            reranked = reranked.sort_values("captain_score", ascending=False).reset_index(drop=True)
        t_captain_elapsed = time.time() - t_captain_start

        # ---- Phase 4: AGENTIC CRAG (L11) ----
        print("[Alliance] Phase 4: Agentic CRAG refinement...")
        t_crag_start = time.time()
        refined = self.crag_agent.refine_query(reranked, min_sharpe, max_drawdown)
        t_crag_elapsed = time.time() - t_crag_start

        # ---- Log timing to experiments_log.jsonl ----
        self._log_search_phase(query, {
            "n_rrf_candidates": len(fused_ids),
            "recall_latency_ms": round(t_recall_elapsed * 1000, 1),
            "feature_latency_ms": round(t_feat_elapsed * 1000, 1),
            "captain_latency_ms": round(t_captain_elapsed * 1000, 1),
            "crag_latency_ms": round(t_crag_elapsed * 1000, 1),
            "total_latency_ms": round((t_recall_elapsed + t_feat_elapsed + t_captain_elapsed + t_crag_elapsed) * 1000, 1),
        })

        # ---- Assemble final results ----
        final_results = []
        for _, row in refined.head(top_k).iterrows():
            doc_id = int(row["doc_id"])
            doc = self.df.iloc[doc_id]
            final_results.append({
                "rank": len(final_results) + 1,
                "doc_id": doc_id,
                "text": doc["text"],
                "sharpe": doc["sharpe"],
                "max_drawdown": doc["max_drawdown"],
                "total_return": doc["total_return"],
                "equity_curve": doc["equity_curve"],
                "captain_score": row["captain_score"],
                "crag_action": row.get("crag_action", "none"),
                "rrf_score": fused_scores[np.where(fused_ids == doc_id)[0][0]]
                if doc_id in fused_ids else 0.0,
            })

        result_df = pd.DataFrame(final_results)
        print(f"[Alliance] Returning {len(result_df)} results.\n")
        return result_df


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    alliance = AllianceRetriever()
    results = alliance.search("трендовая стратегия пробой SMA", top_k=5)
    print(results[["rank", "doc_id", "sharpe", "max_drawdown", "captain_score"]])