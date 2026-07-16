"""
ltr_captain.py — Шаг 5: The Captain — LTR с LambdaMART (LightGBM).

Course: Deep Learning for Search
Lecture refs: L08 (LambdaMART / The Captain), L09 (Feature Engineering for LTR)

This module:
  1. Extracts LTR features for (Query, Candidate_Doc) pairs:
     - bm25_score, splade_cos, dense_text_cos, dense_curve_cos
     - diff_sharpe, diff_drawdown (user query preferences vs candidate metrics)
  2. Trains lightgbm.LGBMRanker (LambdaMART) on a validation set.
  3. Provides a predict() interface for reranking.

"The Captain" (L08) sits on top of the Alliance retrieval and makes the
final precision-oriented reranking decisions.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from tqdm import tqdm

from sparse_index import BM25Index, SPLADEIndex, simple_tokenize
from dense_index import DenseIndexSearcher
from models import TextBiEncoder, CurveEncoder
import torch

_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
BM25_PATH = _SCRIPT_DIR / "bm25_index.pkl"
SPLADE_MATRIX_PATH = _SCRIPT_DIR / "splade_sparse_matrix.npz"
CAPTAIN_MODEL_PATH = _SCRIPT_DIR / "captain_lgbm.txt"
FEATURE_NAMES = [
    "bm25_score",
    "splade_cos",
    "dense_text_cos",
    "dense_curve_cos",
    "diff_sharpe",
    "diff_drawdown",
]


# ------------------------------------------------------------------ #
# Feature Extraction
# ------------------------------------------------------------------ #

def _batch_curve_cosine(dense_searcher: DenseIndexSearcher,
                         candidate_ids: np.ndarray,
                         batch_size: int = 512) -> Dict[int, float]:
    """
    Compute curve cosine similarity for all candidates in batches.

    Uses FAISS index.reconstruct() to get pre-computed embeddings
    (O(1) per document, NO GPU encoding needed).
    """
    ref_curve = np.linspace(100, 120, 252).astype(np.float32)
    with torch.no_grad():
        ref_tensor = torch.tensor(ref_curve).unsqueeze(0).to(dense_searcher.device)
        ref_emb = dense_searcher.curve_encoder(ref_tensor).cpu().numpy()[0]

    curve_index = dense_searcher.curve_index
    scores_map: Dict[int, float] = {}
    n = len(candidate_ids)

    for start in range(0, n, batch_size):
        batch_ids = candidate_ids[start:start + batch_size]
        # Use pre-computed embeddings from FAISS index (instant, no GPU)
        embs = np.vstack([
            curve_index.reconstruct(int(doc_id)) for doc_id in batch_ids
        ])  # (batch, 128)
        cos_sims = embs @ ref_emb  # (batch,)
        for doc_id, sim in zip(batch_ids, cos_sims):
            scores_map[int(doc_id)] = float(sim)

    return scores_map


def extract_features(
    query: str,
    candidate_ids: np.ndarray,
    df: pd.DataFrame,
    bm25_index: BM25Index,
    splade_index: Optional[SPLADEIndex],
    dense_searcher: DenseIndexSearcher,
    query_sharpe: Optional[float] = None,
    query_max_drawdown: Optional[float] = None,
) -> pd.DataFrame:
    """
    Extract LTR features for (query, candidate_doc) pairs.

    Features (L09):
      1. bm25_score — BM25 lexical score
      2. splade_cos — SPLADE cosine similarity
      3. dense_text_cos — Dense text Bi-encoder cosine similarity
      4. dense_curve_cos — Dense curve encoder cosine similarity
      5. diff_sharpe — |candidate_sharpe - user_desired_sharpe|
      6. diff_drawdown — |candidate_drawdown - user_desired_drawdown|
    """
    # Default user preferences to corpus median
    if query_sharpe is None:
        query_sharpe = df["sharpe"].median()
    if query_max_drawdown is None:
        query_max_drawdown = df["max_drawdown"].median()

    # ---- BM25 scores ----
    bm25_scores = bm25_index.search(query, top_k=len(candidate_ids))
    bm25_score_map = dict(zip(bm25_scores[0].astype(int), bm25_scores[1]))

    # ---- SPLADE scores (optional) ----
    splade_score_map: Dict[int, float] = {}
    if splade_index is not None:
        splade_scores = splade_index.search(query, top_k=len(candidate_ids))
        splade_score_map = dict(zip(splade_scores[0].astype(int), splade_scores[1]))

    # ---- Dense text scores ----
    dense_text_ids, dense_text_scores = dense_searcher.search_text(query, top_k=len(candidate_ids))
    dense_text_map = dict(zip(dense_text_ids.astype(int), dense_text_scores))

    # ---- Dense curve scores (pre-computed from FAISS, NO GPU) ----
    curve_score_map = _batch_curve_cosine(dense_searcher, candidate_ids)

    # ---- Build features (pure numpy, no GPU) ----
    n = len(candidate_ids)
    doc_ids_int = candidate_ids.astype(int)
    sharpes = df.iloc[doc_ids_int]["sharpe"].values
    drawdowns = df.iloc[doc_ids_int]["max_drawdown"].values

    features_list = []
    for i in range(n):
        did = int(doc_ids_int[i])
        features_list.append({
            "doc_id": did,
            "bm25_score": bm25_score_map.get(did, 0.0),
            "splade_cos": splade_score_map.get(did, 0.0),
            "dense_text_cos": dense_text_map.get(did, 0.0),
            "dense_curve_cos": curve_score_map.get(did, 0.0),
            "diff_sharpe": abs(sharpes[i] - query_sharpe),
            "diff_drawdown": abs(drawdowns[i] - query_max_drawdown),
        })

    return pd.DataFrame(features_list)


# ------------------------------------------------------------------ #
# Training Data Generation
# ------------------------------------------------------------------ #

def generate_training_data(
    df: pd.DataFrame,
    bm25_index: BM25Index,
    splade_index: Optional[SPLADEIndex],
    dense_searcher: DenseIndexSearcher,
    n_queries: int = 100,
    top_k: int = 50,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate LTR training data with pseudo-labels.
    Uses pre-computed FAISS embeddings for fast feature extraction.
    """
    print(f"[Captain] Generating {n_queries} training queries...")
    all_features = []
    all_groups = []

    query_indices = np.random.choice(len(df), size=min(n_queries, len(df)), replace=False)

    for q_idx in tqdm(query_indices, desc="[Captain] Queries", ncols=80):
        query_text = df.iloc[q_idx]["text"]
        query_sharpe = df.iloc[q_idx]["sharpe"]
        query_dd = df.iloc[q_idx]["max_drawdown"]

        bm25_ids, _ = bm25_index.search(query_text, top_k=top_k)
        candidate_ids = np.array(list(set(bm25_ids.tolist()))[:top_k])

        if len(candidate_ids) == 0:
            continue

        feats = extract_features(
            query_text, candidate_ids, df,
            bm25_index, splade_index, dense_searcher,
            query_sharpe, query_dd,
        )

        sharpes = df.iloc[candidate_ids.astype(int)]["sharpe"].values
        if sharpes.max() > sharpes.min():
            relevance = ((sharpes - sharpes.min()) / (sharpes.max() - sharpes.min()) * 4).astype(int)
        else:
            relevance = np.full(len(sharpes), 2, dtype=int)

        feats["relevance"] = relevance
        all_features.append(feats)
        all_groups.append(len(feats))

    train_df = pd.concat(all_features, ignore_index=True)
    print(f"[Captain] Training data: {len(train_df)} samples, {len(all_groups)} queries")
    return train_df, all_groups


# ------------------------------------------------------------------ #
# The Captain: LambdaMART (L08)
# ------------------------------------------------------------------ #

class TheCaptain:
    """
    LambdaMART ranker (The Captain, L08).

    Uses LightGBM's LGBMRanker which implements LambdaMART:
    a gradient-boosted decision tree algorithm that directly optimizes
    NDCG (Normalized Discounted Cumulative Gain) using lambda gradients.

    The Captain takes the Alliance's top-100 candidates and makes
    the final precision-oriented reranking decisions.
    """

    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES

    def train(self, train_df: pd.DataFrame, group_sizes: List[int]):
        """
        Train LambdaMART ranker.

        Args:
            train_df: DataFrame with feature columns + 'relevance' column
            group_sizes: list of group sizes (one per query)
        """
        X = train_df[self.feature_names].values.astype(np.float32)
        y = train_df["relevance"].values.astype(np.int32)

        # Create dataset for ranking
        train_data = lgb.Dataset(X, label=y, group=group_sizes,
                                 feature_name=self.feature_names)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "verbose": -1,
        }

        print("[Captain] Training LambdaMART (LightGBM LGBMRanker)...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            valid_names=["train"],
            callbacks=[lgb.log_evaluation(period=20)],
        )

        self.model.save_model(str(CAPTAIN_MODEL_PATH))
        print(f"[Captain] Model saved to {CAPTAIN_MODEL_PATH}")

        # Feature importance
        importance = self.model.feature_importance()
        print("\n[Captain] Feature Importance:")
        for fname, imp in sorted(zip(self.feature_names, importance), key=lambda x: -x[1]):
            print(f"  {fname:25s}: {imp}")

    def load(self, model_path: Path = CAPTAIN_MODEL_PATH):
        """Load a pre-trained Captain model."""
        self.model = lgb.Booster(model_file=str(model_path))
        print(f"[Captain] Model loaded from {model_path}")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores for candidates.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of predicted scores (higher = more relevant)
        """
        if self.model is None:
            raise RuntimeError("Captain model not loaded. Call train() or load() first.")
        X = features_df[self.feature_names].values.astype(np.float32)
        scores = self.model.predict(X)
        return scores

    def rerank(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank candidates by predicted score (descending).

        Returns:
            DataFrame sorted by predicted relevance (best first)
        """
        features_df = features_df.copy()
        features_df["captain_score"] = self.predict(features_df)
        features_df = features_df.sort_values("captain_score", ascending=False).reset_index(drop=True)
        return features_df



if __name__ == "__main__":
    # This requires pre-built indices
    print("[ltr_captain] This module requires pre-built BM25, SPLADE, and FAISS indices.")
    print("[ltr_captain] Run: python data_parser.py && python sparse_index.py && python train.py && python dense_index.py")
    print("[ltr_captain] Then run this module to train The Captain.")
    print("[ltr_captain] Note: In the full pipeline (pipeline.py), training is automatic.")