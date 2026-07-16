"""
dense_index.py — Шаг 4: Векторное пространство и FAISS.

Course: Deep Learning for Search
Lecture refs: L02 (Vector Space Model, Cosine Similarity), L05 (Dense Retrieval)

This module:
  1. Loads the trained CurveEncoder and TextBiEncoder.
  2. Encodes the entire clean corpus into dense vectors.
  3. L2-normalizes all vectors (Sir Cosine, L02/L05).
  4. Builds two FAISS IndexFlatIP indices:
     - text_index: for text embeddings
     - curve_index: for equity curve embeddings

FAISS IndexFlatIP computes inner product, which equals cosine similarity
when vectors are L2-normalized.
"""

import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path
from typing import Tuple

from models import CurveEncoder, TextBiEncoder

_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
CURVE_ENCODER_PATH = _SCRIPT_DIR / "curve_encoder.pth"
TEXT_PROJ_PATH = _SCRIPT_DIR / "text_proj_head.pth"
TEXT_INDEX_PATH = _SCRIPT_DIR / "faiss_text.index"
CURVE_INDEX_PATH = _SCRIPT_DIR / "faiss_curve.index"
TEXT_EMB_PATH = _SCRIPT_DIR / "text_embeddings.npy"
CURVE_EMB_PATH = _SCRIPT_DIR / "curve_embeddings.npy"

CURVE_LEN = 252
EMBEDDING_DIM = 128
BATCH_SIZE = 128


def build_dense_indices(
    clean_path: Path = CLEAN_PATH,
    curve_encoder_path: Path = CURVE_ENCODER_PATH,
    text_proj_path: Path = TEXT_PROJ_PATH,
    text_index_path: Path = TEXT_INDEX_PATH,
    curve_index_path: Path = CURVE_INDEX_PATH,
    text_emb_path: Path = TEXT_EMB_PATH,
    curve_emb_path: Path = CURVE_EMB_PATH,
) -> Tuple[faiss.Index, faiss.Index, np.ndarray, np.ndarray]:
    """
    Build FAISS indices for dense retrieval.

    Returns:
        text_index, curve_index, text_embeddings, curve_embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dense_index] Device: {device}")

    # Load clean data
    print("[dense_index] Loading clean data...")
    df = pd.read_parquet(clean_path)
    texts = df["text"].tolist()
    curves = np.array(df["equity_curve"].tolist(), dtype=np.float32)
    doc_ids = df["id"].values
    n_docs = len(df)
    print(f"[dense_index] {n_docs} documents loaded.")

    # ---- Text Embeddings ----
    print("\n[dense_index] Encoding texts with Bi-encoder (Scout)...")
    text_encoder = TextBiEncoder()
    text_embeddings = text_encoder.encode(texts, batch_size=BATCH_SIZE, normalize=True)
    print(f"[dense_index] Text embeddings shape: {text_embeddings.shape}")

    # Verify L2 normalization
    norms = np.linalg.norm(text_embeddings, axis=1)
    print(f"[dense_index] Text embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")

    # Build FAISS IndexFlatIP for text
    dim_text = text_embeddings.shape[1]
    text_index = faiss.IndexFlatIP(dim_text)
    text_index.add(text_embeddings)
    print(f"[dense_index] Text FAISS index built: {text_index.ntotal} vectors, dim={dim_text}")
    faiss.write_index(text_index, str(text_index_path))
    print(f"[dense_index] Saved to {text_index_path}")

    # ---- Curve Embeddings ----
    print("\n[dense_index] Encoding curves with CurveEncoder...")
    curve_encoder = CurveEncoder(input_len=CURVE_LEN, output_dim=EMBEDDING_DIM).to(device)

    # Load trained weights if available
    if curve_encoder_path.exists():
        curve_encoder.load_state_dict(torch.load(str(curve_encoder_path), map_location=device))
        print(f"[dense_index] Loaded trained CurveEncoder from {curve_encoder_path}")
    else:
        print("[dense_index] WARNING: No trained weights found, using random initialization!")

    curve_encoder.eval()
    curve_embeddings_list = []

    with torch.no_grad():
        for i in range(0, n_docs, BATCH_SIZE):
            batch_curves = torch.tensor(
                curves[i:i + BATCH_SIZE], dtype=torch.float32
            ).to(device)
            batch_embs = curve_encoder(batch_curves).cpu().numpy()
            curve_embeddings_list.append(batch_embs)

    curve_embeddings = np.vstack(curve_embeddings_list)
    print(f"[dense_index] Curve embeddings shape: {curve_embeddings.shape}")

    # Verify L2 normalization
    norms = np.linalg.norm(curve_embeddings, axis=1)
    print(f"[dense_index] Curve embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")

    # Build FAISS IndexFlatIP for curves
    curve_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    curve_index.add(curve_embeddings)
    print(f"[dense_index] Curve FAISS index built: {curve_index.ntotal} vectors, dim={EMBEDDING_DIM}")
    faiss.write_index(curve_index, str(curve_index_path))
    print(f"[dense_index] Saved to {curve_index_path}")

    # Save embeddings for reuse
    np.save(str(text_emb_path), text_embeddings)
    np.save(str(curve_emb_path), curve_embeddings)

    return text_index, curve_index, text_embeddings, curve_embeddings


class DenseIndexSearcher:
    """
    Wrapper for searching the FAISS dense indices.
    """

    def __init__(self,
                 text_index_path: Path = TEXT_INDEX_PATH,
                 curve_index_path: Path = CURVE_INDEX_PATH,
                 text_encoder=None):
        """
        Load pre-built FAISS indices.

        Args:
            text_encoder: TextBiEncoder instance for encoding queries on-the-fly.
                         If None, will create one.
        """
        self.text_index = faiss.read_index(str(text_index_path))
        self.curve_index = faiss.read_index(str(curve_index_path))
        print(f"[DenseSearch] Text index loaded: {self.text_index.ntotal} vectors")
        print(f"[DenseSearch] Curve index loaded: {self.curve_index.ntotal} vectors")

        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            from models import TextBiEncoder
            self.text_encoder = TextBiEncoder()

        # Curve encoder for query encoding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curve_encoder = CurveEncoder(input_len=CURVE_LEN, output_dim=EMBEDDING_DIM).to(device)
        if CURVE_ENCODER_PATH.exists():
            self.curve_encoder.load_state_dict(torch.load(str(CURVE_ENCODER_PATH), map_location=device))
        self.curve_encoder.eval()
        self.device = device

    def search_text(self, query: str, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search text FAISS index.

        Returns: (doc_ids, scores)
        """
        query_emb = self.text_encoder.encode([query], normalize=True)  # (1, dim)
        scores, ids = self.text_index.search(query_emb, top_k)
        return ids[0], scores[0]

    def search_curve(self, query_curve: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search curve FAISS index.

        Args:
            query_curve: np.ndarray of shape (252,)

        Returns: (doc_ids, scores)
        """
        with torch.no_grad():
            curve_tensor = torch.tensor(query_curve, dtype=torch.float32).unsqueeze(0).to(self.device)
            query_emb = self.curve_encoder(curve_tensor).cpu().numpy()  # (1, 128)
        scores, ids = self.curve_index.search(query_emb, top_k)
        return ids[0], scores[0]


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    text_idx, curve_idx, text_embs, curve_embs = build_dense_indices()

    # Test search
    searcher = DenseIndexSearcher(text_encoder=None)

    query = "трендовая стратегия пробой SMA 50"
    ids, scores = searcher.search_text(query, top_k=5)
    print(f"\n[Dense Text] Query: '{query}'")
    for i, (doc_id, score) in enumerate(zip(ids, scores)):
        print(f"  {i+1}. Doc {doc_id} (score: {score:.4f})")

    print("\n[dense_index] All done.")