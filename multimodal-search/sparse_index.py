"""
sparse_index.py — Шаг 2: Sparse Retrieval & SPLADE (Learned Sparse).

Course: Deep Learning for Search
Lecture refs: L02 (Inverted Index, BM25), L05 (SPLADE / Learned Sparse Retrieval)

Two sparse retrieval systems:
  1. BM25 (rank_bm25) — classic lexical Inverted Index approach.
     Tokenization: word-level (nltk-style simple tokenizer).
  2. SPLADE (nreimers/splade-cocondenser-ense) — Learned Sparse Retrieval.
     Produces sparse vectors via log(1 + ReLU(logit)) per BERT vocab term.

Both return (doc_ids, scores) for a query.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from scipy import sparse as sp
import re
import pickle
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
SPLADE_VOCAB_PATH = _SCRIPT_DIR / "splade_vocab.pkl"
SPLADE_MATRIX_PATH = _SCRIPT_DIR / "splade_sparse_matrix.npz"


# ------------------------------------------------------------------ #
# Simple tokenizer (word-level, for BM25 Inverted Index)
# ------------------------------------------------------------------ #

def simple_tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, filter short tokens."""
    text = text.lower()
    tokens = re.findall(r'[a-zа-яё0-9]+', text, flags=re.UNICODE)
    return [t for t in tokens if len(t) > 1]


# ------------------------------------------------------------------ #
# BM25 Index (L02: Inverted Index + TF-IDF + BM25 scoring)
# ------------------------------------------------------------------ #

class BM25Index:
    """
    Classic BM25 Inverted Index.
    Uses rank_bm25.BM25Okapi internally.
    """

    def __init__(self):
        self.bm25 = None
        self.doc_ids = None
        self.tokenized_corpus = None

    def build(self, texts: List[str], doc_ids: List[int]):
        """
        Build BM25 Inverted Index from document texts.
        texts: list of clean strategy descriptions
        doc_ids: corresponding document IDs
        """
        print("[BM25] Tokenizing corpus for Inverted Index...")
        self.tokenized_corpus = [simple_tokenize(t) for t in texts]
        self.doc_ids = np.array(doc_ids)
        print("[BM25] Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] Index built on {len(self.doc_ids)} documents.")

    def search(self, query: str, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the Inverted Index.
        Returns: (doc_ids, scores) arrays of length min(top_k, corpus_size).
        """
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        # Get top-k indices
        top_idx = np.argsort(scores)[::-1][:top_k]
        return self.doc_ids[top_idx], scores[top_idx]

    def save(self, path: Path = Path("bm25_index.pkl")):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)
        print(f"[BM25] Saved to {path}")

    def load(self, path: Path = Path("bm25_index.pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        print(f"[BM25] Loaded from {path}")


# ------------------------------------------------------------------ #
# SPLADE (L05: Learned Sparse Retrieval)
# ------------------------------------------------------------------ #

class SPLADEIndex:
    """
    SPLADE — Sparse Lexical and Expansion Model for Dense Retrieval (L05).
    Model: nreimers/splade-cocondenser-ense

    Sparse vector computation:
      For each token in BERT vocab, take the [CLS] logit -> apply ReLU -> log(1 + x)
      This produces a sparse representation where each dimension = weight of a vocab term.

    We build a scipy.sparse matrix for the full corpus and search via cosine similarity.
    """

    def __init__(self, model_name: str = "nreimers/splade-cocondenser-ense"):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        print(f"[SPLADE] Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._torch = torch
        print(f"[SPLADE] Model loaded on {self.device}.")

        self.sparse_matrix = None  # scipy.sparse.csr_matrix (n_docs x vocab_size)
        self.doc_ids = None
        self.vocab_size = self.tokenizer.vocab_size

    def _text_to_sparse(self, text: str) -> Dict[int, float]:
        """
        Convert a single text to a sparse vector (dict: token_id -> weight).
        Formula (L05): weight_i = log(1 + ReLU(logit_i))
        """
        with self._torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            # logits shape: (1, seq_len, vocab_size)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Max pooling over sequence dimension -> (1, vocab_size)
            max_logits, _ = self._torch.max(logits, dim=1)  # (1, vocab_size)

            # Apply SPLADE activation: log(1 + ReLU(x))
            sparse_weights = self._torch.log1p(self._torch.relu(max_logits)).squeeze(0)  # (vocab_size,)

            # Convert to dense numpy for now (will sparsify later)
            weights_np = sparse_weights.cpu().numpy()

        # Sparsify: keep only non-zero entries
        sparse_dict = {}
        nonzero_idx = np.where(weights_np > 1e-6)[0]
        for idx in nonzero_idx:
            sparse_dict[int(idx)] = float(weights_np[idx])

        return sparse_dict

    def build(self, texts: List[str], doc_ids: List[int],
              save_vocab: Path = SPLADE_VOCAB_PATH,
              save_matrix: Path = SPLADE_MATRIX_PATH):
        """
        Build SPLADE sparse matrix for the full corpus.
        Saves the sparse matrix to disk for efficient loading.
        """
        n_docs = len(texts)
        self.doc_ids = np.array(doc_ids)

        print(f"[SPLADE] Encoding {n_docs} documents...")
        rows, cols, data = [], [], []

        for i, text in enumerate(tqdm(texts, desc="[SPLADE] Encoding")):
            sparse_dict = self._text_to_sparse(text)
            for token_id, weight in sparse_dict.items():
                rows.append(i)
                cols.append(token_id)
                data.append(weight)

        # Build CSR sparse matrix
        self.sparse_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, self.vocab_size),
        )

        # Save to disk
        sp.save_npz(str(save_matrix), self.sparse_matrix)
        print(f"[SPLADE] Sparse matrix saved to {save_matrix}")
        print(f"[SPLADE]  Shape: {self.sparse_matrix.shape}")
        print(f"[SPLADE]  Non-zeros: {self.sparse_matrix.nnz}")
        print(f"[SPLADE]  Density: {self.sparse_matrix.nnz / (self.sparse_matrix.shape[0] * self.sparse_matrix.shape[1]):.6f}")

    def search(self, query: str, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search SPLADE index using cosine similarity between sparse vectors.
        Query vector is computed on-the-fly; corpus vectors are pre-computed.

        Returns: (doc_ids, scores) of length min(top_k, corpus_size).
        """
        if self.sparse_matrix is None:
            raise RuntimeError("SPLADE index not built. Call build() first.")

        # Encode query
        query_sparse = self._text_to_sparse(query)
        if not query_sparse:
            # Empty query vector -> return first top_k docs with score 0
            return self.doc_ids[:top_k], np.zeros(min(top_k, len(self.doc_ids)))

        # Build query sparse vector
        query_cols = list(query_sparse.keys())
        query_data = list(query_sparse.values())
        query_vec = sp.csr_matrix(
            (query_data, ([0] * len(query_cols), query_cols)),
            shape=(1, self.vocab_size),
        )

        # Cosine similarity: normalize both query and docs
        # L2 normalize query
        query_norm = sp.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # L2 normalize each doc row
        row_norms = sp.linalg.norm(self.sparse_matrix, axis=1)
        # Avoid division by zero
        row_norms[row_norms == 0] = 1.0
        normalized_matrix = self.sparse_matrix.multiply(1.0 / row_norms[:, np.newaxis])

        # Compute cosine similarities: (1, vocab) @ (vocab, n_docs) -> (1, n_docs)
        similarities = (query_vec @ normalized_matrix.T).toarray().flatten()

        # Get top-k
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return self.doc_ids[top_idx], similarities[top_idx]

    def load(self, matrix_path: Path = SPLADE_MATRIX_PATH, doc_ids: np.ndarray = None):
        """Load pre-built SPLADE sparse matrix from disk."""
        self.sparse_matrix = sp.load_npz(str(matrix_path))
        if doc_ids is not None:
            self.doc_ids = doc_ids
        print(f"[SPLADE] Loaded sparse matrix from {matrix_path}")
        print(f"[SPLADE]  Shape: {self.sparse_matrix.shape}")


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-splade", action="store_true",
                        help="Skip SPLADE (requires HuggingFace auth for gated model)")
    args = parser.parse_args()

    # Load clean data
    df = pd.read_parquet(CLEAN_PATH)
    texts = df["text"].tolist()
    doc_ids = df["id"].tolist()
    print(f"Loaded {len(df)} documents from {CLEAN_PATH}")

    # Build BM25 Index
    bm25 = BM25Index()
    bm25.build(texts, doc_ids)
    bm25.save()

    # Test BM25 search
    query = "трендовая стратегия пробой SMA"
    ids, scores = bm25.search(query, top_k=5)
    print(f"\n[BM25] Query: '{query}'")
    for i, (doc_id, score) in enumerate(zip(ids, scores)):
        print(f"  {i+1}. Doc {doc_id} (score: {score:.4f}): {texts[int(doc_id)][:80]}...")

    # Build SPLADE Index (optional — model is gated, requires `huggingface-cli login`)
    if not args.skip_splade:
        try:
            splade = SPLADEIndex()
            splade.build(texts, doc_ids)

            # Test SPLADE search
            ids, scores = splade.search(query, top_k=5)
            print(f"\n[SPLADE] Query: '{query}'")
            for i, (doc_id, score) in enumerate(zip(ids, scores)):
                print(f"  {i+1}. Doc {doc_id} (score: {score:.4f}): {texts[int(doc_id)][:80]}...")
        except OSError as e:
            print(f"\n[SPLADE] Не удалось загрузить модель: {e}")
            print("[SPLADE] Решения:")
            print("  1. huggingface-cli login   (ввести токен с huggingface.co/settings/tokens)")
            print("  2. Принять лицензию: https://huggingface.co/nreimers/splade-cocondenser-ense")
            print("  3. Или запустите с --skip-splade (пайплайн будет работать с BM25 + Dense)")
    else:
        print("\n[SPLADE] Пропущен (--skip-splade)")

    print("\n[sparse_index] All done.")