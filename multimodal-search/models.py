"""
models.py — Шаг 3: Deep Learning & Multimodal Encoders.

Course: Deep Learning for Search
Lecture refs:
  L06 (Bi-encoder / Scout), L07 (Cross-encoder / Judge),
  L08 (InfoNCE, Margin-MSE Distillation)

Components:
  1. CurveEncoder — 1D-CNN that encodes equity curves into 128-dim vectors.
  2. TextBiEncoder (Scout) — wraps sentence-transformers/all-MiniLM-L6-v2.
  3. CrossEncoderJudge — wraps cross-encoder/ms-marco-MiniLM-L-6-v2
     with metrics-augmented prompts.
  4. Projection head for InfoNCE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Optional, Dict
import numpy as np


# ------------------------------------------------------------------ #
# 1. Curve Encoder (1D-CNN)
# ------------------------------------------------------------------ #

class CurveEncoder(nn.Module):
    """
    Encodes an equity curve (252 time steps) into a fixed-size dense vector.

    Architecture (L06 — Bi-encoder spirit for non-text modality):
      Input: (batch, 252)  — equity curve
      -> Unsqueeze(1) -> (batch, 1, 252)
      -> Conv1d(1, 64, kernel=5, padding=2) + ReLU + BatchNorm
      -> Conv1d(64, 128, kernel=5, padding=2) + ReLU + BatchNorm
      -> Global Average Pooling -> (batch, 128)
      -> Linear(128, 128) -> (batch, 128)  — output embedding

    This allows us to search equity curves by their shape/pattern,
    complementing the text-based retrieval.
    """

    def __init__(self, input_len: int = 252, output_dim: int = 128):
        super().__init__()
        self.input_len = input_len
        self.output_dim = output_dim

        # Conv layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # Projection to output dim
        self.proj = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 252) — equity curves
        Returns:
            (batch, output_dim) — L2-normalized embeddings
        """
        # (batch, 252) -> (batch, 1, 252)
        x = x.unsqueeze(1)

        # Conv block 1
        x = self.conv1(x)        # (batch, 64, 252)
        x = F.relu(x)
        x = self.bn1(x)

        # Conv block 2
        x = self.conv2(x)        # (batch, 128, 252)
        x = F.relu(x)
        x = self.bn2(x)

        # Global Average Pooling -> (batch, 128)
        x = x.mean(dim=2)

        # Project to output dim
        x = self.proj(x)         # (batch, output_dim)

        # L2 normalize (Sir Cosine, L02/L05)
        x = F.normalize(x, p=2, dim=1)

        return x


# ------------------------------------------------------------------ #
# 2. Text Bi-encoder (Scout — L06)
# ------------------------------------------------------------------ #

class TextBiEncoder:
    """
    Bi-encoder (Scout) for text retrieval (L06).
    Uses sentence-transformers/all-MiniLM-L6-v2.

    The Bi-encoder encodes query and document independently into the same
    vector space, enabling efficient ANN search via FAISS.

    For training with InfoNCE, we add a trainable projection head.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Scout/BiEncoder] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[Scout/BiEncoder] Model loaded on {self.device}.")

    def encode(self, texts: List[str],
               batch_size: int = 64,
               show_progress: bool = True,
               normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: list of strings
            batch_size: encoding batch size
            show_progress: show tqdm bar
            normalize: L2-normalize output (for cosine similarity / FAISS IP)

        Returns:
            np.ndarray of shape (n_texts, embed_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text query."""
        return self.encode([text], normalize=normalize)[0]


# ------------------------------------------------------------------ #
# 3. Cross-encoder Judge (Teacher — L07)
# ------------------------------------------------------------------ #

class CrossEncoderJudge:
    """
    Cross-encoder (Judge / Teacher) for relevance scoring (L07).

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2.
    Unlike Bi-encoder, the Cross-encoder jointly processes (query, doc) pairs,
    enabling deep interaction between query and document tokens.

    For our multimodal case, we augment the document text with numeric metrics:
      [CLS] strategy_text [SEP] Metrics: Sharpe 1.5, Drawdown -10% [SEP]

    This Teacher's scores are used for Margin-MSE Distillation (L08)
    to train the Bi-encoder (Scout) students.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Judge/CrossEncoder] Loading {model_name}...")
        self.model = CrossEncoder(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CrossEncoder handles device internally
        print(f"[Judge/CrossEncoder] Model loaded.")

    def _format_pair(self, query: str, doc_text: str,
                     doc_sharpe: float, doc_drawdown: float) -> Tuple[str, str]:
        """
        Format a (query, doc) pair with metrics augmentation.
        This is how we inject numeric features into the text-based Cross-encoder.

        Format: [CLS] query [SEP] doc_text + metrics [SEP]
        """
        metrics_text = f"Metrics: Sharpe {doc_sharpe:.2f}, Drawdown {doc_drawdown:.1f}%"
        doc_with_metrics = f"{doc_text}. {metrics_text}"
        return (query, doc_with_metrics)

    def predict(self, pairs: List[Tuple[str, str]],
                batch_size: int = 64) -> np.ndarray:
        """
        Score (query, doc) pairs. Higher score = more relevant.
        """
        scores = self.model.predict(pairs, batch_size=batch_size)
        return scores

    def predict_single(self, query: str, doc_text: str,
                       doc_sharpe: float, doc_drawdown: float) -> float:
        """Score a single (query, doc) pair."""
        pair = self._format_pair(query, doc_text, doc_sharpe, doc_drawdown)
        return self.model.predict([pair])[0]


# ------------------------------------------------------------------ #
# 4. Projection Head (for InfoNCE training of Bi-encoders)
# ------------------------------------------------------------------ #

class ProjectionHead(nn.Module):
    """
    Trainable projection head for InfoNCE contrastive learning (L06/L08).
    Maps encoder output to a space where InfoNCE loss is applied.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
# 5. Combined Bi-encoder with projection (for training)
# ------------------------------------------------------------------ #

class TrainableTextEncoder(nn.Module):
    """
    Wraps the sentence-transformer model with a projection head
    for InfoNCE / Margin-MSE training.

    The base transformer is frozen; only the projection head is trained.
    This is efficient and prevents catastrophic forgetting.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim: int = 128):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.transformer = self.encoder[0].auto_model  # get the base transformer
        self.proj = ProjectionHead(input_dim=self.encoder.get_sentence_embedding_dimension(),
                                   output_dim=proj_dim)
        # Freeze transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask):
        """Get base embeddings from frozen transformer."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_emb = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_emb = sum_emb / sum_mask
        return mean_emb

    def forward(self, input_ids, attention_mask):
        """Project embeddings through trainable head."""
        emb = self.get_embeddings(input_ids, attention_mask)
        proj_emb = self.proj(emb)
        return F.normalize(proj_emb, p=2, dim=1)


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Testing Models ===\n")

    # Test CurveEncoder
    print("[Test] CurveEncoder")
    curve_enc = CurveEncoder(input_len=252, output_dim=128)
    dummy_curves = torch.randn(4, 252)
    out = curve_enc(dummy_curves)
    print(f"  Input:  {dummy_curves.shape}")
    print(f"  Output: {out.shape}")
    print(f"  L2 norm: {torch.norm(out[0]):.4f} (should be ~1.0)")
    print()

    # Test TextBiEncoder
    print("[Test] TextBiEncoder (Scout)")
    scout = TextBiEncoder()
    texts = ["трендовая стратегия пробой SMA 50", "скальпинг RSI на M5"]
    emb = scout.encode(texts)
    print(f"  Encoded {len(texts)} texts -> {emb.shape}")
    cos_sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print()

    # Test CrossEncoderJudge
    print("[Test] CrossEncoderJudge (Judge/Teacher)")
    judge = CrossEncoderJudge()
    score = judge.predict_single(
        query="трендовая стратегия",
        doc_text="Пробой SMA 50 с подтверждением объёма",
        doc_sharpe=1.5,
        doc_drawdown=-10.0,
    )
    print(f"  Relevance score: {score:.4f}")
    print()

    print("=== All model tests passed ===")