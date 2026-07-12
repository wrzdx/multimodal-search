"""
train.py — Шаг 3 (продолжение): Margin-MSE Distillation Training Loop.

Course: Deep Learning for Search
Lecture refs:
  L06 (Bi-encoder / Scout), L07 (Cross-encoder / Judge),
  L08 (InfoNCE, Margin-MSE Distillation)

Training pipeline:
  1. Load clean data + pre-built sparse indices.
  2. For each training epoch:
     a. Form training pairs: (query, positive_doc, hard_negative_doc).
        Hard negatives: semantically similar text (from BM25/SPLADE) but
        with significantly different metrics.
     b. In-batch negatives for InfoNCE loss on both Text and Curve Bi-encoders.
     c. Teacher (Cross-encoder Judge) scores the positive and negative pairs.
     d. Margin-MSE Distillation loss (L08):
        MSE((s_pos - s_neg)_student, (t_pos - t_neg)_teacher)
        where s = student (Bi-encoder) scores, t = teacher (Cross-encoder) scores.
  3. Train for 3 epochs.

Output: trained CurveEncoder weights (curve_encoder.pth) and
        trained TextBiEncoder projection head (text_proj_head.pth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import random

from models import CurveEncoder, TrainableTextEncoder, ProjectionHead

# Paths
_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
BM25_PATH = _SCRIPT_DIR / "bm25_index.pkl"
SPLADE_MATRIX_PATH = _SCRIPT_DIR / "splade_sparse_matrix.npz"
CURVE_ENCODER_PATH = _SCRIPT_DIR / "curve_encoder.pth"
TEXT_PROJ_PATH = _SCRIPT_DIR / "text_proj_head.pth"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3
INFO_NCE_TEMPERATURE = 0.07
EMBEDDING_DIM = 128
CURVE_LEN = 252
NUM_NEGATIVES = 4  # hard negatives per positive pair


# ------------------------------------------------------------------ #
# Training Dataset
# ------------------------------------------------------------------ #

class TripletDataset(Dataset):
    """
    Dataset for Margin-MSE distillation.
    Each sample: (query_text, pos_text, pos_curve, pos_sharpe, pos_drawdown,
                  neg_text, neg_curve, neg_sharpe, neg_drawdown)
    """

    def __init__(self, df: pd.DataFrame, bm25_index, splade_index=None):
        self.df = df
        self.texts = df["text"].tolist()
        self.curves = np.array(df["equity_curve"].tolist(), dtype=np.float32)
        self.sharpes = df["sharpe"].values
        self.drawdowns = df["max_drawdown"].values
        self.ids = df["id"].values
        self.bm25 = bm25_index
        self.n = len(df)

    def __len__(self):
        return self.n

    def _find_hard_negative(self, query_idx: int) -> int:
        """
        Find a hard negative: semantically similar (from BM25) but with
        significantly different metrics (Sharpe or Drawdown).

        This mimics the hard negative mining strategy from L08.
        """
        query_text = self.texts[query_idx]
        query_sharpe = self.sharpes[query_idx]
        query_dd = self.drawdowns[query_idx]

        # Get BM25 top-20 candidates
        ids, _ = self.bm25.search(query_text, top_k=20)

        # Filter for candidates with different metrics
        for doc_id in ids:
            if int(doc_id) == query_idx:
                continue
            doc_sharpe = self.sharpes[doc_id]
            doc_dd = self.drawdowns[doc_id]
            # "Different metrics" = Sharpe differs by > 0.5 or Drawdown by > 10%
            if abs(doc_sharpe - query_sharpe) > 0.5 or abs(doc_dd - query_dd) > 10:
                return int(doc_id)

        # Fallback: random document with different metrics
        for _ in range(50):
            neg_idx = random.randint(0, self.n - 1)
            if neg_idx != query_idx:
                if abs(self.sharpes[neg_idx] - query_sharpe) > 0.3:
                    return neg_idx
        return (query_idx + 1) % self.n

    def __getitem__(self, idx):
        neg_idx = self._find_hard_negative(idx)
        return {
            "query_text": self.texts[idx],
            "pos_text": self.texts[idx],  # In our setting, query IS the doc
            "pos_curve": self.curves[idx],
            "pos_sharpe": self.sharpes[idx],
            "pos_drawdown": self.drawdowns[idx],
            "neg_text": self.texts[neg_idx],
            "neg_curve": self.curves[neg_idx],
            "neg_sharpe": self.sharpes[neg_idx],
            "neg_drawdown": self.drawdowns[neg_idx],
        }


def collate_fn(batch):
    """Custom collate for variable-length text + fixed-size curves."""
    query_texts = [item["query_text"] for item in batch]
    pos_texts = [item["pos_text"] for item in batch]
    neg_texts = [item["neg_text"] for item in batch]

    pos_curves = torch.tensor([item["pos_curve"] for item in batch], dtype=torch.float32)
    neg_curves = torch.tensor([item["neg_curve"] for item in batch], dtype=torch.float32)

    pos_sharpes = torch.tensor([item["pos_sharpe"] for item in batch], dtype=torch.float32)
    pos_drawdowns = torch.tensor([item["pos_drawdown"] for item in batch], dtype=torch.float32)
    neg_sharpes = torch.tensor([item["neg_sharpe"] for item in batch], dtype=torch.float32)
    neg_drawdowns = torch.tensor([item["neg_drawdown"] for item in batch], dtype=torch.float32)

    return {
        "query_texts": query_texts,
        "pos_texts": pos_texts,
        "neg_texts": neg_texts,
        "pos_curves": pos_curves,
        "neg_curves": neg_curves,
        "pos_sharpes": pos_sharpes,
        "pos_drawdowns": pos_drawdowns,
        "neg_sharpes": neg_sharpes,
        "neg_drawdowns": neg_drawdowns,
    }


# ------------------------------------------------------------------ #
# Margin-MSE Loss (L08)
# ------------------------------------------------------------------ #

def margin_mse_loss(student_pos_scores: torch.Tensor,
                    student_neg_scores: torch.Tensor,
                    teacher_pos_scores: torch.Tensor,
                    teacher_neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Margin-MSE Distillation Loss (L08):

    L = MSE( (s_pos - s_neg), (t_pos - t_neg) )

    Where:
      s_pos, s_neg = Student (Bi-encoder) cosine similarity scores
      t_pos, t_neg = Teacher (Cross-encoder) relevance scores

    The idea: the Student should learn to reproduce the Teacher's
    margin between positive and negative pairs.
    """
    student_margins = student_pos_scores - student_neg_scores  # (batch,)
    teacher_margins = teacher_pos_scores - teacher_neg_scores  # (batch,)

    loss = F.mse_loss(student_margins, teacher_margins)
    return loss


def info_nce_loss(embeddings: torch.Tensor, temperature: float = INFO_NCE_TEMPERATURE) -> torch.Tensor:
    """
    InfoNCE contrastive loss (L06/L08).

    For a batch of N embeddings, each embedding is a "query" and
    all other embeddings in the batch serve as negatives.
    The diagonal (self-similarity) is the positive pair.

    L = -log(exp(sim(q,k+)/τ) / Σ_j exp(sim(q,k_j)/τ))

    In our case, for each anchor i, the positive is i itself (identity),
    and negatives are all other items in the batch.

    Args:
        embeddings: (batch, dim) — L2-normalized
        temperature: τ scaling factor

    Returns:
        scalar InfoNCE loss
    """
    # Cosine similarity matrix (since embeddings are L2-normalized, dot product = cos)
    sim_matrix = embeddings @ embeddings.T  # (batch, batch)

    # Scale by temperature
    sim_matrix = sim_matrix / temperature

    # Labels: diagonal (each item matches itself)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Cross-entropy loss (equivalent to InfoNCE with in-batch negatives)
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


# ------------------------------------------------------------------ #
# Training Loop
# ------------------------------------------------------------------ #

def train(distill: bool = True,
          curve_encoder_path: Path = CURVE_ENCODER_PATH,
          text_proj_path: Path = TEXT_PROJ_PATH,
          epochs: int = EPOCHS):
    """
    Main training function.

    1. Train CurveEncoder with InfoNCE (in-batch negatives).
    2. Train Text Bi-encoder projection with InfoNCE + Margin-MSE Distillation.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Load data
    print("[train] Loading clean data...")
    df = pd.read_parquet(CLEAN_PATH)

    # Load BM25 for hard negative mining
    print("[train] Loading BM25 index for hard negative mining...")
    import pickle
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_index = bm25_data["bm25"]

    # Create dataset
    print("[train] Creating training dataset...")
    dataset = TripletDataset(df, bm25_index)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---- Initialize models ----
    curve_encoder = CurveEncoder(input_len=CURVE_LEN, output_dim=EMBEDDING_DIM).to(device)
    text_encoder = TrainableTextEncoder(proj_dim=EMBEDDING_DIM).to(device)

    # Optimizers (only train projection heads / curve encoder)
    curve_optimizer = torch.optim.Adam(curve_encoder.parameters(), lr=LEARNING_RATE)
    text_optimizer = torch.optim.Adam(text_encoder.proj.parameters(), lr=LEARNING_RATE)

    # Load Cross-encoder Teacher if doing distillation
    teacher = None
    if distill:
        print("[train] Loading Cross-encoder Teacher (Judge) for Margin-MSE...")
        from sentence_transformers import CrossEncoder as CE
        teacher = CE("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Tokenizer for text encoder
    tokenizer = text_encoder.encoder.tokenizer

    # ---- Training ----
    print(f"\n[train] Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        curve_encoder.train()
        text_encoder.proj.train()

        total_info_nce_curve = 0.0
        total_info_nce_text = 0.0
        total_margin_mse = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            pos_curves = batch["pos_curves"].to(device)
            neg_curves = batch["neg_curves"].to(device)

            # ---- Step 1: InfoNCE on Curve Encoder ----
            curve_optimizer.zero_grad()

            # Encode all curves in batch (positive + negative) as in-batch negatives
            all_curves = torch.cat([pos_curves, neg_curves], dim=0)  # (2*batch, 252)
            curve_embeddings = curve_encoder(all_curves)  # (2*batch, 128)

            # InfoNCE: each curve's embedding should be close to itself
            loss_curve = info_nce_loss(curve_embeddings, temperature=INFO_NCE_TEMPERATURE)
            loss_curve.backward()
            curve_optimizer.step()

            # ---- Step 2: InfoNCE + Margin-MSE on Text Bi-encoder ----
            text_optimizer.zero_grad()

            # Tokenize all texts (query + pos + neg)
            all_texts = batch["query_texts"] + batch["pos_texts"] + batch["neg_texts"]
            encoded = tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            text_embeddings = text_encoder(input_ids, attention_mask)  # (3*batch, 128)

            # InfoNCE on text embeddings
            loss_text = info_nce_loss(text_embeddings, temperature=INFO_NCE_TEMPERATURE)

            # ---- Step 3: Margin-MSE Distillation ----
            loss_mse = torch.tensor(0.0, device=device)
            if teacher is not None:
                # Get Teacher scores for (query, pos) and (query, neg) pairs
                teacher_pairs = []
                for i in range(len(batch["query_texts"])):
                    # Positive pair
                    pos_metrics = f"Metrics: Sharpe {batch['pos_sharpes'][i]:.2f}, Drawdown {batch['pos_drawdowns'][i]:.1f}%"
                    pos_doc = f"{batch['pos_texts'][i]}. {pos_metrics}"
                    teacher_pairs.append((batch["query_texts"][i], pos_doc))

                    # Negative pair
                    neg_metrics = f"Metrics: Sharpe {batch['neg_sharpes'][i]:.2f}, Drawdown {batch['neg_drawdowns'][i]:.1f}%"
                    neg_doc = f"{batch['neg_texts'][i]}. {neg_metrics}"
                    teacher_pairs.append((batch["query_texts"][i], neg_doc))

                with torch.no_grad():
                    teacher_scores = teacher.predict(teacher_pairs, batch_size=len(teacher_pairs))
                    teacher_pos_scores = torch.tensor(
                        teacher_scores[0::2], dtype=torch.float32, device=device
                    )
                    teacher_neg_scores = torch.tensor(
                        teacher_scores[1::2], dtype=torch.float32, device=device
                    )

                # Student scores: cosine similarity between query and pos/neg embeddings
                batch_size = len(batch["query_texts"])
                query_embs = text_embeddings[:batch_size]        # (batch, 128)
                pos_embs = text_embeddings[batch_size:2*batch_size]   # (batch, 128)
                neg_embs = text_embeddings[2*batch_size:]        # (batch, 128)

                student_pos_scores = (query_embs * pos_embs).sum(dim=1)  # (batch,)
                student_neg_scores = (query_embs * neg_embs).sum(dim=1)  # (batch,)

                loss_mse = margin_mse_loss(
                    student_pos_scores, student_neg_scores,
                    teacher_pos_scores, teacher_neg_scores,
                )

            # Combined text loss
            total_text_loss = loss_text + 0.5 * loss_mse
            total_text_loss.backward()
            text_optimizer.step()

            # Log
            total_info_nce_curve += loss_curve.item()
            total_info_nce_text += loss_text.item()
            total_margin_mse += loss_mse.item()
            n_batches += 1

            pbar.set_postfix({
                "CE": f"{loss_curve.item():.4f}",
                "TE": f"{loss_text.item():.4f}",
                "MSE": f"{loss_mse.item():.4f}",
            })

        print(f"\n  Epoch {epoch + 1} averages:")
        print(f"    InfoNCE (Curve): {total_info_nce_curve / n_batches:.4f}")
        print(f"    InfoNCE (Text):  {total_info_nce_text / n_batches:.4f}")
        print(f"    Margin-MSE:      {total_margin_mse / n_batches:.4f}")

    # ---- Save models ----
    torch.save(curve_encoder.state_dict(), str(curve_encoder_path))
    torch.save(text_encoder.proj.state_dict(), str(text_proj_path))
    print(f"\n[train] Curve encoder saved to {curve_encoder_path}")
    print(f"[train] Text projection head saved to {text_proj_path}")

    return curve_encoder, text_encoder


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    curve_enc, text_enc = train(distill=True, epochs=EPOCHS)
    print("\n[train] All done.")