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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from models import CurveEncoder, TrainableTextEncoder
import pandas as pd

# Paths
_SCRIPT_DIR = Path(__file__).parent
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
BM25_PATH = _SCRIPT_DIR / "bm25_index.pkl"
SPLADE_MATRIX_PATH = _SCRIPT_DIR / "splade_sparse_matrix.npz"
CURVE_ENCODER_PATH = _SCRIPT_DIR / "curve_encoder.pth"
TEXT_PROJ_PATH = _SCRIPT_DIR / "text_proj_head.pth"

# Hyperparameters
BATCH_SIZE = 64  # increased from 32 — AMP allows larger batches
EPOCHS = 3
LEARNING_RATE = 2e-3  # increased for faster convergence with larger batches
INFO_NCE_TEMPERATURE = 0.07
EMBEDDING_DIM = 128
CURVE_LEN = 252
NUM_NEGATIVES = 4  # hard negatives per positive pair
GRAD_ACCUM_STEPS = 2  # gradient accumulation steps (effective batch = 64*2 = 128)
WARMUP_STEPS = 100  # linear warmup steps
USE_AMP = True  # Automatic Mixed Precision for GPU speedup (~1.5-2x)


# ------------------------------------------------------------------ #
# Training Dataset
# ------------------------------------------------------------------ #

class TripletDataset(Dataset):
    """
    Dataset for Margin-MSE distillation.
    Each sample: (query_text, pos_text, pos_curve, pos_sharpe, pos_drawdown,
                  neg_text, neg_curve, neg_sharpe, neg_drawdown)

    Hard negatives are pre-computed using vectorized metric-based sampling
    (no per-document BM25 search — that was O(n²) and took 600+ hours for 205K docs).
    Results are cached to disk so this only runs once.
    """

    _CACHE_PATH = _SCRIPT_DIR / "hard_negatives_cache.npy"

    def __init__(self, df: pd.DataFrame, bm25_index=None, splade_index=None):
        self.texts = df["text"].tolist()
        self.curves = np.array(df["equity_curve"].tolist(), dtype=np.float32)
        self.sharpes = df["sharpe"].values.astype(np.float32)
        self.drawdowns = df["max_drawdown"].values.astype(np.float32)
        self.n = len(df)

        # Try loading cached hard negatives
        if TripletDataset._CACHE_PATH.exists():
            cached = np.load(str(TripletDataset._CACHE_PATH))
            if cached.shape == (self.n,):
                print(f"[Dataset] Loaded cached hard negatives ({self.n:,} docs)")
                self.neg_indices = cached
                return

        # Vectorized hard negative mining using metric differences (~3 seconds for 205K)
        print(f"[Dataset] Pre-computing hard negatives for {self.n:,} docs (vectorized)...")
        t0 = time.time()

        # Sort by sharpe to create "metric buckets"
        sharpe_order = np.argsort(self.sharpes)
        # Map from sorted position back to original index
        rank = np.empty(self.n, dtype=np.int64)
        rank[sharpe_order] = np.arange(self.n, dtype=np.int64)

        self.neg_indices = np.empty(self.n, dtype=np.int64)
        rng = np.random.default_rng(42)

        # For each doc, pick a negative from a DIFFERENT sharpe bucket
        # Offset: at least 20% away in the sorted order
        min_offset = max(1, self.n // 5)

        for i in range(self.n):
            r = rank[i]
            # Pick random offset far away in sharpe ranking
            for _ in range(10):
                offset = rng.integers(min_offset, self.n)
                # Pick direction: above or below in sharpe ranking
                target_rank = (r + offset) % self.n
                candidate = sharpe_order[target_rank]
                if candidate != i:
                    self.neg_indices[i] = candidate
                    break
            else:
                self.neg_indices[i] = sharpe_order[(r + min_offset) % self.n]

        # Cache to disk
        np.save(str(TripletDataset._CACHE_PATH), self.neg_indices)
        elapsed = time.time() - t0
        print(f"[Dataset] Done in {elapsed:.1f}s (cached to {TripletDataset._CACHE_PATH.name})")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        neg_idx = int(self.neg_indices[idx])
        return {
            "query_text": self.texts[idx],
            "pos_text": self.texts[idx],
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

    # Fast: numpy stack first, then single torch.tensor call
    pos_curves = torch.tensor(np.stack([item["pos_curve"] for item in batch]), dtype=torch.float32)
    neg_curves = torch.tensor(np.stack([item["neg_curve"] for item in batch]), dtype=torch.float32)

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

def _get_lr_lambda(current_step: int, warmup: int, total: int) -> float:
    """Linear warmup + linear decay learning rate schedule."""
    if current_step < warmup:
        return float(current_step + 1) / float(max(1, warmup))
    # Linear decay after warmup
    progress = (current_step - warmup) / max(1, total - warmup)
    return max(0.1, 1.0 - progress)


def train(distill: bool = True,
          curve_encoder_path: Path = CURVE_ENCODER_PATH,
          text_proj_path: Path = TEXT_PROJ_PATH,
          epochs: int = EPOCHS):
    """
    Main training function.

    1. Train CurveEncoder with InfoNCE (in-batch negatives).
    2. Train Text Bi-encoder projection with InfoNCE + Margin-MSE Distillation.

    Optimizations for real data (60k+ docs):
      - AMP (Automatic Mixed Precision) for GPU: ~1.5-2x speedup
      - Gradient accumulation for larger effective batch
      - Linear warmup + decay LR schedule
      - Larger batch size (64 vs 32)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"[train] Device: {device}")
    print(f"[train] AMP (Mixed Precision): {'ON' if use_amp else 'OFF'}")

    # Load data
    print("[train] Loading clean data...")
    df = pd.read_parquet(CLEAN_PATH)

    # Create dataset (BM25 no longer needed — vectorized metric-based sampling)
    print("[train] Creating training dataset...")
    dataset = TripletDataset(df)
    # Adaptive batch size: must be <= dataset size
    actual_batch = min(BATCH_SIZE, len(df))
    effective_batch = actual_batch * GRAD_ACCUM_STEPS
    print(f"[train]  Dataset: {len(df)} samples")
    print(f"[train]  Batch size: {actual_batch} × {GRAD_ACCUM_STEPS} accum = {effective_batch} effective")
    if len(df) < 4:
        print("[train] ERROR: Too few samples for training. Need at least 4.")
        print("[train] Generate real data first:")
        print("[train]   python real_parser.py           # 60k real docs")
        print("[train]   python real_parser.py --fast     # ~25k docs quickly")
        print("[train]   python data_parser.py --skip-generate")
        return None, None
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch,
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

    # Learning rate schedulers with warmup
    total_steps = len(dataloader) * epochs
    curve_scheduler = torch.optim.lr_scheduler.LambdaLR(
        curve_optimizer,
        lr_lambda=lambda s: _get_lr_lambda(s, WARMUP_STEPS, total_steps),
    )
    text_scheduler = torch.optim.lr_scheduler.LambdaLR(
        text_optimizer,
        lr_lambda=lambda s: _get_lr_lambda(s, WARMUP_STEPS, total_steps),
    )

    # AMP GradScaler for mixed precision
    # Use torch.cuda.amp.GradScaler for compatibility with PyTorch 2.0+
    if use_amp:
        try:
            scaler = torch.amp.GradScaler(device)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Load Cross-encoder Teacher if doing distillation (GPU only — too slow on CPU)
    teacher = None
    if distill and device.type == "cuda":
        print("[train] Loading Cross-encoder Teacher (Judge) for Margin-MSE...")
        from sentence_transformers import CrossEncoder as CE
        teacher = CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
    elif distill and device.type == "cpu":
        print("[train] WARNING: Distillation disabled on CPU (too slow). Use --no-distill to suppress.")
        print("[train]   Training with InfoNCE only. For distillation, use a GPU.")
        distill = False

    # Tokenizer for text encoder
    tokenizer = text_encoder.encoder.tokenizer

    # ---- Training ----
    print(f"\n[train] Starting training for {epochs} epochs ({total_steps} total steps)...")
    global_step = 0
    epoch_start_time = time.time()

    for epoch in range(epochs):
        curve_encoder.train()
        text_encoder.proj.train()

        total_info_nce_curve = 0.0
        total_info_nce_text = 0.0
        total_margin_mse = 0.0
        n_batches = 0

        # Gradient accumulation buffers
        accum_curve_loss = 0.0
        accum_text_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            pos_curves = batch["pos_curves"].to(device)
            neg_curves = batch["neg_curves"].to(device)

            # ---- Step 1: InfoNCE on Curve Encoder ----
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                all_curves = torch.cat([pos_curves, neg_curves], dim=0)
                curve_embeddings = curve_encoder(all_curves)
                loss_curve = info_nce_loss(curve_embeddings, temperature=INFO_NCE_TEMPERATURE)
                loss_curve_scaled = loss_curve / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss_curve_scaled).backward()
            else:
                loss_curve_scaled.backward()

            # ---- Step 2: InfoNCE + Margin-MSE on Text Bi-encoder ----
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
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

                text_embeddings = text_encoder(input_ids, attention_mask)

                loss_text = info_nce_loss(text_embeddings, temperature=INFO_NCE_TEMPERATURE)

                # ---- Step 3: Margin-MSE Distillation ----
                loss_mse = torch.tensor(0.0, device=device)
                if teacher is not None:
                    teacher_pairs = []
                    for i in range(len(batch["query_texts"])):
                        pos_metrics = f"Metrics: Sharpe {batch['pos_sharpes'][i]:.2f}, Drawdown {batch['pos_drawdowns'][i]:.1f}%"
                        pos_doc = f"{batch['pos_texts'][i]}. {pos_metrics}"
                        teacher_pairs.append((batch["query_texts"][i], pos_doc))

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

                    batch_size = len(batch["query_texts"])
                    query_embs = text_embeddings[:batch_size]
                    pos_embs = text_embeddings[batch_size:2*batch_size]
                    neg_embs = text_embeddings[2*batch_size:]

                    student_pos_scores = (query_embs * pos_embs).sum(dim=1)
                    student_neg_scores = (query_embs * neg_embs).sum(dim=1)

                    loss_mse = margin_mse_loss(
                        student_pos_scores, student_neg_scores,
                        teacher_pos_scores, teacher_neg_scores,
                    )

                total_text_loss = loss_text + 0.5 * loss_mse
                total_text_loss_scaled = total_text_loss / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(total_text_loss_scaled).backward()
            else:
                total_text_loss_scaled.backward()

            # ---- Gradient accumulation step ----
            accum_step = (batch_idx + 1) % GRAD_ACCUM_STEPS
            if accum_step == 0 or batch_idx == len(dataloader) - 1:
                if scaler is not None:
                    scaler.unscale_(curve_optimizer)
                    scaler.unscale_(text_optimizer)
                    torch.nn.utils.clip_grad_norm_(curve_encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(text_encoder.proj.parameters(), 1.0)
                    scaler.step(curve_optimizer)
                    scaler.step(text_optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(curve_encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(text_encoder.proj.parameters(), 1.0)
                    curve_optimizer.step()
                    text_optimizer.step()

                curve_optimizer.zero_grad()
                text_optimizer.zero_grad()

                # Step schedulers AFTER optimizer.step()
                curve_scheduler.step()
                text_scheduler.step()
            global_step += 1

            # Log (unscaled losses)
            total_info_nce_curve += loss_curve.item()
            total_info_nce_text += loss_text.item()
            total_margin_mse += loss_mse.item()
            n_batches += 1

            pbar.set_postfix({
                "CE": f"{loss_curve.item():.4f}",
                "TE": f"{loss_text.item():.4f}",
                "MSE": f"{loss_mse.item():.4f}",
                "lr": f"{curve_scheduler.get_last_lr()[0]:.1e}",
            })

        epoch_time = time.time() - epoch_start_time
        epoch_start_time = time.time()
        print(f"\n  Epoch {epoch + 1} averages ({epoch_time:.1f}s):")
        print(f"    InfoNCE (Curve): {total_info_nce_curve / n_batches:.4f}")
        print(f"    InfoNCE (Text):  {total_info_nce_text / n_batches:.4f}")
        print(f"    Margin-MSE:      {total_margin_mse / n_batches:.4f}")
        print(f"    LR (curve):      {curve_scheduler.get_last_lr()[0]:.2e}")

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-distill", action="store_true", help="Disable Margin-MSE distillation (InfoNCE only)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    curve_enc, text_enc = train(distill=not args.no_distill, epochs=args.epochs)
    print("\n[train] All done.")