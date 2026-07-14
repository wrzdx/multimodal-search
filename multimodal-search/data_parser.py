"""
data_parser.py — Шаг 1: Имитация реального парсинга и очистка данных.

Course: Deep Learning for Search
Lecture refs: L02 (Inverted Index), L03 (Lexical Gap), L11 (Agentic CRAG)

This module:
  1. Generates a dirty dump (raw_strategies.jsonl) simulating a real forum parser
     with HTML junk, missing values, malformed strings.
  2. Cleans the text (BeautifulSoup), parses metrics (regex), reconstructs
     missing equity curves via Geometric Brownian Motion calibrated to
     total_return and max_drawdown.
  3. Saves clean dataset to clean_strategies.parquet.
"""

import json
import re
import random
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- Constants ----------
_SCRIPT_DIR = Path(__file__).parent
RAW_PATH = _SCRIPT_DIR / "raw_strategies.jsonl"
CLEAN_PATH = _SCRIPT_DIR / "clean_strategies.parquet"
N_RECORDS = 5000
CURVE_LEN = 252  # ~1 year of daily data

# ------------------------------------------------------------------ #
# 1. DATA GENERATOR: create_raw_dump
# ------------------------------------------------------------------ #

STRATEGY_TEMPLATES = [
    "Пробой {indicator} {period} с подтверждением {confirmation}",
    "Возврат к {indicator} {period}, тейк-профит {tp}%",
    "Трендовая стратегия на {timeframe} по {indicator} и {indicator2}",
    "Mean-reversion на {indicator} {period}, стоп-лосс {sl}%",
    "Мартингейл с фильтром {indicator} {period}, риск {risk}%",
    "Скальпинг по {indicator} на {timeframe}, цель {tp} п.п.",
    "Сетка ордеров вокруг {indicator} {period}, шаг {step}%",
    "Momentum breakout: {indicator} {period} + volume spike",
    "Парный трейдинг {asset1}/{asset2} по {indicator} {period}",
    "Канальная стратегия {indicator} {period}, выход на границе канала",
]

INDICATORS = [
    "SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR",
    "Stochastic", "CCI", "Williams %R", "ADX", "Ichimoku",
    "VWAP", "Parabolic SAR", "Donchian Channel", "Keltner Channel",
]

CONFIRMATIONS = [
    "объёмом", "RSI дивергенцией", "MACD кроссовером", "пробоем ATR",
    "возвратом к средней", "уровнем поддержки", "фигурой теханализа",
    "дивергенцией на старшем таймфрейме", "нет", "трендовым фильтром",
]

TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1", "W1"]
ASSETS = ["EUR/USD", "BTC/USD", "S&P 500", "AAPL", "Gold", "Oil", "ETH/USD", "GBP/JPY"]


def _pick(lst):
    return random.choice(lst)


def _generate_strategy_text() -> str:
    """Generate a realistic strategy description with random HTML junk."""
    template = _pick(STRATEGY_TEMPLATES)
    text = template.format(
        indicator=_pick(INDICATORS),
        indicator2=_pick(INDICATORS),
        period=random.choice([10, 14, 20, 21, 50, 100, 200]),
        confirmation=_pick(CONFIRMATIONS),
        tp=round(random.uniform(0.5, 10.0), 1),
        sl=round(random.uniform(0.5, 8.0), 1),
        risk=round(random.uniform(0.5, 5.0), 1),
        timeframe=_pick(TIMEFRAMES),
        step=round(random.uniform(0.1, 2.0), 2),
        asset1=_pick(ASSETS),
        asset2=_pick(ASSETS),
    )

    # Inject random HTML junk to simulate real parser output (Lexical Gap source)
    html_injections = [
        '<b>{}</b>', '<br/>', '<span class="highlight">{}</span>',
        '&nbsp;', '<div align="center">{}</div>', '<i>{}</i>',
        '\n', '\t', '&amp;', '&lt;', '&gt;',
        '<p style="color:red">{}</p>', '<a href="/strategy/123">{}</a>',
    ]
    # Wrap some words in HTML
    words = text.split()
    for i in range(0, len(words), random.randint(2, 5)):
        if i < len(words):
            injection = random.choice(html_injections)
            words[i] = injection.format(words[i])
    return " ".join(words)


def _generate_metrics_str() -> str:
    """Generate a metrics string — sometimes with typos, sometimes empty."""
    roll = random.random()
    if roll < 0.05:
        return ""  # completely empty
    elif roll < 0.15:
        # Partial / broken
        parts = []
        if random.random() < 0.5:
            parts.append(f"Sharpe: {round(random.uniform(-1.0, 3.0), 2)}")
        if random.random() < 0.3:
            parts.append(f"Drwdown: {round(random.uniform(-50, -2), 1)}%")  # typo
        return " | ".join(parts)
    else:
        sharpe = round(random.uniform(-1.0, 3.0), 2)
        drawdown = round(random.uniform(-50, -2), 1)
        ret = round(random.uniform(-30, 80), 1)
        # Occasional typos
        s_label = "Sharpe" if random.random() > 0.05 else "Sharpr"
        d_label = "Drawdown" if random.random() > 0.08 else "Drwdown"
        r_label = "Return" if random.random() > 0.05 else "Retrn"
        return f"{s_label}: {sharpe} | {d_label}: {drawdown}% | {r_label}: {ret}%"


def _generate_curve_str(known_return: float, known_drawdown: float) -> str:
    """
    Generate an equity curve string.
    70% — valid JSON array of floats
    20% — empty string (curve lost)
    10% — broken string
    """
    roll = random.random()
    if roll < 0.70:
        # Generate a realistic-looking equity curve
        curve = _simulate_equity_curve(known_return, known_drawdown)
        arr_str = "[" + ", ".join(f"{v:.4f}" for v in curve) + "]"
        # Occasionally add trailing noise
        if random.random() < 0.1:
            arr_str = arr_str.rstrip("]") + ", " + str(random.random()) + "]"
        return arr_str
    elif roll < 0.90:
        return ""  # empty — curve lost
    else:
        # Broken: missing bracket, non-numeric, truncated
        broken = [
            "[100.0, 101.5, 99.2",  # missing bracket
            "100.0, 101.5, NaN, 99.2]",  # non-numeric
            "[100.0; 101.5; 99.2]",  # wrong separator
            "[]",  # empty array
            "[100.0, 101.5, " + "..." * random.randint(1, 3),  # truncated
        ]
        return random.choice(broken)


def _simulate_equity_curve(total_return: float, max_drawdown: float) -> np.ndarray:
    """
    Simulate an equity curve using Geometric Brownian Motion (GBM),
    then calibrate drift to match total_return and scale volatility
    to approximate max_drawdown.

    GBM: dS = mu * S * dt + sigma * S * dW
    Discrete: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Fast path: uses a closed-form heuristic for sigma calibration
    instead of expensive Monte Carlo binary search.
    """
    dt = 1.0 / 252
    n_steps = CURVE_LEN
    S0 = 100.0

    # Target final value
    S_target = S0 * (1 + total_return / 100.0)

    # Calibrate mu to achieve target return
    # E[S_T] = S0 * exp(mu * T) => mu = log(S_target/S0) / T
    T = n_steps * dt
    mu = np.log(max(S_target, 1.0) / S0) / T

    # Calibrate sigma using a fast heuristic:
    # For GBM, expected max drawdown ~ sigma * sqrt(T) * C where C ≈ 1.5 (empirical)
    abs_dd = abs(max_drawdown) / 100.0
    sigma = abs_dd / (1.5 * np.sqrt(T))
    sigma = np.clip(sigma, 0.01, 2.0)

    # Generate curve
    Z = np.random.randn(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    S = S0 * np.cumprod(np.exp(log_returns))
    S = np.insert(S, 0, S0)  # prepend initial value

    return S[1:]  # return CURVE_LEN points


def create_raw_dump(output_path: Path = RAW_PATH, n: int = N_RECORDS):
    """
    Generate a dirty JSONL dump simulating a real forum parser output.
    Each line is a JSON object with fields: text, metrics_str, curve_str.
    """
    print(f"[data_parser] Generating {n} dirty records to {output_path}...")
    records = []
    for i in range(n):
        text = _generate_strategy_text()
        metrics_str = _generate_metrics_str()

        # Parse metrics to use for curve generation
        parsed = _quick_parse_metrics(metrics_str)
        total_ret = parsed.get("total_return", random.uniform(-20, 40))
        max_dd = parsed.get("max_drawdown", random.uniform(-30, -5))
        curve_str = _generate_curve_str(total_ret, max_dd)

        record = {
            "id": i,
            "text": text,
            "metrics_str": metrics_str,
            "curve_str": curve_str,
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[data_parser] Done. {len(records)} records written.")


def _quick_parse_metrics(metrics_str: str) -> dict:
    """Quick and dirty metrics parse — used internally during generation."""
    m = {}
    sharpe_m = re.search(r'[Ss]harp[er]*\s*:\s*([-+]?\d*\.?\d+)', metrics_str)
    dd_m = re.search(r'[Dd]rw?down\s*:\s*([-+]?\d*\.?\d+)%?', metrics_str)
    ret_m = re.search(r'[Rr]etr?n?\s*:\s*([-+]?\d*\.?\d+)%?', metrics_str)
    if sharpe_m:
        m["sharpe"] = float(sharpe_m.group(1))
    if dd_m:
        m["max_drawdown"] = float(dd_m.group(1))
    if ret_m:
        m["total_return"] = float(ret_m.group(1))
    return m


# ------------------------------------------------------------------ #
# 2. CLEANING PIPELINE
# ------------------------------------------------------------------ #

def clean_text(raw_text: str) -> str:
    """
    Remove HTML tags and entities using BeautifulSoup.
    This addresses the surface-level noise; deeper Lexical Gap (L03)
    is handled by the dense encoders (Bi-encoder / Cross-encoder).
    """
    soup = BeautifulSoup(raw_text, "lxml")
    clean = soup.get_text(separator=" ", strip=True)
    # Collapse multiple spaces
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Remove HTML entities leftovers
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return clean


def parse_metrics(metrics_str: str) -> dict:
    """
    Parse the messy metrics string into structured dict.
    Uses regex with fuzzy label matching to handle typos.

    Returns: {"sharpe": float, "max_drawdown": float, "total_return": float}
    """
    defaults = {"sharpe": np.nan, "max_drawdown": np.nan, "total_return": np.nan}

    if not metrics_str or not metrics_str.strip():
        return defaults

    # Sharpe — match "Sharpe", "Sharpr", "sharpe", etc.
    m = re.search(r'[Ss]harp[er]*\s*:\s*([-+]?\d*\.?\d+)', metrics_str)
    if m:
        try:
            defaults["sharpe"] = float(m.group(1))
        except ValueError:
            pass

    # Drawdown — match "Drawdown", "Drwdown", "drawdown", etc.
    # [Dd]r matches "Dr" or "dr", \w* handles optional chars like 'a' in "Drawdown" or missing 'a' in "Drwdown"
    m = re.search(r'[Dd]r\w*down\s*:\s*([-+]?\d*\.?\d+)%?', metrics_str)
    if m:
        try:
            defaults["max_drawdown"] = float(m.group(1))
        except ValueError:
            pass

    # Return — match "Return", "Retrn", "return", "Total Return", etc.
    # [Rr]et\w*\s*:\s* handles "Return:", "Retrn:", "Total Return:", etc.
    m = re.search(r'(?:(?:[Tt]otal\s+)?[Rr]et\w*)\s*:\s*([-+]?\d*\.?\d+)', metrics_str)
    if m:
        try:
            defaults["total_return"] = float(m.group(1))
        except (ValueError, IndexError):
            pass

    return defaults


def parse_or_reconstruct_curve(
    curve_str: str,
    total_return: float,
    max_drawdown: float,
    curve_len: int = CURVE_LEN,
) -> np.ndarray:
    """
    Parse curve_str into numpy array.
    If empty or broken -> RECONSTRUCT via GBM simulator (Lecture: data imputation).

    CRITICAL: We do NOT discard strategies with missing curves.
    Instead we reconstruct them using a Geometric Brownian Motion simulator
    calibrated so that the final total_return and max_drawdown match
    the parsed metrics. This is a key engineering decision for robust search.
    """
    if curve_str and curve_str.strip() and curve_str.strip() != "[]":
        try:
            # Try to parse as JSON
            curve = json.loads(curve_str)
            if isinstance(curve, list) and len(curve) >= 10:
                arr = np.array(curve, dtype=np.float64)
                # Filter out NaN/Inf
                valid_mask = np.isfinite(arr)
                if np.sum(valid_mask) > 10:
                    arr = arr[valid_mask]
                    # Pad or truncate to CURVE_LEN
                    if len(arr) < curve_len:
                        # Interpolate to fill
                        x_old = np.linspace(0, 1, len(arr))
                        x_new = np.linspace(0, 1, curve_len)
                        arr = np.interp(x_new, x_old, arr)
                    elif len(arr) > curve_len:
                        # Downsample evenly
                        idx = np.linspace(0, len(arr) - 1, curve_len).astype(int)
                        arr = arr[idx]
                    return arr
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # fall through to reconstruction

    # RECONSTRUCTION via GBM
    return _simulate_equity_curve(total_return, max_drawdown)


def clean_dataset(
    input_path: Path = RAW_PATH,
    output_path: Path = CLEAN_PATH,
    n: int = N_RECORDS,
) -> pd.DataFrame:
    """
    Full cleaning pipeline: read raw JSONL -> clean text -> parse metrics ->
    reconstruct curves -> save to Parquet.
    """
    print(f"[data_parser] Reading dirty data from {input_path}...")

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            rows.append(rec)

    df_raw = pd.DataFrame(rows)
    print(f"[data_parser] Loaded {len(df_raw)} raw records.")

    # --- Clean text (remove HTML junk) ---
    print("[data_parser] Cleaning text (BeautifulSoup)...")
    df_raw["clean_text"] = df_raw["text"].apply(clean_text)

    # --- Parse metrics ---
    print("[data_parser] Parsing metrics (regex)...")
    metrics_df = df_raw["metrics_str"].apply(parse_metrics).apply(pd.Series)
    df_raw = pd.concat([df_raw, metrics_df], axis=1)

    # --- Fill missing metrics with median or random values ---
    for col in ["sharpe", "max_drawdown", "total_return"]:
        median_val = df_raw[col].median()
        if pd.isna(median_val):
            if col == "sharpe":
                median_val = 1.0
            elif col == "max_drawdown":
                median_val = -15.0
            else:
                median_val = 20.0
        count_nan = df_raw[col].isna().sum()
        if count_nan > 0:
            print(f"[data_parser]  {col}: {count_nan} NaN values -> filling with median ({median_val:.2f})")
            df_raw[col] = df_raw[col].fillna(median_val)

    # --- Parse / reconstruct equity curves ---
    print("[data_parser] Parsing / reconstructing equity curves (GBM)...")
    curves = []
    n_reconstructed = 0
    for _, row in df_raw.iterrows():
        curve = parse_or_reconstruct_curve(
            row["curve_str"],
            row["total_return"],
            row["max_drawdown"],
        )
        # Check if we reconstructed (empty or broken input)
        if not row["curve_str"] or not row["curve_str"].strip():
            n_reconstructed += 1
        curves.append(curve.tolist())

    df_raw["equity_curve"] = curves
    print(f"[data_parser]  Reconstructed {n_reconstructed} missing curves via GBM.")

    # --- Build final clean DataFrame ---
    df_clean = df_raw[["id", "clean_text", "sharpe", "max_drawdown", "total_return", "equity_curve"]].copy()
    df_clean.rename(columns={"clean_text": "text"}, inplace=True)

    # Save
    df_clean.to_parquet(output_path, index=False)
    print(f"[data_parser] Clean dataset saved to {output_path}")
    print(f"[data_parser]  Shape: {df_clean.shape}")
    print(f"[data_parser]  Sharpe  — mean: {df_clean['sharpe'].mean():.2f}, std: {df_clean['sharpe'].std():.2f}")
    print(f"[data_parser]  Drawdown — mean: {df_clean['max_drawdown'].mean():.2f}")
    print(f"[data_parser]  Return   — mean: {df_clean['total_return'].mean():.2f}")

    return df_clean


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Генерация и очистка данных")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Пропустить генерацию синтетики (использовать raw_strategies.jsonl от real_parser.py)",
    )
    parser.add_argument("--n", type=int, default=N_RECORDS, help="Количество записей для генерации")
    args = parser.parse_args()

    if args.skip_generate:
        if not RAW_PATH.exists():
            print(f"[data_parser] {RAW_PATH} не найден. Сначала запустите:")
            print(f"  python real_parser.py")
            raise SystemExit(1)
        print(f"[data_parser] Пропуск генерации — используем реальный дамп из {RAW_PATH}")
    else:
        create_raw_dump(n=args.n)

    # Step 1b: Clean and save
    clean_dataset()
    print("[data_parser] All done.")