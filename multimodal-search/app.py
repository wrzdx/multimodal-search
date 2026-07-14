"""
app.py — Шаг 8: UI и Оценка Оракула (Streamlit).

Course: Deep Learning for Search
Lecture refs: L11 (Agentic CRAG, RAGAS / Faithfulness)

Streamlit application:
  1. Text input for strategy search query.
  2. Slider filters: Min Sharpe, Max Drawdown (Agentic CRAG triggers).
  3. Top-5 results: cleaned text, metrics table, equity curve plot (plotly).
  4. RAGAS Faithfulness Check (L11): Heuristic check — if text claims
     "high profit" / "low risk" but metrics contradict, show warning.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Optional

# ---- RAGAS Faithfulness (L11) ----

FAITHFULNESS_PATTERNS = {
    "high_profit": [
        r"высок[аоый].*прибыл", r"high\s*profit", r"больш[аоый].*доход",
        r"выгодн", r"эффективн", r"много.*заработ",
        r"тейк.*профит\s*\d{2,}", r"return.*\d{2,}",
    ],
    "low_risk": [
        r"низк[аоий].*риск", r"low\s*risk", r"безопасн",
        r"мал[аоый].*просад", r"стабильн", r"надёжн",
        r"conservative", r"drawdown.*-[0-5]%",
    ],
}


def ragas_faithfulness_check(text: str, sharpe: float, max_drawdown: float) -> Dict:
    """
    RAGAS-inspired Faithfulness heuristic (L11).

    Checks whether the strategy text's claims align with its actual metrics.
    This is a simplified version of the RAGAS Faithfulness metric.

    Faithfulness = (number of supported claims) / (total claims)

    In our heuristic:
      - If text mentions "high profit" but total_return < 10% -> unsupported claim
      - If text mentions "low risk" but max_drawdown < -25% -> unsupported claim

    Returns:
        dict with "score" (0.0-1.0), "issues" list
    """
    import re

    text_lower = text.lower()
    issues = []
    supported = 0
    total_claims = 0

    # Check high profit claims
    for pattern in FAITHFULNESS_PATTERNS["high_profit"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            total_claims += 1
            # "High profit" claim — check if Sharpe > 1.0
            if sharpe >= 1.0:
                supported += 1
            else:
                issues.append(f"Текст заявляет о высокой прибыли, но Sharpe = {sharpe:.2f}")

    # Check low risk claims
    for pattern in FAITHFULNESS_PATTERNS["low_risk"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            total_claims += 1
            # "Low risk" claim — check if drawdown > -15%
            if max_drawdown > -15.0:
                supported += 1
            else:
                issues.append(
                    f"Текст заявляет о низком риске, но Max Drawdown = {max_drawdown:.1f}%"
                )

    if total_claims == 0:
        return {"score": 1.0, "issues": [], "label": "OK"}

    score = supported / total_claims
    if score < 1.0:
        label = "Низкая согласованность (Low Faithfulness)"
    elif score < 0.75:
        label = "Средняя согласованность"
    else:
        label = "Высокая согласованность (High Faithfulness)"

    return {"score": score, "issues": issues, "label": label}


# ---- Plot equity curve ----

def plot_equity_curve(curve: list, title: str = "Equity Curve"):
    """Plot equity curve using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=curve,
        mode="lines",
        name="Equity",
        line=dict(color="#10b981", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Day",
        yaxis_title="Equity",
        height=200,
        margin=dict(l=40, r=20, t=30, b=30),
        template="plotly_white",
    )
    return fig


# ---- Main Streamlit App ----

def main():
    st.set_page_config(
        page_title="Мультимодальный поиск стратегий",
        page_icon="🔍",
        layout="wide",
    )

    # ---- Header ----
    st.title("🔍 Мультимодальный поиск стратегий")
    st.caption(
        "Альянс текста и временных рядов с обратной связью — "
        "BM25 + SPLADE + Dense Bi-encoder + Curve CNN → RRF → LambdaMART → Agentic CRAG"
    )

    # ---- Sidebar: Alliance Architecture ----
    with st.sidebar:
        st.header("Архитектура Альянса")
        st.markdown("""
        **Phase 1: Recall (RRF)**
        - BM25 (Inverted Index, L02)
        - SPLADE (Learned Sparse, L05)
        - Dense Text (Bi-encoder / Scout, L06)
        - Dense Curve (CNN Encoder, L06)
        → RRF Fusion (k=60)

        **Phase 2: Features (L09)**
        - 6 LTR features per candidate

        **Phase 3: Precision (L08)**
        - The Captain (LambdaMART)

        **Phase 4: Agentic CRAG (L11)**
        - Post-retrieval corrections
        - Slider-based feedback

        ---
        *Course: Deep Learning for Search*
        """)

    # ---- Query Input ----
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "📝 Введите описание стратегии:",
            value="трендовая стратегия пробой SMA 50 с подтверждением объёма",
            placeholder="Например: скальпинг RSI на M5...",
        )
    with col2:
        search_btn = st.button("🔍 Найти", type="primary", use_container_width=True)

    # ---- Filters (CRAG triggers, L11) ----
    col_a, col_b = st.columns(2)
    with col_a:
        min_sharpe = st.slider(
            "📈 Минимальный Sharpe Ratio",
            min_value=-1.0, max_value=3.0, value=0.0, step=0.1,
            help="Agentic CRAG (L11): отфильтрует стратегии с Sharpe ниже порога"
        )
    with col_b:
        max_drawdown = st.slider(
            "📉 Максимальный Drawdown (%)",
            min_value=-50.0, max_value=-1.0, value=-50.0, step=1.0,
            help="Agentic CRAG (L11): отфильтрует стратегии с Drawdown хуже порога"
        )

    # ---- Search ----
    if search_btn or query:
        with st.spinner("Alliance检索中... BM25 + SPLADE + Dense → RRF → Captain → CRAG"):
            try:
                from pipeline import AllianceRetriever

                # Initialize alliance (lazy, cached in session state)
                if "alliance" not in st.session_state:
                    st.session_state.alliance = AllianceRetriever()

                alliance = st.session_state.alliance

                results = alliance.search(
                    query,
                    top_k=5,
                    min_sharpe=min_sharpe if min_sharpe > 0.0 else None,
                    max_drawdown=max_drawdown if max_drawdown > -50.0 else None,
                )

                # Display results
                st.success(f"Найдено результатов: {len(results)}")

                if len(results) == 0:
                    st.warning("По вашему запросу ничего не найдено. Попробуйте ослабить фильтры.")
                else:
                    for _, row in results.iterrows():
                        rank = row["rank"]
                        doc_id = row["doc_id"]
                        text = row["text"]
                        sharpe = row["sharpe"]
                        dd = row["max_drawdown"]
                        ret = row["total_return"]
                        curve = row["equity_curve"]
                        captain_score = row["captain_score"]
                        crag_action = row.get("crag_action", "none")

                        # ---- Result Card ----
                        st.divider()
                        header_col1, header_col2 = st.columns([1, 5])

                        with header_col1:
                            st.markdown(f"### #{rank}")
                        with header_col2:
                            # CRAG badge
                            if crag_action != "none" and crag_action != "ok":
                                st.markdown(
                                    f"⚠️ **CRAG**: {crag_action}"
                                )

                        # Strategy text
                        st.markdown(f"**Стратегия (Doc {doc_id}):** {text}")

                        # Metrics row
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                        with met_col1:
                            st.metric("Sharpe", f"{sharpe:.2f}")
                        with met_col2:
                            st.metric("Max Drawdown", f"{dd:.1f}%",
                                      delta=f"{dd:.1f}%" if dd < -20 else None)
                        with met_col3:
                            st.metric("Total Return", f"{ret:.1f}%",
                                      delta=f"{ret:.1f}%" if ret > 0 else None,
                                      delta_color="normal")
                        with met_col4:
                            st.metric("Captain Score", f"{captain_score:.2f}")

                        # Equity curve plot
                        st.plotly_chart(
                            plot_equity_curve(curve, f"Equity Curve — Doc {doc_id}"),
                            use_container_width=True,
                        )

                        # ---- RAGAS Faithfulness Check (L11) ----
                        faithfulness = ragas_faithfulness_check(text, sharpe, dd)
                        if faithfulness["issues"]:
                            st.warning(
                                f"🚨 **{faithfulness['label']}** (score: {faithfulness['score']:.2f})\n\n"
                                + "\n".join(f"• {issue}" for issue in faithfulness["issues"])
                            )
                        else:
                            st.info(f"✅ {faithfulness['label']} (score: {faithfulness['score']:.2f})")

            except FileNotFoundError as e:
                st.error(f"❌ Необходимые файлы не найдены. Сначала запустите пайплайн:\n\n"
                         f"```bash\npython data_parser.py\npython sparse_index.py\n"
                         f"python train.py\npython dense_index.py\n```\n\n"
                         f"Ошибка: {e}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ---- Footer ----
    st.divider()
    st.caption(
        "Мультимодальный поиск стратегий | Deep Learning for Search | "
        "BM25 · SPLADE · Bi-encoder · Cross-encoder · InfoNCE · "
        "Margin-MSE · RRF · LambdaMART · Agentic CRAG · RAGAS/Faithfulness\n"
        "Evaluation: NDCG@k, MRR, Recall@k, Precision@k | "
        "Comparative Study: 8 approaches benchmarked"
    )


if __name__ == "__main__":
    main()