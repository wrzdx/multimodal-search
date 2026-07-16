"""
Streamlit UI — Поиск торговых стратегий.
Run: streamlit run app.py
"""

import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict

st.set_page_config(page_title="Поиск стратегий", page_icon="🔍", layout="wide")


# ── Faithfulness check ────────────────────────────────────────────

_PATTERNS_HIGH_PROFIT = [
    r"высок[аоый].*прибыл", r"high\s*profit", r"больш[аоый].*доход",
    r"выгодн", r"эффективн", r"много.*заработ",
    r"тейк.*профит\s*\d{2,}", r"return.*\d{2,}",
]
_PATTERNS_LOW_RISK = [
    r"низк[аоий].*риск", r"low\s*risk", r"безопасн",
    r"мал[аоый].*просад", r"стабильн", r"надёжн",
    r"conservative", r"drawdown.*-[0-5]%",
]


def faithfulness_check(text: str, sharpe: float, max_drawdown: float) -> Dict:
    text_lower = text.lower()
    issues, supported, total = [], 0, 0
    for p in _PATTERNS_HIGH_PROFIT:
        if re.search(p, text_lower, re.IGNORECASE):
            total += 1
            if sharpe >= 1.0:
                supported += 1
            else:
                issues.append(f"Заявлена высокая прибыль, но Sharpe = {sharpe:.2f}")
    for p in _PATTERNS_LOW_RISK:
        if re.search(p, text_lower, re.IGNORECASE):
            total += 1
            if max_drawdown > -15.0:
                supported += 1
            else:
                issues.append(f"Заявлен низкий риск, но Drawdown = {max_drawdown:.1f}%")
    if total == 0:
        return {"score": 1.0, "issues": []}
    return {"score": supported / total, "issues": issues}


# ── Equity curve plot ────────────────────────────────────────────

def plot_equity(curve, accent_color: str = "#10b981"):
    r, g, b = int(accent_color[1:3], 16), int(accent_color[3:5], 16), int(accent_color[5:7], 16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=curve, mode="lines",
        line=dict(color=accent_color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.10)",
        hovertemplate="День %{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=35, r=10, t=5, b=20),
        height=140,
        xaxis_title=None, yaxis_title=None,
        template="plotly_white",
        font=dict(size=10, color="#475569"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", showticklabels=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e9e5f5", gridwidth=1, linecolor="#e2e8f0")
    return fig


# ── Color helpers ─────────────────────────────────────────────────

def _sharpe_color(v): return "#059669" if v >= 2.0 else ("#d97706" if v >= 1.0 else "#dc2626")
def _dd_color(v): return "#059669" if v >= -10 else ("#d97706" if v >= -25 else "#dc2626")
def _ret_color(v): return "#059669" if v >= 50 else ("#d97706" if v >= 10 else "#64748b")
def _accent(rank): return {1: "#f59e0b", 2: "#8b5cf6", 3: "#06b6d4"}.get(rank, "#64748b")

def _score_info(s):
    if s >= 3.0: return "#059669", "#ecfdf5", "Отлично"
    if s >= 1.5: return "#7c3aed", "#f5f3ff", "Хорошо"
    return "#64748b", "#f1f5f9", "Средне"


# ── HTML render helpers (NO commas in inline values!) ────────────

def _badge(rank):
    c = {1: "#f59e0b", 2: "#8b5cf6", 3: "#06b6d4"}.get(rank, "#94a3b8")
    return (
        '<span style="display:inline-flex;align-items:center;justify-content:center;'
        'width:30px;height:30px;border-radius:8px;font-weight:700;font-size:0.82rem;'
        f'color:#fff;background:{c};box-shadow:0 2px 8px rgba(0,0,0,0.15);">{rank}</span>'
    )

def _pill(score):
    c, bg, lbl = _score_info(score)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 10px;'
        f'border-radius:20px;font-size:0.78rem;font-weight:600;background:{bg};color:{c};'
        f'border:1px solid {c}22;">★ {score:.2f} {lbl}</span>'
    )

def _mbox(label, value, color, icon):
    return (
        f'<div style="background:#faf8ff;border:1px solid #ede9fe;border-radius:10px;'
        f'padding:8px 10px;text-align:center;">'
        f'<div style="font-size:0.68rem;color:#7c3aed;text-transform:uppercase;'
        f'letter-spacing:0.05em;font-weight:700;margin-bottom:2px;">{icon} {label}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;color:{color};">{value}</div></div>'
    )

def _card(row):
    rank = int(row["rank"])
    ac = _accent(rank)
    s, dd, r, sc = row["sharpe"], row["max_drawdown"], row["total_return"], row["captain_score"]
    m = "".join([
        _mbox("Sharpe", f"{s:.2f}", _sharpe_color(s), "📊"),
        _mbox("Drawdown", f"{dd:.1f}%", _dd_color(dd), "📉"),
        _mbox("Return", f"{r:.1f}%", _ret_color(r), "📈"),
        _mbox("Score", f"{sc:.2f}", "#7c3aed", "⭐"),
    ])
    ca = row.get("crag_action", "none")
    warn = ""
    if ca not in ("none", "ok"):
        warn = (
            '<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 8px;'
            'border-radius:6px;background:#fef2f2;border:1px solid #fecaca;'
            'color:#dc2626;font-size:0.73rem;font-weight:600;">⚠ ' + ca + '</span>'
        )
    return (
        f'<div style="background:#fff;border:1px solid #ede9fe;border-left:5px solid {ac};'
        'border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:0.75rem;'
        'box-shadow:0 2px 12px rgba(124,58,237,0.06);">'
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        '<div style="display:flex;align-items:center;gap:8px;">'
        f'{_badge(rank)}'
        f'<span style="font-weight:600;font-size:0.92rem;color:#1e293b;">Стратегия #{int(row["doc_id"])}</span>'
        '</div>'
        f'<div style="display:flex;align-items:center;gap:6px;">{_pill(sc)} {warn}</div>'
        '</div>'
        f'<div style="font-size:0.86rem;color:#475569;line-height:1.55;margin-bottom:0.8rem;">{row["text"]}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;">{m}</div>'
        '</div>'
    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():

    st.markdown("""<style>
    /* ── Hide chrome ── */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(180deg,
            #0f172a 0%, #0f172a 35%,
            #ede9fe 35.5%, #f5f3ff 65%,
            #eef2ff 100%) !important;
    }

    /* ── Container ── */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 920px !important;
    }

    /* ══════════ HERO (custom HTML, works fine) ══════════ */
    .hero { text-align: center; padding: 1.5rem 0 1.8rem; }
    .hero h1 {
        font-size: 2.2rem; font-weight: 800; color: #f8fafc;
        margin: 0 0 0.3rem; letter-spacing: -0.03em;
    }
    .hero h1 span {
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { font-size: 0.92rem; color: #94a3b8; margin: 0; }

    /* ══════════ FORM (real DOM parent for all widgets) ══════════ */
    [data-testid="stForm"] {
        background: #1e293b !important;
        border: 1.5px solid #334155 !important;
        border-radius: 18px !important;
        padding: 1.5rem 1.8rem 1.2rem !important;
        max-width: 700px !important;
        margin: 0 auto 2rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    }

    /* ── Text input inside form ── */
    [data-testid="stForm"] [data-testid="stTextInput"] > div > div {
        background: #0f172a !important;
        border: 2px solid #475569 !important;
        border-radius: 12px !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] > div > div:focus-within {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        color: #f1f5f9 !important;
        font-size: 15px !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
        color: #64748b !important;
    }

    /* ── Selectbox inside form (filters) ── */
    [data-testid="stForm"] [data-testid="stSelectbox"] > div > div {
        background: #0f172a !important;
        border: 2px solid #475569 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #7c3aed !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] svg {
        fill: #a78bfa !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] .stSelectbox label p {
        color: #a78bfa !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] div[class*="placeholder"] {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #e2e8f0 !important;
    }

    /* ── Expander inside form ── */
    [data-testid="stForm"] [data-testid="stExpander"] {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    [data-testid="stForm"] [data-testid="stExpander"] details { border: none !important; }
    [data-testid="stForm"] [data-testid="stExpander"] summary {
        color: #a78bfa !important; font-size: 0.85rem !important; font-weight: 600 !important;
    }
    [data-testid="stForm"] [data-testid="stExpander"] summary:hover {
        color: #c4b5fd !important;
    }

    /* ── Submit button ── */
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] > button {
        background: #7c3aed !important;
        border: none !important; border-radius: 12px !important;
        color: #fff !important; font-weight: 700 !important;
        font-size: 1rem !important; padding: 0.65rem 2rem !important;
        box-shadow: 0 4px 16px rgba(124,58,237,0.4) !important;
        width: 100% !important;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] > button:hover {
        background: #6d28d9 !important;
        box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
    }

    /* ══════════ EMPTY STATE ══════════ */
    .empty { text-align: center; padding: 3rem 1rem 2rem; }
    .empty .pulse-icon {
        width: 80px; height: 80px; border-radius: 24px;
        background: #7c3aed;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 2rem; margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(124,58,237,0.3);
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .empty .main-text { color: #a78bfa; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem; }
    .empty .hint-text { color: #7c7c9a; font-size: 0.88rem; }

    /* ══════════ RESULTS COUNT ══════════ */
    .results-header {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 1rem; padding-left: 2px;
    }
    .results-header .dot { width: 8px; height: 8px; border-radius: 50%; background: #7c3aed; }
    .results-header span { font-size: 0.88rem; color: #6d28d9; font-weight: 600; }

    /* ══════════ PLOTLY ══════════ */
    .stPlotlyChart { margin-top: -0.4rem; margin-bottom: 0.4rem; }

    /* ══════════ FAITHFULNESS ══════════ */
    [data-testid="stExpander"] {
        border: 1px solid #fde68a !important; border-radius: 10px !important;
        background: #fffbeb !important; margin-top: 0.3rem;
    }
    [data-testid="stExpander"] details { border: none !important; }
    [data-testid="stExpander"] summary { color: #92400e !important; font-size: 0.82rem; font-weight: 500; }

    /* ══════════ MISC ══════════ */
    hr { display: none; }
    [data-testid="stMetricContainer"],
    [data-testid="stMetric"] { background: transparent !important; }
    .stSpinner > div { border-top-color: #7c3aed !important; }

    @media (max-width: 640px) {
        .hero h1 { font-size: 1.6rem; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        [data-testid="stForm"] { padding: 1rem 1rem 1rem !important; }
    }
    </style>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # HERO (pure HTML — no Streamlit widgets, renders fine)
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="hero">'
        '<h1>Поиск <span>торговых стратегий</span></h1>'
        '<p>Мультимодальный поиск по тексту, метрикам и кривым доходности</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════
    # SEARCH FORM — st.form creates a REAL <form> DOM parent,
    # so CSS selectors like [data-testid="stForm"] [data-testid="stSelectbox"]
    # actually work (child IS inside parent).
    # ══════════════════════════════════════════════════════════════
    with st.form("search_form", clear_on_submit=False):

        # Search input
        query = st.text_input(
            "Запрос",
            label_visibility="collapsed",
            placeholder="🔍  Индикаторы, паттерны, таймфреймы...",
        )

        # Filters (selectbox — always visible, no slider CSS issues)
        with st.expander("⚙️ Фильтры", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                sharpe_opt = st.selectbox(
                    "Мин. Sharpe Ratio",
                    options=["Любой", "≥ 0.5", "≥ 1.0", "≥ 1.5", "≥ 2.0", "≥ 2.5"],
                    index=0,
                )
            with c2:
                dd_opt = st.selectbox(
                    "Макс. Drawdown",
                    options=["Любой", "≥ -5%", "≥ -10%", "≥ -15%", "≥ -20%", "≥ -30%"],
                    index=0,
                )

        # Submit button
        search_pressed = st.form_submit_button("Найти стратегии", type="primary")

    # ══════════════════════════════════════════════════════════════
    # EMPTY STATE
    # ══════════════════════════════════════════════════════════════
    if not (search_pressed or query):
        st.markdown(
            '<div class="empty">'
            '<div class="pulse-icon">🔍</div>'
            '<div class="main-text">Начните поиск стратегий</div>'
            '<div class="hint-text">'
            'Например: «трендовая стратегия пробой SMA»<br>'
            'или «низкая просадка скальпинг»</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if not query.strip():
        st.warning("Введите поисковый запрос")
        return

    # ══════════════════════════════════════════════════════════════
    # Parse filter values
    # ══════════════════════════════════════════════════════════════
    _sharpe_map = {
        "Любой": None, "≥ 0.5": 0.5, "≥ 1.0": 1.0,
        "≥ 1.5": 1.5, "≥ 2.0": 2.0, "≥ 2.5": 2.5,
    }
    _dd_map = {
        "Любой": None, "≥ -5%": -5.0, "≥ -10%": -10.0,
        "≥ -15%": -15.0, "≥ -20%": -20.0, "≥ -30%": -30.0,
    }
    min_sharpe = _sharpe_map.get(sharpe_opt)
    max_drawdown = _dd_map.get(dd_opt)

    # ══════════════════════════════════════════════════════════════
    # RUN PIPELINE
    # ══════════════════════════════════════════════════════════════
    with st.spinner("Ищем стратегии..."):
        try:
            from pipeline import AllianceRetriever
            if "alliance" not in st.session_state:
                st.session_state.alliance = AllianceRetriever()
            results = st.session_state.alliance.search(
                query,
                top_k=5,
                min_sharpe=min_sharpe,
                max_drawdown=max_drawdown,
            )
        except FileNotFoundError:
            st.error(
                "Файлы индексов не найдены. Сначала выполните:\n\n"
                "```\n"
                "python data_parser.py\n"
                "python sparse_index.py\n"
                "python train.py\n"
                "python dense_index.py\n"
                "```"
            )
            return
        except Exception as e:
            st.error(f"Ошибка: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ══════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════
    if results.empty:
        st.info("Ничего не найдено. Попробуйте изменить запрос или ослабить фильтры.")
        return

    st.markdown(
        f'<div class="results-header">'
        f'<div class="dot"></div>'
        f'<span>Найдено стратегий: {len(results)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for _, row in results.iterrows():
        st.markdown(_card(row), unsafe_allow_html=True)
        st.plotly_chart(
            plot_equity(row["equity_curve"], _accent(int(row["rank"]))),
            use_container_width=True,
        )
        fc = faithfulness_check(row["text"], row["sharpe"], row["max_drawdown"])
        if fc["issues"]:
            with st.expander("⚠ Несоответствия в описании"):
                for issue in fc["issues"]:
                    st.markdown(f"- {issue}")


if __name__ == "__main__":
    main()