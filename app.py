import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM  — Deep navy + electric teal
# ─────────────────────────────────────────────
BG       = "#0a0e1a"
SURFACE  = "#111827"
SURFACE2 = "#1a2236"
BORDER   = "#1e2d45"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
TEAL     = "#06b6d4"
VIOLET   = "#8b5cf6"
AMBER    = "#f59e0b"
GREEN    = "#10b981"
RED      = "#ef4444"

LR_COL  = TEAL
RF_COL  = VIOLET
XGB_COL = AMBER

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                font=dict(size=11)),
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1526 0%, {BG} 100%) !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] hr {{ border-color: {BORDER} !important; }}

header[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDeployButton"],
[data-testid="stBaseButton-header"],
.stDeployButton,
#MainMenu {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1400px; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* KPI Grid */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
.kpi-grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px; }}
.kpi-card {{
    background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE2} 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}}
.kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-bar);
}}
.kpi-card:hover {{ border-color: {TEAL}44; }}
.kpi-label {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 2.1rem; font-weight: 700; line-height: 1;
    background: var(--val-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.kpi-sub {{ font-size: 0.73rem; color: {MUTED}; margin-top: 6px; display: flex; align-items: center; gap: 4px; }}
.kpi-badge {{
    display: inline-block;
    background: var(--badge-bg); color: var(--badge-color);
    border-radius: 4px; padding: 1px 7px; font-size: 0.65rem; font-weight: 600;
}}

/* Section heading */
.section-heading {{
    font-size: 0.95rem; font-weight: 600; color: {TEXT};
    display: flex; align-items: center; gap: 8px;
    margin: 24px 0 12px;
    border-left: 3px solid {TEAL};
    padding-left: 10px;
}}

/* Model comparison table */
.model-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
.model-table th {{
    background: {SURFACE2}; color: {MUTED};
    font-weight: 600; letter-spacing: .06em; font-size: .68rem; text-transform: uppercase;
    padding: 10px 14px; border-bottom: 1px solid {BORDER}; text-align: left;
}}
.model-table td {{ padding: 11px 14px; border-bottom: 1px solid {BORDER}55; color: {TEXT}; }}
.model-table tr:last-child td {{ border-bottom: none; }}
.model-table tr:hover td {{ background: {SURFACE2}; }}
.model-row-best td {{ color: {TEAL} !important; }}

/* Pill badges */
.pill {{ display: inline-block; border-radius: 20px; padding: 2px 10px; font-size: 0.65rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }}
.pill-teal   {{ background: {TEAL}22;   color: {TEAL};   border: 1px solid {TEAL}44; }}
.pill-violet {{ background: {VIOLET}22; color: {VIOLET}; border: 1px solid {VIOLET}44; }}
.pill-amber  {{ background: {AMBER}22;  color: {AMBER};  border: 1px solid {AMBER}44; }}
.pill-green  {{ background: {GREEN}22;  color: {GREEN};  border: 1px solid {GREEN}44; }}
.pill-red    {{ background: {RED}22;    color: {RED};    border: 1px solid {RED}44; }}

/* Insight box */
.insight-box {{
    background: {SURFACE2}; border: 1px solid {BORDER};
    border-left: 3px solid {AMBER}; border-radius: 8px;
    padding: 14px 16px; font-size: 0.82rem; color: {MUTED};
    margin-top: 8px; line-height: 1.6;
}}
.insight-box b {{ color: {TEXT}; }}

/* Progress bar */
.prog-bar-bg {{ background: {SURFACE2}; border-radius: 4px; height: 6px; overflow: hidden; }}
.prog-bar-fill {{
    height: 6px; border-radius: 4px;
    background: linear-gradient(90deg, var(--bar-start), var(--bar-end));
}}

/* Brand */
.brand {{
    font-size: 1.05rem; font-weight: 700; color: {TEAL};
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}}
.brand-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {TEAL}; }}

/* Mini stat card */
.mini-stat {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 12px; text-align: center;
}}

/* Streamlit component overrides */
[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background: {SURFACE} !important; border-color: {BORDER} !important; color: {TEXT} !important;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER} !important; border-radius: 10px !important; overflow: hidden; }}
.stRadio label {{ color: {TEXT} !important; font-size: 0.88rem !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    X_test  = pd.read_csv("data/processed/X_test.csv").fillna(0)
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    X_train = pd.read_csv("data/processed/X_train.csv").fillna(0)
    raw     = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    return X_test, y_test, X_train, raw

@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.joblib")
    lr     = joblib.load("models/logistic_regression.joblib")
    rf     = joblib.load("models/random_forest.joblib")
    xgb    = joblib.load("models/xgboost.joblib")
    return scaler, lr, rf, xgb

X_test, y_test, X_train, raw_df = load_data()
scaler, lr_model, rf_model, xgb_model = load_models()
X_scaled = scaler.transform(X_test)

lr_proba  = lr_model.predict_proba(X_scaled)[:, 1]
lr_pred   = lr_model.predict(X_scaled)
rf_proba  = rf_model.predict_proba(X_scaled)[:, 1]
rf_pred   = rf_model.predict(X_scaled)
xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
xgb_pred  = xgb_model.predict(X_scaled)

lr_auc  = roc_auc_score(y_test, lr_proba)
rf_auc  = roc_auc_score(y_test, rf_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)

MODEL_META = {
    "Logistic Regression": dict(proba=lr_proba,  pred=lr_pred,  auc=lr_auc,  color=LR_COL,  obj=lr_model),
    "Random Forest":       dict(proba=rf_proba,  pred=rf_pred,  auc=rf_auc,  color=RF_COL,  obj=rf_model),
    "XGBoost":             dict(proba=xgb_proba, pred=xgb_pred, auc=xgb_auc, color=XGB_COL, obj=xgb_model),
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def chart(fig, **extra):
    merged = {**PLOTLY_BASE, **extra}
    fig.update_layout(**merged)
    return fig

def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-dot"></div>Churn Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};margin-bottom:20px;">Telco Customer Analytics</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=["Overview", "CSV Explorer", "Model Performance", "Model Deep-Dive"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    total_s = len(raw_df)
    churned_s = (raw_df["Churn"] == "Yes").sum()
    st.markdown(f"""
    <div style="font-size:.72rem;color:{MUTED};line-height:2;">
        <span style="color:{TEXT};font-weight:600;">Dataset</span><br>
        📁 Telco-Customer-Churn<br>
        👥 {total_s:,} customers<br>
        🔴 {churned_s:,} churned ({churned_s/total_s*100:.1f}%)<br>
        🤖 3 ML models trained
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════

if page == "Overview":

    st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{TEXT}">Customer Churn Overview</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.82rem;color:{MUTED};margin-bottom:24px;">Telco customer base — 7,043 accounts — full dataset analysis</div>',
                unsafe_allow_html=True)

    total_c  = len(raw_df)
    churn_n  = (raw_df["Churn"] == "Yes").sum()
    churn_r  = churn_n / total_c * 100
    retained = total_c - churn_n
    avg_ten  = round(raw_df[raw_df["Churn"] == "Yes"]["tenure"].mean(), 1)
    avg_monthly = round(raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"].mean(), 2)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{TEAL},{VIOLET})">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{TEAL},{TEXT})">{total_c:,}</div>
        <div class="kpi-sub">full dataset</div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{RED},{AMBER})">
        <div class="kpi-label">Churned</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{RED},{AMBER})">{churn_n:,}</div>
        <div class="kpi-sub">
          <span class="kpi-badge" style="--badge-bg:{RED}22;--badge-color:{RED}">{churn_r:.1f}%</span>
          of base
        </div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{GREEN},{TEAL})">
        <div class="kpi-label">Retained</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{GREEN},{TEAL})">{retained:,}</div>
        <div class="kpi-sub">
          <span class="kpi-badge" style="--badge-bg:{GREEN}22;--badge-color:{GREEN}">{100-churn_r:.1f}%</span>
          of base
        </div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{AMBER},{VIOLET})">
        <div class="kpi-label">Avg Tenure at Churn</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{AMBER},{TEXT})">{avg_ten}</div>
        <div class="kpi-sub">months · <b style="color:{TEXT}">${avg_monthly}</b>/mo avg</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Donut + Contract churn
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-heading">Churn Distribution</div>', unsafe_allow_html=True)
        vc = raw_df["Churn"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[vc.get("No", 0), vc.get("Yes", 0)],
            hole=0.62,
            marker=dict(colors=[GREEN, RED], line=dict(color=BG, width=3)),
            textinfo="percent",
            textfont=dict(size=13, color=TEXT),
        ))
        fig.add_annotation(
            text=f"<b>{churn_r:.1f}%</b><br><span style='font-size:11px'>Churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=TEXT), align="center"
        )
        chart(fig, height=320, showlegend=True,
              legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    with col2:
        st.markdown('<div class="section-heading">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        ct = raw_df.groupby("Contract")["Churn"].apply(
            lambda s: (s == "Yes").sum() / len(s) * 100
        ).reset_index()
        ct.columns = ["Contract", "Churn Rate"]
        ct = ct.sort_values("Churn Rate", ascending=True)
        fig = go.Figure(go.Bar(
            x=ct["Churn Rate"], y=ct["Contract"], orientation="h",
            marker=dict(color=[TEAL, AMBER, RED][:len(ct)], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in ct["Churn Rate"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        chart(fig, height=320,
              xaxis=dict(title="Churn Rate (%)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=12)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Row 2: Tenure bar + Monthly charges histogram
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-heading">Churn Rate by Tenure</div>', unsafe_allow_html=True)
        df2 = raw_df.copy()
        df2["Tenure Group"] = pd.cut(
            df2["tenure"], bins=[0, 12, 24, 48, 72],
            labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"],
            include_lowest=True
        )
        tg = df2.groupby("Tenure Group", observed=True)["Churn"].apply(
            lambda s: (s == "Yes").sum() / len(s) * 100
        ).reset_index()
        tg.columns = ["Group", "Churn Rate"]
        fig = go.Figure(go.Bar(
            x=tg["Group"].astype(str), y=tg["Churn Rate"],
            marker=dict(color=[RED, AMBER, VIOLET, TEAL][:len(tg)], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in tg["Churn Rate"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        chart(fig, height=320,
              yaxis=dict(title="Churn Rate (%)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    with col4:
        st.markdown('<div class="section-heading">Monthly Charges by Churn Status</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_df[raw_df["Churn"] == "No"]["MonthlyCharges"],
            name="Retained", marker_color=rgba(GREEN, 0.6), nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"],
            name="Churned", marker_color=rgba(RED, 0.8), nbinsx=30,
        ))
        chart(fig, height=320, barmode="overlay",
              xaxis=dict(title="Monthly Charges ($)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(title="Customers", gridcolor=BORDER, tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Row 3: Internet service + Payment method
    col5, col6 = st.columns([1, 1])

    with col5:
        st.markdown('<div class="section-heading">Churn by Internet Service</div>', unsafe_allow_html=True)
        inet = raw_df.groupby("InternetService").agg(
            Customers=("Churn", "count"),
            Churned=("Churn", lambda s: (s == "Yes").sum())
        ).reset_index()
        inet["Churn Rate"] = (inet["Churned"] / inet["Customers"] * 100).round(1)
        inet = inet.sort_values("Churn Rate", ascending=False)

        rows = ""
        for _, r in inet.iterrows():
            bw = int(r["Churn Rate"] * 1.8)
            rows += f"""<tr>
              <td>{r["InternetService"]}</td>
              <td>{r["Customers"]:,}</td>
              <td>{r["Churned"]:,}</td>
              <td>
                <div style="display:flex;align-items:center;gap:8px;">
                  <div class="prog-bar-bg" style="flex:1">
                    <div class="prog-bar-fill" style="width:{bw}%;--bar-start:{RED};--bar-end:{AMBER}"></div>
                  </div>
                  <span style="font-size:.78rem;color:{TEXT};min-width:36px">{r["Churn Rate"]}%</span>
                </div>
              </td>
            </tr>"""

        st.markdown(f"""
        <table class="model-table">
          <thead><tr><th>Service</th><th>Customers</th><th>Churned</th><th>Churn Rate</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="section-heading">Churn by Payment Method</div>', unsafe_allow_html=True)
        pm = raw_df.groupby("PaymentMethod")["Churn"].apply(
            lambda s: (s == "Yes").sum() / len(s) * 100
        ).reset_index()
        pm.columns = ["Method", "Churn Rate"]
        pm = pm.sort_values("Churn Rate", ascending=True)
        fig = go.Figure(go.Bar(
            x=pm["Churn Rate"], y=pm["Method"], orientation="h",
            marker=dict(color=[TEAL, VIOLET, AMBER, RED][:len(pm)], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in pm["Churn Rate"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        chart(fig, height=280,
              xaxis=dict(title="Churn Rate (%)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=10)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    fiber_rate = raw_df[raw_df["InternetService"] == "Fiber optic"]["Churn"].apply(lambda x: x == "Yes").mean() * 100
    st.markdown(f"""
    <div class="insight-box">
      <b>Key Findings:</b> Month-to-month customers churn at ~43% vs 3% for two-year contracts.
      <b>Fiber optic</b> internet shows the highest churn ({fiber_rate:.1f}%).
      Short-tenure customers (0–12 months) are 3× more likely to churn.
      <b>Electronic check</b> payers churn significantly more than any other payment method.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — CSV EXPLORER
# ═══════════════════════════════════════════════════════

elif page == "CSV Explorer":

    st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{TEXT}">CSV Data Explorer</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.82rem;color:{MUTED};margin-bottom:24px;">Upload any CSV or explore the built-in Telco dataset interactively</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a CSV file (optional)", type=["csv"])
    explore_df = pd.read_csv(uploaded) if uploaded else raw_df.copy()

    total_rows, total_cols = explore_df.shape
    missing_pct = round(explore_df.isnull().sum().sum() / (total_rows * total_cols) * 100, 2)
    num_cols = explore_df.select_dtypes(include="number").columns.tolist()
    cat_cols = explore_df.select_dtypes(exclude="number").columns.tolist()

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{TEAL},{VIOLET})">
        <div class="kpi-label">Rows</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{TEAL},{TEXT})">{total_rows:,}</div>
        <div class="kpi-sub">records</div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{VIOLET},{AMBER})">
        <div class="kpi-label">Columns</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{VIOLET},{TEXT})">{total_cols}</div>
        <div class="kpi-sub">{len(num_cols)} numeric · {len(cat_cols)} categorical</div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{GREEN},{TEAL})">
        <div class="kpi-label">Missing Data</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{GREEN},{TEXT})">{missing_pct}%</div>
        <div class="kpi-sub">across all cells</div>
      </div>
      <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{AMBER},{RED})">
        <div class="kpi-label">Memory</div>
        <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{AMBER},{TEXT})">{explore_df.memory_usage(deep=True).sum()/1024:.0f}</div>
        <div class="kpi-sub">KB in memory</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Data Preview</div>', unsafe_allow_html=True)
    n_rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(explore_df.head(n_rows), use_container_width=True,
                 height=min(n_rows * 38 + 40, 420))

    if num_cols:
        st.markdown('<div class="section-heading">Numeric Distributions</div>', unsafe_allow_html=True)
        sel_num = st.selectbox("Select numeric column", num_cols, key="csv_num")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=explore_df[sel_num].dropna(),
            marker_color=TEAL, marker_line=dict(color=BG, width=0.5),
            nbinsx=40, name=sel_num,
        ))
        fig.add_vline(x=explore_df[sel_num].mean(), line_dash="dash",
                      line_color=AMBER, annotation_text="Mean",
                      annotation_font_color=AMBER)
        fig.add_vline(x=explore_df[sel_num].median(), line_dash="dot",
                      line_color=VIOLET, annotation_text="Median",
                      annotation_font_color=VIOLET)
        chart(fig, height=300, title=f"Distribution of {sel_num}",
              xaxis=dict(title=sel_num, gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(title="Count", gridcolor=BORDER, tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        stats = explore_df[sel_num].describe().round(3)
        stat_html = "".join(
            f'<div class="mini-stat"><div style="font-size:.68rem;color:{MUTED};text-transform:uppercase">{k}</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:{TEXT}">{v}</div></div>'
            for k, v in stats.items()
        )
        st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">{stat_html}</div>',
                    unsafe_allow_html=True)

    if cat_cols:
        st.markdown('<div class="section-heading">Categorical Distributions</div>', unsafe_allow_html=True)
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="csv_cat")
        vc2 = explore_df[sel_cat].value_counts().head(15)
        palette_cat = [TEAL, VIOLET, AMBER, GREEN, RED] * 5
        fig = go.Figure(go.Bar(
            x=vc2.values, y=vc2.index.astype(str), orientation="h",
            marker=dict(color=palette_cat[:len(vc2)], line=dict(width=0)),
            text=vc2.values, textposition="outside",
            textfont=dict(color=TEXT, size=11),
        ))
        chart(fig, height=max(280, len(vc2) * 34), title=f"Value Counts: {sel_cat}",
              xaxis=dict(title="Count", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT), autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    if len(num_cols) >= 2:
        st.markdown('<div class="section-heading">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = explore_df[num_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, RED], [0.5, SURFACE2], [1, TEAL]],
            zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=9), hoverongaps=False,
        ))
        chart(fig, height=max(400, len(num_cols) * 36),
              xaxis=dict(tickangle=-30, tickfont=dict(size=9, color=MUTED)),
              yaxis=dict(tickfont=dict(size=9, color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    if len(num_cols) >= 2:
        st.markdown('<div class="section-heading">Scatter Explorer</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_col = st.selectbox("X axis", num_cols, index=0, key="sc_x")
        with sc2:
            y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        with sc3:
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="sc_c")

        sdf = explore_df[[x_col, y_col] + ([color_col] if color_col != "None" else [])].dropna()
        if color_col != "None":
            fig = px.scatter(sdf, x=x_col, y=y_col, color=color_col,
                             color_discrete_sequence=[TEAL, RED, VIOLET, AMBER, GREEN],
                             opacity=0.65)
        else:
            fig = go.Figure(go.Scattergl(
                x=sdf[x_col], y=sdf[y_col], mode="markers",
                marker=dict(color=TEAL, opacity=0.55, size=5, line=dict(width=0))
            ))
        chart(fig, height=380,
              xaxis=dict(title=x_col, gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(title=y_col, gridcolor=BORDER, tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))


# ═══════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════

elif page == "Model Performance":

    st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{TEXT}">Model Performance</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.82rem;color:{MUTED};margin-bottom:24px;">Logistic Regression · Random Forest · XGBoost — 20% holdout test set</div>',
                unsafe_allow_html=True)

    best_name = max(MODEL_META, key=lambda m: MODEL_META[m]["auc"])
    kpi_html = ""
    for name, meta in MODEL_META.items():
        is_best = name == best_name
        badge = f'<span class="kpi-badge" style="--badge-bg:{TEAL}22;--badge-color:{TEAL}">BEST</span>' if is_best else ""
        kpi_html += f"""
        <div class="kpi-card" style="--accent-bar:linear-gradient(90deg,{meta["color"]},{BG})">
          <div class="kpi-label">{name}</div>
          <div class="kpi-value" style="--val-gradient:linear-gradient(135deg,{meta["color"]},{TEXT})">{meta["auc"]:.4f}</div>
          <div class="kpi-sub">AUC-ROC {badge}</div>
        </div>"""
    st.markdown(f'<div class="kpi-grid-3">{kpi_html}</div>', unsafe_allow_html=True)

    # Full comparison table
    st.markdown('<div class="section-heading">Model Comparison</div>', unsafe_allow_html=True)
    rows_html = ""
    for name, meta in MODEL_META.items():
        report = classification_report(y_test, meta["pred"],
                                       target_names=["No Churn", "Churn"], output_dict=True)
        acc = round((meta["pred"] == y_test.values).mean() * 100, 2)
        p1  = round(report["Churn"]["precision"] * 100, 1)
        r1  = round(report["Churn"]["recall"] * 100, 1)
        f1  = round(report["Churn"]["f1-score"] * 100, 1)
        ap  = round(average_precision_score(y_test, meta["proba"]) * 100, 1)
        rc  = "model-row-best" if name == best_name else ""
        star = " ⭐" if name == best_name else ""
        rows_html += f"""<tr class="{rc}">
          <td><b>{name}{star}</b></td>
          <td>{meta["auc"]:.4f}</td><td>{acc}%</td>
          <td>{p1}%</td><td>{r1}%</td><td>{f1}%</td><td>{ap}%</td>
        </tr>"""

    st.markdown(f"""
    <table class="model-table">
      <thead><tr>
        <th>Model</th><th>AUC-ROC</th><th>Accuracy</th>
        <th>Precision</th><th>Recall</th><th>F1</th><th>Avg Precision</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROC + PR curves side by side
    c_roc, c_pr = st.columns(2)

    with c_roc:
        st.markdown('<div class="section-heading">ROC Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=MUTED, dash="dash", width=1), name="Random (0.5000)",
        ))
        for name, meta in MODEL_META.items():
            fpr, tpr, _ = roc_curve(y_test, meta["proba"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} ({meta['auc']:.4f})",
                line=dict(color=meta["color"], width=2.5),
                fill="tozeroy" if name == best_name else None,
                fillcolor=rgba(meta["color"], 0.09) if name == best_name else None,
            ))
        chart(fig, height=380,
              xaxis=dict(title="False Positive Rate", gridcolor=BORDER,
                         tickfont=dict(color=MUTED), range=[0, 1]),
              yaxis=dict(title="True Positive Rate", gridcolor=BORDER,
                         tickfont=dict(color=MUTED), range=[0, 1]),
              legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    with c_pr:
        st.markdown('<div class="section-heading">Precision–Recall Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, meta in MODEL_META.items():
            prec, rec, _ = precision_recall_curve(y_test, meta["proba"])
            ap = average_precision_score(y_test, meta["proba"])
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{name} (AP={ap:.3f})",
                line=dict(color=meta["color"], width=2.5),
            ))
        chart(fig, height=380,
              xaxis=dict(title="Recall", gridcolor=BORDER,
                         tickfont=dict(color=MUTED), range=[0, 1]),
              yaxis=dict(title="Precision", gridcolor=BORDER,
                         tickfont=dict(color=MUTED), range=[0, 1]),
              legend=dict(x=0.02, y=0.05))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Prediction score distributions
    st.markdown('<div class="section-heading">Predicted Probability Distributions</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=list(MODEL_META.keys()),
                        horizontal_spacing=0.06)
    for i, (name, meta) in enumerate(MODEL_META.items(), 1):
        proba_df = pd.DataFrame({"proba": meta["proba"], "true": y_test.values})
        for label, color, opacity in [(0, GREEN, 0.7), (1, RED, 0.85)]:
            subset = proba_df[proba_df["true"] == label]["proba"]
            fig.add_trace(go.Histogram(
                x=subset, nbinsx=25,
                marker_color=color, opacity=opacity,
                name=("No Churn" if label == 0 else "Churn"),
                showlegend=(i == 1),
                legendgroup=str(label),
            ), row=1, col=i)
    fig.update_layout(**{
        **PLOTLY_BASE,
        "height": 300,
        "barmode": "overlay",
        "legend": dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
    })
    for i in range(1, 4):
        xk = "xaxis" if i == 1 else f"xaxis{i}"
        yk = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(**{
            xk: dict(title="P(Churn)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
            yk: dict(gridcolor=BORDER, tickfont=dict(color=MUTED)),
        })
    fig.update_annotations(font=dict(color=MUTED, size=11))
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))


# ═══════════════════════════════════════════════════════
# PAGE 4 — MODEL DEEP-DIVE
# ═══════════════════════════════════════════════════════

elif page == "Model Deep-Dive":

    st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{TEXT}">Model Deep-Dive</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.82rem;color:{MUTED};margin-bottom:24px;">Confusion matrix, classification report, threshold analysis, and feature insights</div>',
                unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model", list(MODEL_META.keys()),
                                label_visibility="collapsed")
    meta = MODEL_META[model_choice]

    # Confusion matrix + report
    c_cm, c_report = st.columns([1, 1])

    with c_cm:
        st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, meta["pred"])
        tn, fp, fn, tp = cm.ravel()
        z_text = [[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]]
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            text=z_text, texttemplate="%{text}",
            textfont=dict(size=14, color="#ffffff"),
            colorscale=[[0, SURFACE], [0.4, rgba(meta["color"], 0.33)], [1, meta["color"]]],
            showscale=False, hoverongaps=False,
        ))
        chart(fig, height=310,
              xaxis=dict(tickfont=dict(color=TEXT, size=11), gridcolor="rgba(0,0,0,0)"),
              yaxis=dict(tickfont=dict(color=TEXT, size=11), gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        acc  = (tp + tn) / (tp + tn + fp + fn) * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px">
          <div class="mini-stat">
            <div style="font-size:.68rem;color:{MUTED};text-transform:uppercase">Accuracy</div>
            <div style="font-size:1.4rem;font-weight:700;color:{meta["color"]}">{acc:.1f}%</div>
          </div>
          <div class="mini-stat">
            <div style="font-size:.68rem;color:{MUTED};text-transform:uppercase">Precision</div>
            <div style="font-size:1.4rem;font-weight:700;color:{meta["color"]}">{prec:.1f}%</div>
          </div>
          <div class="mini-stat">
            <div style="font-size:.68rem;color:{MUTED};text-transform:uppercase">Recall</div>
            <div style="font-size:1.4rem;font-weight:700;color:{meta["color"]}">{rec:.1f}%</div>
          </div>
          <div class="mini-stat">
            <div style="font-size:.68rem;color:{MUTED};text-transform:uppercase">Specificity</div>
            <div style="font-size:1.4rem;font-weight:700;color:{meta["color"]}">{spec:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c_report:
        st.markdown('<div class="section-heading">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, meta["pred"],
                                       target_names=["No Churn", "Churn"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(index=["accuracy"], errors="ignore")
        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        report_df["support"] = report_df["support"].astype("Int64")
        report_df = report_df.round(3)

        rows_html = ""
        for idx, row in report_df.iterrows():
            bp = int(row["precision"] * 100)
            br = int(row["recall"] * 100)
            bf = int(row["f1-score"] * 100)
            rows_html += f"""<tr>
              <td><b>{idx}</b></td>
              <td>
                <div style="display:flex;align-items:center;gap:6px">
                  <div class="prog-bar-bg" style="width:56px">
                    <div class="prog-bar-fill" style="width:{bp}%;--bar-start:{meta["color"]};--bar-end:{meta["color"]}88"></div>
                  </div>
                  {row["precision"]:.3f}
                </div>
              </td>
              <td>
                <div style="display:flex;align-items:center;gap:6px">
                  <div class="prog-bar-bg" style="width:56px">
                    <div class="prog-bar-fill" style="width:{br}%;--bar-start:{VIOLET};--bar-end:{VIOLET}88"></div>
                  </div>
                  {row["recall"]:.3f}
                </div>
              </td>
              <td>
                <div style="display:flex;align-items:center;gap:6px">
                  <div class="prog-bar-bg" style="width:56px">
                    <div class="prog-bar-fill" style="width:{bf}%;--bar-start:{AMBER};--bar-end:{AMBER}88"></div>
                  </div>
                  {row["f1-score"]:.3f}
                </div>
              </td>
              <td>{row["support"]}</td>
            </tr>"""

        st.markdown(f"""
        <table class="model-table" style="margin-top:8px">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    # Threshold analysis
    st.markdown('<div class="section-heading">Threshold Analysis</div>', unsafe_allow_html=True)
    thresholds = np.linspace(0.1, 0.9, 81)
    precs2, recs2, f1s2, accs2 = [], [], [], []
    for t in thresholds:
        p_t = (meta["proba"] >= t).astype(int)
        r = classification_report(y_test, p_t, output_dict=True, zero_division=0)
        churn_key = "1" if "1" in r else list(r.keys())[1]
        precs2.append(r[churn_key]["precision"])
        recs2.append(r[churn_key]["recall"])
        f1s2.append(r[churn_key]["f1-score"])
        accs2.append((p_t == y_test.values).mean())

    fig = go.Figure()
    for y_vals, name, color in [
        (precs2, "Precision", meta["color"]),
        (recs2,  "Recall",    VIOLET),
        (f1s2,   "F1-Score",  AMBER),
        (accs2,  "Accuracy",  GREEN),
    ]:
        fig.add_trace(go.Scatter(
            x=thresholds, y=y_vals, mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=MUTED,
                  annotation_text="Default 0.5", annotation_font_color=MUTED)
    chart(fig, height=320,
          xaxis=dict(title="Classification Threshold", gridcolor=BORDER,
                     tickfont=dict(color=MUTED)),
          yaxis=dict(title="Score", gridcolor=BORDER,
                     tickfont=dict(color=MUTED), range=[0, 1]),
          legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18))
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Feature importance / coefficients
    if model_choice in ("Random Forest", "XGBoost"):
        st.markdown('<div class="section-heading">Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        importances = meta["obj"].feature_importances_
        fi_df = pd.DataFrame({"Feature": X_test.columns, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, SURFACE2], [0.5, rgba(meta["color"], 0.53)], [1, meta["color"]]],
                line=dict(width=0), showscale=False,
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        chart(fig, height=440,
              xaxis=dict(title="Importance", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    else:
        st.markdown('<div class="section-heading">Feature Coefficients (Top 15)</div>', unsafe_allow_html=True)
        coef = lr_model.coef_[0]
        coef_df = pd.DataFrame({"Feature": X_test.columns, "Coefficient": coef})
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
            marker=dict(
                color=[RED if v < 0 else TEAL for v in coef_df["Coefficient"]],
                line=dict(width=0)
            ),
            text=[f"{v:+.4f}" for v in coef_df["Coefficient"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        chart(fig, height=440,
              xaxis=dict(title="Coefficient", gridcolor=BORDER, tickfont=dict(color=MUTED)),
              yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # Churn risk buckets
    st.markdown('<div class="section-heading">Churn Risk Buckets</div>', unsafe_allow_html=True)
    risk_df = pd.DataFrame({"proba": meta["proba"], "actual": y_test.values})
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    blabels = ["Very Low\n(0–20%)", "Low\n(20–40%)", "Medium\n(40–60%)",
               "High\n(60–80%)", "Very High\n(80–100%)"]
    risk_df["bucket"] = pd.cut(risk_df["proba"], bins=bins, labels=blabels, include_lowest=True)
    bucket_counts  = risk_df["bucket"].value_counts().sort_index()
    bucket_actual  = risk_df.groupby("bucket", observed=True)["actual"].mean() * 100
    bucket_palette = [GREEN, TEAL, AMBER, VIOLET, RED]

    c_bk1, c_bk2 = st.columns(2)
    with c_bk1:
        fig = go.Figure(go.Bar(
            x=[str(l) for l in bucket_counts.index], y=bucket_counts.values,
            marker=dict(color=bucket_palette, line=dict(width=0)),
            text=bucket_counts.values, textposition="outside",
            textfont=dict(color=TEXT, size=11),
        ))
        chart(fig, height=300, title="Customers per Risk Bucket",
              xaxis=dict(tickfont=dict(color=TEXT, size=10), gridcolor="rgba(0,0,0,0)"),
              yaxis=dict(title="Count", gridcolor=BORDER, tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    with c_bk2:
        fig = go.Figure(go.Bar(
            x=[str(l) for l in bucket_actual.index], y=bucket_actual.values,
            marker=dict(color=bucket_palette, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in bucket_actual.values],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        chart(fig, height=300, title="Actual Churn Rate per Bucket",
              xaxis=dict(tickfont=dict(color=TEXT, size=10), gridcolor="rgba(0,0,0,0)"),
              yaxis=dict(title="Actual Churn Rate (%)", gridcolor=BORDER,
                         tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
