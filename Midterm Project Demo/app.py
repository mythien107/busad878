"""
app.py  —  AI-Powered Operations Dashboard (Streamlit)
======================================================
The interactive WEB-APP version of the midterm demo. Same data + same AI layer
as the notebook; this file is the "working prototype" deliverable.

RUN IT LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py
Then open the URL it prints (default http://localhost:8501).

The dashboard has:
  • sidebar filters that drive every KPI and chart,
  • a KPI strip + four charts (the "insightful dashboard"),
  • three AI features in tabs (ask-a-question / auto-insights / anomalies),
  • a data-quality panel (error handling made visible),
  • a 👍/👎 feedback widget on AI answers (collect + act on user feedback).

Works with NO API key (offline fallback). Add GEMINI_API_KEY for live AI.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import ai
from analytics import compute_metric
from connectors import clean_and_validate, load_data

st.set_page_config(page_title="AI Ops Dashboard — Supply Chain",
                   page_icon="📦", layout="wide")

PLOTLY_CFG = {"displayModeBar": False}


# --------------------------------------------------------------------------- #
# Data (cached so we load + validate once per session)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data():
    raw = load_data("synthetic")          # swap source here for your project
    return clean_and_validate(raw)


df_all, dq_report = get_data()


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("📦 Filters")
st.sidebar.caption("Every KPI and chart below responds to these.")


def _multi(label, col):
    opts = sorted(df_all[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)


regions = _multi("Region", "supplier_region")
modes = _multi("Transport mode", "transport_mode")
categories = _multi("Product category", "product_category")
suppliers = st.sidebar.multiselect(
    "Supplier", sorted(df_all["supplier"].unique()),
    default=sorted(df_all["supplier"].unique()))

months = sorted(df_all["month"].unique())
m_lo, m_hi = st.sidebar.select_slider(
    "Order month range", options=months, value=(months[0], months[-1]))

mask = (
    df_all["supplier_region"].isin(regions)
    & df_all["transport_mode"].isin(modes)
    & df_all["product_category"].isin(categories)
    & df_all["supplier"].isin(suppliers)
    & df_all["month"].between(m_lo, m_hi)
)
df = df_all[mask].copy()

mode_badge = ("🟢 Gemini (live)" if ai.ai_available() else "⚪ Offline mode (rule-based)")
st.sidebar.divider()
st.sidebar.write("**AI status:**", mode_badge)
st.sidebar.caption(f"Model: `{ai.MODEL}`" if ai.ai_available()
                   else "Set GEMINI_API_KEY to enable live AI.")


# --------------------------------------------------------------------------- #
# Header + KPI strip
# --------------------------------------------------------------------------- #
st.title("AI-Powered Operations Dashboard")
st.caption("Supply-chain OTIF, lead times, and cost — with three generative-AI features. "
           "BUSAD 878 midterm demo.")

if df.empty:
    st.warning("No rows match the current filters. Widen them in the sidebar.")
    st.stop()

k = st.columns(5)
k[0].metric("OTIF", f"{compute_metric(df, 'otif_rate'):.1f}%")
k[1].metric("On-time", f"{compute_metric(df, 'on_time_rate'):.1f}%")
k[2].metric("In-full", f"{compute_metric(df, 'in_full_rate'):.1f}%")
k[3].metric("Avg lead time", f"{compute_metric(df, 'avg_lead_time'):.1f} d")
k[4].metric("Total order value", f"${compute_metric(df, 'total_order_value'):,.0f}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def otif_by(frame, col):
    d = frame[frame["is_valid"] & frame["status"].eq("Delivered")]
    g = d.groupby(col)["otif"].mean().mul(100).round(1).reset_index()
    return g.sort_values("otif")


c1, c2 = st.columns(2)
with c1:
    st.subheader("OTIF trend by month")
    trend = otif_by(df, "month").sort_values("month")
    fig = px.line(trend, x="month", y="otif", markers=True,
                  labels={"otif": "OTIF %", "month": "Order month"})
    fig.add_hline(y=95, line_dash="dot", annotation_text="95% target")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
with c2:
    st.subheader("OTIF by supplier")
    g = otif_by(df, "supplier")
    fig = px.bar(g, x="otif", y="supplier", orientation="h",
                 color="otif", color_continuous_scale="RdYlGn",
                 labels={"otif": "OTIF %", "supplier": ""})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

c3, c4 = st.columns(2)
with c3:
    st.subheader("OTIF by region & mode")
    d = df[df["is_valid"] & df["status"].eq("Delivered")]
    g = (d.groupby(["supplier_region", "transport_mode"])["otif"].mean()
         .mul(100).round(1).reset_index())
    fig = px.bar(g, x="supplier_region", y="otif", color="transport_mode",
                 barmode="group", labels={"otif": "OTIF %", "supplier_region": ""})
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
with c4:
    st.subheader("Avg unit cost trend by category")
    d = df[df["is_valid"]]
    g = d.groupby(["month", "product_category"])["unit_cost"].mean().round(2).reset_index()
    fig = px.line(g, x="month", y="unit_cost", color="product_category",
                  labels={"unit_cost": "Avg unit cost ($)", "month": "Order month"})
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)


# --------------------------------------------------------------------------- #
# Feedback helper (collect + act on user feedback)
# --------------------------------------------------------------------------- #
if "feedback" not in st.session_state:
    st.session_state.feedback = []


def feedback_widget(key: str, context: str):
    col_a, col_b, _ = st.columns([1, 1, 8])
    if col_a.button("👍 Helpful", key=f"up_{key}"):
        st.session_state.feedback.append({"item": context, "rating": "up"})
        st.toast("Thanks — logged as helpful.")
    if col_b.button("👎 Off", key=f"down_{key}"):
        st.session_state.feedback.append({"item": context, "rating": "down"})
        st.toast("Logged — answers rated 👎 are queued for prompt review.")


# --------------------------------------------------------------------------- #
# AI features
# --------------------------------------------------------------------------- #
st.divider()
st.header("🤖 AI features")
tab_q, tab_i, tab_a = st.tabs(["Ask a question", "Automated insights", "Anomaly scan"])

with tab_q:
    st.caption("Natural-language querying. The model only proposes a *query plan*; "
               "pandas computes the numbers. Runs on the full dataset.")
    samples = ["Which region has the worst OTIF in peak season?",
               "Average lead time by transport mode",
               "OTIF by supplier for APAC",
               "Total order value by product category"]
    pick = st.selectbox("Try a sample question (or type your own below):",
                        [""] + samples)
    question = st.text_input("Your question:", value=pick,
                             placeholder="e.g. on-time rate by supplier in Q4")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking…"):
            res = ai.nl_query(df_all, question)
        st.session_state["last_q"] = res
    if "last_q" in st.session_state:
        res = st.session_state["last_q"]
        st.info(res["answer"])
        if not res["table"].empty:
            st.dataframe(res["table"], use_container_width=True, hide_index=True)
        with st.expander("How this was answered (transparency)"):
            st.write(f"Mode: **{res['mode']}** · {res['recap']}")
            st.json(res["plan"])
        feedback_widget("q", f"Q: {res['question']}")

with tab_i:
    st.caption("Automated insights about the **current filtered view**.")
    if st.button("Generate insights", type="primary"):
        with st.spinner("Analyzing…"):
            st.session_state["last_i"] = ai.auto_insights(df)
    if "last_i" in st.session_state:
        out = st.session_state["last_i"]
        st.write(f"_Source: {out['mode']}_")
        for b in out["insights"]:
            st.markdown(f"- {b}")
        feedback_widget("i", "auto_insights")

with tab_a:
    st.caption("Anomaly detection (statistical) + plain-English explanation. "
               "Scans the **full dataset** for cost spikes, OTIF drops, and declining trends.")
    if st.button("Scan for anomalies", type="primary"):
        with st.spinner("Scanning…"):
            st.session_state["last_a"] = ai.explain_anomalies(df_all)
    if "last_a" in st.session_state:
        out = st.session_state["last_a"]
        st.write(f"_Source: {out['mode']}_")
        if not out["anomalies"].empty:
            st.dataframe(out["anomalies"], use_container_width=True, hide_index=True)
        st.markdown(out["explanation"])
        feedback_widget("a", "anomalies")


# --------------------------------------------------------------------------- #
# Data quality + feedback log (error handling made visible)
# --------------------------------------------------------------------------- #
st.divider()
with st.expander("🧪 Data quality & validation report"):
    st.write("Bad rows are flagged and **excluded from metrics**, never silently dropped:")
    st.json(dq_report)
    bad = df_all[~df_all["is_valid"]][
        ["order_id", "units_ordered", "units_received", "order_date",
         "actual_delivery_date", "dq_issue"]]
    if not bad.empty:
        st.dataframe(bad, use_container_width=True, hide_index=True)

with st.expander("🗳️ Feedback log"):
    if st.session_state.feedback:
        st.dataframe(pd.DataFrame(st.session_state.feedback),
                     use_container_width=True, hide_index=True)
        st.caption("In production, 👎 items would be sampled to refine prompts and thresholds.")
    else:
        st.caption("No feedback yet. Rate an AI answer above.")

st.caption("⚠️ Teaching demo on synthetic data. AI output can be wrong — verify before acting.")
