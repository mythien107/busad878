# Midterm Demo — AI-Powered Operations Dashboard (Supply Chain)

A complete, runnable reference example for the BUSAD 878 midterm project. It
shows **two ways to build the same dashboard from the same data**:

1. **Code path** — a Jupyter notebook + a Streamlit web app (this folder).
2. **No-code path** — rebuild it in **Lovable** by describing it in English
   (see [`Lovable_NoCode_Dashboard_Guide.md`](Lovable_NoCode_Dashboard_Guide.md)).

This is the "build vs. buy" choice from the Week 6 lesson, made concrete.

> Everything here runs on **synthetic data** and is a teaching demo only.
> Don't use for production purpose.

---

## What it demonstrates (and how it maps to the rubric)

| Midterm requirement | Where it lives |
|---|---|
| Data connection (+ simulate your own) | `connectors.py` (`load_data`), `generate_synthetic_data.py` |
| ≥ 3 AI-powered features | `ai.py` → NL querying, automated insights, anomaly explanation |
| Insightful dashboard / storytelling | `app.py` charts + the "average hides the problem" narrative |
| Error handling (bad inputs, validating AI output, feedback) | `connectors.clean_and_validate`, `analytics.validate_plan`, thumbs up/down widget in `app.py` |
| Documentation (prompts, design decisions, limits) | docstrings everywhere; reflection template |

## The story baked into the data
Headline OTIF looks stable (~80%), but that average **hides** three problems the
AI features surface: a **Q4 peak-season dip**, a **commodity cost shock** in Raw
Materials, and a **gradually degrading supplier** (Pacific Components Co.).

---

## Setup

```bash
# 1. (recommended) Python 3.10–3.12, in a fresh virtual env
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install
pip install -r requirements.txt

# 3. generate the dataset
python generate_synthetic_data.py        # writes data/supply_chain_shipments.csv

# 4. (optional) enable live AI — without this it runs in offline/rule-based mode
setx GEMINI_API_KEY "your-key"            # Windows (reopen terminal after)
# export GEMINI_API_KEY="your-key"        # macOS / Linux
# free key: https://aistudio.google.com/apikey
```

**No API key? No problem.** Every AI feature has a deterministic offline
fallback, so the notebook and app run fully without one. A small badge shows
whether you're in Gemini or offline mode.

## Run it

```bash
# the interactive web app
streamlit run app.py                      # opens http://localhost:8501

# or the guided notebook
jupyter notebook midterm_dashboard_demo.ipynb
```

---

## File guide

| File | Role |
|---|---|
| `generate_synthetic_data.py` | Builds the dataset with **documented assumptions** — the model artifact to submit if you simulate data. |
| `connectors.py` | `load_data()` (synthetic + **placeholder** CSV/Sheet/SQL/API/Kaggle connectors) and `clean_and_validate()`. |
| `analytics.py` | Deterministic pandas engine: metric math, the safe **query-plan executor**, anomaly detection. |
| `ai.py` | Gemini wrapper + offline fallback + the 3 AI features. The LLM proposes plans/narration; pandas does the math. |
| `app.py` | The Streamlit dashboard (your "working prototype"). |
| `midterm_dashboard_demo.ipynb` | Step-by-step teaching notebook that builds up to the app. |
| `Lovable_NoCode_Dashboard_Guide.md` | Rebuild the dashboard with no code, in Lovable. |

## Use your own data
Swap one line. In `connectors.load_data` the default is `source="synthetic"`;
change the call in `app.py`/the notebook to `load_data("csv", path=...)` or fill
in one of the placeholder connectors (`google_sheet`, `sql`, `rest_api`,
`kaggle`). Everything downstream — validation, analytics, AI — is unchanged.

## Architecture (one glance)
```
            ┌─────────────┐   raw df   ┌────────────────────┐ clean df
 source ──▶ │ connectors  │ ─────────▶ │ clean_and_validate │ ─────────▶ ┐
            └─────────────┘            └────────────────────┘             │
                                                                          ▼
   ┌──────────────────────────── analytics.py (pandas, trusted) ─────────────┐
   │  compute_metric · run_query_plan(validated) · detect_anomalies          │
   └─────────────────────────────────────────────────────────────────────────┘
                                     ▲ numbers            ▲ facts
                       plan (JSON)   │                    │  narration
            ┌──────────────────── ai.py (Gemini + offline) ───────────────────┐
            │  nl_query · auto_insights · explain_anomalies                    │
            └──────────────────────────────────────────────────────────────────┘
                                     ▼
                         app.py (Streamlit)  /  notebook
```
