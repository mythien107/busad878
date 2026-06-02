"""
ai.py
=====
The AI LAYER — three generative-AI features for the dashboard, each built on
Google Gemini with a fully functional OFFLINE FALLBACK so the demo never breaks.

    Feature 1  nl_query(df, question)      Natural-language querying
    Feature 2  auto_insights(df)           Automated insight generation
    Feature 3  explain_anomalies(df)       Anomaly explanation + recommendations

DESIGN PRINCIPLES (these map straight onto the midterm rubric)
--------------------------------------------------------------
* CHEAPEST MODEL.  Defaults to `gemini-2.5-flash-lite` (the low-cost tier). Set
  the GEMINI_MODEL env var to change it; check ai.google.dev/pricing for the
  current cheapest option, as names rotate.
* GROUNDING, NOT GUESSING.  For querying, the model only emits a small JSON
  *plan*; analytics.py computes the real numbers. For insights/anomalies, we
  feed the model PRE-COMPUTED aggregates, never raw rows. The model narrates;
  pandas does the math.
* GRACEFUL DEGRADATION.  No API key, no internet, or a quota error -> every
  function falls back to deterministic, rule-based output and tells you it did
  (`result["mode"] == "offline"`). The dashboard keeps working.
* OUTPUT VALIDATION.  Every plan the model proposes is checked against the
  schema (analytics.validate_plan) before it can run.

Set your key once:   setx GEMINI_API_KEY "your-key"   (Windows)
                      export GEMINI_API_KEY="your-key" (macOS/Linux)
Get a free key at https://aistudio.google.com/apikey
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import pandas as pd

from connectors import DIMENSIONS, METRICS
from analytics import compute_metric, detect_anomalies, run_query_plan, validate_plan

# --------------------------------------------------------------------------- #
# Gemini client (optional import so the module loads even without the SDK)
# --------------------------------------------------------------------------- #
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

try:
    from google import genai
    from google.genai import types
    _SDK_OK = True
except Exception:  # pragma: no cover - SDK not installed
    _SDK_OK = False

_client = None


def ai_available() -> bool:
    """True only if the SDK is installed AND a key is configured."""
    return bool(_SDK_OK and _API_KEY)


def _get_client():
    global _client
    if _client is None and ai_available():
        _client = genai.Client(api_key=_API_KEY)
    return _client


def _generate(prompt: str, system: Optional[str] = None,
              temperature: float = 0.2, as_json: bool = False) -> str:
    """One call to Gemini. Raises on any problem so callers can fall back."""
    client = _get_client()
    if client is None:
        raise RuntimeError("Gemini not available")
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        # Keeping the schema tight + temperature low is our main guard against
        # the model wandering off-format. For querying we also force JSON out.
        response_mime_type="application/json" if as_json else "text/plain",
        max_output_tokens=1024,
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    return (resp.text or "").strip()


# --------------------------------------------------------------------------- #
# Shared: a compact, model-friendly profile of the data (aggregates, not rows)
# --------------------------------------------------------------------------- #
def _compact_profile(df: pd.DataFrame) -> dict:
    """Pre-compute the aggregates an analyst would glance at first."""
    d = df[df["is_valid"] & df["status"].eq("Delivered")]
    prof = {
        "overall_otif_pct": round(compute_metric(df, "otif_rate"), 1),
        "overall_on_time_pct": round(compute_metric(df, "on_time_rate"), 1),
        "overall_in_full_pct": round(compute_metric(df, "in_full_rate"), 1),
        "total_orders": int(len(df)),
        "in_transit": int(df["status"].eq("In Transit").sum()),
        "flagged_bad_rows": int((~df["is_valid"]).sum()),
        "otif_by_region": {k: round(compute_metric(g, "otif_rate"), 1)
                           for k, g in d.groupby("supplier_region")},
        "otif_by_supplier": {k: round(compute_metric(g, "otif_rate"), 1)
                             for k, g in d.groupby("supplier")},
        "otif_by_mode": {k: round(compute_metric(g, "otif_rate"), 1)
                         for k, g in d.groupby("transport_mode")},
        "otif_by_month": {k: round(compute_metric(g, "otif_rate"), 1)
                          for k, g in d.groupby("month")},
    }
    return prof


# =========================================================================== #
# FEATURE 1: NATURAL-LANGUAGE QUERYING
# =========================================================================== #
_PLAN_SYSTEM = f"""You translate an operations manager's question about a supply-chain
shipment dataset into a STRICT JSON query plan. You do not compute answers.

Return ONLY this JSON shape:
{{"metric": <one of {list(METRICS)}>,
  "group_by": <one of {DIMENSIONS} or null>,
  "filters": [{{"column": <dimension/date/numeric column>, "op": <"=="|"!="|">"|">="|"<"|"<="|"in">, "value": <string|number|list>}}],
  "sort": <"asc"|"desc">,
  "limit": <integer or null>}}

Rules:
- Pick the single metric that best answers the question.
- Use group_by when the user compares across a dimension ("by supplier", "which region").
- Region values are exactly: "North America", "EMEA", "APAC", "LATAM".
- Transport modes: "Ground", "Air", "Ocean". Statuses: "Delivered", "In Transit".
- "month" values look like "2025-11". "Peak season"/"Q4" means months 2025-10..2025-12 (use op "in").
- Keep filters minimal. Output JSON only, no prose."""


def _offline_plan(question: str) -> dict:
    """Keyword parser used when Gemini is unavailable — handles common questions."""
    q = question.lower()
    # metric
    if "in full" in q or "in-full" in q or "short" in q:
        metric = "in_full_rate"
    elif "otif" in q or "on time in full" in q or "on-time-in-full" in q:
        metric = "otif_rate"
    elif "on time" in q or "on-time" in q or "late" in q:
        metric = "on_time_rate"
    elif "lead time" in q:
        metric = "avg_lead_time"
    elif "delay" in q:
        metric = "avg_delay_days"
    elif "cost" in q or "price" in q:
        metric = "avg_unit_cost"
    elif "value" in q or "spend" in q:
        metric = "total_order_value"
    elif "how many" in q or "count" in q or "number of" in q:
        metric = "order_count"
    else:
        metric = "otif_rate"
    # group_by
    group_by = None
    for key, dim in [("supplier", "supplier"), ("region", "supplier_region"),
                     ("category", "product_category"), ("mode", "transport_mode"),
                     ("carrier", "carrier"), ("month", "month"),
                     ("warehouse", "destination_dc"), ("dc", "destination_dc")]:
        if key in q:
            group_by = dim
            break
    # filters
    filters = []
    for name, val in [("apac", "APAC"), ("emea", "EMEA"), ("latam", "LATAM"),
                      ("north america", "North America")]:
        if name in q:
            filters.append({"column": "supplier_region", "op": "==", "value": val})
    for name, val in [("ocean", "Ocean"), ("air", "Air"), ("ground", "Ground")]:
        if re.search(rf"\b{name}\b", q):
            filters.append({"column": "transport_mode", "op": "==", "value": val})
    if "q4" in q or "peak" in q:
        filters.append({"column": "month", "op": "in",
                        "value": ["2025-10", "2025-11", "2025-12"]})
    return {"metric": metric, "group_by": group_by, "filters": filters,
            "sort": "asc" if ("worst" in q or "lowest" in q) else "desc",
            "limit": 10 if group_by else None}


def nl_query(df: pd.DataFrame, question: str) -> dict:
    """Answer a natural-language question. Returns dict with table + narration."""
    mode, plan, plan_err = "gemini", None, None
    if ai_available():
        try:
            raw = _generate(f"Question: {question}", system=_PLAN_SYSTEM, as_json=True)
            plan = json.loads(raw)
        except Exception as e:  # bad JSON, quota, no internet...
            plan_err, mode = str(e), "offline"
    else:
        mode = "offline"
    if plan is None:
        plan = _offline_plan(question)

    # VALIDATE before running (catches hallucinated columns/metrics)
    ok, msg = validate_plan(plan)
    if not ok:
        plan, mode = _offline_plan(question), "offline"  # repair with safe parser
        ok, msg = validate_plan(plan)
        if not ok:
            return {"mode": mode, "question": question, "plan": plan,
                    "table": pd.DataFrame(), "recap": "",
                    "answer": f"Sorry — I couldn't turn that into a valid query ({msg}). "
                              f"Try naming a metric like OTIF, lead time, or cost."}

    table, recap = run_query_plan(df, plan)
    if table.empty:
        return {"mode": mode, "question": question, "plan": plan, "table": table,
                "recap": recap,
                "answer": "No rows matched that query. Try widening the filters."}

    # Narrate the (real, computed) result
    answer = _narrate_result(question, plan, table, mode)
    return {"mode": mode, "question": question, "plan": plan,
            "table": table, "recap": recap, "answer": answer}


def _narrate_result(question, plan, table, mode) -> str:
    facts = table.to_dict(orient="records")
    if mode == "gemini" and ai_available():
        try:
            sys = ("You are an operations analyst. In 2-3 sentences, answer the "
                   "user's question using ONLY the provided computed results. "
                   "Cite the actual numbers. Do not invent figures.")
            prompt = (f"Question: {question}\nMetric: {METRICS[plan['metric']]}\n"
                      f"Computed results (authoritative): {json.dumps(facts)}")
            return _generate(prompt, system=sys, temperature=0.3)
        except Exception:
            pass
    # offline narration
    label = METRICS[plan["metric"]]
    if plan.get("group_by"):
        gb = plan["group_by"]
        top = facts[0]
        return (f"{label} {('was lowest' if plan.get('sort') == 'asc' else 'was highest')} "
                f"for {top[gb]} at {top[plan['metric']]}. "
                f"Full breakdown of {len(facts)} {gb} group(s) shown below.")
    return f"{label}: {facts[0][plan['metric']]}."


# =========================================================================== #
# FEATURE 2: AUTOMATED INSIGHTS
# =========================================================================== #
def auto_insights(df: pd.DataFrame, n: int = 5) -> dict:
    """Surface the n most important things an analyst should notice."""
    profile = _compact_profile(df)
    if ai_available():
        try:
            sys = ("You are a supply-chain analyst. From the computed metrics below, "
                   f"write the {n} most decision-relevant insights for an operations "
                   "manager. Each insight: one sentence, lead with the number, name the "
                   "supplier/region/month involved, and imply the 'so what'. Use ONLY "
                   "these numbers. Return a JSON array of strings.")
            raw = _generate(f"Computed metrics:\n{json.dumps(profile, indent=2)}",
                            system=sys, as_json=True, temperature=0.4)
            bullets = json.loads(raw)
            if isinstance(bullets, list) and bullets:
                return {"mode": "gemini", "insights": [str(b) for b in bullets[:n]],
                        "profile": profile}
        except Exception:
            pass
    return {"mode": "offline", "insights": _offline_insights(profile, n),
            "profile": profile}


def _offline_insights(profile: dict, n: int) -> list:
    out = [f"Overall OTIF is {profile['overall_otif_pct']}% "
           f"(on-time {profile['overall_on_time_pct']}%, in-full {profile['overall_in_full_pct']}%) "
           f"across {profile['total_orders']:,} orders."]
    reg = profile["otif_by_region"]
    if reg:
        worst = min(reg, key=reg.get); best = max(reg, key=reg.get)
        out.append(f"{worst} is the weakest region at {reg[worst]}% OTIF, "
                   f"{best} the strongest at {reg[best]}% — a {reg[best] - reg[worst]:.0f}-point gap.")
    sup = profile["otif_by_supplier"]
    if sup:
        ws = min(sup, key=sup.get)
        out.append(f"{ws} is the lowest-performing supplier at {sup[ws]}% OTIF and warrants a review.")
    months = profile["otif_by_month"]
    if len(months) >= 4:
        keys = sorted(months)
        early = sum(months[k] for k in keys[:3]) / 3
        late = sum(months[k] for k in keys[-3:]) / 3
        out.append(f"Recent OTIF (~{late:.0f}%) vs early-period (~{early:.0f}%) shows a "
                   f"{late - early:+.0f}-point shift; watch the trend.")
    mode = profile["otif_by_mode"]
    if mode:
        wm = min(mode, key=mode.get)
        out.append(f"{wm} freight has the lowest OTIF ({mode[wm]}%), consistent with higher variability.")
    if profile["flagged_bad_rows"]:
        out.append(f"{profile['flagged_bad_rows']} rows were flagged as data-quality issues "
                   f"and excluded from metrics.")
    return out[:n]


# =========================================================================== #
# FEATURE 3: ANOMALY EXPLANATION
# =========================================================================== #
def explain_anomalies(df: pd.DataFrame) -> dict:
    """Detect anomalies (deterministically) then explain them in business terms."""
    table = detect_anomalies(df)
    if table.empty:
        return {"mode": "offline", "anomalies": table,
                "explanation": "No anomalies crossed the detection thresholds."}
    records = table.to_dict(orient="records")
    if ai_available():
        try:
            sys = ("You are a supply-chain analyst. For each detected anomaly, write one "
                   "short paragraph: what it likely means in business terms and one "
                   "concrete action to investigate or mitigate. Use ONLY the facts given; "
                   "do not invent numbers. Group related anomalies if they share a cause.")
            raw = _generate(f"Detected anomalies (authoritative):\n{json.dumps(records, indent=2)}",
                            system=sys, temperature=0.4)
            return {"mode": "gemini", "anomalies": table, "explanation": raw}
        except Exception:
            pass
    return {"mode": "offline", "anomalies": table,
            "explanation": _offline_anomaly_text(records)}


def _offline_anomaly_text(records: list) -> str:
    lines = []
    for r in records:
        if r["type"] == "Cost spike":
            act = "Confirm whether this is a market move or a contract change; consider hedging or re-sourcing."
        elif r["type"] == "OTIF drop":
            act = "Check for a specific disruption that month (capacity, weather, customs) and confirm it recovered."
        else:  # Declining trend
            act = "Open a supplier performance review; the slide is gradual and easy to miss in the company average."
        lines.append(f"• [{r['type']}] {r['scope']}: {r['detail']}. → {act}")
    return "\n".join(lines)


if __name__ == "__main__":
    from connectors import load_data, clean_and_validate
    df, _ = clean_and_validate(load_data("synthetic"))
    print("AI available:", ai_available(), "| model:", MODEL)
    print("\n--- nl_query: 'Which region has the worst OTIF in peak season?' ---")
    r = nl_query(df, "Which region has the worst OTIF in peak season?")
    print(f"[{r['mode']}] {r['answer']}\n{r['table'].to_string(index=False)}")
    print("\n--- auto_insights ---")
    ins = auto_insights(df)
    print(f"[{ins['mode']}]")
    for b in ins["insights"]:
        print(" •", b)
    print("\n--- explain_anomalies ---")
    ex = explain_anomalies(df)
    print(f"[{ex['mode']}]\n{ex['explanation']}")
