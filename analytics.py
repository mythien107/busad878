"""
analytics.py
============
The DETERMINISTIC analysis engine — plain pandas, no AI yet.

This separation is the most important design idea in the midterm project--you will find
it under "Error Handling & Robustness" in the grading rubric. The language model never 
touches your numbers directly. Instead it proposes a small, structured QUERY PLAN (which 
metric, which filters, which group-by). We VALIDATE that plan against the known schema 
and then compute the answer here, with pandas we can trust. The LLM's job is translation 
and narration—not arithmetic.

So if a user asks "what was APAC's on-time rate after the peak season?", the LLM
turns that into {metric: on_time_rate, filter: region==APAC & month>=2025-10},
and THIS module produces the actual number. A hallucinated column or a nonsense
metric is caught before it can ever reach the user.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from connectors import DIMENSIONS, METRICS, DATE_COLS, NUMERIC_COLS

ALLOWED_FILTER_COLS = set(DIMENSIONS) | set(DATE_COLS) | set(NUMERIC_COLS)
ALLOWED_OPS = {"==", "!=", ">", ">=", "<", "<=", "in"}


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def _valid(df: pd.DataFrame) -> pd.DataFrame:
    """Rows safe to count at all (passed validation)."""
    return df[df["is_valid"]] if "is_valid" in df.columns else df


def _delivered(df: pd.DataFrame) -> pd.DataFrame:
    """Valid + actually delivered (rates are undefined for in-transit rows)."""
    d = _valid(df)
    return d[d["status"].eq("Delivered")]


def compute_metric(df: pd.DataFrame, metric: str) -> float:
    """Compute one metric over a (possibly filtered) frame. Returns NaN if empty."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {list(METRICS)}")
    d = _delivered(df)
    if metric in ("total_order_value", "avg_unit_cost", "order_count"):
        v = _valid(df)  # value/cost/count don't require delivery
        if metric == "total_order_value":
            return float(v["order_value"].sum())
        if metric == "avg_unit_cost":
            return float(v["unit_cost"].mean())
        return int(len(v))
    if len(d) == 0:
        return float("nan")
    if metric == "otif_rate":
        return float(d["otif"].mean() * 100)
    if metric == "on_time_rate":
        return float(d["on_time"].mean() * 100)
    if metric == "in_full_rate":
        return float(d["in_full"].mean() * 100)
    if metric == "avg_lead_time":
        return float(d["actual_lead_time_days"].mean())
    if metric == "avg_delay_days":
        return float(d["delay_days"].mean())
    return float("nan")


# --------------------------------------------------------------------------- #
# Query-plan execution (the safe bridge from AI intent -> numbers)
# --------------------------------------------------------------------------- #
def validate_plan(plan: dict) -> Tuple[bool, str]:
    """Check an AI-proposed plan against the schema. Returns (ok, message)."""
    if not isinstance(plan, dict):
        return False, "Plan is not an object."
    metric = plan.get("metric")
    if metric not in METRICS:
        return False, f"Metric '{metric}' is not allowed."
    gb = plan.get("group_by")
    if gb not in (None, "", *DIMENSIONS):
        return False, f"group_by '{gb}' is not a known dimension."
    for f in plan.get("filters", []) or []:
        if f.get("column") not in ALLOWED_FILTER_COLS:
            return False, f"Filter column '{f.get('column')}' is not allowed."
        if f.get("op") not in ALLOWED_OPS:
            return False, f"Filter op '{f.get('op')}' is not allowed."
    return True, "ok"


def _apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    out = df
    for f in filters or []:
        col, op, val = f["column"], f["op"], f.get("value")
        s = out[col]
        # compare dates/months as strings where appropriate
        if col in DATE_COLS:
            s = s.dt.strftime("%Y-%m-%d")
        if op == "==":
            out = out[s == val]
        elif op == "!=":
            out = out[s != val]
        elif op == ">":
            out = out[s > val]
        elif op == ">=":
            out = out[s >= val]
        elif op == "<":
            out = out[s < val]
        elif op == "<=":
            out = out[s <= val]
        elif op == "in":
            vals = val if isinstance(val, (list, tuple)) else [val]
            out = out[s.isin(vals)]
    return out


def run_query_plan(df: pd.DataFrame, plan: dict) -> Tuple[pd.DataFrame, str]:
    """Execute a validated plan, returning (result_table, plain-English recap)."""
    ok, msg = validate_plan(plan)
    if not ok:
        raise ValueError(msg)

    metric = plan["metric"]
    gb = plan.get("group_by") or None
    filters = plan.get("filters", []) or []
    filtered = _apply_filters(df, filters)

    label = METRICS[metric]
    if gb:
        groups = []
        for key, sub in filtered.groupby(gb):
            groups.append({gb: key, metric: compute_metric(sub, metric)})
        res = pd.DataFrame(groups).dropna(subset=[metric])
        ascending = str(plan.get("sort", "desc")).lower() == "asc"
        res = res.sort_values(metric, ascending=ascending)
        if plan.get("limit"):
            res = res.head(int(plan["limit"]))
        res[metric] = res[metric].round(2)
        recap = f"{label} by {gb}"
    else:
        res = pd.DataFrame([{metric: round(compute_metric(filtered, metric), 2)}])
        recap = label
    if filters:
        recap += " | filters: " + ", ".join(
            f"{f['column']} {f['op']} {f.get('value')}" for f in filters
        )
    recap += f" | rows considered: {len(filtered):,}"
    return res, recap


# --------------------------------------------------------------------------- #
# Statistical anomaly detection (AI explains these later)
# --------------------------------------------------------------------------- #
def detect_anomalies(df: pd.DataFrame, min_orders: int = 8) -> pd.DataFrame:
    """Find unusual cost months and supplier performance collapses.

    Two families, both scored on a comparable 0–1 ``severity`` scale so they can
    share one ranked table:

      * **Cost spike** — a category-month whose average unit cost jumps well
        above that category's own typical level (catches the commodity shock).
      * **OTIF drop**  — a supplier-month whose On-Time-In-Full rate falls far
        below that supplier's own baseline (catches a degrading supplier whose
        decline is otherwise hidden inside the company-wide average).
    """
    records = []
    v = _valid(df)

    # (1) UNIT-COST SPIKES per category vs the category's own median monthly cost.
    cat_month = (v.groupby(["product_category", "month"])["unit_cost"]
                 .mean().reset_index())
    for cat, sub in cat_month.groupby("product_category"):
        baseline = sub["unit_cost"].median()
        if baseline <= 0:
            continue
        for _, r in sub.iterrows():
            ratio = r["unit_cost"] / baseline
            if ratio >= 1.20:
                records.append({
                    "type": "Cost spike",
                    "scope": f"{cat} — {r['month']}",
                    "detail": (f"avg unit cost ${r['unit_cost']:.2f} vs "
                               f"category baseline ${baseline:.2f} "
                               f"({(ratio - 1) * 100:+.0f}%)"),
                    "severity": round(float(ratio - 1), 3),
                })

    # (2) SUPPLIER OTIF COLLAPSE: a month where a supplier's OTIF is >=20 points
    #     below its own median month (and below 60%), on enough orders to matter.
    d = _delivered(df)
    sup_month = (d.groupby(["supplier", "month"])
                 .agg(otif=("otif", "mean"), n=("otif", "size")).reset_index())
    for sup, sub in sup_month.groupby("supplier"):
        sub = sub[sub["n"] >= min_orders]
        if len(sub) < 4:
            continue
        baseline = sub["otif"].median()
        for _, r in sub.iterrows():
            drop = baseline - r["otif"]
            if drop >= 0.20 and r["otif"] < 0.60:
                records.append({
                    "type": "OTIF drop",
                    "scope": f"{sup} — {r['month']}",
                    "detail": (f"OTIF {r['otif']:.0%} vs supplier baseline "
                               f"{baseline:.0%} ({-drop * 100:+.0f} pp) "
                               f"on {int(r['n'])} orders"),
                    "severity": round(float(drop), 3),
                })

    # (3) DECLINING TREND per supplier: a gradual slide is invisible to (2)
    #     because the baseline itself drifts down. Compare the most recent 90
    #     days against everything before to expose slow erosion (the degrader).
    if len(d):
        cutoff = d["order_date"].max() - pd.Timedelta(days=90)
        for sup, sub in d.groupby("supplier"):
            recent, earlier = sub[sub["order_date"] > cutoff], sub[sub["order_date"] <= cutoff]
            if len(recent) >= 20 and len(earlier) >= 20:
                drop = earlier["otif"].mean() - recent["otif"].mean()
                if drop >= 0.15:
                    records.append({
                        "type": "Declining trend",
                        "scope": sup,
                        "detail": (f"OTIF slid from {earlier['otif'].mean():.0%} "
                                   f"(prior) to {recent['otif'].mean():.0%} "
                                   f"(last 90 days), {-drop * 100:+.0f} pp"),
                        "severity": round(float(drop), 3),
                    })

    if not records:
        return pd.DataFrame(columns=["type", "scope", "detail", "severity"])
    return (pd.DataFrame(records)
            .sort_values("severity", ascending=False)
            .reset_index(drop=True))


if __name__ == "__main__":
    from connectors import load_data, clean_and_validate
    df, _ = clean_and_validate(load_data("synthetic"))

    print("overall OTIF:", round(compute_metric(df, "otif_rate"), 1), "%")

    plan = {"metric": "otif_rate", "group_by": "supplier_region",
            "filters": [{"column": "month", "op": ">=", "value": "2025-10"}],
            "sort": "asc"}
    res, recap = run_query_plan(df, plan)
    print("\n" + recap)
    print(res.to_string(index=False))

    print("\nTop anomalies:")
    print(detect_anomalies(df).head(8).to_string(index=False))
