"""
connectors.py
=============
The DATA LAYER for the demo dashboard.

It does two jobs the midterm rubric cares about:

1. DATA CONNECTION (rubric: "Data Connection"). `load_data()` is the single
   front door for getting a DataFrame, with one branch per source. The
   synthetic branch works out of the box; the others (CSV, Google Sheet, SQL,
   REST API, Kaggle) are PLACEHOLDERS showing exactly where you would plug a
   real connector for YOUR project. Swap the source, keep everything downstream.

2. ERROR HANDLING (rubric: "Graceful handling of bad inputs" + "Validation").
   `clean_and_validate()` parses types, then *flags* impossible/garbage rows
   instead of crashing on them — and hands back a human-readable report of what
   it found. Nothing is silently dropped; bad rows are marked and excluded from
   metrics, so a single fat-fingered quantity can't poison your KPIs.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "supply_chain_shipments.csv")

# --------------------------------------------------------------------------- #
# SCHEMA — the contract the rest of the app (and the AI query planner) relies on
# --------------------------------------------------------------------------- #
DATE_COLS = ["order_date", "promised_delivery_date", "actual_delivery_date"]
NUMERIC_COLS = [
    "units_ordered", "units_received", "unit_cost", "order_value",
    "promised_lead_time_days", "actual_lead_time_days", "delay_days",
]
BOOL_COLS = ["on_time", "in_full", "otif"]

# Dimensions the user is allowed to slice / group by (used to VALIDATE AI output)
DIMENSIONS = [
    "supplier", "supplier_region", "product_category", "transport_mode",
    "carrier", "destination_dc", "status", "month",
]

# Metrics the user is allowed to ask for, with plain-English labels.
# (The actual computation lives in analytics.compute_metric.)
METRICS = {
    "otif_rate": "On-Time-In-Full rate (%)",
    "on_time_rate": "On-time delivery rate (%)",
    "in_full_rate": "In-full (no short-ship) rate (%)",
    "avg_lead_time": "Average actual lead time (days)",
    "avg_delay_days": "Average delay vs. promise (days)",
    "total_order_value": "Total order value ($)",
    "avg_unit_cost": "Average unit cost ($)",
    "order_count": "Number of orders",
}


# --------------------------------------------------------------------------- #
# 1. DATA CONNECTION
# --------------------------------------------------------------------------- #
def load_data(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """Load raw shipment data from one of several sources.

    Parameters
    ----------
    source : {"synthetic", "csv", "google_sheet", "sql", "rest_api", "kaggle"}
        Where to pull the data from. Only "synthetic" and "csv" are wired up;
        the rest are documented placeholders for your own project.
    **kwargs : passed through to the specific loader (e.g. path=..., url=...).
    """
    source = source.lower()
    if source == "synthetic":
        return _load_synthetic()
    if source == "csv":
        return _load_csv(kwargs.get("path", DEFAULT_CSV))
    if source == "google_sheet":
        return _load_google_sheet(**kwargs)
    if source == "sql":
        return _load_sql(**kwargs)
    if source == "rest_api":
        return _load_rest_api(**kwargs)
    if source == "kaggle":
        return _load_kaggle(**kwargs)
    raise ValueError(f"Unknown source '{source}'. "
                     f"Choose from: synthetic, csv, google_sheet, sql, rest_api, kaggle.")


def _load_synthetic() -> pd.DataFrame:
    """Read the generated CSV; generate it first if it doesn't exist yet."""
    if not os.path.exists(DEFAULT_CSV):
        import generate_synthetic_data
        generate_synthetic_data.main()
    return _load_csv(DEFAULT_CSV)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No CSV at {path}. Run `python generate_synthetic_data.py` first, "
            f"or pass source='synthetic'."
        )
    return pd.read_csv(path)


# ---- PLACEHOLDERS: copy one of these for your real project --------------- #
def _load_google_sheet(**kwargs) -> pd.DataFrame:
    """TODO(student): connect a published Google Sheet.

    The quickest path needs no auth — File ▸ Share ▸ Publish to web ▸ CSV,
    then read the published URL directly:

        url = kwargs["url"]   # the .../pub?output=csv link
        return pd.read_csv(url)

    For PRIVATE sheets, use gspread + a service account instead.
    """
    raise NotImplementedError(
        "google_sheet connector is a placeholder — see the docstring for the "
        "two-line published-CSV approach, or wire up gspread for private sheets."
    )


def _load_sql(**kwargs) -> pd.DataFrame:
    """TODO(student): pull from your operational database (ERP/WMS extract).

        from sqlalchemy import create_engine
        engine = create_engine(kwargs["conn_str"])     # e.g. postgresql+psycopg://...
        return pd.read_sql(kwargs["query"], engine)     # parametrize, never f-string user input
    """
    raise NotImplementedError("sql connector is a placeholder — see the docstring.")


def _load_rest_api(**kwargs) -> pd.DataFrame:
    """TODO(student): hit a shipping/ERP REST API (e.g. project44, SAP, ShipStation).

        import requests
        headers = {"Authorization": f"Bearer {os.environ['SHIP_API_KEY']}"}
        resp = requests.get(kwargs["url"], headers=headers, params=kwargs.get("params"),
                            timeout=30)
        resp.raise_for_status()
        return pd.json_normalize(resp.json()["data"])

    Remember: keep keys in environment variables, add retries/back-off for rate
    limits, and page through large result sets.
    """
    raise NotImplementedError("rest_api connector is a placeholder — see the docstring.")


def _load_kaggle(**kwargs) -> pd.DataFrame:
    """TODO(student): download a public Kaggle dataset.

        # pip install kaggle ; put kaggle.json in ~/.kaggle/
        import kaggle, glob
        kaggle.api.dataset_download_files(kwargs["dataset"], path="data", unzip=True)
        return pd.read_csv(glob.glob("data/*.csv")[0])
    """
    raise NotImplementedError("kaggle connector is a placeholder — see the docstring.")


# --------------------------------------------------------------------------- #
# 2. ERROR HANDLING / VALIDATION
# --------------------------------------------------------------------------- #
def clean_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Coerce types and flag bad rows WITHOUT crashing or silently dropping them.

    Returns
    -------
    (clean_df, report)
        clean_df : same rows, with parsed types plus two new columns:
                   `is_valid` (bool) and `dq_issue` (text, "" when clean).
        report   : dict of {issue_name: count} for display in the dashboard.
    """
    df = df.copy()

    # --- type coercion (bad values become NaT/NaN rather than raising) ---
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive a month label used as a dimension everywhere downstream
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    issues = pd.Series([""] * len(df), index=df.index)

    def flag(mask: pd.Series, label: str):
        mask = mask.fillna(False)
        issues.loc[mask & issues.eq("")] = label
        issues.loc[mask & issues.ne("") & ~issues.str.contains(label)] += f"; {label}"

    # impossible / nonsensical values
    flag(df["units_ordered"] <= 0, "non-positive units_ordered")
    flag(df["units_received"] < 0, "negative units_received")
    if "actual_delivery_date" in df.columns:
        flag(df["actual_delivery_date"] < df["order_date"], "delivery before order")
    # delivered rows that are missing their actuals are suspect (not the In-Transit ones)
    flag(df["status"].eq("Delivered") & df["actual_delivery_date"].isna(),
         "delivered but no actual date")

    df["dq_issue"] = issues
    df["is_valid"] = issues.eq("")

    report = {
        "total_rows": int(len(df)),
        "valid_rows": int(df["is_valid"].sum()),
        "flagged_rows": int((~df["is_valid"]).sum()),
        "in_transit_rows": int(df["status"].eq("In Transit").sum()),
    }
    # per-issue counts
    for label in ["non-positive units_ordered", "negative units_received",
                  "delivery before order", "delivered but no actual date"]:
        report[label] = int(df["dq_issue"].str.contains(label).sum())

    return df, report


if __name__ == "__main__":
    raw = load_data("synthetic")
    clean, rpt = clean_and_validate(raw)
    print(f"Loaded {len(raw):,} rows.")
    print("Validation report:", rpt)
    print("\nFlagged examples:")
    cols = ["order_id", "units_ordered", "units_received", "order_date",
            "actual_delivery_date", "dq_issue"]
    print(clean.loc[~clean["is_valid"], cols].head(10).to_string(index=False))
