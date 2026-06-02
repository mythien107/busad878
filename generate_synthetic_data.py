"""
generate_synthetic_data.py
===========================
Synthetic SUPPLY CHAIN / INVENTORY dataset generator for the BUSAD 878 Midterm
demo ("AI-Powered Operations Dashboard").

WHY THIS FILE EXISTS
--------------------
The midterm lets you (a) use a public dataset, (b) simulate your own, or
(c) use redacted company data. If you SIMULATE, the rubric asks you to submit
"the prompt and code you used to generate the dataset so I can see the
assumptions you make in your data distribution." THIS FILE IS THAT ARTIFACT —
a worked example of how to do it well: every assumption is written down, the
randomness is seeded so it is reproducible, and a few realistic data-quality
problems are injected on purpose so the dashboard's error-handling has
something real to catch.

THE STORY BAKED INTO THE DATA (so the dashboard has something to discover)
--------------------------------------------------------------------------
1. Overall On-Time-In-Full (OTIF) sits around the high-80s % but is DRIFTING
   DOWN over the trailing 12 months.
2. One supplier — "Pacific Components Co." (APAC / Ocean) — is the culprit:
   its lead times degrade month over month (a supplier-reliability problem),
   dragging APAC's OTIF from the low-90s down toward ~70%.
3. A COMMODITY COST SHOCK hits "Raw Materials" unit costs (~+28%) in the final
   two months — an anomaly an analyst should flag and explain.
4. Q4 (Oct–Dec) peak season adds congestion: longer delays across the board.
5. Ocean freight has structurally higher delay VARIANCE than Air/Ground.

INJECTED DATA-QUALITY ISSUES (for the "graceful handling of bad inputs" demo)
-----------------------------------------------------------------------------
- ~6% of rows are still "In Transit": actual_delivery_date / actual lead time
  are blank (NaN) — the dashboard must not crash or count these as "late".
- A handful of rows have units_ordered <= 0 or units_received < 0 (bad entry).
- A couple of rows have an actual_delivery_date BEFORE the order_date (impossible).

Run:  python generate_synthetic_data.py
Out:  data/supply_chain_shipments.csv  (+ a printed assumptions summary)
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. Reproducibility + horizon
# --------------------------------------------------------------------------- #
SEED = 878  # the course number, so anyone re-running this gets identical data
rng = np.random.default_rng(SEED)

DATA_START = date(2025, 5, 1)
DATA_END = date(2026, 5, 15)          # "as-of" date for the dataset
N_ORDERS = 3600                        # ~ 300 purchase-order lines / month

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(OUT_DIR, "supply_chain_shipments.csv")

# --------------------------------------------------------------------------- #
# 1. Reference dimensions (the "master data")
# --------------------------------------------------------------------------- #
# Each supplier: region, the transport mode they typically use, a baseline
# promised lead time (days), a base reliability (lower = more on-time), and a
# monthly degradation rate (extra delay added per elapsed month).
SUPPLIERS = {
    #  name                       region          mode      lead  reliab  degrade/mo  short_ship
    "Allegheny Metals":         ("North America", "Ground",  7,    0.35,   0.00,       0.03),
    "Great Lakes Packaging":    ("North America", "Ground",  6,    0.30,   0.00,       0.02),
    "Rhine Components GmbH":     ("EMEA",          "Air",     9,    0.40,   0.02,       0.04),
    "Iberia Logistics SA":      ("EMEA",          "Ocean",   24,   0.55,   0.05,       0.05),
    "Pacific Components Co.":    ("APAC",          "Ocean",   34,   0.60,   0.45,       0.09),  # <-- the degrader
    "Shenzhen Precision Ltd":   ("APAC",          "Ocean",   32,   0.50,   0.08,       0.06),
    "Bangalore ElectroTech":    ("APAC",          "Air",     12,   0.45,   0.03,       0.04),
    "Monterrey Assembly":       ("LATAM",         "Ground",  11,   0.45,   0.04,       0.05),
    "São Paulo Supply":         ("LATAM",         "Ocean",   28,   0.58,   0.06,       0.06),
}

CATEGORIES = {
    #  category          unit_cost_mean  unit_cost_sd   order_qty_lambda
    "Electronics":      (42.0,          9.0,           120),
    "Raw Materials":    (8.5,           2.0,           2200),
    "Packaging":        (1.2,           0.3,           5400),
    "Components":       (15.0,          4.0,           600),
    "Finished Goods":   (78.0,          18.0,          90),
}

# Which categories each supplier tends to ship (keeps the data coherent)
SUPPLIER_CATEGORIES = {
    "Allegheny Metals":      ["Raw Materials", "Components"],
    "Great Lakes Packaging": ["Packaging"],
    "Rhine Components GmbH":  ["Components", "Electronics"],
    "Iberia Logistics SA":   ["Finished Goods", "Packaging"],
    "Pacific Components Co.": ["Components", "Electronics"],
    "Shenzhen Precision Ltd":["Electronics", "Components"],
    "Bangalore ElectroTech": ["Electronics"],
    "Monterrey Assembly":    ["Finished Goods", "Components"],
    "São Paulo Supply":      ["Raw Materials", "Finished Goods"],
}

CARRIERS_BY_MODE = {
    "Ground": ["RoadRunner Freight", "Continental Trucking"],
    "Air":    ["SkyBridge Air Cargo", "AeroFast"],
    "Ocean":  ["BlueWave Lines", "Meridian Shipping"],
    "Rail":   ["IronHorse Rail"],
}

DEST_DCS = ["DC-East (PA)", "DC-Central (TX)", "DC-West (CA)", "DC-South (GA)"]

SUPPLIER_NAMES = list(SUPPLIERS.keys())
CATEGORY_NAMES = list(CATEGORIES.keys())

# --------------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------------- #
HORIZON_DAYS = (DATA_END - DATA_START).days


def _months_elapsed(d: date) -> float:
    """Fractional months since DATA_START (drives the degradation trend)."""
    return (d - DATA_START).days / 30.0


def _seasonal_bump(d: date) -> float:
    """Extra expected delay (days) from Q4 peak-season congestion."""
    return 1.3 if d.month in (10, 11, 12) else 0.0


def _sample_order_dates(n: int) -> list[date]:
    """Order dates over the horizon, weighted slightly toward Q4 (peak season)."""
    offsets = rng.integers(0, HORIZON_DAYS + 1, size=n * 3)  # oversample, then thin
    dates = [DATA_START + timedelta(days=int(o)) for o in offsets]
    weights = np.array([1.6 if dt.month in (10, 11, 12) else 1.0 for dt in dates])
    weights = weights / weights.sum()
    chosen = rng.choice(len(dates), size=n, replace=False, p=weights)
    return sorted(dates[i] for i in chosen)


# --------------------------------------------------------------------------- #
# 3. Generate the rows
# --------------------------------------------------------------------------- #
def generate() -> pd.DataFrame:
    order_dates = _sample_order_dates(N_ORDERS)
    rows = []

    for i, od in enumerate(order_dates, start=1):
        supplier = rng.choice(SUPPLIER_NAMES)
        region, mode, base_lead, reliab, degrade, short_ship_p = SUPPLIERS[supplier]
        category = rng.choice(SUPPLIER_CATEGORIES[supplier])
        cost_mean, cost_sd, qty_lambda = CATEGORIES[category]

        # --- promised lead time (what the supplier committed to) ---
        promised_lead = max(2, int(round(base_lead + rng.normal(0, 1.0))))

        # --- actual delay drivers ---
        # Suppliers commit conservatively, so a healthy shipment lands ON or
        # slightly BEFORE the promise: the baseline delay term is negative.
        # On top of that sit (1) a per-supplier (un)reliability offset,
        # (2) a degradation TREND that grows month over month for weak
        # suppliers, (3) Q4 peak-season congestion, and (4) larger random
        # VARIANCE on Ocean freight. The result: ~mid-80s% on-time overall,
        # drifting down, with the decline concentrated in the degrader + Q4.
        trend_delay = degrade * _months_elapsed(od)
        season_delay = _seasonal_bump(od)
        mode_sigma = 3.2 if mode == "Ocean" else (2.0 if mode == "Air" else 1.6)
        base_delay = rng.normal(-3.5 + reliab * 1.5, mode_sigma)
        delay_component = base_delay + trend_delay + season_delay
        actual_lead = max(1, int(round(promised_lead + delay_component)))

        # --- quantities & cost ---
        units_ordered = int(max(1, rng.poisson(qty_lambda)))
        # commodity cost shock on Raw Materials in the final two months
        shock = 1.28 if (category == "Raw Materials" and od >= date(2026, 3, 15)) else 1.0
        unit_cost = round(max(0.2, rng.normal(cost_mean, cost_sd)) * shock, 2)

        # short-shipments (received < ordered) at the supplier's short-ship rate
        if rng.random() < short_ship_p:
            units_received = int(units_ordered * rng.uniform(0.80, 0.98))
        else:
            units_received = units_ordered

        # --- dates derived from leads ---
        promised_delivery = od + timedelta(days=promised_lead)
        actual_delivery = od + timedelta(days=actual_lead)

        # --- status: anything that would deliver after the as-of date is "In Transit" ---
        if actual_delivery > DATA_END:
            status = "In Transit"
            actual_delivery = pd.NaT
            actual_lead_val = np.nan
            units_received = np.nan          # not yet received
            delay_days = np.nan
            on_time = np.nan
            in_full = np.nan
        else:
            status = "Delivered"
            actual_lead_val = actual_lead
            delay_days = max(0, actual_lead - promised_lead)
            on_time = actual_lead <= promised_lead
            in_full = units_received >= units_ordered

        carrier = rng.choice(CARRIERS_BY_MODE[mode])
        dc = rng.choice(DEST_DCS)

        rows.append(
            {
                "order_id": f"PO-{od.year}-{i:05d}",
                "order_date": od,
                "supplier": supplier,
                "supplier_region": region,
                "product_category": category,
                "sku": f"{category[:3].upper()}-{rng.integers(1000, 9999)}",
                "transport_mode": mode,
                "carrier": carrier,
                "destination_dc": dc,
                "units_ordered": units_ordered,
                "units_received": units_received,
                "unit_cost": unit_cost,
                "promised_lead_time_days": promised_lead,
                "actual_lead_time_days": actual_lead_val,
                "promised_delivery_date": promised_delivery,
                "actual_delivery_date": actual_delivery,
                "delay_days": delay_days,
                "status": status,
                "on_time": on_time,
                "in_full": in_full,
            }
        )

    df = pd.DataFrame(rows)

    # order_value uses the ORDERED quantity (committed spend) so it is always present
    df["order_value"] = (df["units_ordered"] * df["unit_cost"]).round(2)
    # OTIF only defined for delivered rows
    df["otif"] = np.where(
        df["status"].eq("Delivered"),
        (df["on_time"].fillna(False) & df["in_full"].fillna(False)),
        np.nan,
    )

    # ----------------------------------------------------------------------- #
    # 4. Inject realistic data-quality problems (on purpose!)
    # ----------------------------------------------------------------------- #
    delivered_idx = df.index[df["status"].eq("Delivered")].to_numpy()

    # (a) a few impossible deliveries (actual before order)
    bad_dates = rng.choice(delivered_idx, size=2, replace=False)
    for ix in bad_dates:
        df.loc[ix, "actual_delivery_date"] = df.loc[ix, "order_date"] - timedelta(days=3)

    # (b) a few non-positive ordered quantities (data-entry errors)
    bad_qty = rng.choice(delivered_idx, size=4, replace=False)
    df.loc[bad_qty, "units_ordered"] = rng.choice([0, -5, -1, 0], size=4)

    # (c) a couple of negative received quantities
    bad_recv = rng.choice(delivered_idx, size=2, replace=False)
    df.loc[bad_recv, "units_received"] = -3

    return df


# --------------------------------------------------------------------------- #
# 5. Write + summarize
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_CSV, index=False)

    delivered = df[df["status"].eq("Delivered")].copy()
    otif_rate = delivered["otif"].mean()

    print("=" * 70)
    print("SYNTHETIC SUPPLY-CHAIN DATASET GENERATED")
    print("=" * 70)
    print(f"File           : {OUT_CSV}")
    print(f"Rows           : {len(df):,}")
    print(f"Date range     : {df['order_date'].min()}  ->  {df['order_date'].max()}")
    print(f"Suppliers      : {df['supplier'].nunique()}   Regions: {df['supplier_region'].nunique()}")
    print(f"In Transit     : {(df['status'].eq('In Transit')).sum()} rows (blank actuals)")
    print(f"Overall OTIF   : {otif_rate:0.1%} (delivered rows)")
    print("-" * 70)
    print("OTIF by region (delivered):")
    print((delivered.groupby('supplier_region')['otif'].mean().sort_values()
           .apply(lambda x: f'{x:0.1%}')).to_string())
    print("-" * 70)
    print("Injected issues to catch: 2 impossible dates, 4 bad order qtys, "
          "2 negative receipts, plus In-Transit blanks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
