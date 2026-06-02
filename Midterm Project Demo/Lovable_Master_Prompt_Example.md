# Master Prompt



Build a clean, professional, single-page "AI-Powered Operations Dashboard" for a supply-chain operations manager. Use React with the Recharts charting library and a modern card-based layout (Tailwind). Make it fully responsive.

Use EXACTLY this data as a hardcoded constant (do not invent other numbers):

const DATA = {

"kpis": { "otif": 80.3, "on_time": 84.8, "in_full": 94.8, "avg_lead_days": 15.6, "total_order_value": 30022163, "orders": 3600 },

"otif_by_month": [ {"month":"2025-05","otif":84.5},{"month":"2025-06","otif":92.5},{"month":"2025-07","otif":87.5},{"month":"2025-08","otif":85.9},{"month":"2025-09","otif":87.8},{"month":"2025-10","otif":67.2},{"month":"2025-11","otif":69.0},{"month":"2025-12","otif":70.9},{"month":"2026-01","otif":85.4},{"month":"2026-02","otif":82.6},{"month":"2026-03","otif":79.2},{"month":"2026-04","otif":86.9},{"month":"2026-05","otif":95.6} ],

"otif_by_region": [ {"region":"APAC","otif":71.5},{"region":"EMEA","otif":78.9},{"region":"LATAM","otif":82.1},{"region":"North America","otif":91.8} ],

"otif_by_supplier": [ {"supplier":"Pacific Components Co.","otif":50.0},{"supplier":"Iberia Logistics SA","otif":73.5},{"supplier":"Sao Paulo Supply","otif":74.2},{"supplier":"Shenzhen Precision Ltd","otif":74.7},{"supplier":"Rhine Components GmbH","otif":84.4},{"supplier":"Bangalore ElectroTech","otif":88.8},{"supplier":"Monterrey Assembly","otif":88.9},{"supplier":"Allegheny Metals","otif":91.6},{"supplier":"Great Lakes Packaging","otif":92.1} ],

"raw_materials_cost_by_month": [ {"month":"2025-05","cost":8.78},{"month":"2025-06","cost":8.40},{"month":"2025-07","cost":8.88},{"month":"2025-08","cost":8.52},{"month":"2025-09","cost":8.49},{"month":"2025-10","cost":8.74},{"month":"2025-11","cost":8.36},{"month":"2025-12","cost":9.03},{"month":"2026-01","cost":8.58},{"month":"2026-02","cost":8.78},{"month":"2026-03","cost":10.21},{"month":"2026-04","cost":10.89},{"month":"2026-05","cost":11.18} ],

"anomalies": [ {"type":"OTIF drop","scope":"Iberia Logistics SA - 2025-11","detail":"OTIF 47% vs baseline 77% (-30 pp)","action":"Check for a one-off disruption that month; confirm it recovered."},{"type":"Cost spike","scope":"Raw Materials - 2026-04 to 2026-05","detail":"avg unit cost rose ~+27% vs the $8.78 baseline","action":"Confirm market vs contract change; consider hedging or re-sourcing."},{"type":"OTIF drop","scope":"Shenzhen Precision Ltd - 2025-10","detail":"OTIF 54% vs baseline 77% (-23 pp)","action":"Peak-season congestion; verify capacity plan for next Q4."},{"type":"Declining trend","scope":"Pacific Components Co.","detail":"OTIF slid from 52% to 36% over the last 90 days","action":"Open a supplier performance review - the slow slide is hidden by the company-wide average."} ]

};

LAYOUT (top to bottom):

1. Header: title "AI-Powered Operations Dashboard", subtitle "Supply-chain OTIF, lead times & cost - synthetic demo data".

   

2. A row of 5 KPI cards: OTIF 80.3%, On-time 84.8%, In-full 94.8%, Avg lead time 15.6 days, Total order value $30.0M. Color the OTIF card amber (it is below a 95% target).

   

3. A 2-column chart grid: (a) "OTIF trend by month" line chart from otif_by_month, y-axis 0-100, with a dashed reference line at 95 labeled "Target" - note the clear dip in Oct-Dec 2025; (b) "OTIF by supplier" horizontal bar chart from otif_by_supplier sorted ascending, bars colored red->yellow->green by value (lowest = red); (c) "OTIF by region" vertical bar chart from otif_by_region; (d) "Raw Materials unit cost by month" line chart from raw_materials_cost_by_month, highlighting the jump at the end.

   

4. An "AI Insights" panel: render each item in anomalies as a card with a colored badge for type, bold scope, the detail, and the recommended action.

   

5. An "Ask a question" box: a text input where the user types things like "worst region", "worst supplier", or "cost trend", and below it show a plain-English answer derived from DATA using simple keyword matching (no external API). Example: "worst region" -> "APAC has the lowest OTIF at 71.5%."

   

STYLE: executive-friendly, lots of white space, rounded cards with subtle shadows, a calm blue/slate palette, good typography. Add a small footer: "Teaching demo on synthetic data - verify AI output before acting."