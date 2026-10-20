"""
budget_assistant.py
Interactive / batch AI Budget Assistant with rich tables + charts.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate

from config import (
    BUDGET_CSV_PATH, OUTPUT_DIR,
    WARN_THRESHOLD, DANGER_THRESHOLD,
)
from predict import predict_one

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# =====================================================================
# Load budget
# =====================================================================
budget_df = pd.read_csv(BUDGET_CSV_PATH)
budget_df.columns = [c.strip().lower() for c in budget_df.columns]
BUDGET = dict(zip(budget_df["category"], budget_df["monthly_budget"]))


# =====================================================================
# Helpers
# =====================================================================
def money(v):
    return f"R{v:,.2f}"


def show_table(title, df):
    print(f"\n=== {title} ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid",
                   floatfmt=",.2f", showindex=False))

                   # =====================================================================
# Core analysis
# =====================================================================
def analyse(transactions):
    totals = {cat: 0.0 for cat in BUDGET}
    details = []

    for t in transactions:
        cat = predict_one(
            t["merchant"], float(t["amount"]),
            t.get("day_of_month", 15), t.get("day_of_week", 2),
            t.get("is_weekend", 0), t.get("is_month_end", 0),
        )
        totals[cat] = totals.get(cat, 0.0) + float(t["amount"])
        details.append({**t, "category": cat})

    alerts = []
    for cat, spent in totals.items():
        budget = BUDGET.get(cat, 0)
        if budget <= 0:
            continue
        ratio = spent / budget
        remaining = budget - spent
        if ratio >= DANGER_THRESHOLD:
            alerts.append({
                "level": "OVER",
                "category": cat,
                "spent": spent,
                "budget": budget,
                "remaining": remaining,
                "pct": ratio * 100,
                "message": f"OVER BUDGET  - {cat}: spent {money(spent)} of "
                           f"{money(budget)} (over by {money(abs(remaining))})",
            })
        elif ratio >= WARN_THRESHOLD:
            alerts.append({
                "level": "WARN",
                "category": cat,
                "spent": spent,
                "budget": budget,
                "remaining": remaining,
                "pct": ratio * 100,
                "message": f"APPROACHING - {cat}: spent {money(spent)} of "
                           f"{money(budget)} (remaining {money(remaining)})",
            })

    return {"details": details, "totals": totals, "alerts": alerts}