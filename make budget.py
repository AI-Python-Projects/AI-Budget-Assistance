"""
make_budget.py
Derives a monthly budget for each category from your historical transactions
and writes data/budget.csv. Also prints rich tables + saves charts.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate

from config import TRAIN_CSV_PATH, BUDGET_CSV_PATH, OUTPUT_DIR
from preprocessing import load_and_prepare

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BUFFER = 1.10     # 10 % buffer over average monthly spend


# =====================================================================
# Helper
# =====================================================================
def show_table(title, df):
    print(f"\n=== {title} ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid",
                   floatfmt=",.2f", showindex=False))


# =====================================================================
# 1. Load
# =====================================================================
print("=" * 70)
print("AI BUDGET ASSISTANT — AUTO BUDGET GENERATION")
print("=" * 70)
print(f"\nReading transactions from: {TRAIN_CSV_PATH}")

df = load_and_prepare(TRAIN_CSV_PATH)

# Make sure date is proper
df = df.dropna(subset=["date"]).copy()
df["month"] = df["date"].dt.to_period("M")

# =====================================================================
# 2. EDA tables
# =====================================================================
overview = pd.DataFrame({
    "metric": [
        "total transactions",
        "unique categories",
        "months covered",
        "date range start",
        "date range end",
        "total spend (R)",
    ],

     "value": [
        len(df),
        df["category"].nunique(),
        df["month"].nunique(),
        str(df["date"].min().date()),
        str(df["date"].max().date()),
        round(df["amount"].sum(), 2),
    ],
})
show_table("Data Overview", overview)

# Monthly spend per category
monthly = (df.groupby(["month", "category"])["amount"]
             .sum()
             .reset_index())

monthly_pivot = (monthly.pivot(index="month", columns="category", values="amount")
                        .fillna(0)
                        .round(2))

show_table("Monthly Spend per Category (R)",
           monthly_pivot.reset_index().astype({"month": str}))

# Average & std per category
avg   = monthly.groupby("category")["amount"].mean().round(2)
std   = monthly.groupby("category")["amount"].std().round(2).fillna(0)
count = monthly.groupby("category")["amount"].count()

stats = pd.DataFrame({
    "category": avg.index,
    "months_seen": count.values,
    "avg_monthly_spend": avg.values,
    "std_dev": std.values,
    "min_month": monthly.groupby("category")["amount"].min().values,
    "max_month": monthly.groupby("category")["amount"].max().values,
}).sort_values("avg_monthly_spend", ascending=False)

show_table("Monthly Spending Statistics per Category", stats)

# =====================================================================
# 3. Chart — monthly spend per category (line chart)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))
for cat in monthly_pivot.columns:
    ax.plot(monthly_pivot.index.astype(str),
            monthly_pivot[cat],
            marker="o", label=cat, linewidth=1.5)
ax.set_title("Monthly Spend by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Spend (R)")
ax.legend(loc="upper left", ncol=2, fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
line_png = os.path.join(OUTPUT_DIR, "budget_monthly_spend_lines.png")
plt.savefig(line_png, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved] {line_png}")

# =====================================================================
# 4. Chart — avg spend per category (bar)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(stats))))
stats_sorted = stats.sort_values("avg_monthly_spend")
ax.barh(stats_sorted["category"], stats_sorted["avg_monthly_spend"],
        color="#4C72B0", label="Avg monthly spend")
ax.errorbar(stats_sorted["avg_monthly_spend"],
            stats_sorted["category"],
            xerr=stats_sorted["std_dev"],
            fmt="none", ecolor="red", capsize=4, label="±1 std dev")
ax.set_title("Average Monthly Spend per Category",
             fontsize=14, fontweight="bold")
ax.set_xlabel("R")
ax.legend()
plt.tight_layout()
avg_png = os.path.join(OUTPUT_DIR, "budget_avg_monthly_spend.png")
plt.savefig(avg_png, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {avg_png}")


# =====================================================================
# 5. Compute budget = avg × buffer
# =====================================================================
budget = (avg * BUFFER).round(0).astype(int).reset_index()
budget.columns = ["category", "monthly_budget"]
budget = budget.sort_values("category").reset_index(drop=True)

# Add comparison columns
compare = pd.DataFrame({
    "category":          budget["category"],
    "avg_spend":         avg.reindex(budget["category"]).values,
    "buffer_%":          [f"+{int((BUFFER - 1) * 100)}%"] * len(budget),
    "budget (R)":        budget["monthly_budget"].values,
    "increase (R)":      (budget["monthly_budget"].values
                          - avg.reindex(budget["category"]).values).round(2),
})
show_table("Budget Derivation (avg × 1.10)", compare)

# =====================================================================
# 6. Chart — avg vs budget side by side
# =====================================================================
fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(budget))))
y = np.arange(len(budget))
h = 0.4
ax.barh(y - h/2, avg.reindex(budget["category"]).values,
        height=h, label="Avg spend", color="#4C72B0")
ax.barh(y + h/2, budget["monthly_budget"].values,
        height=h, label=f"Budget (×{BUFFER})", color="#DD8452")
ax.set_yticks(y)
ax.set_yticklabels(budget["category"])
ax.set_xlabel("R")
ax.set_title("Average Spend vs Recommended Budget",
             fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
cmp_png = os.path.join(OUTPUT_DIR, "budget_avg_vs_budget.png")
plt.savefig(cmp_png, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {cmp_png}")