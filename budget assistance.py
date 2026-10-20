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
    
# =====================================================================
# Report + charts
# =====================================================================
def report(result, tag="report"):
    details = result["details"]
    totals  = result["totals"]
    alerts  = result["alerts"]

    # ---------------------------------------------------------------
    # 1. Categorised transactions table
    # ---------------------------------------------------------------
    txn_df = pd.DataFrame(details)[["merchant", "amount", "category"]]
    txn_df.columns = ["Merchant", "Amount (R)", "Category"]
    txn_df["Amount (R)"] = txn_df["Amount (R)"].round(2)
    show_table("Categorised Transactions", txn_df)

    # ---------------------------------------------------------------
    # 2. Budget summary table
    # ---------------------------------------------------------------
    rows = []
    total_spent = total_budget = 0.0
    for cat, spent in totals.items():
        budget = BUDGET.get(cat, 0)
        total_spent  += spent
        total_budget += budget
        pct = (spent / budget * 100) if budget else 0
        remaining = budget - spent
        status = "OK"
        if budget > 0:
            if pct >= DANGER_THRESHOLD * 100:
                status = "OVER"
            elif pct >= WARN_THRESHOLD * 100:
                status = "WARN"
        rows.append({
            "Category": cat,
            "Budget (R)": round(budget, 2),
            "Spent (R)": round(spent, 2),
            "Remaining (R)": round(remaining, 2),
            "% Used": round(pct, 1),
            "Status": status,
        })

    summary_df = pd.DataFrame(rows).sort_values("% Used", ascending=False)
    show_table("Budget Summary", summary_df)

    totals_df = pd.DataFrame({
        "Metric": ["Total spent", "Total budget", "Remaining"],
        "Value (R)": [round(total_spent, 2),
                      round(total_budget, 2),
                      round(total_budget - total_spent, 2)],
    })
    show_table("Overall Totals", totals_df)

    # ---------------------------------------------------------------
    # 3. Alerts table
    # ---------------------------------------------------------------
    if alerts:
        alerts_df = pd.DataFrame([
            {"Level": a["level"], "Category": a["category"],
             "Spent (R)": round(a["spent"], 2),
             "Budget (R)": round(a["budget"], 2),
             "Remaining (R)": round(a["remaining"], 2),
             "% Used": round(a["pct"], 1)}
            for a in alerts
        ])
        show_table("Alerts", alerts_df)
    else:
        print("\n=== Alerts ===\n  All categories within budget.")

    # ---------------------------------------------------------------
    # 4. Chart — Budget vs Spent per category
    # ---------------------------------------------------------------
    chart_df = summary_df.copy()
    chart_df = chart_df[chart_df["Budget (R)"] > 0]

    if not chart_df.empty:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.5 * len(chart_df))))
        y = np.arange(len(chart_df))
        h = 0.4
        ax.barh(y - h/2, chart_df["Budget (R)"],  height=h,
                label="Budget",  color="#4C72B0")
        ax.barh(y + h/2, chart_df["Spent (R)"],   height=h,
                label="Spent",   color="#DD8452")
        ax.set_yticks(y)
        ax.set_yticklabels(chart_df["Category"])
        ax.set_xlabel("Rand (R)")
        ax.set_title("Budget vs Spent per Category",
                     fontsize=14, fontweight="bold")
        ax.legend()

        # Highlight over-budget bars in red
        for i, row in enumerate(chart_df.itertuples()):
            if row._5 >= DANGER_THRESHOLD * 100:   # % Used column
                ax.text(row._3 + 20, i + h/2, "OVER",
                        color="red", va="center", fontsize=9, fontweight="bold")
            elif row._5 >= WARN_THRESHOLD * 100:
                ax.text(row._3 + 20, i + h/2, "WARN",
                        color="orange", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, f"assistant_{tag}_budget_vs_spent.png")
        plt.savefig(p1, bbox_inches="tight")
        plt.close()
        print(f"\n[Chart saved] {p1}")

    # ---------------------------------------------------------------
    # 5. Chart — % Used gauge (horizontal bars)
    # ---------------------------------------------------------------
    if not chart_df.empty:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(chart_df))))
        colors = ["#d62728" if p >= 100
                  else "#ff7f0e" if p >= WARN_THRESHOLD * 100
                  else "#2ca02c"
                  for p in chart_df["% Used"]]
        bars = ax.barh(chart_df["Category"], chart_df["% Used"],
                       color=colors)
        ax.axvline(100, color="red", linestyle="--", label="Budget limit (100%)")
        ax.axvline(WARN_THRESHOLD * 100, color="orange",
                   linestyle=":", label=f"Warning ({int(WARN_THRESHOLD*100)}%)")
        ax.set_xlabel("% of budget used")
        ax.set_title("Budget Utilisation per Category",
                     fontsize=14, fontweight="bold")
        ax.set_xlim(0, max(120, chart_df["% Used"].max() + 20))

        for b, pct in zip(bars, chart_df["% Used"]):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2,
                    f"{pct:.1f}%", va="center", fontsize=9)

        ax.legend(loc="lower right")
        plt.tight_layout()
        p2 = os.path.join(OUTPUT_DIR, f"assistant_{tag}_utilisation.png")
        plt.savefig(p2, bbox_inches="tight")
        plt.close()
        print(f"[Chart saved] {p2}")

    # ---------------------------------------------------------------
    # 6. Chart — Pie of spending by category
    # ---------------------------------------------------------------
    pie_df = summary_df[summary_df["Spent (R)"] > 0]
    if not pie_df.empty:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.pie(pie_df["Spent (R)"],
               labels=pie_df["Category"],
               autopct="%1.1f%%",
               startangle=140,
               textprops={"fontsize": 9})
        ax.set_title("Spending Distribution",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        p3 = os.path.join(OUTPUT_DIR, f"assistant_{tag}_spending_pie.png")
        plt.savefig(p3, bbox_inches="tight")
        plt.close()
        print(f"[Chart saved] {p3}")

    # ---------------------------------------------------------------
    # 7. Chart — Spend over time (if dates are present)
    # ---------------------------------------------------------------
    if any("day_of_month" in d for d in details):
        txns_by_day = (
            pd.DataFrame(details)
              .groupby("day_of_month")["amount"].sum()
              .reset_index()
              .sort_values("day_of_month")
        )
        if not txns_by_day.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.lineplot(data=txns_by_day, x="day_of_month", y="amount",
                         marker="o", color="#4C72B0", ax=ax)
            ax.set_title("Daily Spend over the Month",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Day of month")
            ax.set_ylabel("Total spent (R)")
            plt.tight_layout()
            p4 = os.path.join(OUTPUT_DIR,
                              f"assistant_{tag}_daily_spend.png")
            plt.savefig(p4, bbox_inches="tight")
            plt.close()
            print(f"[Chart saved] {p4}")

    # ---------------------------------------------------------------
     # 8. Console alert block (text)
    # ---------------------------------------------------------------
    print("\n========== AI BUDGET ASSISTANT REPORT ==========")
    if alerts:
        for a in alerts:
            prefix = "🚨" if a["level"] == "OVER" else "⚠️ "
            print(f"  {prefix} {a['message']}")
    else:
        print("  ✅ All categories within budget.")


# =====================================================================
# Interactive + Batch modes
# =====================================================================
def interactive():
    print("\n=== AI BUDGET ASSISTANT (interactive) ===")
    print("Type 'done' when finished.\n")
    txns = []
    while True:
        m = input("Merchant : ").strip()
        if m.lower() == "done":
            break
        try:
            a = float(input("Amount   : ").strip())
        except ValueError:
            print("Invalid amount.\n")
            continue
        txns.append({"merchant": m, "amount": a})
        print("Added.\n")
    report(analyse(txns), tag="interactive")


def csv_batch(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Description": "merchant", "Amount": "amount"})
    if "Transaction Type" in df.columns:
        df = df[df["Transaction Type"].str.lower() == "debit"]

    # keep date features if present
    txns = df.to_dict(orient="records")
    report(analyse(txns), tag=os.path.splitext(os.path.basename(path))[0])


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_batch(sys.argv[1])
    else:
        interactive()