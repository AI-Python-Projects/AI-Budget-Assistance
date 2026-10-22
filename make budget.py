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