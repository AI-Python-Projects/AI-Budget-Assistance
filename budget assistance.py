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