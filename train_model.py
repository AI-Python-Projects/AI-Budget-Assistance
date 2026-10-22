"""
train_model.py
Trains the AI Budget Assistant categoriser and prints rich tables + charts.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")               # safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)

from config import TRAIN_CSV_PATH, TEST_CSV_PATH, MODEL_PATH, OUTPUT_DIR
from preprocessing import load_and_prepare

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# =====================================================================
# Helper: pretty console table
# =====================================================================
from tabulate import tabulate

def show_table(title, df, float_fmt="{:,.2f}"):
    print(f"\n=== {title} ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid",
                   floatfmt=",.2f", showindex=False))


# =====================================================================
# 1. Load + clean
# =====================================================================
print("=" * 70)
print("AI BUDGET ASSISTANT — MODEL TRAINING")
print("=" * 70)
print(f"\nLoading training data from: {TRAIN_CSV_PATH}")

df = load_and_prepare(TRAIN_CSV_PATH)

# =====================================================================
# 2. EDA tables
# =====================================================================
show_table("Dataset Snapshot (first 10 rows)",
           df[["merchant", "amount", "day_of_month",
               "day_of_week", "is_weekend", "is_month_end", "category"]].head(10))

show_table("Numeric Summary",
           df[["amount", "day_of_month", "day_of_week",
               "is_weekend", "is_month_end"]].describe().round(2).reset_index()
             .rename(columns={"index": "stat"}))

# Class distribution
class_counts = df["category"].value_counts().reset_index()
class_counts.columns = ["category", "count"]
class_counts["% of total"] = (class_counts["count"] / class_counts["count"].sum() * 100).round(2)

show_table("Class Distribution", class_counts)

# Average spend per category
avg_spend = (df.groupby("category")["amount"]
               .agg(["mean", "median", "min", "max", "sum", "count"])
               .round(2)
               .reset_index()
               .sort_values("mean", ascending=False))
show_table("Spend per Category (R)", avg_spend)

# =====================================================================
# 3. Graph: Class distribution (bar + pie)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart
sns.barplot(data=class_counts.sort_values("count", ascending=False),
            x="count", y="category", ax=axes[0], palette="viridis")
axes[0].set_title("Transactions per Category", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of transactions")
axes[0].set_ylabel("")
for i, v in enumerate(class_counts.sort_values("count", ascending=False)["count"]):
    axes[0].text(v + 0.5, i, str(v), va="center", fontsize=9)