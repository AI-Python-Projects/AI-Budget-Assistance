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
    
# Pie chart
axes[1].pie(class_counts["count"],
            labels=class_counts["category"],
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 9})
axes[1].set_title("Category Share", fontsize=14, fontweight="bold")

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(bar_path, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved] {bar_path}")

# =====================================================================
# 4. Graph: Average spend per category
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 7))
avg_sorted = avg_spend.sort_values("mean", ascending=True)
sns.barplot(data=avg_sorted, x="mean", y="category", palette="magma", ax=ax)
ax.set_title("Average Spend per Category (R)", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean amount (R)")
ax.set_ylabel("")
for i, v in enumerate(avg_sorted["mean"]):
    ax.text(v + 5, i, f"R{v:,.0f}", va="center", fontsize=9)
plt.tight_layout()
spend_path = os.path.join(OUTPUT_DIR, "avg_spend_per_category.png")
plt.savefig(spend_path, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {spend_path}")

# =====================================================================
# 5. Drop rare categories
# =====================================================================
MIN_SAMPLES = 5
counts = df["category"].value_counts()
rare = counts[counts < MIN_SAMPLES].index.tolist()

print("\n" + "=" * 70)
if rare:
    print(f"Dropping rare categories (< {MIN_SAMPLES} samples): {rare}")
    df = df[~df["category"].isin(rare)].reset_index(drop=True)
else:
    print(f"No categories with fewer than {MIN_SAMPLES} samples — nothing dropped.")

print(f"Kept {len(df)} rows across {df['category'].nunique()} categories.")

# =====================================================================
# 6. Hold-out split
# =====================================================================
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category"]
)
test_df.to_csv(TEST_CSV_PATH, index=False)

split_table = pd.DataFrame({
    "split": ["train", "test"],
    "rows":  [len(train_df), len(test_df)],
    "%":     [round(len(train_df)/len(df)*100, 1),
              round(len(test_df)/len(df)*100, 1)],
})
show_table("Train / Test Split", split_table)
print(f"Testing split saved to: {TEST_CSV_PATH}")

# =====================================================================
# 7. Preprocessing + Model
# =====================================================================
numeric_cols = ["amount", "day_of_month", "day_of_week",
                "is_weekend", "is_month_end"]

preprocessor = ColumnTransformer(
    transformers=[
        ("merchant_tfidf",
         TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
         "merchant"),
        ("num", StandardScaler(), numeric_cols),
    ]
)

X_train = train_df.drop(columns=["category", "date"])
y_train = train_df["category"]
X_test  = test_df.drop(columns=["category", "date"])
y_test  = test_df["category"]

# =====================================================================
# 8. GridSearch
# =====================================================================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_grid = {
    "classifier__n_estimators":      [50, 100, 150],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__max_features":      ["sqrt", "log2"],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf":  [1, 2],
}

print("\n" + "=" * 70)
print("Running GridSearchCV (this may take a minute)...")
print("=" * 70)

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

# Pretty print best params
params_table = pd.DataFrame(
    list(grid_search.best_params_.items()),
    columns=["parameter", "value"]
)
show_table("Best Hyperparameters", params_table)
print(f"Best CV score: {grid_search.best_score_:.4f}")

# =====================================================================
# 9. Evaluate on train + test
# =====================================================================
y_train_pred = best_clf.predict(X_train)
y_test_pred  = best_clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)

metrics_table = pd.DataFrame({
    "metric": ["train accuracy", "test accuracy", "overfit gap"],
    "value":  [round(train_acc, 4),
               round(test_acc, 4),
               round(train_acc - test_acc, 4)],
})
show_table("Model Accuracy", metrics_table)

# Full classification report (test)
report_dict = classification_report(y_test, y_test_pred,
                                    output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T.reset_index()
report_df = report_df.rename(columns={"index": "class"})
report_df = report_df.round(3)
show_table("Classification Report — Test Set", report_df)

# =====================================================================
# 10. Graph: Confusion matrix heatmap
# =====================================================================
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_test_pred, labels=labels)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "count"}, ax=ax)
ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved] {cm_path}")