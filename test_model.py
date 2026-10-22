"""
test_model.py
Evaluates the trained AI Budget Assistantrich tables and saves charts to outputs/.
model on Testing.csv.
Prints 
""" 

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)

from tabulate import tabulate

from config import TEST_CSV_PATH, MODEL_PATH, OUTPUT_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


# =====================================================================
# Helper: pretty table
# =====================================================================
def show_table(title, df):
    print(f"\n=== {title} ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid",
                   floatfmt=",.3f", showindex=False))


# =====================================================================
# 1. Load model + test data
# =====================================================================
print("=" * 70)
print("AI BUDGET ASSISTANT — MODEL TESTING")
print("=" * 70)

print(f"\nLoading model  : {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading test set: {TEST_CSV_PATH}")
df = pd.read_csv(TEST_CSV_PATH)

if "date" in df.columns:
    df = df.drop(columns=["date"])

X_test = df.drop(columns=["category"])
y_true = df["category"]

show_table("Test Set Snapshot (first 10 rows)",
           df.head(10))

# =====================================================================
# 2. Predict
# =====================================================================
print("\nRunning predictions on test set...")
y_pred = model.predict(X_test)

# =====================================================================
# 3. Global metrics
# =====================================================================
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

metrics_table = pd.DataFrame({
    "metric": ["Accuracy", "Precision (weighted)",
               "Recall (weighted)", "F1-score (weighted)"],
    "value":  [round(acc, 4), round(prec, 4),
               round(rec, 4), round(f1, 4)],
})
show_table("Overall Test Performance", metrics_table)

# =====================================================================
# 4. Classification report as a table
# =====================================================================
report_dict = classification_report(
    y_true, y_pred, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.reset_index()
report_df = report_df.rename(columns={"index": "class"}).round(3)

show_table("Classification Report (per class)", report_df)

# Also keep the plain-text version for the saved file
report_text = classification_report(y_true, y_pred, zero_division=0)

# =====================================================================
# 5. Confusion matrix — table + heatmap
# =====================================================================
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

show_table("Confusion Matrix (counts)", cm_df.reset_index()
             .rename(columns={"index": "actual\\predicted"}))

# Normalised matrix (row = actual, values sum to 1)
cm_norm = pd.DataFrame(
    cm.astype(float) / cm.sum(axis=1, keepdims=True),
    index=labels, columns=labels
).round(3)

show_table("Confusion Matrix (normalised)",
           cm_norm.reset_index().rename(columns={"index": "actual\\predicted"}))

# Heatmap (counts)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "count"}, ax=ax)
ax.set_title("Confusion Matrix — Test Set (counts)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_png = os.path.join(OUTPUT_DIR, "test_confusion_matrix.png")
plt.savefig(cm_png, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved] {cm_png}")

# Heatmap (normalised)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "proportion"}, ax=ax)
ax.set_title("Confusion Matrix — Test Set (normalised)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_norm_png = os.path.join(OUTPUT_DIR, "test_confusion_matrix_normalised.png")
plt.savefig(cm_norm_png, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {cm_norm_png}")

