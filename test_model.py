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

# =====================================================================
# 6. Per-class bar chart (precision / recall / f1)
# =====================================================================
per_class = report_df[report_df["class"].isin(labels)][
    ["class", "precision", "recall", "f1-score"]
]
per_class_melt = per_class.melt(
    id_vars="class", var_name="metric", value_name="score"
)

fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * len(labels))))
sns.barplot(data=per_class_melt, x="score", y="class",
            hue="metric", palette="Set2", ax=ax)
ax.set_title("Precision / Recall / F1 per Category (Test Set)",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Score")
ax.set_ylabel("")
ax.legend(loc="lower right")
plt.tight_layout()
per_class_png = os.path.join(OUTPUT_DIR, "test_per_class_scores.png")
plt.savefig(per_class_png, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {per_class_png}")

# =====================================================================
# 7. Actual vs Predicted distribution
# =====================================================================
actual_counts    = y_true.value_counts().reindex(labels).fillna(0)
predicted_counts = pd.Series(y_pred).value_counts().reindex(labels).fillna(0)

compare_df = pd.DataFrame({
    "category":  labels,
    "actual":    actual_counts.values.astype(int),
    "predicted": predicted_counts.values.astype(int),
    "diff":      (predicted_counts.values - actual_counts.values).astype(int),
})
show_table("Actual vs Predicted Counts", compare_df)

comp_long = compare_df.melt(id_vars="category",
                            value_vars=["actual", "predicted"],
                            var_name="type", value_name="count")

fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * len(labels))))
sns.barplot(data=comp_long, y="category", x="count",
            hue="type", palette="coolwarm", ax=ax)
ax.set_title("Actual vs Predicted Category Counts",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Number of transactions")
ax.set_ylabel("")
plt.tight_layout()
counts_png = os.path.join(OUTPUT_DIR, "test_actual_vs_predicted.png")
plt.savefig(counts_png, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {counts_png}")

# =====================================================================
# 8. Confidence of predictions
# =====================================================================
try:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        confidence = proba.max(axis=1)

        conf_df = pd.DataFrame({
            "stat":  ["min", "25%", "median", "mean", "75%", "max"],
            "confidence": [
                round(float(np.min(confidence)), 4),
                round(float(np.percentile(confidence, 25)), 4),
                round(float(np.median(confidence)), 4),
                round(float(np.mean(confidence)), 4),
                round(float(np.percentile(confidence, 75)), 4),
                round(float(np.max(confidence)), 4),
            ],
        })
        show_table("Prediction Confidence (max probability)", conf_df)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(confidence, bins=20, kde=True,
                     color="#4C72B0", ax=ax)
        ax.axvline(float(np.mean(confidence)),
                   color="red", linestyle="--",
                   label=f"Mean = {np.mean(confidence):.3f}")
        ax.set_title("Distribution of Prediction Confidence",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Max predicted probability")
        ax.set_ylabel("Number of test samples")
        ax.legend()
        plt.tight_layout()
        conf_png = os.path.join(OUTPUT_DIR, "test_confidence_distribution.png")
        plt.savefig(conf_png, bbox_inches="tight")
        plt.close()
        print(f"[Chart saved] {conf_png}")
except Exception as e:
    print(f"(Confidence chart skipped: {e})")

# =====================================================================
# 9. Errors breakdown — which categories get confused the most
# =====================================================================
wrong = y_true != y_pred
error_df = pd.DataFrame({
    "merchant":  X_test["merchant"].astype(str).values[wrong],
    "amount":    X_test["amount"].values[wrong],
    "actual":    y_true.values[wrong],
    "predicted": np.asarray(y_pred)[wrong],
})
error_df = error_df.sort_values("actual").reset_index(drop=True)

show_table("Misclassified Transactions (sample of up to 30)",
           error_df.head(30))

err_path = os.path.join(OUTPUT_DIR, "test_misclassified.csv")
error_df.to_csv(err_path, index=False)
print(f"\nFull error list saved: {err_path}")

# =====================================================================
# 10. Save summary files
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
cm_path     = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
cm_norm_path= os.path.join(OUTPUT_DIR, "confusion_matrix_normalised.csv")
metrics_path= os.path.join(OUTPUT_DIR, "test_metrics.csv")

with open(report_path, "w") as f:
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1       : {f1:.4f}\n\n")
    f.write(report_text)

cm_df.to_csv(cm_path)
cm_norm.to_csv(cm_norm_path)
metrics_table.to_csv(metrics_path, index=False)

print(f"\n{'=' * 70}")
print("SUMMARY OF OUTPUTS")
print("=" * 70)
for label, p in [
    ("Report            ", report_path),
    ("Confusion matrix  ", cm_path),
    ("Confusion (norm)  ", cm_norm_path),
    ("Metrics           ", metrics_path),
    ("Misclassified     ", err_path),
    ("CM chart          ", cm_png),
    ("CM chart (norm)   ", cm_norm_png),
    ("Per-class scores  ", per_class_png),
    ("Actual vs predict ", counts_png),
]:
    print(f"  {label} {p}")
print("=" * 70)