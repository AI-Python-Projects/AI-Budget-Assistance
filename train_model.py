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

# =====================================================================
# 11. Graph: Per-class F1 scores
# =====================================================================
per_class = report_df[report_df["class"].isin(labels)][["class", "precision",
                                                        "recall", "f1-score"]]
per_class_melt = per_class.melt(id_vars="class",
                                var_name="metric", value_name="score")

fig, ax = plt.subplots(figsize=(13, 8))
sns.barplot(data=per_class_melt, x="score", y="class",
            hue="metric", palette="Set2", ax=ax)
ax.set_title("Precision / Recall / F1 per Category (Test Set)",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Score")
ax.set_ylabel("")
ax.legend(loc="lower right")
plt.tight_layout()
f1_path = os.path.join(OUTPUT_DIR, "per_class_scores.png")
plt.savefig(f1_path, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {f1_path}")

# =====================================================================
# 12. Graph: Feature importance (top-N numeric + top-N merchant n-grams)
# =====================================================================
try:
    classifier = best_clf.named_steps["classifier"]
    pre = best_clf.named_steps["preprocessor"]

    # Names from the ColumnTransformer
    tfidf_names = pre.named_transformers_["merchant_tfidf"].get_feature_names_out()
    num_names   = numeric_cols
    feature_names = np.concatenate([tfidf_names, num_names])

    importances = classifier.feature_importances_
    top_n = 25
    idx = np.argsort(importances)[-top_n:]

    imp_df = pd.DataFrame({
        "feature": feature_names[idx],
        "importance": importances[idx],
    })

    show_table(f"Top {top_n} Most Important Features",
               imp_df.sort_values("importance", ascending=False))

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=imp_df, x="importance", y="feature",
                palette="crest", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[Chart saved] {fi_path}")
except Exception as e:
    print(f"(Feature-importance chart skipped: {e})")

# =====================================================================
# 13. Cross-validation
# =====================================================================
print("\n" + "=" * 70)
print("Running 5-fold cross-validation on full dataset...")
cv_scores = cross_val_score(best_clf, df.drop(columns=["category", "date"]),
                            df["category"], cv=5)

cv_table = pd.DataFrame({
    "fold":     [1, 2, 3, 4, 5],
    "accuracy": np.round(cv_scores, 4),
})
show_table("5-Fold Cross-Validation", cv_table)
print(f"Mean CV accuracy: {cv_scores.mean():.4f}  (± {cv_scores.std():.4f})")

# CV scores plot
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores, palette="viridis", ax=ax)
ax.axhline(cv_scores.mean(), color="red", linestyle="--",
           label=f"Mean = {cv_scores.mean():.4f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Cross-Validation Accuracy per Fold",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
cv_path = os.path.join(OUTPUT_DIR, "cv_scores.png")
plt.savefig(cv_path, bbox_inches="tight")
plt.close()
print(f"[Chart saved] {cv_path}")

# =====================================================================
# 14. Save model
# =====================================================================
joblib.dump(best_clf, MODEL_PATH)
print(f"\n{'=' * 70}")
print(f"Model saved at: {MODEL_PATH}")
print(f"All charts saved under: {OUTPUT_DIR}")
print("=" * 70)