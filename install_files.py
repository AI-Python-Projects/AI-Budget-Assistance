"""
install_files.py
One-shot bootstrapper — writes every project file with the correct content.
Run once with:   python install_files.py
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent
(BASE / "data").mkdir(exist_ok=True)
(BASE / "models").mkdir(exist_ok=True)
(BASE / "outputs").mkdir(exist_ok=True)


FILES = {}

# =====================================================================
# config.py
# =====================================================================
FILES["config.py"] = r'''import os

# ---------- Paths ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_CSV_PATH  = os.path.join(DATA_DIR, "personal_transactions.csv")
TEST_CSV_PATH   = os.path.join(DATA_DIR, "Testing.csv")
BUDGET_CSV_PATH = os.path.join(DATA_DIR, "budget.csv")
MODEL_PATH      = os.path.join(MODEL_DIR, "budget_model.pkl")

for d in (MODEL_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------- Column names in your real CSV ----------
COL_DATE     = "Date"
COL_MERCHANT = "Description"
COL_AMOUNT   = "Amount"
COL_TYPE     = "Transaction Type"
COL_TARGET   = "Category"
COL_ACCOUNT  = "Account Name"

# Only these count as expenses
EXPENSE_TYPES = ["debit"]

# ---------- Alert thresholds ----------
WARN_THRESHOLD   = 0.80
DANGER_THRESHOLD = 1.00
'''


# =====================================================================
# preprocessing.py
# =====================================================================
FILES["preprocessing.py"] = r'''import pandas as pd

from config import (
    COL_DATE, COL_MERCHANT, COL_AMOUNT, COL_TYPE, COL_TARGET,
    EXPENSE_TYPES,
)


def load_and_prepare(csv_path, require_target=True):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    # Keep expenses only
    if COL_TYPE in df.columns:
        df = df[df[COL_TYPE].astype(str).str.lower().isin(EXPENSE_TYPES)].copy()

    # Clean amounts
    amt = (df[COL_AMOUNT].astype(str)
           .str.replace("R", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.replace(",", "", regex=False))
    df[COL_AMOUNT] = pd.to_numeric(amt, errors="coerce").fillna(0).abs()

    # Date features
    dt = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=False)
    df["date"]         = dt
    df["day_of_month"] = dt.dt.day.fillna(15).astype(int)
    df["day_of_week"]  = dt.dt.dayofweek.fillna(2).astype(int)
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    # Standard names used by the model
    df = df.rename(columns={
        COL_MERCHANT: "merchant",
        COL_AMOUNT:   "amount",
        COL_TARGET:   "category",
    })

    if require_target and "category" not in df.columns:
        raise ValueError("Target column 'category' not found in CSV.")

    return df.reset_index(drop=True)
'''


# =====================================================================
# train_model.py
# =====================================================================
FILES["train_model.py"] = r'''import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from config import TRAIN_CSV_PATH, TEST_CSV_PATH, MODEL_PATH
from preprocessing import load_and_prepare

# ---------- Load + clean ----------
print("Loading:", TRAIN_CSV_PATH)
df = load_and_prepare(TRAIN_CSV_PATH)

print("\n--- Dataset Summary ---")
print(df.describe(include="all"))

print("\n--- Class Distribution ---")
print(df["category"].value_counts())

print("\n--- Average spend per category ---")
print(df.groupby("category")["amount"].mean().round(2))

# ---------- Hold-out split ----------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category"]
)
test_df.to_csv(TEST_CSV_PATH, index=False)
print(f"\nTesting split saved to: {TEST_CSV_PATH}  ({len(test_df)} rows)")

# ---------- Preprocessing ----------
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

X = train_df.drop(columns=["category", "date"])
y = train_df["category"]

# ---------- GridSearch ----------
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

print("\nRunning GridSearchCV (may take a minute)...")
grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X, y)

best_clf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# ---------- Score ----------
y_pred = best_clf.predict(X)
print(f"\nTraining Accuracy: {accuracy_score(y, y_pred):.4f}")
print(classification_report(y, y_pred, zero_division=0))

cv_scores = cross_val_score(best_clf, X, y, cv=5)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}")

# ---------- Save ----------
joblib.dump(best_clf, MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")
'''


# =====================================================================
# test_model.py
# =====================================================================
FILES["test_model.py"] = r'''import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import TEST_CSV_PATH, MODEL_PATH, OUTPUT_DIR

print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

print("Loading testing data from:", TEST_CSV_PATH)
df = pd.read_csv(TEST_CSV_PATH)

if "date" in df.columns:
    df = df.drop(columns=["date"])

X_test = df.drop(columns=["category"])
y_true = df["category"]

print("\nPredicting on test set...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_true, y_pred)
print(f"\n--- Testing Performance ---")
print(f"Accuracy: {acc:.4f}")

report = classification_report(y_true, y_pred, zero_division=0)
print("\nClassification Report:\n", report)

labels = sorted(y_true.unique())
cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred),
                     index=labels, columns=labels)
print("\nConfusion Matrix:")
print(cm_df)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
cm_path     = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")

with open(report_path, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n{report}")
cm_df.to_csv(cm_path)

print(f"\nReport saved: {report_path}")
print(f"Confusion matrix saved: {cm_path}")
'''