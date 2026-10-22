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


# =====================================================================
# predict.py
# =====================================================================
FILES["predict.py"] = r'''import os
import sys
import pandas as pd
import joblib

from config import MODEL_PATH, OUTPUT_DIR

model = joblib.load(MODEL_PATH)

FEATURES = ["merchant", "amount",
            "day_of_month", "day_of_week", "is_weekend", "is_month_end"]


def predict_one(merchant, amount,
                day_of_month=15, day_of_week=2,
                is_weekend=0, is_month_end=0):
    row = {
        "merchant": merchant, "amount": amount,
        "day_of_month": day_of_month, "day_of_week": day_of_week,
        "is_weekend": is_weekend, "is_month_end": is_month_end,
    }
    return model.predict(pd.DataFrame([row]))[0]


def predict_batch(df):
    for col, default in [
        ("day_of_month", 15), ("day_of_week", 2),
        ("is_weekend", 0), ("is_month_end", 0),
    ]:
        if col not in df.columns:
            df[col] = default
    df["predicted_category"] = model.predict(df[FEATURES])
    return df


def interactive():
    print("\n=== PREDICT MODE ===")
    print("Type 'done' when finished.\n")
    while True:
        m = input("Merchant : ").strip()
        if m.lower() == "done":
            break
        try:
            a = float(input("Amount   : ").strip())
        except ValueError:
            print("Invalid amount.\n")
            continue
        print(f"  -> Predicted category: {predict_one(m, a)}\n")


def csv_batch(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "Description": "merchant",
        "Amount": "amount",
    })

    out = predict_batch(df)
    out_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Predictions saved: {out_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_batch(sys.argv[1])
    else:
        interactive()
'''


# =====================================================================
# make_budget.py
# =====================================================================
FILES["make_budget.py"] = r'''import pandas as pd

from config import TRAIN_CSV_PATH, BUDGET_CSV_PATH
from preprocessing import load_and_prepare

df = load_and_prepare(TRAIN_CSV_PATH)

df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby(["month", "category"])["amount"].sum().reset_index()
avg = monthly.groupby("category")["amount"].mean().round(2)

budget = (avg * 1.10).round(0).astype(int).reset_index()
budget.columns = ["category", "monthly_budget"]
budget = budget.sort_values("category")

budget.to_csv(BUDGET_CSV_PATH, index=False)

print(f"Wrote {len(budget)} categories to: {BUDGET_CSV_PATH}\n")
print(budget.to_string(index=False))
'''


# =====================================================================
# budget_assistant.py
# =====================================================================
FILES["budget_assistant.py"] = r'''import sys
import pandas as pd

from config import BUDGET_CSV_PATH, WARN_THRESHOLD, DANGER_THRESHOLD
from predict import predict_one

budget_df = pd.read_csv(BUDGET_CSV_PATH)
budget_df.columns = [c.strip().lower() for c in budget_df.columns]
BUDGET = dict(zip(budget_df["category"], budget_df["monthly_budget"]))


def money(v):
    return f"R{v:,.2f}"


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
            alerts.append(f"OVER BUDGET  - {cat}: spent {money(spent)} of "
                          f"{money(budget)} (over by {money(abs(remaining))})")
        elif ratio >= WARN_THRESHOLD:
            alerts.append(f"APPROACHING - {cat}: spent {money(spent)} of "
                          f"{money(budget)} (remaining {money(remaining)})")

    return {"details": details, "totals": totals, "alerts": alerts}


def report(result):
    print("\n========== AI BUDGET ASSISTANT REPORT ==========")
    print("\n--- Categorised Transactions ---")
    for d in result["details"]:
        print(f"  {d['merchant']:<30} {money(d['amount']):>12}  ->  {d['category']}")

    print("\n--- Budget Summary ---")
    total_spent = total_budget = 0.0
    for cat, spent in result["totals"].items():
        budget = BUDGET.get(cat, 0)
        total_spent  += spent
        total_budget += budget
        pct = (spent / budget * 100) if budget else 0
        print(f"  {cat:<22} Spent {money(spent):>12} / Budget {money(budget):>12}  ({pct:5.1f}%)")

    print(f"\n  TOTAL SPENT : {money(total_spent)} / {money(total_budget)}")
    print(f"  REMAINING   : {money(total_budget - total_spent)}")

    print("\n--- Alerts ---")
    if result["alerts"]:
        for a in result["alerts"]:
            print(" ", a)
    else:
        print("  All categories within budget.")


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
    report(analyse(txns))


def csv_batch(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Description": "merchant", "Amount": "amount"})
    if "Transaction Type" in df.columns:
        df = df[df["Transaction Type"].str.lower() == "debit"]
    report(analyse(df.to_dict(orient="records")))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_batch(sys.argv[1])
    else:
        interactive()
'''


# =====================================================================
# app.py  (Streamlit)
# =====================================================================
FILES["app.py"] = r'''import pandas as pd
import streamlit as st

from config import BUDGET_CSV_PATH, WARN_THRESHOLD, DANGER_THRESHOLD
from predict import predict_one

st.set_page_config(page_title="AI Budget Assistant", page_icon="", layout="wide")
st.title("AI Budget Assistant")

budget_df = pd.read_csv(BUDGET_CSV_PATH)
budget_df.columns = [c.strip().lower() for c in budget_df.columns]
BUDGET = dict(zip(budget_df["category"], budget_df["monthly_budget"]))


def money(v):
    return f"R{v:,.2f}"


st.subheader("1. Enter Transactions")
default = pd.DataFrame({
    "merchant": ["Netflix", "Thai Restaurant", "Gas Company"],
    "amount":   [11.76, 24.22, 45.00],
})
edited = st.data_editor(default, num_rows="dynamic", use_container_width=True)

if st.button("Analyse Spending", type="primary"):
    txns = [t for t in edited.to_dict(orient="records")
            if t["merchant"] and t["amount"] > 0]

    totals = {cat: 0.0 for cat in BUDGET}
    rows = []
    for t in txns:
        cat = predict_one(t["merchant"], float(t["amount"]))
        totals[cat] = totals.get(cat, 0.0) + float(t["amount"])
        rows.append({"Merchant": t["merchant"],
                     "Amount (R)": round(float(t["amount"]), 2),
                     "Category": cat})

    st.subheader("2. Categorised Transactions")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("3. Budget Summary")
    summary = []
    for cat, spent in totals.items():
        budget = BUDGET.get(cat, 0)
        pct = (spent / budget * 100) if budget else 0
        summary.append({
            "Category": cat,
            "Spent (R)": round(spent, 2),
            "Budget (R)": budget,
            "Remaining (R)": round(budget - spent, 2),
            "% Used": round(pct, 1),
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    st.subheader("4. Alerts")
    any_alert = False
    for cat, spent in totals.items():
        budget = BUDGET.get(cat, 0)
        if budget <= 0:
            continue
        ratio = spent / budget
        if ratio >= DANGER_THRESHOLD:
            st.error(f"OVER BUDGET - {cat}: {money(spent)} of {money(budget)}")
            any_alert = True
        elif ratio >= WARN_THRESHOLD:
            st.warning(f"Approaching - {cat}: {money(spent)} of {money(budget)}")
            any_alert = True
    if not any_alert:
        st.success("All categories within budget.")
'''


# =====================================================================
# inspect_data.py
# =====================================================================
FILES["inspect_data.py"] = r'''import pandas as pd
from config import TRAIN_CSV_PATH

print("Reading:", TRAIN_CSV_PATH)
df = pd.read_csv(TRAIN_CSV_PATH)

print("\n--- Columns ---")
for i, c in enumerate(df.columns):
    print(f"  {i}: {c!r}")

print("\n--- Shape ---")
print(df.shape)

print("\n--- First 10 rows ---")
print(df.head(10).to_string())

print("\n--- Dtypes ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())
'''


# =====================================================================
# requirements.txt
# =====================================================================
FILES["requirements.txt"] = r'''pandas
numpy
scikit-learn
joblib
streamlit
'''


# =====================================================================
# WRITE ALL FILES
# =====================================================================
print("Writing project files to:", BASE)
for name, content in FILES.items():
    path = BASE / name
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {name}")

print("\nDone. All files created.")
print("\nNext steps:")
print("  1.  python train_model.py")
print("  2.  python test_model.py")
print("  3.  python make_budget.py")
print("  4.  python budget_assistant.py")
print("  5.  python -m streamlit run app.py")