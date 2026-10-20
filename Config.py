import os

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