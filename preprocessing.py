import pandas as pd

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
