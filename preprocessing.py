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