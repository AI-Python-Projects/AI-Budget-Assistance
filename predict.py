import os
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
