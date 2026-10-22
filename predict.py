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