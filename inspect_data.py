import pandas as pd
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