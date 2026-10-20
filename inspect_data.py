import pandas as pd
from config import TRAIN_CSV_PATH

print("Reading:", TRAIN_CSV_PATH)
df = pd.read_csv(TRAIN_CSV_PATH)