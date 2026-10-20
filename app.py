import pandas as pd
import streamlit as st

from config import BUDGET_CSV_PATH, WARN_THRESHOLD, DANGER_THRESHOLD
from predict import predict_one

st.set_page_config(page_title="AI Budget Assistant", page_icon="", layout="wide")
st.title("AI Budget Assistant")

budget_df = pd.read_csv(BUDGET_CSV_PATH)
budget_df.columns = [c.strip().lower() for c in budget_df.columns]
BUDGET = dict(zip(budget_df["category"], budget_df["monthly_budget"]))