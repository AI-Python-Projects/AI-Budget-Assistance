import pandas as pd
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