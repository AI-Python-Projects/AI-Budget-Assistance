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