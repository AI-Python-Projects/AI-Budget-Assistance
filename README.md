
# AI-Budget-Assistance

An AI-powered personal budgeting assistant that uses **Machine Learning** to
automatically categorise bank transactions and monitor spending against a
monthly budget.

## Problem

Manual tracking of personal expenses causes **late detection of overspending**.

## Solution
A Random Forest classifier is trained to categorise each transaction
(Food, Transport, Rent, …). The assistant then:
- Tracks cumulative spending per category
- Warns the user at **80 %** of budget
- Alerts the user at **100 %** of budget

## Project layout
```
data/
  personal_transactions.csv   ← you provide (must have 'category' column)
  budget.csv                  ← monthly budget per category
models/
  budget_model.pkl            ← generated
config.py                     ← column names, thresholds, paths
train_model.py                ← trains the ML categoriser
budget_assistant.py           ← CLI
app.py                        ← Streamlit web UI
```