
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
## Setup
```bash
pip install -r requirements.txt
```

## 1. Add your dataset
Place your transaction dataset at `data/personal_transactions.csv`.
It must contain at least:
- a **merchant** (text) column — e.g. `merchant`
- an **amount** (numeric) column — e.g. `amount`
- a **category** column — e.g. `category`

If your column names differ, update them once in **`config.py`**.

## 2. Train
```bash
python train_model.py
```

## 3. Run
CLI (interactive):
```bash
python budget_assistant.py
```

CLI (batch on a CSV of transactions):
```bash
python budget_assistant.py data/my_month.csv
```

Web UI:
```bash
streamlit run app.py
```
## Example Output
```
Shoprite      R350.00   →   Food
Uber          R120.00   →   Transport

Food       Spent R1,850.00 / Budget R2,000.00  ( 92.5%)
⚠️  APPROACHING — Food: spent R1,850.00 of R2,000.00 (remaining R150.00)
```