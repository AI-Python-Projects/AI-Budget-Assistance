"""
test_model.py
Evaluates the trained AI Budget Assistant model on Testing.csv.
Prints rich tables and saves charts to outputs/.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)

from tabulate import tabulate

from config import TEST_CSV_PATH, MODEL_PATH, OUTPUT_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
