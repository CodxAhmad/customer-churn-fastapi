"""
src/monitor.py
==============
Model Monitoring with Evidently AI.

This script simulates a production monitoring workflow:
  1. Loads the original training data as the "reference" dataset.
  2. Simulates a "current" production batch by introducing realistic drift
     (new customers skewed toward high-risk profiles).
  3. Generates two HTML reports saved to reports/:
       - data_drift_report.html  -> feature-level drift analysis
       - model_performance_report.html -> classification performance on current batch

Run:
    python src/monitor.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from data_prep import load_and_prep_data

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "voting_soft.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts", "feature_names.pkl")


# ─── 1. Load reference data ───────────────────────────────────────────────────
def load_reference(n=1000, random_state=42):
    """Sample n rows of the processed training data as the reference set."""
    X, y = load_and_prep_data(DATA_PATH)
    ref = X.copy()
    ref["target"] = y.values
    return ref.sample(n=n, random_state=random_state).reset_index(drop=True)


# ─── 2. Simulate production drift ─────────────────────────────────────────────
def simulate_current(reference: pd.DataFrame, n=500, random_state=7) -> pd.DataFrame:
    """
    Simulate a realistic production batch with drift:
      - Higher proportion of Month-to-month contracts  -> more churn
      - Shorter tenures (new customers)
      - Higher monthly charges (price increases)
    """
    rng = np.random.default_rng(random_state)
    current = reference.sample(n=n, replace=True, random_state=random_state).copy().reset_index(drop=True)

    # Drift 1: shorter tenures (new customer wave)
    current["tenure"] = np.clip(
        current["tenure"] * rng.uniform(0.3, 0.7, size=n), 0, 72
    ).astype(float)

    # Drift 2: higher monthly charges (price hike)
    current["MonthlyCharges"] = np.clip(
        current["MonthlyCharges"] * rng.uniform(1.1, 1.3, size=n), 0, 200
    ).astype(float)

    # Drift 3: higher total charges correlation break
    current["TotalCharges"] = (
        current["tenure"] * current["MonthlyCharges"]
        + rng.normal(0, 50, size=n)
    ).clip(0).astype(float)

    return current


# ─── 3. Get model predictions ─────────────────────────────────────────────────
def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add model prediction column to a dataset (for classification report)."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    X = df[feature_names]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    result = df.copy()
    result["prediction"] = preds
    return result


# ─── 4. Generate Reports ──────────────────────────────────────────────────────
def generate_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Full data drift report across all features."""
    print("Generating data drift report...")

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
    ])

    report.run(reference_data=reference.drop(columns=["target"]),
               current_data=current.drop(columns=["target"]))

    out_path = os.path.join(REPORTS_DIR, "data_drift_report.html")
    report.save_html(out_path)
    print(f"  [OK] Saved -> {out_path}")
    return report


def generate_model_performance_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Classification performance report comparing ref vs current batch."""
    print("Generating model performance report...")

    # Add predictions to both sets
    ref_with_preds = add_predictions(reference)
    cur_with_preds = add_predictions(current)

    report = Report(metrics=[
        ClassificationPreset(),
    ])

    report.run(
        reference_data=ref_with_preds,
        current_data=cur_with_preds,
        column_mapping=None
    )

    out_path = os.path.join(REPORTS_DIR, "model_performance_report.html")
    report.save_html(out_path)
    print(f"  [OK] Saved -> {out_path}")
    return report


def generate_key_feature_drift(reference: pd.DataFrame, current: pd.DataFrame):
    """Focused drift report on the 3 most important churn features."""
    print("Generating key feature drift report...")

    key_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    report = Report(metrics=[
        ColumnDriftMetric(column_name=col) for col in key_features
    ])

    report.run(
        reference_data=reference.drop(columns=["target"]),
        current_data=current.drop(columns=["target"])
    )

    out_path = os.path.join(REPORTS_DIR, "key_feature_drift_report.html")
    report.save_html(out_path)
    print(f"  [OK] Saved -> {out_path}")
    return report


# ─── 5. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Evidently AI — Customer Churn Model Monitoring")
    print("=" * 60)

    print("\n[1/4] Loading reference data...")
    reference = load_reference(n=1000)
    print(f"      Reference shape: {reference.shape}")

    print("\n[2/4] Simulating production drift...")
    current = simulate_current(reference, n=500)
    print(f"      Current batch shape: {current.shape}")
    print(f"      Avg tenure  -> ref: {reference['tenure'].mean():.1f}  cur: {current['tenure'].mean():.1f}")
    print(f"      Avg monthly -> ref: {reference['MonthlyCharges'].mean():.1f}  cur: {current['MonthlyCharges'].mean():.1f}")

    print("\n[3/4] Running Evidently reports...")
    generate_data_drift_report(reference, current)
    generate_key_feature_drift(reference, current)
    generate_model_performance_report(reference, current)

    print("\n[4/4] All reports saved to reports/")
    print("      Open in browser:")
    print("      -> reports/data_drift_report.html")
    print("      -> reports/model_performance_report.html")
    print("\n" + "=" * 60)
