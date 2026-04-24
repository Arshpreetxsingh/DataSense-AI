from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_preprocessing import load_dataset
from src.feature_engineering import (
    build_feature_frame,
    summarize_dataframe,
    clean_churn_dataframe
)
from src.model_training import train_and_select_model
from src.predict import (
    is_valid_model_artifact,
    load_prediction_artifact,
    predict_records
)

DATA_PATH = ROOT_DIR / "data" / "raw" / "churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"


# -------------------------------
# Helper
# -------------------------------
def build_default_customer(features: pd.DataFrame) -> dict:
    default_customer = {}
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            default_customer[column] = float(features[column].median())
        else:
            mode = features[column].mode(dropna=True)
            default_customer[column] = mode.iloc[0] if not mode.empty else ""
    return default_customer


# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="DataSense AI", layout="wide")
st.title("DataSense AI")
st.caption("Customer churn analysis and prediction pipeline")


# -------------------------------
# Load + preprocess data
# -------------------------------
df = load_dataset(str(DATA_PATH))
df = clean_churn_dataframe(df)

features, _ = build_feature_frame(df)
summary = summarize_dataframe(df)


# -------------------------------
# Dataset Overview
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Rows", summary["rows"])
col2.metric("Columns", summary["columns"])
col3.metric("Churn Rate", f"{summary['target_rate'] * 100:.2f}%")

with st.expander("Dataset Preview", expanded=True):
    st.dataframe(df.head(20), width="stretch")

with st.expander("Data Quality Summary"):
    st.write("Missing values")
    if summary["missing_values"]:
        st.json(summary["missing_values"])
    else:
        st.success("No missing values detected.")

    st.write("Feature types")
    st.write({
        "numeric_columns": len(summary["numeric_columns"]),
        "categorical_columns": len(summary["categorical_columns"]),
    })


# -------------------------------
# Model Training
# -------------------------------
st.subheader("Model Training")

if st.button("Train Pipeline", type="primary"):
    with st.spinner("Training models..."):
        artifact = train_and_select_model(str(ROOT_DIR / "config.yaml"))

    st.success(f"Best model: {artifact['model_name']}")
    st.json({
        "best_model_metrics": artifact["metrics"],
        "all_model_results": artifact["all_results"],
    })


# -------------------------------
# Prediction
# -------------------------------
st.subheader("Single Customer Prediction")

submitted = False
input_payload = {}

if not MODEL_PATH.exists() or not is_valid_model_artifact(str(MODEL_PATH)):
    st.info("Train the pipeline first to enable predictions.")
else:
    artifact = load_prediction_artifact(str(MODEL_PATH))
    default_customer = build_default_customer(features)

    st.caption(f"Loaded model: {artifact['model_name']}")

    with st.form("prediction_form"):
        left_col, right_col = st.columns(2)
        feature_names = list(features.columns)
        midpoint = (len(feature_names) + 1) // 2

        for i, column in enumerate(feature_names):
            target_col = left_col if i < midpoint else right_col
            value = default_customer[column]

            if pd.api.types.is_numeric_dtype(features[column]):
                input_payload[column] = target_col.number_input(
                    column, value=float(value)
                )
            else:
                options = sorted(features[column].dropna().astype(str).unique())
                fallback = str(value)
                if fallback not in options:
                    options = [fallback] + options

                input_payload[column] = target_col.selectbox(
                    column,
                    options=options,
                    index=options.index(fallback)
                )

        submitted = st.form_submit_button("Predict Churn")


# -------------------------------
# Prediction Output
# -------------------------------
if submitted:
    prediction = predict_records([input_payload], str(MODEL_PATH))

    churn_prob = float(prediction.loc[0, "churn_probability"])
    churn_label = "Likely to churn" if int(prediction.loc[0, "prediction"]) else "Likely to stay"

    st.metric("Prediction", churn_label)
    st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
    st.dataframe(prediction, width="stretch")


# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance")

if MODEL_PATH.exists() and is_valid_model_artifact(str(MODEL_PATH)):
    artifact = load_prediction_artifact(str(MODEL_PATH))
    model_pipeline = artifact["model"]

    try:
        model = model_pipeline.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            # SAFE CHECK (due to encoding mismatch)
            if len(importances) == len(artifact["feature_columns"]):
                importance_df = pd.DataFrame({
                    "feature": artifact["feature_columns"],
                    "importance": importances
                }).sort_values(by="importance", ascending=False).head(10)

                st.bar_chart(importance_df.set_index("feature"))
            else:
                st.info("Feature importance not aligned due to encoding (advanced fix later).")

        else:
            st.info("Feature importance not available for this model.")

    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")