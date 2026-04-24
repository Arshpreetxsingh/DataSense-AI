from __future__ import annotations

import pandas as pd

from utils.helper import load_model


def load_prediction_artifact(model_path: str = "models/trained_model.pkl") -> dict:
    return load_model(model_path)


def predict_records(records, model_path: str = "models/trained_model.pkl") -> pd.DataFrame:
    artifact = load_prediction_artifact(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    input_df = pd.DataFrame(records)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(input_df)[:, 1]
    predictions = model.predict(input_df)

    result = input_df.copy()
    result["prediction"] = predictions
    result["churn_probability"] = probabilities
    return result


def is_valid_model_artifact(model_path: str = "models/trained_model.pkl") -> bool:
    try:
        artifact = load_prediction_artifact(model_path)
        return isinstance(artifact, dict) and "model" in artifact and "feature_columns" in artifact
    except Exception:
        return False
