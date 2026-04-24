from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import build_feature_frame, clean_churn_dataframe


def load_dataset(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def prepare_training_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = load_dataset(data_path)
    df = clean_churn_dataframe(df)
    features, target = build_feature_frame(df)
    preprocessor = build_preprocessor(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return {
        "dataframe": df,
        "cleaned_dataframe": df,
        "features": features,
        "target": target,
        "preprocessor": preprocessor,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def export_processed_data(dataset: dict, output_dir: str) -> dict:
    cleaned_path = f"{output_dir}/cleaned_churn.csv"
    train_path = f"{output_dir}/train_features.csv"
    test_path = f"{output_dir}/test_features.csv"

    cleaned = dataset["cleaned_dataframe"].copy()
    cleaned["Churn Value"] = dataset["target"]

    train_export = dataset["x_train"].copy()
    train_export["Churn Value"] = dataset["y_train"]

    test_export = dataset["x_test"].copy()
    test_export["Churn Value"] = dataset["y_test"]

    cleaned.to_csv(cleaned_path, index=False)
    train_export.to_csv(train_path, index=False)
    test_export.to_csv(test_path, index=False)

    return {
        "cleaned_data_path": cleaned_path,
        "train_data_path": train_path,
        "test_data_path": test_path,
    }
