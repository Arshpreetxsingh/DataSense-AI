import pandas as pd


TARGET_COLUMN = "Churn Value"

LEAKAGE_COLUMNS = [
    "Churn Label",
    "Churn Score",
    "Churn Reason",
]

ID_COLUMNS = [
    "CustomerID",
    "Count",
]


def clean_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if "Total Charges" in cleaned.columns:
        cleaned["Total Charges"] = pd.to_numeric(cleaned["Total Charges"], errors="coerce")
    # Tenure groups
    if "Tenure Months" in cleaned.columns:
        cleaned["tenure_group"] = pd.cut(
            cleaned["Tenure Months"],
            bins=[0, 12, 36, 72, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        )

    # Avg monthly spend
    if "Total Charges" in cleaned.columns and "Tenure Months" in cleaned.columns:
        cleaned["avg_monthly_spend"] = cleaned["Total Charges"] / (cleaned["Tenure Months"] + 1)

    # Binary flag for long-term customer
    if "Tenure Months" in cleaned.columns:
        cleaned["is_long_term"] = (cleaned["Tenure Months"] > 24).astype(int)
    
    return cleaned


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set([TARGET_COLUMN, *LEAKAGE_COLUMNS, *ID_COLUMNS])
    return [column for column in df.columns if column not in excluded]


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_churn_dataframe(df)
    feature_columns = get_feature_columns(cleaned)
    features = cleaned[feature_columns].copy()
    target = cleaned[TARGET_COLUMN].astype(int)
    return features, target


def summarize_dataframe(df: pd.DataFrame) -> dict:
    cleaned = clean_churn_dataframe(df)
    target_rate = None
    if TARGET_COLUMN in cleaned.columns:
        target_rate = float(cleaned[TARGET_COLUMN].mean())

    missing_values = cleaned.isna().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    return {
        "rows": int(cleaned.shape[0]),
        "columns": int(cleaned.shape[1]),
        "target_rate": target_rate,
        "missing_values": missing_values.to_dict(),
        "numeric_columns": cleaned.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": cleaned.select_dtypes(exclude=["number"]).columns.tolist(),
    }
