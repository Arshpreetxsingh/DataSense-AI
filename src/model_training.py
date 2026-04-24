from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import export_processed_data, prepare_training_data
from src.feature_engineering import summarize_dataframe
from src.model_evaluation import evaluate_model
from utils.helper import create_directories, load_config, save_json, save_model
import datetime
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def build_candidate_models(config: dict) -> dict:
    candidates = {}
    model_flags = config["model"]["models"]
    params = config["model"].get("hyperparameters", {})
    random_state = config["model"]["random_state"]

    if model_flags.get("logistic_regression", False):
        candidates["logistic_regression"] = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    if model_flags.get("random_forest", False):
        rf_params = params.get("random_forest", {})
        candidates["random_forest"] = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth"),
            random_state=random_state,
            n_jobs=-1
        )

    if model_flags.get("xgboost", False) and XGBClassifier is not None:
        xgb_params = params.get("xgboost", {})
        candidates["xgboost"] = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 100),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            max_depth=xgb_params.get("max_depth", 6),
            random_state=random_state,
            eval_metric="logloss",
        )

    return candidates


def train_and_select_model(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    create_directories(config["paths"])

    dataset = prepare_training_data(
        data_path=f"{config['paths']['raw_data']}churn.csv",
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )
    processed_paths = export_processed_data(dataset, config["paths"]["processed_data"])

    candidates = build_candidate_models(config)
    if not candidates:
        raise ValueError("No models are enabled in config.yaml or available in the environment.")

    results = {}
    best_name = None
    best_score = float("-inf")
    best_pipeline = None

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", dataset["preprocessor"]),
                ("model", estimator),
            ]
        )

        try:
            pipeline.fit(dataset["x_train"], dataset["y_train"])

            metrics = evaluate_model(
                pipeline,
                dataset["x_test"],
                dataset["y_test"]
            )

            results[model_name] = metrics

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_name = model_name
                best_pipeline = pipeline

        except Exception as e:
            print(f"Model {model_name} failed: {e}")

        results[model_name] = metrics

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_name = model_name
            best_pipeline = pipeline
            if best_pipeline is None:
                raise ValueError("All models failed. Check preprocessing or dataset.")
    artifact = {
        "model_name": best_name,
        "model": best_pipeline,
        "metrics": results[best_name] if best_name else {},
        "all_results": results,
        "feature_columns": dataset["features"].columns.tolist(),
        "train_shape": dataset["x_train"].shape,
        "test_shape": dataset["x_test"].shape,
        "processed_paths": processed_paths,
    }

    save_model(artifact, f"{config['paths']['model_dir']}trained_model.pkl")
    
    save_json(
        {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model": artifact["model_name"],
            "best_metrics": artifact["metrics"],
            "all_results": artifact["all_results"],
            "dataset_summary": summarize_dataframe(dataset["dataframe"]),
            "processed_paths": processed_paths,
        },
        f"{config['paths']['processed_data']}training_summary.json",
    )
    return artifact


if __name__ == "__main__":
    artifact = train_and_select_model()
    print(f"Best model: {artifact['model_name']}")
    print(f"ROC-AUC: {artifact['metrics']['roc_auc']:.4f}")
