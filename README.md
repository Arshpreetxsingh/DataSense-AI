# DataSense AI

DataSense AI is a customer churn analysis app built around the Telco churn dataset in [data/raw/churn.csv](D:\Projects\DataSense-AI\data\raw\churn.csv). The project trains multiple machine learning models, compares them, saves the best pipeline, exports processed datasets, and serves predictions in a Streamlit interface.

## What the system does

- Loads and cleans the churn dataset
- Removes obvious target leakage columns before training
- Builds preprocessing for numeric and categorical features
- Trains Logistic Regression, Random Forest, and optionally XGBoost
- Selects the best model using ROC-AUC
- Saves the best model to `models/trained_model.pkl`
- Exports processed training files to `data/processed/`
- Provides a Streamlit UI for training and single-customer prediction

## Project structure

- [app/app.py](D:\Projects\DataSense-AI\app\app.py): Streamlit dashboard
- [src/data_preprocessing.py](D:\Projects\DataSense-AI\src\data_preprocessing.py): dataset loading, split, preprocessing, processed data export
- [src/feature_engineering.py](D:\Projects\DataSense-AI\src\feature_engineering.py): leakage removal, feature selection, data summary
- [src/model_training.py](D:\Projects\DataSense-AI\src\model_training.py): model training and best-model selection
- [src/model_evaluation.py](D:\Projects\DataSense-AI\src\model_evaluation.py): evaluation metrics
- [src/predict.py](D:\Projects\DataSense-AI\src\predict.py): inference helpers
- [test.py](D:\Projects\DataSense-AI\test.py): CLI pipeline run
- [config.yaml](D:\Projects\DataSense-AI\config.yaml): paths and training config

## How to run

Your machine currently has a broken Python setup, so `streamlit` is not available in PowerShell yet. Once Python 3.11+ is installed correctly and visible on PATH, run:

```powershell
cd D:\Projects\DataSense-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
.\train_pipeline.ps1
.\run_app.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Expected outputs

After training succeeds, you should see:

- `models/trained_model.pkl`
- `data/processed/cleaned_churn.csv`
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `data/processed/training_summary.json`

## Notes

- `models/trained_model.pkl` in the repo is currently an empty placeholder and will be overwritten by training.
- The app now guards against invalid or empty model files and asks you to train first instead of crashing.
- `xgboost`, `matplotlib`, and `seaborn` are optional and can be installed later if you want notebook visuals or extra experimentation.
