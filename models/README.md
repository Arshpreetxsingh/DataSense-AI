# Models Directory

This folder stores trained model artifacts created by the pipeline.

## Expected artifact

- `trained_model.pkl`: a saved dictionary containing:
  - the selected model name
  - the fitted preprocessing + model pipeline
  - evaluation metrics
  - feature column order
  - processed-data export paths

The checked-in `trained_model.pkl` is currently empty and should be replaced by running:

```powershell
python test.py
```
