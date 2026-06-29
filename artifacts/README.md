# Task 8 - End-to-End ML Pipeline

## Objective
Built a complete reproducible machine learning pipeline that connects data loading, preprocessing, model training, evaluation, and artifact saving.

## Pipeline Flow

Data Loading
↓
Data Splitting
↓
Preprocessing (StandardScaler)
↓
Model Training (Logistic Regression)
↓
Evaluation
↓
Saving Model and Metrics


## Tools Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib


## Model Details

Dataset:
- Iris Dataset

Model:
- Logistic Regression

Preprocessing:
- StandardScaler

Evaluation Metric:
- Accuracy Score


## Output Artifacts

The pipeline generates:

- `artifacts/model.pkl` - trained ML model
- `artifacts/metrics.json` - evaluation metrics
- `artifacts/experiment_log.txt` - experiment details


## How to Run

Install dependencies:

pip install -r requirements.txt


Run pipeline:


python run_pipeline.py



## Result

The pipeline produces reproducible results using a fixed random state.