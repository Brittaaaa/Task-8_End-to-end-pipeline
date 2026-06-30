# Task 8 - End-to-End ML Pipeline

## Overview

This project implements a complete end-to-end Machine Learning pipeline that connects the entire ML workflow:

Data → Features → Model → Evaluation → Output

The pipeline is reproducible and can be executed using a single command.

---

## Objective

The goal of this task is to build a production-style ML pipeline where preprocessing, model training, evaluation, and artifact saving are connected together.

---

## Pipeline Workflow

1. Load dataset
2. Split data into training and testing sets
3. Apply preprocessing using StandardScaler
4. Train Machine Learning model
5. Evaluate model performance
6. Save trained model and evaluation metrics
7. Generate experiment log

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Dataset

Dataset used:

Iris Dataset

Features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Target:
- Flower species classification

---

## Machine Learning Pipeline

The pipeline contains:

### Preprocessing
- StandardScaler

### Model
- Logistic Regression

The preprocessing and model are combined using Scikit-learn Pipeline.

---

## Evaluation

Metric used:

- Accuracy Score

The model is evaluated on unseen test data.

---

## Reproducibility

A fixed random state is used during data splitting to ensure consistent results across multiple runs.

---

## Generated Artifacts

After execution, the pipeline creates:

artifacts/

├── model.pkl
├── metrics.json
└── experiment_log.txt


### model.pkl
Contains the trained machine learning pipeline.

### metrics.json
Stores evaluation results.

### experiment_log.txt
Stores experiment information.

---

## Installation

Install required libraries:


pip install -r requirements.txt


---

## Running the Pipeline

Execute:


python run_pipeline.py


The complete ML workflow will run automatically and generate the required artifacts.

---

## Result

## Task 9 - Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV on a Random Forest Classifier.

Steps performed:
- Created a baseline model
- Selected important hyperparameters
- Applied GridSearchCV with cross-validation
- Selected the best configuration
- Evaluated the tuned model on test data

Results:
- Baseline Accuracy: 1.0
- Tuned Accuracy: 1.0
- Best Parameters:{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 50}