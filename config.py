"""
Configuration file for Customer Churn Prediction Project.

Contains project paths, model parameters, and experiment settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "raw_data_file": "ecommerce_customer_churn.csv",
    "processed_data_file": "processed_data.csv",
    "engineered_features_file": "engineered_features.csv",
    "sample_predictions_file": "sample_predictions.csv"
}

# Model configuration
MODEL_CONFIG = {
    "experiment_name": "customer_churn_prediction",
    "random_state": 42,
    "test_size": 0.15,
    "validation_size": 0.15,
    "cv_folds": 5,
    "scoring_metric": "f1",
    "feature_selection_k": 20
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": "sqlite:///mlflow.db",
    "artifact_location": str(MODELS_DIR / "mlflow_artifacts"),
    "registry_uri": None
}

# Hyperparameter grids for different models
HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [42]
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "random_state": [42]
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "probability": [True],
        "random_state": [42]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "random_state": [42]
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "categorical_columns": [
        "preferedordercat", "maritalstatus", "complain"
    ],
    "numerical_columns": [
        "tenure", "numberofdeviceregistered", "satisfactionscore",
        "daysincelastorder", "cashbackamount"
    ],
    "target_column": "churn",
    "drop_columns": ["customerid"]
}

# SHAP configuration
SHAP_CONFIG = {
    "sample_size": 100,
    "output_dir": "shap_outputs",
    "plot_types": ["summary", "feature_importance", "individual"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/pipeline.log"
} 