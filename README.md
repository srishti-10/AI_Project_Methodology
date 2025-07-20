# Customer Churn Prediction with Explainable AI

## Project Overview

This project implements a comprehensive customer churn prediction system using machine learning and explainable AI techniques. The system predicts which customers are likely to churn and provides interpretable insights using SHAP (SHapley Additive exPlanations).

## Key Features

- **Machine Learning Pipeline**: Complete data preprocessing, feature engineering, and model training
- **Multiple Model Evaluation**: Comparison of Random Forest, XGBoost, SVM, and Logistic Regression
- **SHAP Explainability**: Detailed model interpretation and feature importance analysis
- **MLflow Integration**: Experiment tracking and model management
- **Comprehensive Documentation**: Detailed case study and technical documentation

## Project Structure

```
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                # Trained models and metadata
├── shap_outputs/          # SHAP visualization outputs
├── src/                   # Source code modules
├── main.py               # Main pipeline script
├── shap_explainability_simple.py  # SHAP analysis script
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── UPDATED_CASE_STUDY.tex # Comprehensive case study document
```

## Setup Instructions

1. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ai_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the complete pipeline:**
   ```bash
   python main.py
   ```

2. **Generate SHAP explanations:**
   ```bash
   python shap_explainability_simple.py
   ```

3. **View MLflow experiments:**
   ```bash
   mlflow ui
   ```

4. **Start FastAPI server:**
   ```bash
   python api/app.py
   ```
   
   Then visit http://localhost:8000/docs for interactive API documentation.

## Model Performance

The Random Forest model achieved excellent performance:
- **Accuracy**: 97.9%
- **Precision**: 95.6%
- **Recall**: 91.6%
- **F1-Score**: 93.5%
- **ROC-AUC**: 99.7%

## Key Insights

The SHAP analysis revealed:
- Customer tenure is the most important predictor of churn
- Satisfaction scores strongly influence churn probability
- Recent activity patterns are critical indicators
- Cashback amounts serve as loyalty indicators

## Documentation

- **UPDATED_CASE_STUDY.tex**: Comprehensive case study document with detailed methodology, results, and insights
- **shap_outputs/**: Generated SHAP visualizations and explanation reports
- **models/**: Trained model artifacts and metadata

## Technologies Used

- Python 3.9+
- Scikit-learn for machine learning
- SHAP for explainable AI
- MLflow for experiment tracking
- FastAPI for REST API
- Pandas, NumPy for data processing
- Matplotlib, Seaborn for visualization

 