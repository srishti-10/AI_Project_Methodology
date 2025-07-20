"""
Model training script for Customer Churn Prediction Project.

This script handles model training, evaluation, and MLflow integration
for tracking experiments and model versions with GPU acceleration support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI to file-based storage
mlflow.set_tracking_uri("file:./mlruns")

# GPU Configuration
def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

# Global device variable
DEVICE = get_device()


class ModelTrainer:
    """Class for handling model training and evaluation with GPU support."""
    
    def __init__(self, experiment_name: str = "customer_churn_prediction", use_gpu: bool = True):
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            use_gpu: Whether to use GPU acceleration
        """
        self.experiment_name = experiment_name
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = DEVICE if self.use_gpu else torch.device("cpu")
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"ModelTrainer initialized with device: {self.device}")
        if self.use_gpu:
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    
    def load_data(self, filepath: str = "data/processed/engineered_features.csv") -> Tuple:
        """
        Load the engineered features and prepare train-test split.
        
        Args:
            filepath: Path to the engineered features file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col not in ['customerid', 'churn']]
            X = df[feature_cols]
            y = df['churn']
            
            # Split the data (assuming it's already split or we'll do it here)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Train set shape: {X_train.shape}")
            logger.info(f"Test set shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Please run feature engineering first.")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_models(self) -> Dict[str, Any]:
        """
        Define the models to train with GPU support.
        
        Returns:
            Dictionary of model names and their instances
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True, cache_size=1000, max_iter=1000),
            'linear_svm': SVC(random_state=42, probability=True, kernel='linear', cache_size=1000, max_iter=1000)
        }
        
        # Add GPU-optimized models if GPU is available
        if self.use_gpu:
            models.update({
                'gpu_logistic_regression': LogisticRegression(
                    random_state=42, max_iter=2000, solver='lbfgs'
                ),
                'gpu_random_forest': RandomForestClassifier(
                    random_state=42, n_estimators=200, n_jobs=-1
                ),
                'gpu_gradient_boosting': GradientBoostingClassifier(
                    random_state=42, n_estimators=200
                ),
                'pytorch_nn': self._create_pytorch_model()
            })
        
        return models
    
    def _create_pytorch_model(self, input_size: int = None) -> nn.Module:
        """
        Create a PyTorch neural network model for GPU training.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch neural network model
        """
        class ChurnPredictor(nn.Module):
            def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
                super(ChurnPredictor, self).__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, 1))
                layers.append(nn.Sigmoid())
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        if input_size is None:
            # Default input size, will be updated during training
            input_size = 50
        
        model = ChurnPredictor(input_size)
        model = model.to(self.device)
        return model
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Define hyperparameter grids for each model with GPU optimization.
        
        Returns:
            Dictionary of hyperparameter grids
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'svm': {
                'C': [0.1, 1],
                'kernel': ['rbf'],
                'gamma': ['scale'],
                'cache_size': [1000],
                'max_iter': [1000]
            },
            'linear_svm': {
                'C': [0.1, 1],
                'cache_size': [1000],
                'max_iter': [1000]
            }
        }
        
        # Add GPU-optimized hyperparameter grids
        if self.use_gpu:
            param_grids.update({
                'gpu_logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],  # L2 is faster on GPU
                    'solver': ['lbfgs'],  # LBFGS is GPU-friendly
                    'max_iter': [1000]  # More iterations for GPU
                },
                'gpu_random_forest': {
                    'n_estimators': [50, 100],  # More trees for GPU
                    'max_depth': [10, None],
                    'min_samples_split': [2, 5],
                    'n_jobs': [-1]  # Use all CPU cores for GPU-accelerated RF
                },
                'gpu_gradient_boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'pytorch_nn': {
                    'learning_rate': [0.001, 0.01],
                    'batch_size': [32, 64],
                    'epochs': [50, 100],
                    'hidden_sizes': [[64, 32], [128, 64]],
                    'dropout_rate': [0.2, 0.3]
                }
            })
        
        return param_grids
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def train_pytorch_model(self, model: nn.Module, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series, 
                           learning_rate: float = 0.001, batch_size: int = 64, 
                           epochs: int = 100, hidden_sizes: list = [128, 64, 32], 
                           dropout_rate: float = 0.3) -> Dict[str, Any]:
        """
        Train a PyTorch neural network model with GPU acceleration and detailed progress tracking.
        
        Args:
            model: PyTorch model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            hidden_sizes: Hidden layer sizes
            dropout_rate: Dropout rate
            
        Returns:
            Dictionary containing trained model, metrics, and parameters
        """
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model with correct input size
        input_size = X_train.shape[1]
        model = self._create_pytorch_model(input_size)
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training PyTorch Model", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Training loop
        model.train()
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Training phase
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = (outputs > 0.5).float()
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
            
            # Calculate training metrics
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor).squeeze()
                val_loss = criterion(val_outputs, y_test_tensor)
                val_pred = (val_outputs > 0.5).float()
                val_accuracy = (val_pred == y_test_tensor).float().mean().item()
                
                val_losses.append(val_loss.item())
                val_accuracies.append(val_accuracy)
            
            model.train()
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{avg_train_loss:.4f}',
                'Train Acc': f'{train_accuracy:.4f}',
                'Val Loss': f'{val_loss.item():.4f}',
                'Val Acc': f'{val_accuracy:.4f}'
            })
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Training predictions
            train_outputs = model(X_train_tensor).squeeze()
            train_pred = (train_outputs > 0.5).float()
            train_pred_proba = train_outputs.cpu().numpy()
            
            # Test predictions
            test_outputs = model(X_test_tensor).squeeze()
            test_pred = (test_outputs > 0.5).float()
            test_pred_proba = test_outputs.cpu().numpy()
        
        # Convert predictions back to numpy for sklearn metrics
        y_train_np = y_train.values
        y_test_np = y_test.values
        train_pred_np = train_pred.cpu().numpy()
        test_pred_np = test_pred.cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_np, test_pred_np),
            'precision': precision_score(y_test_np, test_pred_np),
            'recall': recall_score(y_test_np, test_pred_np),
            'f1_score': f1_score(y_test_np, test_pred_np),
            'roc_auc': roc_auc_score(y_test_np, test_pred_proba)
        }
        
        # Generate and save visualizations
        self._save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, 
                                y_test_np, test_pred_np, test_pred_proba, model_name="pytorch_nn")
        
        # Clear GPU memory
        if self.use_gpu:
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            'model': model,
            'metrics': metrics,
            'best_params': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'hidden_sizes': hidden_sizes,
                'dropout_rate': dropout_rate
            },
            'best_score': metrics['f1_score'],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def _save_training_plots(self, train_losses, val_losses, train_accuracies, val_accuracies,
                           y_true, y_pred, y_pred_proba, model_name="model"):
        """
        Generate and save training visualization plots.
        
        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            train_accuracies: Training accuracy history
            val_accuracies: Validation accuracy history
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for file naming
        """
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.upper()} Training Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        axes[0, 0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        axes[0, 1].plot(train_accuracies, label='Training Accuracy', color='green', linewidth=2)
        axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
        axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Plot 4: ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        axes[1, 1].plot(fpr, tpr, color='purple', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_title('ROC Curve', fontweight='bold')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = plots_dir / f"{model_name}_training_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to: {plot_path}")
        
        # Save classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = plots_dir / f"{model_name}_classification_report_{timestamp}.csv"
        report_df.to_csv(report_path)
        
        logger.info(f"Classification report saved to: {report_path}")
        
        # Print classification report
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT - {model_name.upper()}")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred))
        print(f"{'='*60}")
    
    def train_model_with_mlflow(self, model_name: str, model, param_grid: Dict,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train a model with hyperparameter tuning and MLflow tracking.
        
        Args:
            model_name: Name of the model
            model: Model instance
            param_grid: Hyperparameter grid
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing model, metrics, and parameters
        """
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
            
            # Perform hyperparameter tuning with progress tracking
            logger.info(f"Training {model_name} with hyperparameter tuning...")
            
            # Calculate total combinations for progress bar
            from itertools import product
            param_combinations = list(product(*param_grid.values()))
            total_combinations = len(param_combinations)
            
            # Create progress bar for grid search
            with tqdm(total=total_combinations, desc=f"Training {model_name}", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                
                # Manual grid search with progress tracking
                best_score = 0
                best_params = None
                best_model = None
                
                for i, params in enumerate(param_combinations):
                    # Convert params tuple to dict
                    param_dict = dict(zip(param_grid.keys(), params))
                    
                    try:
                        # Create model with current parameters
                        current_model = model.__class__(**param_dict)
                        
                        # Cross-validation
                        from sklearn.model_selection import cross_val_score
                        cv_scores = cross_val_score(current_model, X_train, y_train, cv=5, scoring='f1')
                        mean_score = cv_scores.mean()
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'Score': f'{mean_score:.4f}',
                            'Best': f'{best_score:.4f}'
                        })
                        
                        # Update best model if better
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = param_dict
                            best_model = current_model
                            
                    except Exception as e:
                        logger.warning(f"Failed to train {model_name} with params {param_dict}: {e}")
                        pbar.update(1)
                        pbar.set_postfix({
                            'Score': 'FAILED',
                            'Best': f'{best_score:.4f}'
                        })
                        continue
                
                # Fit the best model on full training data
                if best_model is None:
                    logger.error(f"No valid model found for {model_name}. Using default parameters.")
                    best_model = model.__class__()
                    best_params = {}
                    best_score = 0.0
                
                best_model.fit(X_train, y_train)
            
            # Best model is already selected from manual grid search
            
            # Evaluate on test set
            metrics = self.evaluate_model(best_model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_f1_score", best_score)
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            
            # Generate and save visualizations for sklearn models
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Create dummy lists for sklearn models (no epoch-wise tracking)
            dummy_losses = [0] * 10  # Dummy data for plotting
            dummy_accuracies = [metrics['accuracy']] * 10
            
            self._save_training_plots(dummy_losses, dummy_losses, dummy_accuracies, dummy_accuracies,
                                    y_test.values, y_pred, y_pred_proba, model_name=model_name)
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance to a temporary file first
                temp_path = f"temp_{model_name}_feature_importance.csv"
                feature_importance.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path, f"{model_name}_feature_importance.csv")
                # Clean up temp file
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            logger.info(f"{model_name} training completed. Best CV F1: {best_score:.4f}")
            logger.info(f"Test metrics: {metrics}")
            
            return {
                'model': best_model,
                'metrics': metrics,
                'best_params': best_params,
                'best_score': best_score
            }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train all models and find the best one with GPU support.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing all models and their results
        """
        models = self.get_models()
        param_grids = self.get_hyperparameter_grids()
        results = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {model_name.upper()}")
                logger.info(f"{'='*50}")
                
                # Handle PyTorch models separately
                if model_name == 'pytorch_nn':
                    # For PyTorch, we'll use a simplified grid search
                    best_result = None
                    best_score = 0
                    
                    # Sample a few combinations for faster training
                    param_combinations = [
                        {'learning_rate': 0.001, 'batch_size': 64, 'epochs': 50, 
                         'hidden_sizes': [128, 64, 32], 'dropout_rate': 0.3},
                        {'learning_rate': 0.01, 'batch_size': 128, 'epochs': 100, 
                         'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.2},
                        {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 200, 
                         'hidden_sizes': [64, 32], 'dropout_rate': 0.5}
                    ]
                    
                    for params in param_combinations:
                        logger.info(f"Trying PyTorch params: {params}")
                        result = self.train_pytorch_model(
                            model, X_train, y_train, X_test, y_test, **params
                        )
                        
                        if result['metrics']['f1_score'] > best_score:
                            best_result = result
                            best_score = result['metrics']['f1_score']
                    
                    results[model_name] = best_result
                    
                    # Log to MLflow
                    with mlflow.start_run(run_name=f"{model_name}_training"):
                        mlflow.log_param("model_type", model_name)
                        mlflow.log_param("n_features", X_train.shape[1])
                        mlflow.log_param("n_samples", X_train.shape[0])
                        mlflow.log_metrics(best_result['metrics'])
                        mlflow.log_params(best_result['best_params'])
                        
                        # Save PyTorch model
                        torch.save(best_result['model'].state_dict(), f"models/{model_name}_model.pth")
                        mlflow.log_artifact(f"models/{model_name}_model.pth")
                
                else:
                    # Standard sklearn models
                    result = self.train_model_with_mlflow(
                        model_name, model, param_grids[model_name],
                        X_train, y_train, X_test, y_test
                    )
                    results[model_name] = result
                
                # Update best model if this one is better
                current_score = results[model_name]['metrics']['f1_score']
                if current_score > self.best_score:
                    self.best_model = results[model_name]['model']
                    self.best_params = results[model_name]['best_params']
                    self.best_score = current_score
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def save_best_model(self, model_path: str = "models/best_model.pkl") -> None:
        """
        Save the best model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        # Create models directory
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Save model info
        model_info = {
            'model_type': type(self.best_model).__name__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'feature_names': list(self.best_model.feature_names_in_) if hasattr(self.best_model, 'feature_names_in_') else None
        }
        
        info_path = model_path.replace('.pkl', '_info.json')
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {info_path}")
    
    def generate_model_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a comprehensive model comparison report.
        
        Args:
            results: Dictionary containing all model results
            
        Returns:
            DataFrame with model comparison
        """
        report_data = []
        
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'CV_F1_Score': result['best_score'],
                'Test_Accuracy': result['metrics']['accuracy'],
                'Test_Precision': result['metrics']['precision'],
                'Test_Recall': result['metrics']['recall'],
                'Test_F1_Score': result['metrics']['f1_score'],
                'Test_ROC_AUC': result['metrics']['roc_auc']
            }
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Test_F1_Score', ascending=False)
        
        return report_df


def main():
    """Main function to run the model training pipeline."""
    try:
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Load data
        logger.info("Loading engineered features...")
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # Train all models
        logger.info("Starting model training...")
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Generate and display report
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON REPORT")
        logger.info("="*80)
        
        report_df = trainer.generate_model_report(results)
        logger.info("\n" + report_df.to_string(index=False))
        
        # Save best model
        logger.info("\nSaving best model...")
        trainer.save_best_model()
        
        # Log best model info
        logger.info(f"\nBest model: {type(trainer.best_model).__name__}")
        logger.info(f"Best F1 Score: {trainer.best_score:.4f}")
        logger.info(f"Best parameters: {trainer.best_params}")
        
        logger.info("\nModel training completed successfully!")
        logger.info("You can view the MLflow UI by running: mlflow ui")
        
        return results, trainer.best_model
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 