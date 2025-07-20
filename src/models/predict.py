"""
Prediction script for Customer Churn Prediction Project.

This script handles making predictions using the trained model
and provides a simple API for churn prediction with GPU support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from typing import Dict, List, Union, Optional
import warnings
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# GPU Configuration
def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

# Global device variable
DEVICE = get_device()


class ChurnPredictor:
    """Class for making churn predictions using trained models with GPU support."""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize the ChurnPredictor.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = None
        self.feature_names = None
        self.model_type = None  # 'sklearn' or 'pytorch'
        self.device = DEVICE
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and its metadata."""
        try:
            # Check if it's a PyTorch model
            if self.model_path.suffix == '.pth':
                self.model_type = 'pytorch'
                self._load_pytorch_model()
            else:
                self.model_type = 'sklearn'
                self._load_sklearn_model()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model type: {self.model_type}")
            logger.info(f"Device: {self.device}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_sklearn_model(self) -> None:
        """Load sklearn model."""
        self.model = joblib.load(self.model_path)
        
        # Load model info
        info_path = self.model_path.with_suffix('.json').with_name(self.model_path.stem + '_info.json')
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
            logger.info(f"Model info loaded from {info_path}")
        
        # Get feature names
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif self.model_info and 'feature_names' in self.model_info:
            self.feature_names = self.model_info['feature_names']
        
        logger.info(f"Number of features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        # Load model architecture and weights
        model_info_path = self.model_path.with_suffix('.json').with_name(self.model_path.stem + '_info.json')
        
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
            
            # Create model with same architecture
            input_size = self.model_info.get('input_size', 50)
            hidden_sizes = self.model_info.get('hidden_sizes', [128, 64, 32])
            dropout_rate = self.model_info.get('dropout_rate', 0.3)
            
            self.model = self._create_pytorch_model(input_size, hidden_sizes, dropout_rate)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.feature_names = self.model_info.get('feature_names', [])
            logger.info(f"PyTorch model loaded with {input_size} input features")
        else:
            raise FileNotFoundError(f"Model info file not found: {model_info_path}")
    
    def _create_pytorch_model(self, input_size: int, hidden_sizes: list, dropout_rate: float) -> nn.Module:
        """Create PyTorch model architecture."""
        class ChurnPredictor(nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout_rate):
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
        
        return ChurnPredictor(input_size, hidden_sizes, dropout_rate)
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Preprocess input data for prediction using the same pipeline as training.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame")
        
        # Standardize column names (convert to lowercase with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Simple preprocessing to match training pipeline
        # Step 1: Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Step 2: Create basic engineered features (simplified version)
        # Cashback per order
        if 'cashbackamount' in df.columns and 'ordercount' in df.columns:
            df['cashback_per_order'] = df['cashbackamount'] / (df['ordercount'] + 1)
        
        # Hours per device
        if 'hourspendonapp' in df.columns and 'numberofdeviceregistered' in df.columns:
            df['hours_per_device'] = df['hourspendonapp'] / (df['numberofdeviceregistered'] + 1)
        
        # Orders per tenure
        if 'ordercount' in df.columns and 'tenure' in df.columns:
            df['orders_per_tenure'] = df['ordercount'] / (df['tenure'] + 1)
        
        # Days per order
        if 'daysincelastorder' in df.columns and 'ordercount' in df.columns:
            df['days_per_order'] = df['daysincelastorder'] / (df['ordercount'] + 1)
        
        # Risk indicators
        if 'complain' in df.columns:
            df['risk_score'] = df['complain'] * 0.5
        if 'satisfactionscore' in df.columns:
            df['high_satisfaction'] = (df['satisfactionscore'] >= 4).astype(int)
        if 'cashbackamount' in df.columns:
            df['high_cashback'] = (df['cashbackamount'] > df['cashbackamount'].quantile(0.75)).astype(int)
        if 'tenure' in df.columns:
            df['short_tenure'] = (df['tenure'] < 12).astype(int)
        
        # Customer value
        if 'ordercount' in df.columns and 'cashbackamount' in df.columns:
            df['customer_value'] = df['ordercount'] * df['cashbackamount'] / 100
        
        # Categorical encoding (simple label encoding)
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        # Step 3: Feature selection (use the same features as training)
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    df[feature] = 0
            
            # Select only the features used by the model
            df = df[self.feature_names]
        
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict[str, Union[int, float, List]]:
        """
        Make churn predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        try:
            # Preprocess input
            X = self.preprocess_input(data)
            
            # Make predictions based on model type
            if self.model_type == 'sklearn':
                predictions = self.model.predict(X)
                probabilities = self.model.predict_proba(X)[:, 1]  # Probability of churning
            elif self.model_type == 'pytorch':
                with torch.no_grad():
                    X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
                    outputs = self.model(X_tensor).cpu().numpy()
                    probabilities = outputs.flatten()
                    predictions = (probabilities > 0.5).astype(int)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Prepare results
            results = {
                'predictions': predictions.tolist() if len(predictions) > 1 else int(predictions[0]),
                'probabilities': probabilities.tolist() if len(probabilities) > 1 else float(probabilities[0]),
                'churn_risk': self._get_churn_risk_level(probabilities)
            }
            
            logger.info(f"Predictions completed for {len(X)} samples using {self.model_type} model")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def _get_churn_risk_level(self, probabilities: np.ndarray) -> Union[str, List[str]]:
        """
        Convert probabilities to risk levels.
        
        Args:
            probabilities: Churn probabilities
            
        Returns:
            Risk levels (Low, Medium, High)
        """
        def get_risk_level(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"
        
        if len(probabilities) == 1:
            return get_risk_level(probabilities[0])
        else:
            return [get_risk_level(p) for p in probabilities]
    
    def predict_batch(self, filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from a file.
        
        Args:
            filepath: Path to the input data file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with original data and predictions
        """
        try:
            # Load data
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} samples from {filepath}")
            
            # Make predictions
            results = self.predict(df)
            
            # Add predictions to original data
            df['churn_prediction'] = results['predictions']
            df['churn_probability'] = results['probabilities']
            df['churn_risk'] = results['churn_risk']
            
            # Save results if output path provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Union[str, float]]:
        """
        Get model performance information.
        
        Returns:
            Dictionary with model performance metrics
        """
        if not self.model_info:
            return {"error": "No model info available"}
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "best_score": self.model_info.get("best_score", 0.0),
            "best_params": self.model_info.get("best_params", {}),
            "feature_count": len(self.feature_names) if self.feature_names else 0
        }


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing predictions using e-commerce features.
    
    Returns:
        Sample DataFrame
    """
    sample_data = {
        'tenure': [12, 24, 6, 36, 18],
        'preferredlogindevice': ['Mobile App', 'Computer', 'Mobile App', 'Computer', 'Mobile App'],
        'citytier': [1, 2, 3, 1, 2],
        'warehousetohome': [15, 25, 8, 30, 20],
        'preferredpaymentmode': ['Credit Card', 'UPI', 'Debit Card', 'Credit Card', 'UPI'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'hourspendonapp': [3.5, 2.1, 4.2, 1.8, 3.0],
        'numberofdeviceregistered': [2, 3, 1, 4, 2],
        'preferedordercat': ['Laptop & Accessory', 'Mobile', 'Fashion', 'Grocery', 'Mobile'],
        'satisfactionscore': [4, 3, 5, 2, 4],
        'maritalstatus': ['Single', 'Married', 'Single', 'Married', 'Single'],
        'numberofaddress': [2, 3, 1, 4, 2],
        'complain': [0, 1, 0, 1, 0],
        'orderamounthikefromlastyear': [15, 25, 10, 30, 20],
        'couponused': [1, 0, 1, 0, 1],
        'ordercount': [8, 12, 5, 15, 10],
        'daysincelastorder': [5, 2, 8, 1, 3],
        'cashbackamount': [45.2, 78.9, 23.1, 95.6, 67.4]
    }
    
    return pd.DataFrame(sample_data)


def main():
    """Main function to demonstrate prediction functionality."""
    try:
        # Initialize predictor
        logger.info("Initializing ChurnPredictor...")
        predictor = ChurnPredictor()
        
        # Create sample data
        logger.info("Creating sample data for testing...")
        sample_data = create_sample_data()
        
        # Make predictions
        logger.info("Making predictions on sample data...")
        results = predictor.predict(sample_data)
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("PREDICTION RESULTS")
        logger.info("="*60)
        
        for i, (pred, prob, risk) in enumerate(zip(
            results['predictions'], 
            results['probabilities'], 
            results['churn_risk']
        )):
            logger.info(f"Customer {i+1}:")
            logger.info(f"  Churn Prediction: {'Yes' if pred else 'No'}")
            logger.info(f"  Churn Probability: {prob:.3f}")
            logger.info(f"  Risk Level: {risk}")
        
        # Get model performance
        performance = predictor.get_model_performance()
        logger.info("Model Performance:")
        logger.info(f"  Model Type: {performance['model_type']}")
        logger.info(f"  Best Score: {performance['best_score']:.4f}")
        logger.info(f"  Feature Count: {performance['feature_count']}")
        
        # Save sample predictions
        output_path = "data/processed/sample_predictions.csv"
        # Save sample data first, then predict
        sample_data.to_csv("temp_sample_data.csv", index=False)
        predictor.predict_batch("temp_sample_data.csv", output_path)
        # Clean up temp file
        import os
        if os.path.exists("temp_sample_data.csv"):
            os.remove("temp_sample_data.csv")
        
        logger.info("Prediction demonstration completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in prediction demonstration: {e}")
        raise


if __name__ == "__main__":
    main() 