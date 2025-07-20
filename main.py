"""
Customer Churn Prediction Pipeline

Main script for training and evaluating machine learning models for customer churn prediction.
"""

import sys
import logging
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from data.prepare_data import DataPreprocessor
from features.build_features import FeatureEngineer
from models.train_model import ModelTrainer
from models.predict import ChurnPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ChurnPredictionPipeline:
    """Main pipeline for customer churn prediction workflow."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.data_preprocessor = None
        self.feature_engineer = None
        self.model_trainer = None
        self.predictor = None
        
    def run_data_preparation(self) -> bool:
        """Run data preparation step."""
        try:
            logger.info("STEP 1: DATA PREPARATION")
            
            self.data_preprocessor = DataPreprocessor()
            self.data_preprocessor.load_data()
            cleaned_data = self.data_preprocessor.clean_data()
            self.data_preprocessor.save_processed_data()
            
            summary = self.data_preprocessor.get_data_summary()
            logger.info(f"Data preparation completed. Shape: {summary['shape']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            return False
    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering step."""
        try:
            logger.info("STEP 2: FEATURE ENGINEERING")
            
            self.feature_engineer = FeatureEngineer()
            df = self.feature_engineer.load_processed_data()
            
            # Build features
            engineered_data = self.feature_engineer.build_features(df)
            
            # Save engineered features
            self.feature_engineer.save_engineered_features()
            
            logger.info("Feature engineering completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
    def run_model_training(self) -> bool:
        """Run model training and evaluation step."""
        try:
            logger.info("STEP 3: MODEL TRAINING")
            
            self.model_trainer = ModelTrainer()
            
            # Load engineered data
            X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.load_data()
            
            # Train models
            models = self.model_trainer.train_models(X_train, y_train)
            
            # Evaluate models
            results = self.model_trainer.evaluate_models(models, X_val, y_val, X_test, y_test)
            
            # Save best model
            self.model_trainer.save_best_model()
            
            logger.info("Model training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False
    
    def run_prediction_demo(self) -> bool:
        """Run prediction demonstration."""
        try:
            logger.info("STEP 4: PREDICTION DEMO")
            
            self.predictor = ChurnPredictor()
            
            # Load test data
            test_data = self.predictor.load_test_data()
            
            # Make predictions
            predictions = self.predictor.predict(test_data)
            
            # Display results
            self.predictor.display_results(predictions)
            
            logger.info("Prediction demo completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in prediction demo: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info("Starting Customer Churn Prediction Pipeline")
        logger.info("="*50)
        
        steps = [
            ("Data Preparation", self.run_data_preparation),
            ("Feature Engineering", self.run_feature_engineering),
            ("Model Training", self.run_model_training),
            ("Prediction Demo", self.run_prediction_demo)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\nRunning {step_name}...")
            if not step_func():
                logger.error(f"Pipeline failed at {step_name}")
                return False
        
        logger.info("\nPipeline completed successfully!")
        return True


def main():
    """Main function to run the pipeline."""
    pipeline = ChurnPredictionPipeline()
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info("All steps completed successfully!")
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 