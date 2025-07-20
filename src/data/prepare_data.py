"""
Data preparation script for Customer Churn Prediction Project.

This script handles the initial data loading, cleaning, and preprocessing
for the e-commerce customer churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class for handling data preprocessing operations."""
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filename: str = "ecommerce_customer_churn.csv") -> pd.DataFrame:
        """
        Load the raw dataset.
        
        Args:
            filename: Name of the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_path = self.data_path / filename
            if file_path.exists():
                self.raw_data = pd.read_csv(file_path)
                logger.info(f"Data loaded successfully from {file_path}")
                logger.info(f"Dataset shape: {self.raw_data.shape}")
                return self.raw_data
            else:
                # Create sample data for demonstration
                logger.warning(f"File {file_path} not found. Creating sample data.")
                self.raw_data = self._create_sample_data()
                return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample e-commerce customer churn data for demonstration.
        
        Returns:
            Sample DataFrame
        """
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'usage_frequency': np.random.randint(1, 31, n_samples),
            'support_calls': np.random.randint(0, 10, n_samples),
            'monthly_bill': np.random.uniform(20, 100, n_samples),
            'total_usage_gb': np.random.uniform(1, 100, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'unlimited_data': np.random.choice(['Yes', 'No'], n_samples),
            'monthly_charge': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(20, 8000, n_samples)
        }
        
        # Create churn target based on some business logic
        churn_prob = (
            (data['support_calls'] > 5) * 0.3 +
            (data['monthly_bill'] > 80) * 0.2 +
            (data['contract_type'] == 'Month-to-month') * 0.4 +
            (data['tenure'] < 12) * 0.3
        )
        
        data['churn'] = np.random.binomial(1, churn_prob, n_samples)
        
        return pd.DataFrame(data)
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, duplicates, and data types.
        
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.raw_data.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # Fill missing values based on data type
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Fill numeric columns with median
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Standardize column names (convert to lowercase with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert data types
        df['customerid'] = df['customerid'].astype(int)
        df['churn'] = df['churn'].astype(int)
        
        # Handle outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['customerid', 'churn']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        self.processed_data = df
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def save_processed_data(self, filename: str = "processed_data.csv") -> None:
        """
        Save the processed data to the processed data directory.
        
        Args:
            filename: Name of the output file
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary containing dataset summary
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        df = self.processed_data
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_distribution': df['churn'].value_counts().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        return summary


def main():
    """Main function to run the data preparation pipeline."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load data
        logger.info("Loading data...")
        preprocessor.load_data()
        
        # Clean data
        logger.info("Cleaning data...")
        cleaned_data = preprocessor.clean_data()
        
        # Save processed data
        logger.info("Saving processed data...")
        preprocessor.save_processed_data()
        
        # Print summary
        summary = preprocessor.get_data_summary()
        logger.info("Data preparation completed successfully!")
        logger.info(f"Final dataset shape: {summary['shape']}")
        logger.info(f"Churn distribution: {summary['churn_distribution']}")
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 