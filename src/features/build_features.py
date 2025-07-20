"""
Feature engineering script for Customer Churn Prediction Project.

This script handles feature creation, transformation, and selection
for the e-commerce customer churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Class for handling feature engineering operations."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = None
        
    def load_processed_data(self, filepath: str = "data/processed/processed_data.csv") -> pd.DataFrame:
        """
        Load the processed data.
        
        Args:
            filepath: Path to the processed data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Processed data loaded successfully from {filepath}")
            logger.info(f"Dataset shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Please run data preparation first.")
            raise
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing e-commerce data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        
        # Create e-commerce specific interaction features
        df['cashback_per_order'] = df['cashbackamount'] / (df['ordercount'] + 1)
        df['hours_per_device'] = df['hourspendonapp'] / (df['numberofdeviceregistered'] + 1)
        df['orders_per_tenure'] = df['ordercount'] / (df['tenure'] + 1)
        df['hike_per_order'] = df['orderamounthikefromlastyear'] / (df['ordercount'] + 1)
        
        # Create ratio features
        df['coupon_usage_rate'] = df['couponused'] / (df['ordercount'] + 1)
        df['days_per_order'] = df['daysincelastorder'] / (df['ordercount'] + 1)
        
        # Create categorical features based on quantiles
        df['high_cashback'] = (df['cashbackamount'] > df['cashbackamount'].quantile(0.75)).astype(int)
        df['high_satisfaction'] = (df['satisfactionscore'] > df['satisfactionscore'].quantile(0.75)).astype(int)
        df['frequent_orders'] = (df['ordercount'] > df['ordercount'].quantile(0.75)).astype(int)
        df['short_tenure'] = (df['tenure'] < df['tenure'].quantile(0.25)).astype(int)
        df['high_hike'] = (df['orderamounthikefromlastyear'] > df['orderamounthikefromlastyear'].quantile(0.75)).astype(int)
        
        # Create tenure groups for e-commerce (as numeric codes)
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                   labels=[0, 1, 2, 3])  # Numeric codes instead of strings
        
        # Create satisfaction groups (as numeric codes)
        df['satisfaction_group'] = pd.cut(df['satisfactionscore'], bins=[0, 3, 4, 5], 
                                         labels=[0, 1, 2])  # Numeric codes instead of strings
        
        # Create city tier risk (higher tier = lower risk)
        df['city_risk'] = 4 - df['citytier']  # Tier 1 = low risk (1), Tier 3 = high risk (3)
        
        # Create device risk (more devices = higher engagement = lower risk)
        df['device_risk'] = 1 / (df['numberofdeviceregistered'] + 1)
        
        # Create address risk (more addresses = higher risk)
        df['address_risk'] = df['numberofaddress']
        
        # Create engagement score
        df['engagement_score'] = (
            df['hourspendonapp'] * 0.3 +
            df['ordercount'] * 0.3 +
            df['satisfactionscore'] * 0.2 +
            df['numberofdeviceregistered'] * 0.2
        )
        
        # Create risk score
        df['risk_score'] = (
            df['complain'] * 0.4 +
            df['daysincelastorder'] * 0.3 +
            df['orderamounthikefromlastyear'] * 0.2 +
            df['city_risk'] * 0.1
        )
        
        # Create customer value score
        df['customer_value'] = (
            df['cashbackamount'] * 0.4 +
            df['ordercount'] * 0.3 +
            df['tenure'] * 0.2 +
            df['satisfactionscore'] * 0.1
        )
        
        # Handle any NaN values that might have been created during feature engineering
        # Fill NaN in numeric columns only (avoid categorical columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill NaN in categorical columns with mode or default value
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col in ['tenure_group', 'satisfaction_group']:
                    # For our custom categorical features, use the most common category
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
                else:
                    # For other categorical columns, use mode
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info("E-commerce feature creation completed")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'customerid':  # Skip customer ID
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical feature: {col}")
        
        return df
    
    def scale_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            DataFrame with scaled numeric features
        """
        df = df.copy()
        
        # Identify numeric columns (excluding target and ID)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['customerid', 'churn']]
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"Fitted scaler on {len(numeric_cols)} numeric features")
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info(f"Transformed {len(numeric_cols)} numeric features")
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'churn', 
                       k: int = 20) -> pd.DataFrame:
        """
        Select the most important features using statistical tests.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        df = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['customer_id', target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # Apply feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        self.feature_names = selected_features
        
        # Create new DataFrame with selected features
        selected_df = df[['customerid', target_col] + selected_features].copy()
        
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_df
    
    def prepare_train_test_split(self, df: pd.DataFrame, target_col: str = 'churn',
                                test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare train-test split for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['customerid', target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Train churn rate: {y_train.mean():.3f}")
        logger.info(f"Test churn rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_engineered_data(self, df: pd.DataFrame, filename: str = "engineered_features.csv") -> None:
        """
        Save the engineered features to the processed data directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
        """
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Engineered features saved to {output_path}")
    
    def get_feature_importance_scores(self) -> pd.DataFrame:
        """
        Get feature importance scores from the feature selector.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted. Run select_features() first.")
        
        # Get all feature names and scores (not just selected ones)
        all_feature_names = self.feature_selector.feature_names_in_
        all_scores = self.feature_selector.scores_
        
        # Create DataFrame with all features and their scores
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance_score': all_scores
        }).sort_values('importance_score', ascending=False)
        
        return importance_df


def main():
    """Main function to run the feature engineering pipeline."""
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Load processed data
        logger.info("Loading processed data...")
        df = engineer.load_processed_data()
        
        # Create features
        logger.info("Creating new features...")
        df = engineer.create_features(df)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        df = engineer.encode_categorical_features(df)
        
        # Scale numeric features
        logger.info("Scaling numeric features...")
        df = engineer.scale_numeric_features(df, fit=True)
        
        # Select features
        logger.info("Selecting important features...")
        df_selected = engineer.select_features(df)
        
        # Prepare train-test split
        logger.info("Preparing train-test split...")
        X_train, X_test, y_train, y_test = engineer.prepare_train_test_split(df_selected)
        
        # Save engineered data
        logger.info("Saving engineered features...")
        engineer.save_engineered_data(df_selected)
        
        # Get feature importance
        importance_df = engineer.get_feature_importance_scores()
        logger.info("Top 10 most important features:")
        logger.info(importance_df.head(10))
        
        logger.info("Feature engineering completed successfully!")
        
        return df_selected, X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 