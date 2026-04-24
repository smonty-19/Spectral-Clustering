"""
Data Loader for Kaggle Heart Disease Dataset

This module handles loading and preprocessing the heart disease dataset
following Step 1 of the mini-project guidelines: Real-World Data

The dataset contains medical measurements that will be represented as
a matrix for linear algebra operations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os


class HeartDiseaseDataLoader:
    """Load and preprocess Kaggle Heart Disease dataset."""
    
    def __init__(self, filepath='heart.csv', verbose=True):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        filepath : str
            Path to heart.csv from Kaggle
        verbose : bool
            Print loading information
        """
        self.filepath = filepath
        self.verbose = verbose
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def check_file_exists(self):
        """Check if the heart.csv file exists."""
        if not os.path.exists(self.filepath):
            print("\n" + "="*70)
            print("ERROR: heart.csv not found!")
            print("="*70)
            return False
        return True
    
    def load_data(self):
        """
        Load the heart disease dataset.
        
        Returns:
        --------
        pd.DataFrame : Raw dataset
        """
        if not self.check_file_exists():
            raise FileNotFoundError(f"Cannot find {self.filepath}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: LOADING REAL-WORLD DATA")
            print("="*70)
            print(f"\nLoading from: {self.filepath}")
        
        self.df = pd.read_csv(self.filepath)
        
        if self.verbose:
            print(f"\n✓ Dataset loaded successfully!")
            print(f"\nDataset Information:")
            print(f"  - Shape: {self.df.shape}")
            print(f"  - Samples: {self.df.shape[0]} patients")
            print(f"  - Features: {self.df.shape[1]} medical measurements")
            print(f"\nColumn Names:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"  {i:2d}. {col}")
            print(f"\nFirst 5 rows:")
            print(self.df.head())
        
        return self.df
    
    def preprocess_data(self, target_column='target', drop_missing=True):
        """
        Preprocess the dataset.
        
        Steps:
        1. Remove missing values
        2. Separate features (X) and target (y)
        3. Standardize features
        
        Parameters:
        -----------
        target_column : str
            Name of target column
        drop_missing : bool
            Remove rows with missing values
        
        Returns:
        --------
        tuple : (X_scaled, y, feature_names)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("PREPROCESSING DATA")
            print("="*70)
        
        df_clean = self.df.copy()
        
        # Handle missing values
        if drop_missing:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            if self.verbose:
                print(f"\nMissing Values Handling:")
                print(f"  - Rows before: {initial_rows}")
                print(f"  - Rows removed: {initial_rows - len(df_clean)}")
                print(f"  - Rows after: {len(df_clean)}")
        
        # Separate features and target
        if target_column in df_clean.columns:
            self.y = df_clean[target_column].values
            X_raw = df_clean.drop(columns=[target_column]).values
            self.feature_names = df_clean.drop(columns=[target_column]).columns.tolist()
        else:
            # Assume last column is target
            self.y = df_clean.iloc[:, -1].values
            X_raw = df_clean.iloc[:, :-1].values
            self.feature_names = df_clean.columns[:-1].tolist()
        
        # Convert target to binary classification
        self.y = (self.y > 0).astype(int)
        
        if self.verbose:
            print(f"\nTarget Variable Conversion:")
            print(f"  - Class 0 (No Disease): {np.sum(self.y == 0)} patients")
            print(f"  - Class 1 (Has Disease): {np.sum(self.y == 1)} patients")
            print(f"  - Ratio: {np.sum(self.y == 1) / len(self.y) * 100:.1f}% have disease")
        
        # Standardize features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X_raw)
        
        if self.verbose:
            print(f"\nFeature Standardization:")
            print(f"  - Original X shape: {X_raw.shape}")
            print(f"  - Standardized X shape: {self.X.shape}")
            print(f"  - Mean (after scaling): {np.mean(self.X, axis=0)}")
            print(f"  - Std (after scaling): {np.std(self.X, axis=0)}")
            print(f"\nFeatures (as matrix representation):")
            for i, name in enumerate(self.feature_names, 1):
                print(f"  {i:2d}. {name}")
            print(f"\nMatrix Representation:")
            print(f"  - Data matrix X: {self.X.shape[0]} patients × {self.X.shape[1]} measurements")
            print(f"  - Target vector y: {self.y.shape[0]} labels")
            print(f"\nFirst 5 rows of standardized data:")
            print(self.X[:5])
        
        return self.X, self.y, self.feature_names
    
    def get_data(self):
        """Get processed data."""
        return self.X, self.y, self.feature_names
    
    def get_feature_info(self):
        """Get information about features."""
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_samples': len(self.X) if self.X is not None else 0,
            'n_classes': len(np.unique(self.y)) if self.y is not None else 0
        }


def load_heart_disease_dataset(filepath='heart.csv', verbose=True):
    """
    Convenience function to load and preprocess heart disease data.
    
    Parameters:
    -----------
    filepath : str
        Path to heart.csv
    verbose : bool
        Print information
    
    Returns:
    --------
    tuple : (X_scaled, y, feature_names)
    """
    loader = HeartDiseaseDataLoader(filepath, verbose)
    loader.load_data()
    X, y, features = loader.preprocess_data()
    return X, y, features


if __name__ == "__main__":
    # Test data loader
    loader = HeartDiseaseDataLoader('heart.csv', verbose=True)
    loader.load_data()
    X, y, features = loader.preprocess_data()
    
    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)
    print(f"\nReady for spectral clustering!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")