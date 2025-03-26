#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering Module

This module handles feature extraction and transformation.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Setup logger
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Class for feature extraction and transformation"""
    
    def __init__(self):
        self.pca = None
        self.selector = None
    
    def extract_features_from_data(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from a single data point
        
        Args:
            data: Dictionary containing data point
            
        Returns:
            List of extracted features
        """
        try:
            features = []
            
            # Extract value
            if 'value' in data:
                features.append(float(data['value']))
            
            # Extract CO2 equivalent if available
            if 'co2Equivalent' in data:
                features.append(float(data['co2Equivalent']))
            
            # Extract time features from timestamp
            if 'timestamp' in data:
                timestamp = pd.to_datetime(data['timestamp'])
                # Hour of day (0-23)
                features.append(timestamp.hour)
                # Day of week (0-6, 0 is Monday)
                features.append(timestamp.dayofweek)
                # Is weekend (0 or 1)
                features.append(1 if timestamp.dayofweek >= 5 else 0)
                # Month (1-12)
                features.append(timestamp.month)
                # Add cyclical features
                features.append(np.sin(2 * np.pi * timestamp.hour / 24))
                features.append(np.cos(2 * np.pi * timestamp.hour / 24))
                features.append(np.sin(2 * np.pi * timestamp.dayofweek / 7))
                features.append(np.cos(2 * np.pi * timestamp.dayofweek / 7))
            
            logger.info(f"Extracted {len(features)} features from data point")
            
            return features
        
        except Exception as e:
            logger.exception(f"Error extracting features: {str(e)}")
            # Return empty list if feature extraction fails
            return []
    
    def create_time_series_features(self, df: pd.DataFrame, target_col: str, window_sizes: List[int] = [1, 6, 12, 24]) -> pd.DataFrame:
        """Create time series features like rolling mean, std, etc.
        
        Args:
            df: DataFrame containing time series data
            target_col: Column to create features for
            window_sizes: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with added time series features
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_ts = df.copy()
            
            # Check if target column exists
            if target_col not in df_ts.columns:
                logger.warning(f"Target column '{target_col}' not found in DataFrame")
                return df_ts
            
            # Ensure DataFrame is sorted by timestamp
            if 'timestamp' in df_ts.columns:
                df_ts = df_ts.sort_values('timestamp')
            
            # Create rolling features for each window size
            for window in window_sizes:
                # Rolling mean
                df_ts[f'{target_col}_rolling_mean_{window}'] = df_ts[target_col].rolling(window=window, min_periods=1).mean()
                
                # Rolling standard deviation
                df_ts[f'{target_col}_rolling_std_{window}'] = df_ts[target_col].rolling(window=window, min_periods=1).std()
                
                # Rolling min and max
                df_ts[f'{target_col}_rolling_min_{window}'] = df_ts[target_col].rolling(window=window, min_periods=1).min()
                df_ts[f'{target_col}_rolling_max_{window}'] = df_ts[target_col].rolling(window=window, min_periods=1).max()
                
                # Lag features
                df_ts[f'{target_col}_lag_{window}'] = df_ts[target_col].shift(window)
            
            # Calculate difference features
            df_ts[f'{target_col}_diff_1'] = df_ts[target_col].diff(1)
            df_ts[f'{target_col}_diff_7'] = df_ts[target_col].diff(7)
            df_ts[f'{target_col}_diff_24'] = df_ts[target_col].diff(24)
            
            # Calculate percentage change
            df_ts[f'{target_col}_pct_change_1'] = df_ts[target_col].pct_change(1)
            df_ts[f'{target_col}_pct_change_7'] = df_ts[target_col].pct_change(7)
            df_ts[f'{target_col}_pct_change_24'] = df_ts[target_col].pct_change(24)
            
            # Fill NaN values created by lag and diff operations
            df_ts = df_ts.fillna(method='bfill').fillna(method='ffill')
            
            logger.info(f"Created time series features for '{target_col}' with window sizes {window_sizes}")
            
            return df_ts
        
        except Exception as e:
            logger.exception(f"Error creating time series features: {str(e)}")
            # Return original DataFrame if creating time series features fails
            return df
    
    def apply_pca(self, df: pd.DataFrame, columns: List[str], n_components: int = 2, fit: bool = True) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction
        
        Args:
            df: DataFrame containing features
            columns: List of column names to apply PCA to
            n_components: Number of principal components to keep
            fit: Whether to fit the PCA or use previously fitted PCA
            
        Returns:
            DataFrame with PCA features
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_pca = df.copy()
            
            # Check if columns exist in DataFrame
            existing_columns = [col for col in columns if col in df_pca.columns]
            if not existing_columns:
                logger.warning(f"None of the specified columns {columns} exist in the DataFrame")
                return df_pca
            
            # Initialize PCA if not already initialized or if fit is True
            if self.pca is None or fit:
                self.pca = PCA(n_components=min(n_components, len(existing_columns)))
            
            # Fit or transform based on the fit parameter
            if fit:
                pca_result = self.pca.fit_transform(df_pca[existing_columns])
            else:
                pca_result = self.pca.transform(df_pca[existing_columns])
            
            # Add PCA components to DataFrame
            for i in range(pca_result.shape[1]):
                df_pca[f'pca_component_{i+1}'] = pca_result[:, i]
            
            # Optionally drop original columns
            # df_pca = df_pca.drop(existing_columns, axis=1)
            
            logger.info(f"Applied PCA to {len(existing_columns)} features, reduced to {pca_result.shape[1]} components")
            
            return df_pca
        
        except Exception as e:
            logger