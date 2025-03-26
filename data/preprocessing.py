#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessing Module

This module handles data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setup logger
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for data preprocessing"""
    
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_cleaned = df.copy()
            
            # Log initial data stats
            logger.info(f"Cleaning data with shape {df.shape}")
            logger.info(f"Missing values before cleaning: {df.isna().sum().sum()}")
            
            # Handle missing values
            # First try forward fill (use previous values)
            df_cleaned = df_cleaned.fillna(method='ffill')
            # Then try backward fill for any remaining NAs
            df_cleaned = df_cleaned.fillna(method='bfill')
            
            # For any remaining NAs, fill with column mean/median/mode as appropriate
            for col in df_cleaned.columns:
                if df_cleaned[col].isna().any():
                    if np.issubdtype(df_cleaned[col].dtype, np.number):
                        # For numeric columns, use median
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    else:
                        # For non-numeric columns, use mode
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            
            # Handle outliers using IQR method
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
            
            # Log cleaning results
            logger.info(f"Missing values after cleaning: {df_cleaned.isna().sum().sum()}")
            logger.info(f"Data cleaned successfully with shape {df_cleaned.shape}")
            
            return df_cleaned
        
        except Exception as e:
            logger.exception(f"Error cleaning data: {str(e)}")
            # Return original DataFrame if cleaning fails
            return df
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler or MinMaxScaler
        
        Args:
            df: DataFrame containing features to scale
            columns: List of column names to scale
            method: Scaling method ('standard' or 'minmax')
            fit: Whether to fit the scaler or use previously fitted scaler
            
        Returns:
            DataFrame with scaled features
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_scaled = df.copy()
            
            # Select scaler based on method
            scaler = self.standard_scaler if method == 'standard' else self.minmax_scaler
            
            # Check if columns exist in DataFrame
            existing_columns = [col for col in columns if col in df_scaled.columns]
            if not existing_columns:
                logger.warning(f"None of the specified columns {columns} exist in the DataFrame")
                return df_scaled
            
            # Fit or transform based on the fit parameter
            if fit:
                scaled_data = scaler.fit_transform(df_scaled[existing_columns])
            else:
                scaled_data = scaler.transform(df_scaled[existing_columns])
            
            # Create a DataFrame with scaled data
            scaled_df = pd.DataFrame(scaled_data, columns=existing_columns, index=df_scaled.index)
            
            # Replace original columns with scaled values
            for col in existing_columns:
                df_scaled[col] = scaled_df[col]
            
            logger.info(f"Scaled {len(existing_columns)} features using {method} scaling")
            
            return df_scaled
        
        except Exception as e:
            logger.exception(f"Error scaling features: {str(e)}")
            # Return original DataFrame if scaling fails
            return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'one-hot') -> pd.DataFrame:
        """Encode categorical variables
        
        Args:
            df: DataFrame containing categorical variables
            columns: List of column names to encode
            method: Encoding method ('one-hot' or 'label')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_encoded = df.copy()
            
            # Check if columns exist in DataFrame
            existing_columns = [col for col in columns if col in df_encoded.columns]
            if not existing_columns:
                logger.warning(f"None of the specified columns {columns} exist in the DataFrame")
                return df_encoded
            
            # Encode categorical variables based on method
            if method == 'one-hot':
                # One-hot encoding
                for col in existing_columns:
                    # Get dummies and add prefix to avoid column name conflicts
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                    # Add dummy columns to DataFrame
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    # Drop original column
                    df_encoded = df_encoded.drop(col, axis=1)
            elif method == 'label':
                # Label encoding
                for col in existing_columns:
                    # Map unique values to integers
                    unique_values = df_encoded[col].unique()
                    mapping = {value: i for i, value in enumerate(unique_values)}
                    # Apply mapping
                    df_encoded[col] = df_encoded[col].map(mapping)
            
            logger.info(f"Encoded {len(existing_columns)} categorical variables using {method} encoding")
            
            return df_encoded
        
        except Exception as e:
            logger.exception(f"Error encoding categorical variables: {str(e)}")
            # Return original DataFrame if encoding fails
            return df
    
    def add_time_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Add time-based features from timestamp column
        
        Args:
            df: DataFrame containing timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with added time features
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_time = df.copy()
            
            # Check if timestamp column exists
            if timestamp_col not in df_time.columns:
                logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
                return df_time
            
            # Ensure timestamp column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_time[timestamp_col]):
                df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
            
            # Extract time features
            df_time['hour'] = df_time[timestamp_col].dt.hour
            df_time['day_of_week'] = df_time[timestamp_col].dt.dayofweek
            df_time['day_of_month'] = df_time[timestamp_col].dt.day
            df_time['month'] = df_time[timestamp_col].dt.month
            df_time['year'] = df_time[timestamp_col].dt.year
            df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
            
            # Add cyclical features for hour, day of week, and month
            # These help capture the cyclical nature of time
            df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
            df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
            df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
            df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
            
            logger.info(f"Added time features based on '{timestamp_col}' column")
            
            return df_time
        
        except Exception as e:
            logger.exception(f"Error adding time features: {str(e)}")
            # Return original DataFrame if adding time features fails
            return df