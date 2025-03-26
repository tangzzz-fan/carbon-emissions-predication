#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Loader Module

This module handles loading data from various sources for training and prediction.
"""

import pandas as pd
import numpy as np
import logging
import requests
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading data from various sources"""
    
    def __init__(self, api_base_url: Optional[str] = None):
        self.api_base_url = api_base_url or "http://localhost:3000"
    
    def load_historical_data(self, device_id: str, data_type: str, hours: int = 24) -> pd.DataFrame:
        """Load historical data from the API
        
        Args:
            device_id: ID of the device
            data_type: Type of data to load
            hours: Number of hours of historical data to load
            
        Returns:
            DataFrame containing the historical data
        """
        try:
            # Call the API to get historical data
            response = requests.get(
                f"{self.api_base_url}/data-collection/historical-data",
                params={
                    "deviceId": device_id,
                    "type": data_type,
                    "hours": hours
                }
            )
            
            if response.status_code == 200:
                # Convert response to DataFrame
                data = response.json()
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Ensure data is complete
                df = self._validate_data(df)
                
                return df
            else:
                logger.error(f"Failed to load historical data: {response.status_code} - {response.text}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.exception(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()
    
    def load_realtime_data(self, device_id: str, data_type: str) -> pd.DataFrame:
        """Load real-time data from the API
        
        Args:
            device_id: ID of the device
            data_type: Type of data to load
            
        Returns:
            DataFrame containing the real-time data
        """
        try:
            # Call the API to get real-time data
            response = requests.get(
                f"{self.api_base_url}/data-collection/realtime-data",
                params={
                    "deviceId": device_id,
                    "type": data_type
                }
            )
            
            if response.status_code == 200:
                # Convert response to DataFrame
                data = response.json()
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure data is complete
                df = self._validate_data(df)
                
                return df
            else:
                logger.error(f"Failed to load real-time data: {response.status_code} - {response.text}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.exception(f"Error loading real-time data: {str(e)}")
            return pd.DataFrame()
    
    def load_from_queue(self, queue_name: str, max_messages: int = 100) -> pd.DataFrame:
        """Load data from a message queue
        
        Args:
            queue_name: Name of the queue to load from
            max_messages: Maximum number of messages to load
            
        Returns:
            DataFrame containing the data from the queue
        """
        try:
            # Call the API to get data from the queue
            response = requests.get(
                f"{self.api_base_url}/data-collection/queue-data",
                params={
                    "queueName": queue_name,
                    "maxMessages": max_messages
                }
            )
            
            if response.status_code == 200:
                # Convert response to DataFrame
                data = response.json()
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp if available
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                
                # Ensure data is complete
                df = self._validate_data(df)
                
                return df
            else:
                logger.error(f"Failed to load data from queue: {response.status_code} - {response.text}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.exception(f"Error loading data from queue: {str(e)}")
            return pd.DataFrame()
    
    def load_from_file(self, file_path: str, file_format: str = 'csv') -> pd.DataFrame:
        """Load data from a file
        
        Args:
            file_path: Path to the file
            file_format: Format of the file (csv, json, etc.)
            
        Returns:
            DataFrame containing the data from the file
        """
        try:
            # Load data based on file format
            if file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_format.lower() == 'json':
                df = pd.read_json(file_path)
            elif file_format.lower() == 'excel' or file_format.lower() == 'xlsx':
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime if available
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Ensure data is complete
            df = self._validate_data(df)
            
            return df
        
        except Exception as e:
            logger.exception(f"Error loading data from file: {str(e)}")
            return pd.DataFrame()
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated and cleaned DataFrame
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning("Empty DataFrame received")
                return df
            
            # Check for required columns
            required_columns = ['timestamp', 'value']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Required column '{col}' not found in data")
            
            # Remove duplicates
            if 'timestamp' in df.columns:
                df = df.drop_duplicates(subset=['timestamp'])
            
            # Handle missing values
            if 'value' in df.columns:
                # Fill missing values with forward fill, then backward fill
                df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers (optional, based on z-score)
            if 'value' in df.columns and len(df) > 10:
                z_scores = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
                df = df[z_scores < 3]  # Keep only values within 3 standard deviations
            
            return df
        
        except Exception as e:
            logger.exception(f"Error validating data: {str(e)}")
            return df
    
    def convert_to_model_format(self, df: pd.DataFrame, target_col: str = 'value', 
                               feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Convert data to format suitable for model training/prediction
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            feature_cols: List of feature column names (optional)
            
        Returns:
            Tuple of (X, y) with features and target
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning("Empty DataFrame received for conversion")
                return df, None
            
            # Extract target if available
            y = None
            if target_col in df.columns:
                y = df[target_col]
            
            # Extract features
            if feature_cols is not None:
                # Use specified feature columns
                available_cols = [col for col in feature_cols if col in df.columns]
                if not available_cols:
                    logger.warning("None of the specified feature columns found in data")
                    X = df.copy()
                else:
                    X = df[available_cols].copy()
            else:
                # Use all columns except target as features
                X = df.copy()
                if target_col in X.columns and y is not None:
                    X = X.drop(columns=[target_col])
            
            # Convert timestamp to datetime features if present
            if 'timestamp' in X.columns:
                # Extract datetime features
                X['hour'] = X['timestamp'].dt.hour
                X['day'] = X['timestamp'].dt.day
                X['day_of_week'] = X['timestamp'].dt.dayofweek
                X['month'] = X['timestamp'].dt.month
                
                # Keep timestamp for reference but not for modeling
                X_for_model = X.drop(columns=['timestamp'])
            else:
                X_for_model = X
            
            return X_for_model, y
        
        except Exception as e:
            logger.exception(f"Error converting data to model format: {str(e)}")
            return df, None