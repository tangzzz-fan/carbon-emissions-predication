#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time Series Models Module

This module implements time series prediction models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from models.base import BaseModel
from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """Facebook Prophet time series model"""
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        """Initialize the Prophet model
        
        Args:
            name: Name of the model
            **kwargs: Additional arguments for Prophet
        """
        super().__init__(name=name or "ProphetModel")
        self.model = Prophet(**kwargs)
        self.forecast = None
    
    def train(self, X, y=None, **kwargs):
        """Train the Prophet model
        
        Args:
            X: DataFrame with 'ds' (datetime) column
               If y is None, X should also contain 'y' column
            y: Series with target values (optional)
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Trained model
        """
        try:
            # Prepare training data
            if isinstance(X, pd.DataFrame):
                train_df = X.copy()
                
                # Check if 'ds' column exists
                if 'ds' not in train_df.columns and 'timestamp' in train_df.columns:
                    train_df['ds'] = train_df['timestamp']
                
                # Add target column if provided separately
                if y is not None:
                    train_df['y'] = y
                
                # Check required columns
                if 'ds' not in train_df.columns or 'y' not in train_df.columns:
                    raise ValueError("Training data must contain 'ds' and 'y' columns")
            else:
                raise ValueError("Training data must be a pandas DataFrame")
            
            # Fit the model
            self.model.fit(train_df, **kwargs)
            
            # Update model state
            self.trained = True
            self.updated_at = datetime.now()
            
            # Update metadata
            self.metadata["train_shape"] = train_df.shape
            self.metadata["train_start"] = train_df['ds'].min().isoformat()
            self.metadata["train_end"] = train_df['ds'].max().isoformat()
            
            logger.info(f"Trained {self.name} on {len(train_df)} samples")
            
            return self
        
        except Exception as e:
            logger.exception(f"Error training {self.name}: {str(e)}")
            raise
    
    def predict(self, X=None, periods: int = 24, freq: str = 'H', **kwargs):
        """Make predictions with the Prophet model
        
        Args:
            X: DataFrame with 'ds' column for future dates (optional)
               If not provided, a future DataFrame will be created
            periods: Number of periods to forecast
            freq: Frequency of forecast
            **kwargs: Additional arguments for model.predict()
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Create future DataFrame if not provided
            if X is None:
                future = self.model.make_future_dataframe(periods=periods, freq=freq)
            else:
                # Use provided DataFrame
                future = X.copy()
                
                # Check if 'ds' column exists
                if 'ds' not in future.columns and 'timestamp' in future.columns:
                    future['ds'] = future['timestamp']
                
                # Check required columns
                if 'ds' not in future.columns:
                    raise ValueError("Prediction data must contain 'ds' column")
            
            # Make predictions
            self.forecast = self.model.predict(future)
            
            logger.info(f"Made predictions with {self.name} for {len(future)} samples")
            
            return self.forecast
        
        except Exception as e:
            logger.exception(f"Error predicting with {self.name}: {str(e)}")
            raise
    
    def get_forecast_components(self):
        """Get forecast components
        
        Returns:
            Dictionary of forecast components
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        return self.model.plot_components(self.forecast)
    
    def format_predictions(self, predictions: pd.DataFrame, device_id: str) -> Dict[str, Any]:
        """Format predictions according to API specification
        
        Args:
            predictions: DataFrame with predictions
            device_id: ID of the device
            
        Returns:
            Dictionary with formatted predictions
        """
        try:
            # Check if predictions exist
            if predictions is None or len(predictions) == 0:
                return {
                    "success": False,
                    "error": "No predictions available",
                    "message": "Failed to generate predictions"
                }
            
            # Format predictions
            formatted_predictions = []
            for _, row in predictions.iterrows():
                formatted_predictions.append({
                    "timestamp": row['ds'].isoformat(),
                    "predictedValue": float(row['yhat']),
                    "confidence": 0.95,  # Fixed confidence for now
                    "unit": "kgCO2"
                })
            
            # Create response
            response = {
                "success": True,
                "deviceId": device_id,
                "predictions": formatted_predictions,
                "modelInfo": {
                    "version": "1.0.0",
                    "type": "Prophet",
                    "trainedOn": self.updated_at.isoformat()
                },
                "metadata": {
                    "factorsConsidered": ["历史数据趋势", "季节性", "节假日"],
                    "anomalyDetected": False
                }
            }
            
            return response
        
        except Exception as e:
            logger.exception(f"Error formatting predictions: {str(e)}")
            return {
                "success": False,
                "error": "Formatting Error",
                "message": str(e)
            }


class ARIMAModel(BaseModel):
    """ARIMA time series model"""
    
    def __init__(self, name: Optional[str] = None, order: Tuple[int, int, int] = (1, 1, 1)):
        """Initialize the ARIMA model
        
        Args:
            name: Name of the model
            order: ARIMA order (p, d, q)
        """
        super().__init__(name=name or f"ARIMA{order}")
        self.order = order
        self.model = None
        self.history = []
    
    def train(self, X, y=None, **kwargs):
        """Train the ARIMA model
        
        Args:
            X: DataFrame or Series with time series data
               If DataFrame, it should contain a target column
            y: Series with target values (optional)
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Trained model
        """
        try:
            # Prepare training data
            if isinstance(X, pd.DataFrame):
                # If y is provided, use it as the target
                if y is not None:
                    train_data = y
                # Otherwise, try to find a target column
                elif 'value' in X.columns:
                    train_data = X['value']
                elif 'y' in X.columns:
                    train_data = X['y']
                else:
                    raise ValueError("Training data must contain a target column or y must be provided")
            else:
                # Use X as the target
                train_data = X
            
            # Convert to list if necessary
            if isinstance(train_data, (pd.Series, np.ndarray)):
                self.history = train_data.tolist()
            else:
                self.history = train_data
            
            # Fit the model
            self.model = ARIMA(self.history, order=self.order)
            self.model_fit = self.model.fit(**kwargs)
            
            # Update model state
            self.trained = True
            self.updated_at = datetime.now()
            
            # Update metadata
            self.metadata["train_size"] = len(self.history)
            self.metadata["order"] = self.order
            
            logger.info(f"Trained {self.name} on {len(self.history)} samples")
            
            return self
        
        except Exception as e:
            logger.exception(f"Error training {self.name}: {str(e)}")
            raise
    
    def predict(self, X=None, steps: int = 24, **kwargs):
        """Make predictions with the ARIMA model
        
        Args:
            X: Not used for ARIMA, included for API consistency
            steps: Number of steps to forecast
            **kwargs: Additional arguments for model.forecast()
            
        Returns:
            Array with predictions
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Make predictions
            forecast = self.model_fit.forecast(steps=steps, **kwargs)
            
            logger.info(f"Made predictions with {self.name} for {steps} steps")
            
            return forecast
        
        except Exception as e:
            logger.exception(f"Error predicting with {self.name}: {str(e)}")
            raise
    
    def update(self, new_observations):
        """Update the model with new observations
        
        Args:
            new_observations: New observations to add to history
            
        Returns:
            Self
        """
        try:
            # Convert to list if necessary
            if isinstance(new_observations, (pd.Series, np.ndarray)):
                new_data = new_observations.tolist()
            else:
                new_data = new_observations
            
            # Add new observations to history
            self.history.extend(new_data)
            
            # Retrain the model
            self.model = ARIMA(self.history, order=self.order)
            self.model_fit = self.model.fit()
            
            # Update model state
            self.updated_at = datetime.now()
            
            # Update metadata
            self.metadata["train_size"] = len(self.history)
            
            logger.info(f"Updated {self.name} with {len(new_data)} new observations")
            
            return self
        
        except Exception as e:
            logger.exception(f"Error updating {self.name}: {str(e)}")
            raise
    
    def format_predictions(self, predictions, device_id: str, start_date: datetime = None) -> Dict[str, Any]:
        """Format predictions according to API specification
        
        Args:
            predictions: Array with predictions
            device_id: ID of the device
            start_date: Start date for predictions
            
        Returns:
            Dictionary with formatted predictions
        """
        try:
            # Check if predictions exist
            if predictions is None or len(predictions) == 0:
                return {
                    "success": False,
                    "error": "No predictions available",
                    "message": "Failed to generate predictions"
                }
            
            # Use current time as start date if not provided
            if start_date is None:
                start_date = datetime.now()
            
            # Format predictions
            formatted_predictions = []
            for i, value in enumerate(predictions):
                prediction_time = start_date + timedelta(hours=i)
                formatted_predictions.append({
                    "timestamp": prediction_time.isoformat(),
                    "predictedValue": float(value),
                    "confidence": 0.9,  # Fixed confidence for now
                    "unit": "kgCO2"
                })
            
            # Create response
            response = {
                "success": True,
                "deviceId": device_id,
                "predictions": formatted_predictions,
                "modelInfo": {
                    "version": "1.0.0",
                    "type": f"ARIMA{self.order}",
                    "trainedOn": self.updated_at.isoformat()
                },
                "metadata": {
                    "factorsConsidered": ["历史数据趋势", "自回归", "移动平均"],
                    "anomalyDetected": False
                }
            }
            
            return response
        
        except Exception as e:
            logger.exception(f"Error formatting predictions: {str(e)}")
            return {
                "success": False,
                "error": "Formatting Error",
                "message": str(e)
            }