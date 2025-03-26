#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Network Models Module

This module implements neural network prediction models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from models.base import BaseModel
from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM neural network model for time series prediction"""
    
    def __init__(self, name: Optional[str] = None, units: int = 50, layers: int = 2, dropout: float = 0.2,
                 sequence_length: int = 24, batch_size: int = 32, epochs: int = 50):
        """Initialize the LSTM model
        
        Args:
            name: Name of the model
            units: Number of LSTM units per layer
            layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Length of input sequences
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        super().__init__(name=name or "LSTMModel")
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=self.units, return_sequences=(self.layers > 1),
                      input_shape=input_shape))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())
        
        # Middle LSTM layers
        for i in range(self.layers - 2):
            model.add(LSTM(units=self.units, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(BatchNormalization())
        
        # Last LSTM layer (if more than one layer)
        if self.layers > 1:
            model.add(LSTM(units=self.units))
            model.add(Dropout(self.dropout))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (X, y) with input sequences and targets
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y=None, validation_split: float = 0.2, **kwargs):
        """Train the LSTM model
        
        Args:
            X: DataFrame or array with features
               If DataFrame, it should contain a target column
            y: Series or array with target values (optional)
            validation_split: Fraction of data to use for validation
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Trained model
        """
        try:
            # Prepare training data
            if isinstance(X, pd.DataFrame):
                # If y is provided, use it as the target
                if y is not None:
                    train_data = y.values.reshape(-1, 1)
                # Otherwise, try to find a target column
                elif 'value' in X.columns:
                    train_data = X['value'].values.reshape(-1, 1)
                elif 'y' in X.columns:
                    train_data = X['y'].values.reshape(-1, 1)
                else:
                    raise ValueError("Training data must contain a target column or y must be provided")
                
                # Add additional features if available
                if len(X.columns) > 1:
                    feature_cols = [col for col in X.columns if col not in ['value', 'y', 'timestamp', 'ds']]
                    if feature_cols:
                        additional_features = X[feature_cols].values
                        train_data = np.hstack((train_data, additional_features))
            else:
                # Use X as the target
                train_data = X.reshape(-1, 1) if len(X.shape) == 1 else X
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(train_data)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(scaled_data)
            
            # Reshape y if it's multivariate (we only predict the first column)
            if len(y_seq.shape) > 2:
                y_seq = y_seq[:, 0].reshape(-1, 1)
            
            # Build model if not already built
            if self.model is None:
                input_shape = (X_seq.shape[1], X_seq.shape[2])
                self.model = self._build_model(input_shape)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=f"{settings.MODEL_SAVE_PATH}/lstm_checkpoint.h5",
                               save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                **kwargs
            )
            
            # Update model state
            self.trained = True
            self.updated_at = datetime.now()
            
            # Update metadata
            self.metadata["train_shape"] = train_data.shape
            self.metadata["sequence_length"] = self.sequence_length
            self.metadata["training_history"] = {
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "epochs_trained": len(history.history['loss'])
            }
            
            logger.info(f"Trained {self.name} on {len(train_data)} samples")
            
            return self
        
        except Exception as e:
            logger.exception(f"Error predicting with {self.name}: {str(e)}")
            raise
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the model to disk
        
        Args:
            filepath: Path to save the model (optional)
                     If None, use the default path from settings
                     
        Returns:
            Path where the model was saved
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Use provided filepath or default
            if filepath is None:
                # Create directory if it doesn't exist
                import os
                os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"{settings.MODEL_SAVE_PATH}/{self.name}_{timestamp}.h5"
            
            # Save Keras model
            self.model.save(filepath)
            
            # Save scaler and metadata
            import pickle
            scaler_path = filepath.replace(".h5", "_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata_path = filepath.replace(".h5", "_metadata.json")
            with open(metadata_path, 'w') as f:
                # Convert numpy types to Python native types for JSON serialization
                metadata_json = {}
                for k, v in self.metadata.items():
                    if isinstance(v, dict):
                        metadata_json[k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                metadata_json[k][kk] = vv.tolist()
                            elif isinstance(vv, np.integer):
                                metadata_json[k][kk] = int(vv)
                            elif isinstance(vv, np.floating):
                                metadata_json[k][kk] = float(vv)
                            else:
                                metadata_json[k][kk] = vv
                    elif isinstance(v, np.ndarray):
                        metadata_json[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        metadata_json[k] = int(v)
                    elif isinstance(v, np.floating):
                        metadata_json[k] = float(v)
                    else:
                        metadata_json[k] = v
                
                json.dump(metadata_json, f, indent=2)
            
            # Update model state
            self.metadata["saved_path"] = filepath
            self.metadata["saved_at"] = datetime.now().isoformat()
            
            logger.info(f"Saved {self.name} to {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.exception(f"Error saving {self.name}: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """Load a model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            # Create a new instance
            model = cls()
            
            # Load Keras model
            model.model = load_model(filepath)
            
            # Load scaler
            import pickle
            scaler_path = filepath.replace(".h5", "_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                model.scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = filepath.replace(".h5", "_metadata.json")
            with open(metadata_path, 'r') as f:
                model.metadata = json.load(f)
            
            # Set model state
            model.trained = True
            if "sequence_length" in model.metadata:
                model.sequence_length = model.metadata["sequence_length"]
            
            logger.info(f"Loaded model from {filepath}")
            
            return model
        
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            raise
    
    def evaluate(self, X, y=None, metrics: List[str] = ['mse', 'mae', 'rmse', 'mape']) -> Dict[str, float]:
        """Evaluate the model on test data
        
        Args:
            X: Test data features
            y: Test data target (optional)
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Get target from X if y is not provided
            if y is None and isinstance(X, pd.DataFrame):
                if 'value' in X.columns:
                    y = X['value']
                elif 'y' in X.columns:
                    y = X['y']
                else:
                    raise ValueError("Target column not found in test data")
            
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            results = {}
            
            if 'mse' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_squared_error
                results['mse'] = float(mean_squared_error(y, y_pred))
            
            if 'mae' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_absolute_error
                results['mae'] = float(mean_absolute_error(y, y_pred))
            
            if 'rmse' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_squared_error
                results['rmse'] = float(np.sqrt(mean_squared_error(y, y_pred)))
            
            if 'mape' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_absolute_percentage_error
                results['mape'] = float(mean_absolute_percentage_error(y, y_pred))
            
            if 'r2' in metrics or 'all' in metrics:
                from sklearn.metrics import r2_score
                results['r2'] = float(r2_score(y, y_pred))
            
            # Update metadata
            self.metadata["evaluation"] = results
            self.metadata["evaluated_at"] = datetime.now().isoformat()
            
            logger.info(f"Evaluated {self.name}: {results}")
            
            return results
        
        except Exception as e:
            logger.exception(f"Error evaluating {self.name}: {str(e)}")
            raise
            logger.exception(f"Error training {self.name}: {str(e)}")
            raise
    
    def predict(self, X=None, steps: int = 24, **kwargs):
        """Make predictions with the LSTM model
        
        Args:
            X: Input data for prediction
               If None, the last sequence from training will be used
            steps: Number of steps to forecast
            **kwargs: Additional arguments
            
        Returns:
            Array with predictions
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Prepare input data
            if X is None:
                # Use the last sequence from training data
                raise ValueError("Input data must be provided for prediction")
            
            # Convert to numpy array if DataFrame or Series
            if isinstance(X, pd.DataFrame):
                # Try to find a target column
                if 'value' in X.columns:
                    input_data = X['value'].values.reshape(-1, 1)
                elif 'y' in X.columns:
                    input_data = X['y'].values.reshape(-1, 1)
                else:
                    # Use all columns
                    input_data = X.values
            elif isinstance(X, pd.Series):
                input_data = X.values.reshape(-1, 1)
            else:
                input_data = X.reshape(-1, 1) if len(X.shape) == 1 else X
            
            # Scale the data
            scaled_data = self.scaler.transform(input_data)
            
            # Make predictions
            predictions = []
            current_batch = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, scaled_data.shape[1])
            
            for i in range(steps):
                # Predict next value
                next_pred = self.model.predict(current_batch)[0]
                
                # Append to predictions
                predictions.append(next_pred)
                
                # Update current batch for next prediction
                next_input = np.zeros((1, 1, scaled_data.shape[1]))
                next_input[0, 0, 0] = next_pred
                
                # Shift the batch and add the new prediction
                current_batch = np.append(current_batch[:, 1:, :], next_input, axis=1)
            
            # Convert predictions to numpy array
            predictions = np.array(predictions).reshape(-1, 1)
            
            # Inverse transform to get original scale
            predictions_rescaled = self.scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), scaled_data.shape[1]-1))]))
            
            # Extract the target column predictions
            final_predictions = predictions_rescaled[:, 0]
            
            logger.info(f"Made predictions with {self.name} for {steps} steps")
            
            return final_predictions
        
        except Exception as e:
            logger.exception(f"Error predicting with {self.name}: {str(e)}")
            raise
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the model to disk
        
        Args:
            filepath: Path to save the model (optional)
                     If None, use the default path from settings
                     
        Returns:
            Path where the model was saved
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Use provided filepath or default
            if filepath is None:
                # Create directory if it doesn't exist
                import os
                os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"{settings.MODEL_SAVE_PATH}/{self.name}_{timestamp}.h5"
            
            # Save Keras model
            self.model.save(filepath)
            
            # Save scaler and metadata
            import pickle
            scaler_path = filepath.replace(".h5", "_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata_path = filepath.replace(".h5", "_metadata.json")
            with open(metadata_path, 'w') as f:
                # Convert numpy types to Python native types for JSON serialization
                metadata_json = {}
                for k, v in self.metadata.items():
                    if isinstance(v, dict):
                        metadata_json[k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                metadata_json[k][kk] = vv.tolist()
                            elif isinstance(vv, np.integer):
                                metadata_json[k][kk] = int(vv)
                            elif isinstance(vv, np.floating):
                                metadata_json[k][kk] = float(vv)
                            else:
                                metadata_json[k][kk] = vv
                    elif isinstance(v, np.ndarray):
                        metadata_json[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        metadata_json[k] = int(v)
                    elif isinstance(v, np.floating):
                        metadata_json[k] = float(v)
                    else:
                        metadata_json[k] = v
                
                json.dump(metadata_json, f, indent=2)
            
            # Update model state
            self.metadata["saved_path"] = filepath
            self.metadata["saved_at"] = datetime.now().isoformat()
            
            logger.info(f"Saved {self.name} to {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.exception(f"Error saving {self.name}: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """Load a model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            # Create a new instance
            model = cls()
            
            # Load Keras model
            model.model = load_model(filepath)
            
            # Load scaler
            import pickle
            scaler_path = filepath.replace(".h5", "_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                model.scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = filepath.replace(".h5", "_metadata.json")
            with open(metadata_path, 'r') as f:
                model.metadata = json.load(f)
            
            # Set model state
            model.trained = True
            if "sequence_length" in model.metadata:
                model.sequence_length = model.metadata["sequence_length"]
            
            logger.info(f"Loaded model from {filepath}")
            
            return model
        
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            raise
    
    def evaluate(self, X, y=None, metrics: List[str] = ['mse', 'mae', 'rmse', 'mape']) -> Dict[str, float]:
        """Evaluate the model on test data
        
        Args:
            X: Test data features
            y: Test data target (optional)
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Check if model is trained
            if not self.trained:
                raise ValueError(f"Model {self.name} has not been trained yet")
            
            # Get target from X if y is not provided
            if y is None and isinstance(X, pd.DataFrame):
                if 'value' in X.columns:
                    y = X['value']
                elif 'y' in X.columns:
                    y = X['y']
                else:
                    raise ValueError("Target column not found in test data")
            
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            results = {}
            
            if 'mse' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_squared_error
                results['mse'] = float(mean_squared_error(y, y_pred))
            
            if 'mae' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_absolute_error
                results['mae'] = float(mean_absolute_error(y, y_pred))
            
            if 'rmse' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_squared_error
                results['rmse'] = float(np.sqrt(mean_squared_error(y, y_pred)))
            
            if 'mape' in metrics or 'all' in metrics:
                from sklearn.metrics import mean_absolute_percentage_error
                results['mape'] = float(mean_absolute_percentage_error(y, y_pred))
            
            if 'r2' in metrics or 'all' in metrics:
                from sklearn.metrics import r2_score
                results['r2'] = float(r2_score(y, y_pred))
            
            # Update metadata
            self.metadata["evaluation"] = results
            self.metadata["evaluated_at"] = datetime.now().isoformat()
            
            logger.info(f"Evaluated {self.name}: {results}")
            
            return results
        
        except Exception as e:
            logger.exception(f"Error evaluating {self.name}: {str(e)}")
            raise