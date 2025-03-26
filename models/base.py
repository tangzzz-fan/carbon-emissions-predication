#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Model Module

This module defines the base class for all prediction models.
"""

import os
import joblib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all prediction models"""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the model
        
        Args:
            name: Name of the model
        """
        self.name = name or self.__class__.__name__
        self.model = None
        self.trained = False
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.metadata = {}
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model
        
        Args:
            X: Features
            y: Target
            **kwargs: Additional arguments for training
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """Make predictions
        
        Args:
            X: Features
            **kwargs: Additional arguments for prediction
            
        Returns:
            Predictions
        """
        pass
    
    def evaluate(self, X, y_true, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate the model
        
        Args:
            X: Features
            y_true: True target values
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric names and values
        """
        from utils.metrics import calculate_metrics
        
        # Check if model is trained
        if not self.trained:
            logger.warning(f"Model {self.name} has not been trained yet")
            return {}
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        results = calculate_metrics(y_true, y_pred, metrics)
        
        logger.info(f"Evaluation results for {self.name}: {results}")
        
        return results
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the model to disk
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        # Check if model is trained
        if not self.trained:
            raise ValueError(f"Model {self.name} has not been trained yet")
        
        # Use default path if not provided
        if path is None:
            # Create directory if it doesn't exist
            os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
            path = os.path.join(settings.MODEL_SAVE_PATH, f"{self.name}.joblib")
        
        # Update metadata
        self.updated_at = datetime.now()
        self.metadata["saved_at"] = self.updated_at.isoformat()
        
        # Save the model
        joblib.dump(self, path)
        
        logger.info(f"Model {self.name} saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str):
        """Load a model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        try:
            # Load the model
            model = joblib.load(path)
            
            logger.info(f"Model loaded from {path}")
            
            return model
        except Exception as e:
            logger.exception(f"Error loading model from {path}: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"{self.name} (trained: {self.trained})"
    
    def __repr__(self) -> str:
        """Representation of the model"""
        return self.__str__()