#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction Endpoints Module

This module implements the prediction API endpoints.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from models.time_series import ProphetModel, ARIMAModel
from models.neural_network import LSTMModel
from data.loader import DataLoader
from data.preprocessing import DataPreprocessor
from data.features import FeatureExtractor
from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize data loader
data_loader = DataLoader(api_base_url=settings.DATA_COLLECTION_API_URL)

# Initialize data preprocessor
data_preprocessor = DataPreprocessor()

# Initialize feature extractor
feature_extractor = FeatureExtractor()


# Define request and response models
class PredictionRequest(BaseModel):
    """Model for prediction request"""
    id: Optional[str] = None
    deviceId: str
    timestamp: str
    value: float
    type: str
    co2Equivalent: Optional[float] = None
    unit: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "deviceId": "energy-meter-001",
                "timestamp": "2023-05-10T08:30:00.000Z",
                "value": 42.5,
                "type": "power_consumption",
                "co2Equivalent": 21.25,
                "unit": "kgCO2"
            }
        }


class PredictionItem(BaseModel):
    """Model for a single prediction"""
    timestamp: str
    predictedValue: float
    confidence: float
    unit: str


class ModelInfo(BaseModel):
    """Model for model information"""
    version: str
    type: str
    trainedOn: str


class PredictionMetadata(BaseModel):
    """Model for prediction metadata"""
    factorsConsidered: List[str]
    anomalyDetected: bool


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    success: bool
    deviceId: str
    predictions: List[PredictionItem]
    modelInfo: ModelInfo
    metadata: PredictionMetadata
    error: Optional[str] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Model for error response"""
    success: bool = False
    error: str
    message: str
    code: Optional[str] = None


# Helper functions
def get_model(model_type: str = None):
    """Get prediction model based on type
    
    Args:
        model_type: Type of model to use
        
    Returns:
        Prediction model
    """
    # Use default model if not specified
    if model_type is None:
        model_type = settings.DEFAULT_MODEL
    
    # Select model based on type
    if model_type.lower() == 'prophet':
        model = ProphetModel()
    elif model_type.lower() == 'arima':
        model = ARIMAModel()
    elif model_type.lower() == 'lstm':
        model = LSTMModel()
    else:
        # Default to Prophet
        model = ProphetModel()
    
    return model


def calculate_co2_equivalent(value: float) -> float:
    """Calculate CO2 equivalent from energy consumption
    
    Args:
        value: Energy consumption value in kWh
        
    Returns:
        CO2 equivalent in kgCO2
    """
    return value * settings.CO2_CONVERSION_FACTOR


# Define API endpoints
@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model_type: Optional[str] = None):
    """Make predictions based on input data
    
    Args:
        request: Prediction request data
        model_type: Type of model to use
        
    Returns:
        Prediction response
    """
    try:
        logger.info(f"Received prediction request for device {request.deviceId}")
        
        # Calculate CO2 equivalent if not provided
        if request.co2Equivalent is None and request.type == 'power_consumption':
            request.co2Equivalent = calculate_co2_equivalent(request.value)
        
        # Get historical data for the device
        historical_data = data_loader.load_historical_data(
            device_id=request.deviceId,
            data_type=request.type,
            hours=24  # Get last 24 hours of data
        )
        
        # Check if we have enough historical data
        if len(historical_data) < 2:
            logger.warning(f"Not enough historical data for device {request.deviceId}")
            # Return error response
            return {
                "success": False,
                "deviceId": request.deviceId,
                "predictions": [],
                "modelInfo": {
                    "version": "1.0.0",
                    "type": "None",
                    "trainedOn": datetime.now().isoformat()
                },
                "metadata": {
                    "factorsConsidered": [],
                    "anomalyDetected": False
                },
                "error": "Insufficient Data",
                "message": "Not enough historical data for prediction"
            }
        
        # Clean and preprocess data
        cleaned_data = data_preprocessor.clean_data(historical_data)
        
        # Add time features
        data_with_time_features = data_preprocessor.add_time_features(cleaned_data)
        
        # Create time series features
        if 'value' in data_with_time_features.columns:
            data_with_ts_features = feature_extractor.create_time_series_features(
                data_with_time_features, 'value'
            )
        else:
            # Use co2Equivalent if value is not available
            data_with_ts_features = feature_extractor.create_time_series_features(
                data_with_time_features, 'co2Equivalent'
            )
        
        # Get prediction model
        model = get_model(model_type)
        
        # Prepare data for Prophet model
        if isinstance(model, ProphetModel):
            # Prophet requires 'ds' and 'y' columns
            train_data = data_with_ts_features.copy()
            train_data['ds'] = train_data['timestamp']
            
            # Use co2Equivalent as target if available, otherwise use value
            if 'co2Equivalent' in train_data.columns:
                train_data['y'] = train_data['co2Equivalent']
            else:
                train_data['y'] = train_data['value']
            
            # Train the model
            model.train(train_data)
            
            # Make predictions for the next 24 hours
            predictions = model.predict(periods=24, freq='H')
            
            # Format predictions
            response = model.format_predictions(predictions, request.deviceId)
        
        # Prepare data for ARIMA model
        elif isinstance(model, ARIMAModel):
            # ARIMA requires a time series
            train_data = data_with_ts_features.copy()
            
            # Use co2Equivalent as target if available, otherwise use value
            if 'co2Equivalent' in train_data.columns:
                target = train_data['co2Equivalent']
            else:
                target = train_data['value']
            
            # Train the model
            model.train(target)
            
            # Make predictions for the next 24 hours
            predictions = model.predict(steps=24)
            
            # Format predictions
            response = model.format_predictions(
                predictions, 
                request.deviceId, 
                start_date=datetime.now()
            )
        
        # Prepare data for LSTM model
        elif isinstance(model, LSTMModel):
            # LSTM requires a time series with features
            train_data = data_with_ts_features.copy()
            
            # Use co2Equivalent as target if available, otherwise use value
            if 'co2Equivalent' in train_data.columns:
                target_col = 'co2Equivalent'
            else:
                target_col = 'value'
            
            # Select features
            feature_cols = [col for col in train_data.columns 
                           if col not in ['timestamp', 'ds', 'id', 'deviceId', 'type', 'unit']]
            
            # Train the model
            model.train(train_data[feature_cols], train_data[target_col])
            
            # Make predictions for the next 24 hours
            predictions = model.predict(train_data[feature_cols], steps=24)
            
            # Format predictions
            response = model.format_predictions(
                predictions, 
                request.deviceId, 
                start_date=datetime.now()
            )
        
        else:
            # This should not happen, but just in case
            raise ValueError(f"Unsupported model type: {type(model).__name__}")
        
        logger.info(f"Generated predictions for device {request.deviceId}")
        
        return response
    
    except Exception as e:
        logger.exception(f"Error generating predictions: {str(e)}")
        
        # Return error response
        return {
            "success": False,
            "deviceId": request.deviceId,
            "predictions": [],
            "modelInfo": {
                "version": "1.0.0",
                "type": "None",
                "trainedOn": datetime.now().isoformat()
            },
            "metadata": {
                "factorsConsidered": [],
                "anomalyDetected": False
            },
            "error": "Prediction Error",
            "message": str(e)
        }


@router.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available prediction models
    
    Returns:
        List of available model types
    """
    return ["prophet", "arima", "lstm"]


@router.post("/batch", response_model=Dict[str, PredictionResponse])
async def batch_predict(requests: List[PredictionRequest], background_tasks: BackgroundTasks, model_type: Optional[str] = None):
    """Make batch predictions for multiple devices
    
    Args:
        requests: List of prediction requests
        background_tasks: Background tasks
        model_type: Type of model to use
        
    Returns:
        Dictionary mapping device IDs to prediction responses
    """
    try:
        logger.info(f"Received batch prediction request for {len(requests)} devices")
        
        # Process each request
        results = {}
        for request in requests:
            # Make prediction
            result = await predict(request, model_type)
            results[request.deviceId] = result
        
        return results
    
    except Exception as e:
        logger.exception(f"Error processing batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))