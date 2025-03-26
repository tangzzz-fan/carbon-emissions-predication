#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Module

This module implements various metrics for model evaluation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logger
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """Calculate evaluation metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        metrics: List of metrics to calculate
            If None, calculate all available metrics
            
    Returns:
        Dictionary of metric names and values
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Default metrics
    available_metrics = {
        'mse': mean_squared_error,
        'rmse': lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mape': mean_absolute_percentage_error,
        'smape': symmetric_mean_absolute_percentage_error
    }
    
    # Use all metrics if not specified
    if metrics is None:
        metrics = list(available_metrics.keys())
    
    # Calculate specified metrics
    results = {}
    for metric in metrics:
        if metric.lower() in available_metrics:
            try:
                results[metric] = float(available_metrics[metric.lower()](y_true, y_pred))
            except Exception as e:
                logger.warning(f"Error calculating {metric}: {str(e)}")
                results[metric] = float('nan')
        else:
            logger.warning(f"Metric '{metric}' not available")
    
    return results


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        MAPE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    
    if not np.any(mask):
        return float('nan')
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        SMAPE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    
    if not np.any(mask):
        return float('nan')
    
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100