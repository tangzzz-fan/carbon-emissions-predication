#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module

This module implements visualization functions for prediction results.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# Setup logger
logger = logging.getLogger(__name__)


def plot_predictions(y_true, y_pred, timestamps=None, title="Prediction Results", save_path=None):
    """Plot actual vs predicted values
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        timestamps: Timestamps for x-axis (optional)
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use timestamps if provided, otherwise use indices
        x = timestamps if timestamps is not None else np.arange(len(y_true))
        
        # Plot actual values
        ax.plot(x, y_true, 'b-', label='Actual')
        
        # Plot predicted values
        ax.plot(x, y_pred, 'r--', label='Predicted')
        
        # Add labels and title
        ax.set_xlabel('Time' if timestamps is not None else 'Index')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if using timestamps
        if timestamps is not None:
            plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is not None:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    except Exception as e:
        logger.exception(f"Error plotting predictions: {str(e)}")
        return None


def plot_forecast(forecast_df, historical_df=None, timestamp_col='ds', value_col='yhat', 
                 ci_upper_col='yhat_upper', ci_lower_col='yhat_lower', 
                 title="Forecast Results", save_path=None):
    """Plot forecast results with confidence intervals
    
    Args:
        forecast_df: DataFrame with forecast results
        historical_df: DataFrame with historical data (optional)
        timestamp_col: Column name for timestamps
        value_col: Column name for forecast values
        ci_upper_col: Column name for upper confidence interval
        ci_lower_col: Column name for lower confidence interval
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot forecast
        ax.plot(forecast_df[timestamp_col], forecast_df[value_col], 'r-', label='Forecast')
        
        # Plot confidence intervals if available
        if ci_upper_col in forecast_df.columns and ci_lower_col in forecast_df.columns:
            ax.fill_between(forecast_df[timestamp_col], 
                           forecast_df[ci_lower_col], 
                           forecast_df[ci_upper_col], 
                           color='r', alpha=0.2, label='95% Confidence Interval')
        
        # Plot historical data if provided
        if historical_df is not None:
            if 'y' in historical_df.columns:
                historical_value_col = 'y'
            elif value_col in historical_df.columns:
                historical_value_col = value_col
            else:
                historical_value_col = historical_df.columns[1]  # Assume second column is value
            
            ax.plot(historical_df[timestamp_col], historical_df[historical_value_col], 
                    'b-', label='Historical')
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is not None:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    except Exception as e:
        logger.exception(f"Error plotting forecast: {str(e)}")
        return None


def create_interactive_plot(y_true, y_pred, timestamps=None, title="Interactive Prediction Results"):
    """Create interactive plot with Plotly
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        timestamps: Timestamps for x-axis (optional)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        # Use timestamps if provided, otherwise use indices
        x = timestamps if timestamps is not None else np.arange(len(y_true))
        
        # Create figure
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=x,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=x,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time' if timestamps is not None else 'Index',
            yaxis_title='Value',
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        logger.exception(f"Error creating interactive plot: {str(e)}")
        return None