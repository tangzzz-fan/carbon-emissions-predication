#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration settings for the Carbon Prediction module.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    
    # API security
    API_KEY: str = Field(default="pred_api_bfd8c9a7e35f4de1b82a6c8d9f0")
    
    # Model settings
    MODEL_DIR: str = Field(default="trained_models")
    DEFAULT_MODEL: str = Field(default="arima")
    
    # Data settings
    DATA_DIR: str = Field(default="data")
    
    # 添加缺失的设置项
    DATA_COLLECTION_API_URL: str = Field(default="http://localhost:3000")
    RABBITMQ_HOST: str = Field(default="localhost")
    RABBITMQ_PORT: str = Field(default="5672")
    RABBITMQ_USER: str = Field(default="guest")
    RABBITMQ_PASSWORD: str = Field(default="guest")
    RABBITMQ_PREDICTION_QUEUE: str = Field(default="data_prediction_queue")
    LOG_FILE: str = Field(default="logs/carbon_prediction.log")
    MODEL_SAVE_PATH: str = Field(default="trained_models/")
    CO2_CONVERSION_FACTOR: float = Field(default=0.5)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        # 如果您想允许未定义的额外字段，取消下面这行的注释
        # "extra": "allow"
    }


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)

# 确保日志目录存在
log_dir = os.path.dirname(settings.LOG_FILE)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)