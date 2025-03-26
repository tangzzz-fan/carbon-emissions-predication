#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging Module

This module sets up and manages application logging.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

from core.config import settings


def setup_logging():
    """Setup application logging"""
    
    # Create logs directory if it doesn't exist
    log_file = Path(settings.LOG_FILE)
    log_dir = log_file.parent
    os.makedirs(log_dir, exist_ok=True)
    
    # Get log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler with rotation
            RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=10485760,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
        ]
    )
    
    # Set log level for external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Log file: {settings.LOG_FILE}")
    
    return logger