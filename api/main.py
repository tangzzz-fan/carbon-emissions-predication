#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Main Module

This module initializes the FastAPI application and sets up routes.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import prediction
from core.config import settings

# Setup logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Carbon Prediction API",
    description="API for carbon emission prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/predict", tags=["prediction"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Carbon Prediction API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}