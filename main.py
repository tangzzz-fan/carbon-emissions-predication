#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Carbon Prediction Module - Main Entry Point

This module provides API endpoints for carbon emission prediction.
"""

import os
import logging
import threading
import json
import pika
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import settings
from models.base import BaseModel as PredictionBaseModel
from models.time_series import ProphetModel, ARIMAModel
from models.neural_network import LSTMModel
from utils.metrics import calculate_metrics

# Setup logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Available models
MODEL_REGISTRY = {
    "prophet": ProphetModel,
    "arima": ARIMAModel,
    "lstm": LSTMModel
}

# 全局变量来保存队列消费线程和连接
queue_consumer_thread = None
rabbitmq_connection = None
rabbitmq_channel = None
should_continue_consuming = True

# Initialize FastAPI app
app = FastAPI(
    title="Carbon Prediction API",
    description="API for predicting carbon emissions in logistics parks",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class DataPoint(BaseModel):
    deviceId: str
    timestamp: str
    value: float
    type: str
    unit: str
    co2Equivalent: Optional[float] = None
    transformedValue: Optional[float] = None

class PredictionRequest(BaseModel):
    deviceId: str
    dataPoints: Optional[List[DataPoint]] = None
    startDate: Optional[str] = None
    horizon: Optional[int] = 24  # Default to 24 hour prediction
    modelType: Optional[str] = "arima"  # Default model

class PredictionResponse(BaseModel):
    success: bool
    deviceId: str
    predictions: Optional[List[Dict[str, Any]]] = None
    modelInfo: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None

# API key validation
def validate_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Carbon Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(validate_api_key)
):
    try:
        device_id = request.deviceId
        model_type = request.modelType.lower()
        
        # Validate model type
        if model_type not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
        # Initialize the model
        model_class = MODEL_REGISTRY[model_type]
        model = model_class(name=f"{model_type}_{device_id}")
        
        # Convert data to proper format for training
        if request.dataPoints:
            # Process historical data for training
            import pandas as pd
            
            data = pd.DataFrame([
                {
                    "timestamp": dp.timestamp,
                    "value": dp.co2Equivalent if dp.co2Equivalent is not None else dp.value
                } for dp in request.dataPoints
            ])
            
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.sort_values("timestamp")
            
            # Train the model
            if model_type == "prophet":
                train_data = data.rename(columns={"timestamp": "ds", "value": "y"})
                model.train(train_data)
            else:
                X = data["timestamp"].values
                y = data["value"].values
                model.train(X, y)
        
        # Generate start date if not provided
        start_date = datetime.now()
        if request.startDate:
            start_date = datetime.fromisoformat(request.startDate.replace('Z', '+00:00'))
        
        # Generate predictions
        predictions = model.predict(horizon=request.horizon)
        
        # Format the response
        response = model.format_predictions(
            predictions=predictions, 
            device_id=device_id, 
            start_date=start_date
        )
        
        return response
        
    except Exception as e:
        logger.exception(f"Prediction error: {str(e)}")
        return {
            "success": False,
            "deviceId": request.deviceId,
            "error": "Prediction Error",
            "message": str(e)
        }

@app.post("/api/v1/train")
async def train_model(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(validate_api_key)
):
    try:
        data = await request.json()
        device_id = data.get("deviceId")
        model_type = data.get("modelType", "arima").lower()
        historical_data = data.get("historicalData", [])
        
        if not device_id:
            raise HTTPException(status_code=400, detail="deviceId is required")
            
        if model_type not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
            
        if not historical_data:
            raise HTTPException(status_code=400, detail="No historical data provided")
        
        # Schedule training task in background
        background_tasks.add_task(
            train_model_task,
            device_id=device_id,
            model_type=model_type,
            historical_data=historical_data
        )
        
        return {
            "success": True,
            "message": f"Training {model_type} model for device {device_id} scheduled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Training request error: {str(e)}")
        return {
            "success": False,
            "error": "Training Request Error",
            "message": str(e)
        }

@app.get("/api/v1/models")
async def list_models(api_key: str = Depends(validate_api_key)):
    """List available prediction models"""
    return {
        "success": True,
        "models": list(MODEL_REGISTRY.keys()),
        "defaultModel": "arima"
    }

# Background task for model training
async def train_model_task(device_id: str, model_type: str, historical_data: List[Dict]):
    try:
        logger.info(f"Training {model_type} model for device {device_id}")
        
        # Initialize the model
        model_class = MODEL_REGISTRY[model_type]
        model = model_class(name=f"{model_type}_{device_id}")
        
        # Convert data to proper format
        import pandas as pd
        
        data = pd.DataFrame(historical_data)
        if "timestamp" not in data.columns:
            data["timestamp"] = data.get("ds", data.get("date", pd.to_datetime("now")))
            
        if "value" not in data.columns:
            data["value"] = data.get("y", data.get("co2Equivalent", 0))
            
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        
        # Train the model
        if model_type == "prophet":
            train_data = data.rename(columns={"timestamp": "ds", "value": "y"})
            model.train(train_data)
        else:
            X = data["timestamp"].values
            y = data["value"].values
            model.train(X, y)
        
        # Save the model
        model_path = os.path.join(settings.MODEL_DIR, f"{model_type}_{device_id}.joblib")
        model.save(model_path)
        
        logger.info(f"Model {model_type}_{device_id} trained and saved successfully")
        
    except Exception as e:
        logger.exception(f"Error training model: {str(e)}")

# 队列消费者相关函数
def load_model_for_queue(model_type: str = None):
    """加载预测模型"""
    try:
        model_type = model_type or settings.DEFAULT_MODEL
        if model_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown model type: {model_type}, using default: {settings.DEFAULT_MODEL}")
            model_type = settings.DEFAULT_MODEL
            
        model_class = MODEL_REGISTRY[model_type]
        model_path = os.path.join(settings.MODEL_DIR, f"{model_type}_model.joblib")
        
        if os.path.exists(model_path):
            logger.info(f"Loading existing {model_type} model from {model_path}")
            return model_class.load(model_path)
        else:
            logger.info(f"Creating new {model_type} model")
            return model_class(name=f"{model_type}_model")
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
        return None

def process_queue_message(data: Dict[str, Any]):
    """处理从队列接收的消息"""
    try:
        logger.info(f"Processing queue message: {json.dumps(data, indent=2)}")
        
        # 提取数据
        device_id = data.get('deviceId')
        data_type = data.get('type')
        value = data.get('value')
        timestamp = data.get('timestamp')
        
        if not all([device_id, timestamp, value is not None]):
            logger.warning(f"Missing required fields in message: {data}")
            return False
            
        # 加载模型
        model = load_model_for_queue()
        if not model:
            logger.error("Failed to load prediction model")
            return False
            
        # 执行预测
        # 这里的预测逻辑需要根据您的实际需求定制
        prediction_result = {
            'deviceId': device_id,
            'timestamp': timestamp,
            'originalValue': value,
            'predictedValue': value * settings.CO2_CONVERSION_FACTOR,
            'unit': 'kgCO2'
        }
        
        logger.info(f"Prediction result: {json.dumps(prediction_result, indent=2)}")
        
        # 这里可以添加将预测结果保存到数据库或通知其他系统的代码
        
        return True
    except Exception as e:
        logger.exception(f"Error processing queue message: {str(e)}")
        return False

def consume_queue():
    """消费RabbitMQ队列的函数，将在单独的线程中运行"""
    global rabbitmq_connection, rabbitmq_channel, should_continue_consuming
    
    logger.info("Queue consumer thread started")
    
    max_retries = 10
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        if not should_continue_consuming:
            logger.info("Queue consumer thread stopping (external request)")
            break
            
        try:
            # 建立连接
            logger.info(f"Connecting to RabbitMQ at {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}...")
            credentials = pika.PlainCredentials(
                settings.RABBITMQ_USER, 
                settings.RABBITMQ_PASSWORD
            )
            connection_params = pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=int(settings.RABBITMQ_PORT),
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            rabbitmq_connection = pika.BlockingConnection(connection_params)
            rabbitmq_channel = rabbitmq_connection.channel()
            
            # 声明队列
            rabbitmq_channel.queue_declare(queue=settings.RABBITMQ_PREDICTION_QUEUE, durable=True)
            
            # 限制每次只处理一条消息
            rabbitmq_channel.basic_qos(prefetch_count=1)
            
            # 定义回调函数
            def callback(ch, method, properties, body):
                if not should_continue_consuming:
                    logger.info("Rejecting message due to shutdown")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                    return
                    
                try:
                    # 解析消息
                    data = json.loads(body)
                    # 处理消息
                    success = process_queue_message(data)
                    
                    if success:
                        # 确认消息已处理
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    else:
                        # 拒绝消息，重新入队
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                except Exception as e:
                    logger.exception(f"Error in callback: {str(e)}")
                    # 出错时拒绝消息
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            # 设置消费者
            rabbitmq_channel.basic_consume(
                queue=settings.RABBITMQ_PREDICTION_QUEUE,
                on_message_callback=callback
            )
            
            # 开始消费
            logger.info(f"Starting to consume from queue: {settings.RABBITMQ_PREDICTION_QUEUE}")
            
            # 此方法会阻塞线程，直到通道关闭
            while should_continue_consuming:
                try:
                    rabbitmq_connection.process_data_events(time_limit=1)  # 定期检查should_continue_consuming标志
                except pika.exceptions.AMQPError:
                    break
                    
            logger.info("Stopping queue consumption")
            
            # 清理连接
            if rabbitmq_channel and rabbitmq_channel.is_open:
                rabbitmq_channel.close()
            if rabbitmq_connection and rabbitmq_connection.is_open:
                rabbitmq_connection.close()
                
            # 如果是外部请求停止，则退出循环
            if not should_continue_consuming:
                break
                
        except pika.exceptions.AMQPConnectionError as e:
            if attempt < max_retries - 1 and should_continue_consuming:
                logger.warning(f"RabbitMQ connection failed. Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts: {str(e)}")
                break
        except Exception as e:
            logger.exception(f"Unexpected error in queue consumer: {str(e)}")
            if should_continue_consuming and attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                break
    
    logger.info("Queue consumer thread exited")

# FastAPI事件处理
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    global queue_consumer_thread, should_continue_consuming
    
    # 确保目录存在
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 启动队列消费线程
    logger.info("Starting queue consumer thread")
    should_continue_consuming = True
    queue_consumer_thread = threading.Thread(target=consume_queue, daemon=True)
    queue_consumer_thread.start()
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    global queue_consumer_thread, rabbitmq_connection, rabbitmq_channel, should_continue_consuming
    
    # 停止队列消费
    logger.info("Shutting down queue consumer")
    should_continue_consuming = False
    
    # 关闭RabbitMQ连接
    if rabbitmq_channel and rabbitmq_channel.is_open:
        try:
            rabbitmq_channel.close()
        except Exception as e:
            logger.error(f"Error closing RabbitMQ channel: {str(e)}")
    
    if rabbitmq_connection and rabbitmq_connection.is_open:
        try:
            rabbitmq_connection.close()
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {str(e)}")
    
    # 等待队列消费线程结束
    if queue_consumer_thread and queue_consumer_thread.is_alive():
        logger.info("Waiting for queue consumer thread to exit...")
        queue_consumer_thread.join(timeout=5)
        if queue_consumer_thread.is_alive():
            logger.warning("Queue consumer thread did not exit in time")
    
    logger.info("Application shutdown complete")

# Run the application
if __name__ == "__main__":
    import uvicorn
    import time
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )