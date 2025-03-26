#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Queue Consumer Module

This module implements a consumer for the RabbitMQ prediction queue.
"""

import json
import logging
import pika
import pandas as pd
import numpy as np
import sys
import os
import time
from typing import Dict, Any

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.config import settings
from models.time_series import ProphetModel, ARIMAModel
from models.neural_network import LSTMModel
from utils.metrics import calculate_metrics

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 可用模型注册表
MODEL_REGISTRY = {
    "prophet": ProphetModel,
    "arima": ARIMAModel,
    "lstm": LSTMModel
}

def load_model(model_type: str = None):
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

def process_message(data: Dict[str, Any]):
    """处理从队列接收的消息"""
    try:
        logger.info(f"Processing message: {json.dumps(data, indent=2)}")
        
        # 提取数据
        device_id = data.get("deviceId")
        data_type = data.get("type")
        value = data.get("value")
        timestamp = data.get("timestamp")
        
        if not all([device_id, data_type, value, timestamp]):
            logger.error("Missing required fields in message")
            return False
            
        # 加载模型
        model = load_model()
        if not model:
            logger.error("Failed to load prediction model")
            return False
            
        # 创建输入数据
        # 这里简化处理，实际应用中可能需要获取历史数据
        df = pd.DataFrame({
            'timestamp': [pd.to_datetime(timestamp)],
            'value': [float(value)]
        })
        df.set_index('timestamp', inplace=True)
        
        # 执行预测
        prediction_result = model.predict(df)
        logger.info(f"Prediction result: {json.dumps(prediction_result, indent=2)}")
        
        # 这里可以添加将预测结果存储到数据库或发送回应用的代码
        
        return True
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        return False

def start_consumer():
    """启动RabbitMQ消费者"""
    # 连接重试参数
    max_retries = 5
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
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
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            
            # 声明队列
            channel.queue_declare(queue=settings.RABBITMQ_PREDICTION_QUEUE, durable=True)
            
            # 限制每次只处理一条消息
            channel.basic_qos(prefetch_count=1)
            
            # 定义回调函数
            def callback(ch, method, properties, body):
                try:
                    # 解析消息
                    data = json.loads(body)
                    # 处理消息
                    success = process_message(data)
                    
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
            channel.basic_consume(
                queue=settings.RABBITMQ_PREDICTION_QUEUE,
                on_message_callback=callback
            )
            
            # 开始消费
            logger.info(f"Starting to consume from queue: {settings.RABBITMQ_PREDICTION_QUEUE}")
            logger.info("Press CTRL+C to exit")
            channel.start_consuming()
            
        except pika.exceptions.AMQPConnectionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"RabbitMQ connection failed. Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts: {str(e)}")
                break
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down...")
            if 'connection' in locals() and connection.is_open:
                connection.close()
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            if 'connection' in locals() and connection.is_open:
                connection.close()
            break

if __name__ == "__main__":
    start_consumer() 