# Python预测模块实时数据获取指南

## 实时数据流向和访问方法

在当前系统架构中，Python预测模块可以通过以下方式获取实时数据：

1. HTTP API接收实时数据

系统会通过HTTP POST请求将电力消耗(power_consumption)类型的数据直接发送到预测服务：

```python
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import json

app = FastAPI()

# 加载预训练模型
model = joblib.load("carbon_prediction_model.pkl")

@app.post("/predict")
async def predict(data: dict):
    try:
        # 记录接收到的数据
        print(f"Received data: {json.dumps(data, indent=2)}")
        
        # 提取必要特征进行预测
        features = extract_features(data)
        
        # 使用模型进行预测
        prediction = model.predict([features])[0]
        
        # 构建响应
        response = {
            "success": True,
            "deviceId": data.get("deviceId"),
            "predictions": [
                {
                    "timestamp": data.get("timestamp"),
                    "predictedValue": float(prediction),
                    "confidence": 0.95,
                    "unit": "kgCO2"
                }
            ],
            "modelInfo": {
                "version": "1.0",
                "type": "RandomForest" 
            }
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_features(data):
    # 从数据中提取预测所需特征
    features = [
        data.get("value", 0),
        # 可以添加其他特征，如时间特征
        hour_of_day(data.get("timestamp")),
        day_of_week(data.get("timestamp")),
        # ...其他特征
    ]
    return features
```

### 2. 消息队列接收实时数据
对于其他类型的数据，预测模块需要从消息队列获取：
```python
import pika
import json
import pandas as pd
import joblib

# 加载预训练模型
model = joblib.load("carbon_prediction_model.pkl")

# 连接到RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 声明预测队列
channel.queue_declare(queue='data_prediction_queue')

def callback(ch, method, properties, body):
    try:
        # 解析消息内容
        data = json.loads(body)
        print(f"Received from queue: {json.dumps(data, indent=2)}")
        
        # 提取特征并预测
        features = extract_features(data)
        prediction = model.predict([features])[0]
        
        # 处理预测结果
        store_prediction_result(data, prediction)
        
        # 确认消息已处理
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message: {e}")
        # 根据需要决定是否重新入队
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# 开始消费消息
channel.basic_consume(queue='data_prediction_queue', on_message_callback=callback)
print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
```

### 3. 获取历史数据进行预测
对于需要历史数据的预测模型（如LSTM、时间序列模型），可以通过以下方式获取：
```python
import requests
import pandas as pd

def get_historical_data(device_id, data_type, hours=24):
    """获取指定设备的历史数据"""
    # 调用系统API获取历史数据
    response = requests.get(
        f"http://localhost:3000/data-collection/historical-data",
        params={
            "deviceId": device_id,
            "type": data_type,
            "hours": hours
        }
    )
    
    if response.status_code == 200:
        # 将返回的JSON数据转换为DataFrame
        data = response.json()
        df = pd.DataFrame(data)
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    else:
        print(f"Error fetching historical data: {response.text}")
        return None

def predict_with_history(current_data):
    # 获取该设备的历史数据
    device_id = current_data.get("deviceId")
    data_type = current_data.get("type")
    
    history = get_historical_data(device_id, data_type)
    
    if history is not None:
        # 准备模型输入
        X = prepare_time_series_features(history, current_data)
        
        # 加载时间序列模型
        lstm_model = joblib.load("lstm_prediction_model.pkl")
        
        # 预测
        prediction = lstm_model.predict(X)
        return prediction
    else:
        # 无法获取历史数据时的备选方案
        return fallback_prediction(current_data)
```


## Mock-IoT模拟数据流向
Mock-IoT生成的模拟时间序列数据的完整流向：
1. 数据生成：
    TimeSeriesGeneratorService、ScenarioGeneratorService等生成模拟数据
    例如，使用generateCarbonEmissionTimeSeries方法生成碳排放时间序列
2. 数据发送至采集服务：
    通过dataCollectionService.create(deviceDataDto)将数据保存到数据库
    数据被发送到处理队列
3. 数据处理：
    DataProcessingService消费队列数据
    数据经过清洗、转换
    根据类型决定是通过HTTP API还是预测队列发送到预测模块
4. 预测处理：
    Python预测服务接收数据并处理
    生成预测结果并返回
5. 预测结果流向：
    预测结果应被保存到数据库
    可能触发预警机制
    可通过WebSocket推送到前端展示

### 整合到系统的建议
1. 创建历史数据API：
```typescript
   @Get('historical-data')
   async getHistoricalData(
     @Query('deviceId') deviceId: string,
     @Query('type') type: string,
     @Query('hours') hours: number
   ) {
     return this.dataCollectionService.getHistoricalData(deviceId, type, hours);
   }
```
2. 设置回调机制：在Python预测服务中增加回调，将结果发送回NestJS服务：
```python
   def send_prediction_result(result):
       requests.post('http://localhost:3000/prediction/results', json=result)
```