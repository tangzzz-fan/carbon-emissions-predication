## Python预测模块架构设计文档

### 1. 概述

#### 1.1 项目目标
- 构建高精度的碳排放预测模型
- 提供灵活的模型训练和预测接口
- 支持多种预测方法和算法
- 与后端系统无缝集成

#### 1.2 技术选型
- 编程语言：Python 3.9+
- 框架：FastAPI（提供API服务）
- 机器学习库：Scikit-learn, TensorFlow/PyTorch, Prophet
- 数据处理：Pandas, NumPy
- 可视化：Matplotlib, Plotly
- 模型持久化：Joblib/Pickle

#### 1.3 架构原则
- **模块化设计**：各组件高内聚低耦合
- **可扩展性**：支持添加新的预测算法和模型
- **可测试性**：易于进行单元测试和集成测试
- **清晰的接口**：提供统一的预测接口
- **数据处理流水线**：标准化的数据处理流程

### 2. 目录结构

```
carbon_prediction/
├── api/                  # API服务
│   ├── __init__.py
│   ├── main.py           # FastAPI主应用
│   ├── endpoints/        # API端点
│   └── models/           # API请求/响应模型
├── core/                 # 核心模块
│   ├── __init__.py
│   ├── config.py         # 配置管理
│   └── logging.py        # 日志管理
├── data/                 # 数据处理
│   ├── __init__.py
│   ├── preprocessing.py  # 数据预处理
│   ├── features.py       # 特征工程
│   └── loader.py         # 数据加载
├── models/               # 预测模型
│   ├── __init__.py
│   ├── base.py           # 模型基类
│   ├── linear.py         # 线性模型
│   ├── time_series.py    # 时间序列模型
│   ├── neural_network.py # 神经网络模型
│   └── ensemble.py       # 集成模型
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── metrics.py        # 评估指标
│   └── visualization.py  # 可视化工具
├── tests/                # 测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── trained_models/       # 存储训练好的模型
├── .env                  # 环境变量
├── requirements.txt      # 依赖管理
└── main.py               # 入口文件
```

### 3. 核心模块设计

#### 3.1 数据处理模块（Data Module）

**功能**：
- 数据加载和解析
- 数据清洗和预处理
- 特征工程
- 数据集分割（训练集/测试集）

**主要组件**：
- DataLoader：从不同源加载数据
- DataPreprocessor：数据清洗和标准化
- FeatureExtractor：特征提取和转换
- DataSplitter：数据集分割

**示例代码**：
```python
# data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """清洗数据，处理缺失值和异常值"""
        # 处理缺失值
        df = df.fillna(method='ffill')  # 前向填充
        
        # 处理异常值（例如使用IQR方法）
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return df
    
    def scale_features(self, df, columns, fit=True):
        """标准化特征"""
        if fit:
            self.scaler.fit(df[columns])
        
        scaled_data = self.scaler.transform(df[columns])
        scaled_df = pd.DataFrame(scaled_data, columns=columns, index=df.index)
        
        # 替换原始特征
        for col in columns:
            df[col] = scaled_df[col]
            
        return df
```

#### 3.2 模型模块（Model Module）

**功能**：
- 定义预测模型基类
- 实现各种预测算法
- 模型训练与评估
- 模型持久化

**主要组件**：
- BaseModel：模型基类
- LinearModel：线性回归/时间序列模型
- NeuralNetworkModel：神经网络模型
- EnsembleModel：集成模型

**示例代码**：
```python
# models/base.py
import joblib
import os
from abc import ABC, abstractmethod
from datetime import datetime

class BaseModel(ABC):
    """预测模型基类"""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.model = None
        self.trained = False
        self.created_at = datetime.now()
    
    @abstractmethod
    def train(self, X, y):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """进行预测"""
        pass
    
    def evaluate(self, X, y_true, metrics=None):
        """评估模型"""
        from utils.metrics import calculate_metrics
        
        y_pred = self.predict(X)
        results = calculate_metrics(y_true, y_pred, metrics)
        return results
    
    def save(self, path):
        """保存模型"""
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        """加载模型"""
        return joblib.load(path)
```

```python
# models/time_series.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from models.base import BaseModel

class ARIMAModel(BaseModel):
    """ARIMA时间序列模型"""
    
    def __init__(self, order=(1,1,1)):
        super().__init__(name=f"ARIMA{order}")
        self.order = order
    
    def train(self, X, y):
        """训练ARIMA模型"""
        # 假设X包含日期索引
        data = pd.Series(y, index=pd.DatetimeIndex(X))
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        self.trained = True
        return self
    
    def predict(self, X):
        """进行预测"""
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        
        # 假设X是预测的步数或日期范围
        steps = len(X) if isinstance(X, (list, np.ndarray)) else X
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

class ProphetModel(BaseModel):
    """Facebook Prophet模型"""
    
    def __init__(self, params=None):
        super().__init__(name="Prophet")
        self.params = params or {}
        self.model = Prophet(**self.params)
    
    def train(self, X, y):
        """训练Prophet模型"""
        # Prophet要求特定的数据格式
        df = pd.DataFrame({'ds': X, 'y': y})
        self.model.fit(df)
        self.trained = True
        return self
    
    def predict(self, X):
        """进行预测"""
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        
        # X可以是日期列表或期间
        if isinstance(X, list):
            future = pd.DataFrame({'ds': X})
        else:
            future = self.model.make_future_dataframe(periods=X)
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values
```

#### 3.3 API服务模块（API Module）

**功能**：
- RESTful API接口提供
- 处理预测请求
- 返回预测结果
- 模型管理接口

**主要组件**：
- FastAPI应用实例
- 预测端点
- 模型管理端点
- 请求/响应模型

**示例代码**：
```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from models.model_manager import ModelManager
from data.preprocessing import DataPreprocessor

app = FastAPI(title="碳排放预测API", description="提供碳排放预测服务")
model_manager = ModelManager()
preprocessor = DataPreprocessor()

class PredictionRequest(BaseModel):
    model_name: str
    data: List[Dict[str, Any]]
    features: List[str]
    
class PredictionResponse(BaseModel):
    predictions: List[float]
    timestamp: datetime
    model_name: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 转换输入数据为DataFrame
        df = pd.DataFrame(request.data)
        
        # 预处理数据
        df = preprocessor.clean_data(df)
        df = preprocessor.scale_features(df, request.features, fit=False)
        
        # 获取模型并预测
        model = model_manager.get_model(request.model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        # 提取特征并预测
        X = df[request.features].values
        predictions = model.predict(X)
        
        return {
            "predictions": predictions.tolist(),
            "timestamp": datetime.now(),
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {"models": model_manager.list_models()}
```

#### 3.4 模型管理模块（Model Manager）

**功能**：
- 模型注册与管理
- 模型版本控制
- 模型选择与加载

**主要组件**：
- ModelManager：管理多个预测模型
- ModelRegistry：模型注册表
- ModelFactory：模型创建工厂

**示例代码**：
```python
# models/model_manager.py
import os
import glob
from typing import Dict, List, Optional, Type

from models.base import BaseModel
from models.linear import LinearRegressionModel
from models.time_series import ARIMAModel, ProphetModel
from models.neural_network import LSTMModel

class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir="trained_models"):
        self.models_dir = models_dir
        self.models: Dict[str, BaseModel] = {}
        self.model_types = {
            "linear": LinearRegressionModel,
            "arima": ARIMAModel,
            "prophet": ProphetModel,
            "lstm": LSTMModel
        }
        self._load_models()
    
    def _load_models(self):
        """加载已保存的模型"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            return
        
        model_files = glob.glob(os.path.join(self.models_dir, "*.joblib"))
        for file in model_files:
            try:
                model = BaseModel.load(file)
                self.models[model.name] = model
            except Exception as e:
                print(f"Error loading model {file}: {e}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """获取指定名称的模型"""
        return self.models.get(name)
    
    def register_model(self, model: BaseModel) -> bool:
        """注册新模型"""
        if model.name in self.models:
            return False
        
        self.models[model.name] = model
        model.save(os.path.join(self.models_dir, f"{model.name}.joblib"))
        return True
    
    def create_model(self, model_type: str, **kwargs) -> BaseModel:
        """创建新模型"""
        if model_type not in self.model_types:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_types[model_type]
        return model_class(**kwargs)
    
    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        return list(self.models.keys())
```

### 4. 集成与部署

#### 4.1 与NestJS后端集成

设计RPC接口，通过HTTP或消息队列与后端通信：

```python
# 集成示例 - HTTP接口
@app.post("/api/v1/predict")
async def predict_emissions(request: PredictionRequest):
    # 处理来自NestJS的预测请求
    # 验证请求数据
    # 调用相应模型进行预测
    # 返回预测结果
    pass
```

#### 4.2 Docker部署配置

提供Dockerfile简化部署：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 总结

以上是针对物流园区碳排放预测和管理系统的三个主要组件的架构设计。这些设计遵循了Clean Architecture和SOLID原则，确保了系统的可维护性、可扩展性和可测试性。

每个组件都有明确的职责划分：
- 前端负责用户交互和数据可视化
- 后端处理业务逻辑和数据存储
- Python预测模块专注于高精度的碳排放预测

通过这样的分层设计，系统各部分可以独立开发和测试，同时又能协同工作，为物流园区提供全面的碳排放监控和管理能力。
