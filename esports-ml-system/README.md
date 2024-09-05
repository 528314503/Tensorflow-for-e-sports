# 电竞赛事智能推荐与定价系统

本项目是一个基于深度学习的电竞赛事推荐和定价系统。它使用TensorFlow框架实现了分布式训练、模型优化和推理服务。

## 功能特点

- 数据处理和特征工程
- 深度学习推荐模型
- 分布式训练
- TensorRT模型优化
- RESTful API for 推荐和定价

## 使用方法

1. 数据处理:
```python
from src.data_processing import DataProcessor

processor = DataProcessor()
data = processor.load_data(['data/raw/user_data.csv', 'data/raw/event_data.csv'])
processed_data = processor.preprocess(data)