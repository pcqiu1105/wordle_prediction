# Wordle玩家表现分布预测项目

## 项目简介
基于双分支BiLSTM的Wordle玩家表现分布预测模型研究

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明
```bash
# 完整流程训练与评估
python main.py --mode train_eval

# 仅训练
python main.py --mode train

# 仅评估
python main.py --mode eval

# 可视化
python main.py --mode visualize
```

## 项目结构
wordle-prediction/
├── README.md                      # 项目说明文档
├── requirements.txt               # 依赖包列表
├── main.py                        # 主运行脚本（入口点）
├── config.yaml                    # 配置文件
├── data/
│   ├── raw/                       # 原始数据
│   │   └── raw_data.xlsx
│   ├── processed/                 # 处理后数据
│   └── embeddings/                # 词嵌入文件（如果需要）
├── src/                           # 源代码
│   ├── __init__.py
│   ├── data_preprocessing.py      # 数据预处理模块
│   ├── feature_engineering.py     # 特征工程模块
│   ├── models.py                  # 模型定义模块
│   ├── train.py                   # 训练模块
│   ├── evaluate.py                # 评估模块
│   ├── visualize.py               # 可视化模块
│   └── utils.py                   # 工具函数模块
├── notebooks/                     # Jupyter Notebook（可选）
│   └── exploration.ipynb          # 探索性分析
├── outputs/                       # 输出结果
│   ├── models/                    # 保存的模型权重
│   ├── figures/                   # 生成的图表
│   └── logs/                      # 训练日志
└── tests/                         # 单元测试
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_utils.py