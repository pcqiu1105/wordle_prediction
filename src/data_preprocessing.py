"""
数据预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, config):
        self.config = config
        
    def load_and_preprocess(self):
        """加载和预处理数据"""
        print("="*60)
        print("数据加载与预处理")
        print("="*60)
        
        # 读取数据
        df = pd.read_excel(self.config['data']['raw_data_path'], 
                          index_col=False, 
                          skiprows=1, 
                          usecols='B:M')
        
        # 重命名列
        df = df.rename(columns={
            'Date': 'date',
            'Contest number': 'contest_number', 
            'Word': 'word',
            'Number of  reported results': 'total_reported_results',
            'Number in hard mode': 'hard_mode_players',
            '1 try': 'attempts_1',
            '2 tries': 'attempts_2', 
            '3 tries': 'attempts_3',
            '4 tries': 'attempts_4',
            '5 tries': 'attempts_5',
            '6 tries': 'attempts_6',
            '7 or more tries (X)': 'attempts_7_or_more'
        })
        
        # 清洗词
        df['word'] = df['word'].apply(lambda x: str(x).strip().lower())
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"数据形状: {df.shape}")
        print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
        
        return df
