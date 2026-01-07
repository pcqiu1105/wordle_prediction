"""
特征工程模块
"""

import pandas as pd
import numpy as np
import wordfreq
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config):
        self.config = config
        self.numeric_features = config['data']['features']['numeric']
        self.categorical_features = config['data']['features']['categorical']
        self.target_columns = config['data']['features']['target']
        
    def create_features(self, df):
        """创建特征"""
        print("\n" + "="*60)
        print("特征工程")
        print("="*60)
        
        # 词频特征
        df['word_freq_zipf'] = df['word'].str.lower().apply(
            lambda x: wordfreq.zipf_frequency(x, 'en'))
        
        # 结构特征
        df['unique_letter_ratio'] = df['word'].apply(
            lambda x: len(set(x.lower())) / len(x))
        df['has_repeated_letters'] = (df['unique_letter_ratio'] < 1.0).astype(int)
        
        # 元音个数
        def get_vowel_count(s):
            vowels = "aeiou"
            return sum(1 for c in s if c in vowels)
        df['vowel_count'] = df['word'].apply(lambda x: get_vowel_count(x.lower()))
        
        # 稀有字母个数
        def get_rare_letter_count(s):
            rare_letters = {'j', 'q', 'x', 'z'}
            return sum(1 for c in s if c in rare_letters)
        df['rare_letter_count'] = df['word'].apply(
            lambda x: get_rare_letter_count(x.lower()))
        
        # 常见字母开头
        common_starts = {'s', 'c', 'b', 't', 'p', 'a'}
        df['common_start'] = df['word'].str.lower().str[0].isin(common_starts).astype(int)
        
        # 时间特征
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        df['month'] = df['date'].dt.month
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # 游戏阶段
        phase_bins = [
            pd.Timestamp('2022-01-01'),
            pd.Timestamp('2022-04-01'),
            pd.Timestamp('2022-10-01'),
            pd.Timestamp('2022-12-31')
        ]
        df['seasonal_phase'] = pd.cut(df['date'], bins=phase_bins, 
                                     labels=[0, 1, 2]).astype(int)
        
        # 困难模式特征
        df['hard_mode_ratio'] = df['hard_mode_players'] / df['total_reported_results']
        df['hard_ratio_rolling'] = df['hard_mode_ratio'].rolling(
            window=7, min_periods=1).mean()
        df['hard_ratio_trend'] = df['hard_mode_ratio'] - df['hard_ratio_rolling']
        
        # 目标变量归一化
        df[self.target_columns] = df[self.target_columns].div(
            df[self.target_columns].sum(axis=1), axis=0)
        
        print(f"✅ 特征工程完成，总特征数: {len(df.columns)}")
        
        return df
    
    def split_data(self, df):
        """分割数据集"""
        print("\n" + "="*60)
        print("数据集分割")
        print("="*60)
        
        train_ratio, val_ratio = 0.7, 0.15
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        
        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size:train_size+val_size].copy()
        df_test = df.iloc[train_size+val_size:].copy()
        
        # 标准化
        df_train_std, df_val_std, df_test_std = self._standardize_features(
            df_train, df_val, df_test)
        
        print(f"训练集: {len(df_train)} 样本")
        print(f"验证集: {len(df_val)} 样本")
        print(f"测试集: {len(df_test)} 样本")
        
        return (df_train_std, df_val_std, df_test_std)
    
    def _standardize_features(self, df_train, df_val, df_test):
        """标准化特征"""
        scaler = StandardScaler()
        scaler.fit(df_train[self.numeric_features])
        
        df_train_std = df_train.copy()
        df_val_std = df_val.copy()
        df_test_std = df_test.copy()
        
        df_train_std[self.numeric_features] = scaler.transform(
            df_train[self.numeric_features])
        df_val_std[self.numeric_features] = scaler.transform(
            df_val[self.numeric_features])
        df_test_std[self.numeric_features] = scaler.transform(
            df_test[self.numeric_features])
        
        return df_train_std, df_val_std, df_test_std
    
    def create_dataloaders(self, train_data, val_data, test_data):
        """创建数据加载器"""
        print("\n" + "="*60)
        print("创建数据加载器")
        print("="*60)
        
        # TODO: 实现序列构建和数据加载器创建
        # 这里需要根据具体的数据格式实现
        
        # 示例代码结构
        train_loader = None
        val_loader = None
        test_loader = None
        
        return train_loader, val_loader, test_loader
