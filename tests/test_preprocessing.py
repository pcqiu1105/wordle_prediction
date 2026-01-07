"""
数据预处理模块的单元测试
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """测试数据预处理类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = {
            'data': {
                'raw_data_path': 'test_data.xlsx',
                'features': {
                    'numeric': [],
                    'categorical': [],
                    'target': []
                }
            }
        }
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=10),
            'Contest number': range(100, 110),
            'Word': ['apple', 'brave', 'crane', 'drain', 'elbow',
                    'flute', 'grape', 'house', 'image', 'jolly'],
            'Number of  reported results': [10000] * 10,
            'Number in hard mode': [2000] * 10,
            '1 try': [1.0, 0.5, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2, 0.1, 0.5],
            '2 tries': [2.0, 1.5, 1.3, 1.2, 1.1, 1.5, 1.3, 1.2, 1.1, 1.5],
            '3 tries': [3.0, 2.5, 2.3, 2.2, 2.1, 2.5, 2.3, 2.2, 2.1, 2.5],
            '4 tries': [4.0, 3.5, 3.3, 3.2, 3.1, 3.5, 3.3, 3.2, 3.1, 3.5],
            '5 tries': [5.0, 4.5, 4.3, 4.2, 4.1, 4.5, 4.3, 4.2, 4.1, 4.5],
            '6 tries': [6.0, 5.5, 5.3, 5.2, 5.1, 5.5, 5.3, 5.2, 5.1, 5.5],
            '7 or more tries (X)': [7.0, 6.5, 6.3, 6.2, 6.1, 6.5, 6.3, 6.2, 6.1, 6.5]
        })
        
        # 保存测试数据到临时文件
        self.test_file = 'test_data_temp.xlsx'
        self.test_data.to_excel(self.test_file, index=False)
        self.config['data']['raw_data_path'] = self.test_file
        
        self.preprocessor = DataPreprocessor(self.config)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_and_preprocess_returns_dataframe(self):
        """测试load_and_preprocess返回DataFrame"""
        df = self.preprocessor.load_and_preprocess()
        
        # 检查返回类型
        self.assertIsInstance(df, pd.DataFrame)
        
        # 检查数据形状
        self.assertEqual(df.shape[0], 10)  # 10行
        self.assertEqual(df.shape[1], 13)  # 原始12列 + 重命名后的列
    
    def test_column_renaming(self):
        """测试列名重命名"""
        df = self.preprocessor.load_and_preprocess()
        
        # 检查新列名是否存在
        expected_columns = ['date', 'contest_number', 'word', 'total_reported_results',
                          'hard_mode_players', 'attempts_1', 'attempts_2', 'attempts_3',
                          'attempts_4', 'attempts_5', 'attempts_6', 'attempts_7_or_more']
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_word_cleaning(self):
        """测试单词清洗"""
        # 创建有问题的数据
        test_data = pd.DataFrame({
            'Date': ['2022-01-01'],
            'Contest number': [100],
            'Word': ['  APPLE  '],  # 有空格和大写
            'Number of  reported results': [10000],
            'Number in hard mode': [2000],
            '1 try': [1.0],
            '2 tries': [2.0],
            '3 tries': [3.0],
            '4 tries': [4.0],
            '5 tries': [5.0],
            '6 tries': [6.0],
            '7 or more tries (X)': [7.0]
        })
        
        test_file = 'test_word_cleaning.xlsx'
        test_data.to_excel(test_file, index=False)
        
        config = self.config.copy()
        config['data']['raw_data_path'] = test_file
        preprocessor = DataPreprocessor(config)
        
        df = preprocessor.load_and_preprocess()
        
        # 检查清洗结果
        self.assertEqual(df['word'].iloc[0], 'apple')
        
        # 清理
        os.remove(test_file)
    
    def test_date_conversion(self):
        """测试日期转换"""
        df = self.preprocessor.load_and_preprocess()
        
        # 检查date列是否为datetime类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))
        
        # 检查日期范围
        self.assertEqual(df['date'].min(), pd.Timestamp('2022-01-01'))
        self.assertEqual(df['date'].max(), pd.Timestamp('2022-01-10'))
    
    def test_data_integrity(self):
        """测试数据完整性"""
        df = self.preprocessor.load_and_preprocess()
        
        # 检查是否有NaN值
        self.assertFalse(df.isnull().any().any())
        
        # 检查数值列的数据类型
        numeric_cols = ['total_reported_results', 'hard_mode_players', 
                       'attempts_1', 'attempts_2', 'attempts_3', 'attempts_4',
                       'attempts_5', 'attempts_6', 'attempts_7_or_more']
        
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))
    
    def test_negative_values_handling(self):
        """测试负值处理"""
        # 创建包含负值的数据
        test_data = self.test_data.copy()
        test_data['1 try'] = [-1.0] * 10
        
        test_file = 'test_negative.xlsx'
        test_data.to_excel(test_file, index=False)
        
        config = self.config.copy()
        config['data']['raw_data_path'] = test_file
        preprocessor = DataPreprocessor(config)
        
        df = preprocessor.load_and_preprocess()
        
        # 检查负值仍然存在（预处理不修改数值）
        self.assertEqual(df['attempts_1'].iloc[0], -1.0)
        
        # 清理
        os.remove(test_file)
    
    def test_empty_data(self):
        """测试空数据处理"""
        # 创建空数据
        empty_data = pd.DataFrame(columns=self.test_data.columns)
        
        test_file = 'test_empty.xlsx'
        empty_data.to_excel(test_file, index=False)
        
        config = self.config.copy()
        config['data']['raw_data_path'] = test_file
        preprocessor = DataPreprocessor(config)
        
        df = preprocessor.load_and_preprocess()
        
        # 检查返回空DataFrame
        self.assertEqual(len(df), 0)
        
        # 清理
        os.remove(test_file)
    
    def test_duplicate_dates(self):
        """测试重复日期处理"""
        # 创建有重复日期的数据
        test_data = pd.DataFrame({
            'Date': ['2022-01-01', '2022-01-01', '2022-01-02'],
            'Contest number': [100, 101, 102],
            'Word': ['apple', 'brave', 'crane'],
            'Number of  reported results': [10000, 11000, 12000],
            'Number in hard mode': [2000, 2100, 2200],
            '1 try': [1.0, 1.1, 1.2],
            '2 tries': [2.0, 2.1, 2.2],
            '3 tries': [3.0, 3.1, 3.2],
            '4 tries': [4.0, 4.1, 4.2],
            '5 tries': [5.0, 5.1, 5.2],
            '6 tries': [6.0, 6.1, 6.2],
            '7 or more tries (X)': [7.0, 7.1, 7.2]
        })
        
        test_file = 'test_duplicate.xlsx'
        test_data.to_excel(test_file, index=False)
        
        config = self.config.copy()
        config['data']['raw_data_path'] = test_file
        preprocessor = DataPreprocessor(config)
        
        df = preprocessor.load_and_preprocess()
        
        # 检查重复日期被保留
        self.assertEqual(len(df[df['date'] == '2022-01-01']), 2)
        
        # 清理
        os.remove(test_file)


class TestFeatureEngineering(unittest.TestCase):
    """测试特征工程类"""
    
    def setUp(self):
        """测试前准备"""
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from feature_engineering import FeatureEngineer
        
        # 创建模拟配置
        self.config = {
            'data': {
                'features': {
                    'numeric': ['word_freq_zipf', 'unique_letter_ratio'],
                    'categorical': ['has_repeated_letters'],
                    'target': ['attempts_1', 'attempts_2']
                }
            }
        }
        
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=5),
            'word': ['apple', 'brave', 'crane', 'drain', 'elbow'],
            'hard_mode_players': [2000, 2100, 2200, 2300, 2400],
            'total_reported_results': [10000, 11000, 12000, 13000, 14000],
            'attempts_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'attempts_2': [0.2, 0.3, 0.4, 0.5, 0.6]
        })
        
        self.engineer = FeatureEngineer(self.config)
    
    def test_create_features_returns_dataframe(self):
        """测试create_features返回DataFrame"""
        df_with_features = self.engineer.create_features(self.test_df.copy())
        
        self.assertIsInstance(df_with_features, pd.DataFrame)
        
        # 检查是否添加了新特征
        self.assertIn('word_freq_zipf', df_with_features.columns)
        self.assertIn('unique_letter_ratio', df_with_features.columns)
        self.assertIn('has_repeated_letters', df_with_features.columns)
    
    def test_word_frequency_feature(self):
        """测试词频特征"""
        df_with_features = self.engineer.create_features(self.test_df.copy())
        
        # 检查词频特征存在且为数值
        self.assertTrue('word_freq_zipf' in df_with_features.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_with_features['word_freq_zipf']))
    
    def test_unique_letter_ratio(self):
        """测试唯一字母比例特征"""
        df_with_features = self.engineer.create_features(self.test_df.copy())
        
        # 检查特征值范围
        self.assertTrue(all(0 <= df_with_features['unique_letter_ratio'] <= 1))
        
        # 检查特定单词
        test_word = 'apple'  # 有重复字母p
        mask = df_with_features['word'] == test_word
        ratio = df_with_features.loc[mask, 'unique_letter_ratio'].iloc[0]
        self.assertAlmostEqual(ratio, 4/5)  # 4个唯一字母 / 5个总字母
    
    def test_repeated_letters_flag(self):
        """测试重复字母标志"""
        df_with_features = self.engineer.create_features(self.test_df.copy())
        
        # 检查特征值
        self.assertTrue(all(df_with_features['has_repeated_letters'].isin([0, 1])))
        
        # 'apple'有重复字母，应该为1
        mask = df_with_features['word'] == 'apple'
        self.assertEqual(df_with_features.loc[mask, 'has_repeated_letters'].iloc[0], 1)
    
    def test_target_normalization(self):
        """测试目标变量归一化"""
        df_with_features = self.engineer.create_features(self.test_df.copy())
        
        # 检查目标变量是否归一化（和为1）
        for idx, row in df_with_features.iterrows():
            target_sum = row['attempts_1'] + row['attempts_2']
            self.assertAlmostEqual(target_sum, 1.0, places=6)
    
    def test_split_data_returns_three_dataframes(self):
        """测试split_data返回三个DataFrame"""
        train, val, test = self.engineer.split_data(self.test_df)
        
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        
        # 检查没有数据泄漏
        self.assertTrue(len(train) + len(val) + len(test) == len(self.test_df))
    
    def test_standardization(self):
        """测试特征标准化"""
        from feature_engineering import FeatureEngineer
        
        # 修改配置以包含更多数值特征
        config = self.config.copy()
        config['data']['features']['numeric'] = ['word_freq_zipf']
        
        engineer = FeatureEngineer(config)
        
        # 添加测试特征
        test_df = self.test_df.copy()
        test_df['word_freq_zipf'] = [5.0, 4.0, 3.0, 2.0, 1.0]
        
        # 执行标准化
        train_df = test_df.iloc[:3].copy()
        val_df = test_df.iloc[3:4].copy()
        test_df_subset = test_df.iloc[4:].copy()
        
        train_std, val_std, test_std = engineer._standardize_features(
            train_df, val_df, test_df_subset)
        
        # 检查训练集均值为0
        self.assertAlmostEqual(train_std['word_freq_zipf'].mean(), 0, places=6)
        
        # 检查训练集标准差为1
        self.assertAlmostEqual(train_std['word_freq_zipf'].std(), 1, places=6)


if __name__ == '__main__':
    unittest.main()
