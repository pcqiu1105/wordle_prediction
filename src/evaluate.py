"""
评估模块
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


class Evaluator:
    """评估器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] 
                                  if torch.cuda.is_available() else 'cpu')
        
    def evaluate(self, model, test_loader):
        """评估模型"""
        model.eval()
        criterion = nn.MSELoss()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 5:
                    hist_word, hist_num, curr_word, curr_num, targets = batch
                    hist_word = hist_word.to(self.device)
                    hist_num = hist_num.to(self.device)
                    curr_word = curr_word.to(self.device)
                    curr_num = curr_num.to(self.device)
                    targets = targets.to(self.device)
                    
                    if isinstance(model, ImprovedBiLSTM):
                        outputs, _ = model(hist_word, hist_num, curr_word, curr_num)
                    else:
                        outputs = model(hist_word, hist_num, curr_word, curr_num)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # 计算各项指标
        metrics = self._calculate_metrics(predictions, targets)
        
        return metrics
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        
        # 各目标变量的MAE
        target_columns = self.config['data']['features']['target']
        mae_per_target = {}
        for i, col in enumerate(target_columns):
            mae_per_target[col] = mean_absolute_error(targets[:, i], predictions[:, i])
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAE_per_target': mae_per_target
        }
        
        return metrics
    
    def save_results(self, metrics):
        """保存结果"""
        results_path = os.path.join(self.config['paths']['logs'], 'evaluation_results.csv')
        
        # 创建DataFrame
        results_df = pd.DataFrame([{
            'Metric': 'MSE',
            'Value': metrics['MSE']
        }, {
            'Metric': 'MAE',
            'Value': metrics['MAE']
        }, {
            'Metric': 'RMSE',
            'Value': metrics['RMSE']
        }])
        
        results_df.to_csv(results_path, index=False)
        
        # 保存各目标变量的MAE
        mae_df = pd.DataFrame(list(metrics['MAE_per_target'].items()),
                             columns=['Target', 'MAE'])
        mae_path = os.path.join(self.config['paths']['logs'], 'mae_per_target.csv')
        mae_df.to_csv(mae_path, index=False)
        
        print(f"✅ 结果已保存到 {results_path}")
