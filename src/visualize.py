"""
可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


class Visualizer:
    """可视化类"""
    
    def __init__(self, config):
        self.config = config
        self.figure_path = config['paths']['figures']
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_model_comparison(self, results):
        """绘制模型对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 模型对比
        models = list(results.keys())
        mse_values = [results[model]['MSE'] for model in models]
        
        axes[0].bar(range(len(models)), mse_values, alpha=0.8)
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('模型性能对比')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(mse_values):
            axes[0].text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 提升百分比
        baseline_mse = results['Baseline LSTM']['MSE']
        improvement = [(baseline_mse - mse) / baseline_mse * 100 for mse in mse_values]
        
        axes[1].bar(range(len(models)), improvement, alpha=0.8)
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('相对于基线的提升 (%)')
        axes[1].set_title('相对提升')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        # 从保存的历史中加载数据
        history_path = os.path.join(self.config['paths']['logs'], 
                                   'training_history.npy')
        
        if os.path.exists(history_path):
            history = np.load(history_path, allow_pickle=True).item()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(history['train_losses'], label='训练损失', linewidth=2)
            ax.plot(history['val_losses'], label='验证损失', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('训练曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figure_path, 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_predictions(self, predictions, targets):
        """绘制预测散点图"""
        target_columns = self.config['data']['features']['target']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(target_columns):
            ax = axes[i]
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=20)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel(f'真实 {col}')
            ax.set_ylabel(f'预测 {col}')
            ax.set_title(f'{col}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, 'predictions_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ablation_study(self, results):
        """绘制消融实验图"""
        # 提取消融实验数据
        ablation_groups = [
            ('Improved BiLSTM', 'BiLSTM w/o Embeddings'),
            ('Transformer', 'Transformer w/o Embeddings')
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (with_emb, without_emb) in enumerate(ablation_groups):
            if with_emb in results and without_emb in results:
                mse_values = [results[with_emb]['MSE'], results[without_emb]['MSE']]
                labels = [f'有词嵌入\n({with_emb})', f'无词嵌入\n({without_emb})']
                
                axes[i].bar([0, 1], mse_values, alpha=0.8, 
                           color=['#4ECDC4', '#FF6B6B'])
                axes[i].set_xticks([0, 1])
                axes[i].set_xticklabels(labels)
                axes[i].set_ylabel('MSE')
                axes[i].set_title(f'{with_emb} 消融实验')
                axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, 'ablation_study.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_results(self):
        """加载保存的结果"""
        results_path = os.path.join(self.config['paths']['logs'], 
                                   'evaluation_results.csv')
        
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        return None"""
可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


class Visualizer:
    """可视化类"""
    
    def __init__(self, config):
        self.config = config
        self.figure_path = config['paths']['figures']
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_model_comparison(self, results):
        """绘制模型对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 模型对比
        models = list(results.keys())
        mse_values = [results[model]['MSE'] for model in models]
        
        axes[0].bar(range(len(models)), mse_values, alpha=0.8)
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('模型性能对比')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(mse_values):
            axes[0].text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 提升百分比
        baseline_mse = results['Baseline LSTM']['MSE']
        improvement = [(baseline_mse - mse) / baseline_mse * 100 for mse in mse_values]
        
        axes[1].bar(range(len(models)), improvement, alpha=0.8)
        axes[1].set_xticks
