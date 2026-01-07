"""
训练模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] 
                                  if torch.cuda.is_available() else 'cpu')
        
    def train(self, model, train_loader, val_loader):
        """训练模型"""
        print(f"使用设备: {self.device}")
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.config['training']['learning_rate'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(self.config['training']['num_epochs']):
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # 根据数据格式处理batch
                if len(batch) == 5:  # 改进模型
                    hist_word, hist_num, curr_word, curr_num, targets = batch
                    hist_word = hist_word.to(self.device)
                    hist_num = hist_num.to(self.device)
                    curr_word = curr_word.to(self.device)
                    curr_num = curr_num.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    if isinstance(model, ImprovedBiLSTM):
                        outputs, _ = model(hist_word, hist_num, curr_word, curr_num)
                    else:
                        outputs = model(hist_word, hist_num, curr_word, curr_num)
                else:  # 基线模型
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_loss = self._evaluate(model, val_loader, criterion)
            val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["training"]["num_epochs"]}], '
                     f'Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}')
            
            # 早停和保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(model, "best_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['patience']:
                print(f'早停在 epoch {epoch+1}')
                break
        
        # 保存训练历史
        self._save_training_history(train_losses, val_losses)
        
        print(f'✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}')
        
        return best_val_loss, train_losses, val_losses
    
    def _evaluate(self, model, val_loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_model(self, model, filename):
        """保存模型"""
        model_path = os.path.join(self.config['paths']['models'], filename)
        torch.save(model.state_dict(), model_path)
    
    def _save_training_history(self, train_losses, val_losses):
        """保存训练历史"""
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        history_path = os.path.join(self.config['paths']['logs'], 'training_history.npy')
        np.save(history_path, history)
