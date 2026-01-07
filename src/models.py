"""
模型定义模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseLSTM(nn.Module):
    """基线LSTM模型"""
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=1, dropout=0.2):
        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        output = self.fc2(x)
        output = torch.softmax(output, dim=1)
        return output


class ImprovedBiLSTM(nn.Module):
    """改进的双分支BiLSTM模型"""
    def __init__(self, embedding_matrix, num_features, hidden_size=64,
                 output_size=7, num_layers=1, dropout=0.2, 
                 use_embeddings=True):
        super().__init__()
        
        vocab_size, emb_dim = embedding_matrix.shape
        self.use_embeddings = use_embeddings
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.word_embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.word_embedding.weight.requires_grad = True
        
        # 历史编码器
        input_dim = (emb_dim if use_embeddings else 0) + num_features
        self.hist_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 当前词编码器
        curr_input_dim = (emb_dim if use_embeddings else 0) + num_features
        self.curr_encoder = nn.Sequential(
            nn.Linear(curr_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat):
        # 历史编码
        if self.use_embeddings:
            hist_word_emb = self.word_embedding(hist_word_ids)
            hist_input = torch.cat([hist_word_emb, hist_num_feat], dim=-1)
        else:
            hist_input = hist_num_feat
        
        lstm_out, _ = self.hist_encoder(hist_input)
        
        # Attention
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # 当前词编码
        if self.use_embeddings:
            curr_word_emb = self.word_embedding(curr_word_id)
            curr_input = torch.cat([curr_word_emb, curr_num_feat], dim=-1)
        else:
            curr_input = curr_num_feat
        
        curr_encoded = self.curr_encoder(curr_input)
        
        # 融合预测
        combined = torch.cat([context, curr_encoded], dim=-1)
        output = self.fusion_layer(combined)
        output = torch.softmax(output, dim=1)
        
        return output, attn_weights


class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, embedding_matrix, num_features, d_model=64, nhead=4, 
                 num_layers=2, output_size=7, dropout=0.2, use_embeddings=True):
        super().__init__()
        
        vocab_size, emb_dim = embedding_matrix.shape
        self.use_embeddings = use_embeddings
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.word_embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.word_embedding.weight.requires_grad = True
        
        # 输入投影
        input_dim = (emb_dim if use_embeddings else 0) + num_features
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                        num_layers=num_layers)
        
        # 当前词编码器
        curr_input_dim = (emb_dim if use_embeddings else 0) + num_features
        self.curr_encoder = nn.Sequential(
            nn.Linear(curr_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat):
        # 历史编码
        if self.use_embeddings:
            hist_word_emb = self.word_embedding(hist_word_ids)
            hist_input = torch.cat([hist_word_emb, hist_num_feat], dim=-1)
        else:
            hist_input = hist_num_feat
        
        hist_input = self.input_projection(hist_input)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(hist_input)
        context = transformer_out[:, -1, :]
        
        # 当前词编码
        if self.use_embeddings:
            curr_word_emb = self.word_embedding(curr_word_id)
            curr_input = torch.cat([curr_word_emb, curr_num_feat], dim=-1)
        else:
            curr_input = curr_num_feat
        
        curr_encoded = self.curr_encoder(curr_input)
        
        # 融合预测
        combined = torch.cat([context, curr_encoded], dim=-1)
        output = self.fusion_layer(combined)
        output = torch.softmax(output, dim=1)
        
        return output


class ModelFactory:
    """模型工厂类"""
    
    def __init__(self, config):
        self.config = config
        
    def create_model(self, model_type, **kwargs):
        """创建模型"""
        if model_type == "BaseLSTM":
            return BaseLSTM(
                input_size=len(self.config['data']['features']['numeric']) + 
                         len(self.config['data']['features']['categorical']),
                hidden_size=self.config['model']['hidden_size'],
                output_size=len(self.config['data']['features']['target']),
                **kwargs
            )
        elif model_type == "ImprovedBiLSTM":
            # TODO: 需要embedding_matrix
            embedding_matrix = None  # 从外部传入
            return ImprovedBiLSTM(
                embedding_matrix=embedding_matrix,
                num_features=len(self.config['data']['features']['numeric']) + 
                            len(self.config['data']['features']['categorical']),
                hidden_size=self.config['model']['hidden_size'],
                output_size=len(self.config['data']['features']['target']),
                **kwargs
            )
        elif model_type == "Transformer":
            # TODO: 需要embedding_matrix
            embedding_matrix = None  # 从外部传入
            return TransformerModel(
                embedding_matrix=embedding_matrix,
                num_features=len(self.config['data']['features']['numeric']) + 
                            len(self.config['data']['features']['categorical']),
                d_model=self.config['model']['hidden_size'],
                output_size=len(self.config['data']['features']['target']),
                **kwargs
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def load_model(self, model_path):
        """加载预训练模型"""
        # TODO: 根据模型类型加载
        pass
