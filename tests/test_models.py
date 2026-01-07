"""
模型模块的单元测试
"""

import unittest
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import BaseLSTM, ImprovedBiLSTM, TransformerModel, ModelFactory


class TestBaseLSTM(unittest.TestCase):
    """测试BaseLSTM模型"""
    
    def setUp(self):
        """测试前准备"""
        self.input_size = 10
        self.hidden_size = 32
        self.output_size = 7
        self.batch_size = 8
        self.sequence_length = 7
        
        self.model = BaseLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=1,
            dropout=0.2
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        self.assertIsInstance(self.model.lstm, nn.LSTM)
        self.assertIsInstance(self.model.fc1, nn.Linear)
        self.assertIsInstance(self.model.fc2, nn.Linear)
        self.assertIsInstance(self.model.dropout, nn.Dropout)
    
    def test_forward_pass_shape(self):
        """测试前向传播输出形状"""
        # 创建模拟输入
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # 前向传播
        output = self.model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
    
    def test_forward_pass_values(self):
        """测试前向传播输出值范围"""
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        output = self.model(x)
        
        # 检查输出是否为概率分布（和为1）
        for i in range(self.batch_size):
            row_sum = output[i].sum().item()
            self.assertAlmostEqual(row_sum, 1.0, places=5)
        
        # 检查值在0-1之间
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_different_batch_sizes(self):
        """测试不同批次大小的处理"""
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, self.sequence_length, self.input_size)
            output = self.model(x)
            
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], self.output_size)
    
    def test_gradient_flow(self):
        """测试梯度流"""
        self.model.train()
        
        # 创建模拟数据
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        y = torch.randn(self.batch_size, self.output_size)
        
        # 归一化y使其成为概率分布
        y = torch.softmax(y, dim=1)
        
        # 计算损失
        criterion = nn.MSELoss()
        output = self.model(x)
        loss = criterion(output, y)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"参数 {name} 没有梯度")
    
    def test_model_parameters(self):
        """测试模型参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 计算期望的参数数量
        # LSTM参数
        lstm_params = 4 * ((self.input_size * self.hidden_size) + 
                          (self.hidden_size * self.hidden_size) + 
                          self.hidden_size)
        
        # 全连接层参数
        fc1_params = (self.hidden_size * (self.hidden_size // 2) + 
                     (self.hidden_size // 2))
        fc2_params = ((self.hidden_size // 2) * self.output_size + 
                     self.output_size)
        
        expected_params = lstm_params + fc1_params + fc2_params
        
        self.assertEqual(total_params, expected_params)
    
    def test_dropout_training_eval(self):
        """测试训练和评估模式下的Dropout行为"""
        # 训练模式
        self.model.train()
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # 多次前向传播，输出应该不同（由于dropout）
        outputs = []
        for _ in range(5):
            output = self.model(x)
            outputs.append(output)
        
        # 检查输出有差异
        differences = []
        for i in range(len(outputs)-1):
            diff = torch.mean(torch.abs(outputs[i] - outputs[i+1])).item()
            differences.append(diff)
        
        self.assertTrue(any(diff > 1e-6 for diff in differences))
        
        # 评估模式
        self.model.eval()
        outputs_eval = []
        for _ in range(5):
            output = self.model(x)
            outputs_eval.append(output)
        
        # 评估模式下输出应该相同（无dropout）
        for i in range(len(outputs_eval)-1):
            diff = torch.mean(torch.abs(outputs_eval[i] - outputs_eval[i+1])).item()
            self.assertAlmostEqual(diff, 0, places=6)


class TestImprovedBiLSTM(unittest.TestCase):
    """测试ImprovedBiLSTM模型"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟embedding矩阵
        vocab_size = 100
        emb_dim = 50
        self.embedding_matrix = np.random.randn(vocab_size, emb_dim)
        
        self.num_features = 8
        self.hidden_size = 32
        self.output_size = 7
        self.batch_size = 8
        self.sequence_length = 7
        
        # 创建两个版本的模型
        self.model_with_emb = ImprovedBiLSTM(
            embedding_matrix=self.embedding_matrix,
            num_features=self.num_features,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            use_embeddings=True
        )
        
        self.model_without_emb = ImprovedBiLSTM(
            embedding_matrix=self.embedding_matrix,
            num_features=self.num_features,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            use_embeddings=False
        )
    
    def test_model_components(self):
        """测试模型组件"""
        # 检查有embedding的模型
        self.assertIsInstance(self.model_with_emb.word_embedding, nn.Embedding)
        self.assertIsInstance(self.model_with_emb.hist_encoder, nn.LSTM)
        self.assertIsInstance(self.model_with_emb.attention, nn.Sequential)
        self.assertIsInstance(self.model_with_emb.curr_encoder, nn.Sequential)
        self.assertIsInstance(self.model_with_emb.fusion_layer, nn.Sequential)
        
        # 检查无embedding的模型
        self.assertIsInstance(self.model_without_emb.word_embedding, nn.Embedding)
    
    def test_forward_pass_with_embeddings(self):
        """测试带embedding的前向传播"""
        # 创建模拟输入
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        # 前向传播
        output, attn_weights = self.model_with_emb(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # 检查attention权重形状
        self.assertEqual(attn_weights.shape, 
                        (self.batch_size, self.sequence_length, 1))
        
        # 检查输出是否为概率分布
        for i in range(self.batch_size):
            row_sum = output[i].sum().item()
            self.assertAlmostEqual(row_sum, 1.0, places=5)
    
    def test_forward_pass_without_embeddings(self):
        """测试不带embedding的前向传播"""
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        output, attn_weights = self.model_without_emb(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        self.assertEqual(attn_weights.shape, 
                        (self.batch_size, self.sequence_length, 1))
    
    def test_attention_weights_normalization(self):
        """测试Attention权重归一化"""
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        output, attn_weights = self.model_with_emb(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        # 检查attention权重和为1
        for i in range(self.batch_size):
            weight_sum = attn_weights[i].sum().item()
            self.assertAlmostEqual(weight_sum, 1.0, places=5)
    
    def test_different_input_sizes(self):
        """测试不同输入大小"""
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            hist_word_ids = torch.randint(0, 100, (batch_size, self.sequence_length))
            hist_num_feat = torch.randn(batch_size, self.sequence_length, self.num_features)
            curr_word_id = torch.randint(0, 100, (batch_size,))
            curr_num_feat = torch.randn(batch_size, self.num_features)
            
            output, attn_weights = self.model_with_emb(
                hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
            
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(attn_weights.shape[0], batch_size)
    
    def test_embedding_matrix_initialization(self):
        """测试embedding矩阵初始化"""
        # 检查embedding权重是否被正确复制
        model_emb_weights = self.model_with_emb.word_embedding.weight.data.cpu().numpy()
        
        # 检查形状匹配
        self.assertEqual(model_emb_weights.shape, self.embedding_matrix.shape)
        
        # 检查值近似相等（允许微小差异）
        diff = np.mean(np.abs(model_emb_weights - self.embedding_matrix))
        self.assertLess(diff, 1e-6)
    
    def test_gradient_computation(self):
        """测试梯度计算"""
        self.model_with_emb.train()
        
        # 创建模拟数据
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        y = torch.randn(self.batch_size, self.output_size)
        y = torch.softmax(y, dim=1)
        
        # 计算损失
        criterion = nn.MSELoss()
        output, _ = self.model_with_emb(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        loss = criterion(output, y)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for name, param in self.model_with_emb.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"参数 {name} 没有梯度")
                self.assertFalse(torch.all(param.grad == 0), f"参数 {name} 梯度全为零")


class TestTransformerModel(unittest.TestCase):
    """测试TransformerModel"""
    
    def setUp(self):
        """测试前准备"""
        vocab_size = 100
        emb_dim = 50
        self.embedding_matrix = np.random.randn(vocab_size, emb_dim)
        
        self.num_features = 8
        self.d_model = 32
        self.output_size = 7
        self.batch_size = 8
        self.sequence_length = 7
        
        self.model = TransformerModel(
            embedding_matrix=self.embedding_matrix,
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            output_size=self.output_size,
            use_embeddings=True
        )
    
    def test_model_components(self):
        """测试模型组件"""
        self.assertIsInstance(self.model.word_embedding, nn.Embedding)
        self.assertIsInstance(self.model.input_projection, nn.Linear)
        self.assertIsInstance(self.model.transformer_encoder, nn.TransformerEncoder)
        self.assertIsInstance(self.model.curr_encoder, nn.Sequential)
        self.assertIsInstance(self.model.fusion_layer, nn.Sequential)
    
    def test_forward_pass(self):
        """测试前向传播"""
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        output = self.model(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # 检查概率分布
        for i in range(self.batch_size):
            row_sum = output[i].sum().item()
            self.assertAlmostEqual(row_sum, 1.0, places=5)
    
    def test_without_embeddings(self):
        """测试不带embedding的版本"""
        model_no_emb = TransformerModel(
            embedding_matrix=self.embedding_matrix,
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            output_size=self.output_size,
            use_embeddings=False
        )
        
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        output = model_no_emb(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
    
    def test_transformer_encoder_layers(self):
        """测试Transformer编码器层数"""
        self.assertEqual(len(self.model.transformer_encoder.layers), 2)
    
    def test_positional_encoding(self):
        """测试位置编码（通过Transformer内部处理）"""
        # Transformer模型默认没有显式的位置编码
        # 检查是否能处理序列数据
        hist_word_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length))
        hist_num_feat = torch.randn(self.batch_size, self.sequence_length, self.num_features)
        curr_word_id = torch.randint(0, 100, (self.batch_size,))
        curr_num_feat = torch.randn(self.batch_size, self.num_features)
        
        # 应该能正常处理
        output = self.model(
            hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_size))


class TestModelFactory(unittest.TestCase):
    """测试ModelFactory"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'data': {
                'features': {
                    'numeric': ['feat1', 'feat2', 'feat3'],
                    'categorical': ['cat1', 'cat2'],
                    'target': ['target1', 'target2', 'target3']
                }
            },
            'model': {
                'hidden_size': 32
            }
        }
        
        self.factory = ModelFactory(self.config)
    
    def test_create_baselstm(self):
        """测试创建BaseLSTM"""
        model = self.factory.create_model("BaseLSTM", num_layers=2, dropout=0.3)
        
        self.assertIsInstance(model, BaseLSTM)
        
        # 检查参数
        expected_input_size = 5  # 3个numeric + 2个categorical
        expected_output_size = 3  # 3个target
        
        # 检查LSTM输入维度
        self.assertEqual(model.lstm.input_size, expected_input_size)
        self.assertEqual(model.lstm.hidden_size, self.config['model']['hidden_size'])
        
        # 检查输出维度
        self.assertEqual(model.fc2.out_features, expected_output_size)
    
    def test_create_improvedbilstm(self):
        """测试创建ImprovedBiLSTM"""
        # 需要先创建embedding矩阵
        vocab_size = 100
        emb_dim = 50
        embedding_matrix = np.random.randn(vocab_size, emb_dim)
        
        # 由于ModelFactory.create_model需要embedding_matrix，我们直接测试
        model = ImprovedBiLSTM(
            embedding_matrix=embedding_matrix,
            num_features=5,  # 3 numeric + 2 categorical
            hidden_size=32,
            output_size=3,
            use_embeddings=True
        )
        
        self.assertIsInstance(model, ImprovedBiLSTM)
    
    def test_create_transformer(self):
        """测试创建TransformerModel"""
        vocab_size = 100
        emb_dim = 50
        embedding_matrix = np.random.randn(vocab_size, emb_dim)
        
        model = TransformerModel(
            embedding_matrix=embedding_matrix,
            num_features=5,
            d_model=32,
            nhead=4,
            num_layers=2,
            output_size=3,
            use_embeddings=True
        )
        
        self.assertIsInstance(model, TransformerModel)
    
    def test_invalid_model_type(self):
        """测试无效模型类型"""
        with self.assertRaises(ValueError):
            self.factory.create_model("InvalidModel")
    
    def test_model_device_movement(self):
        """测试模型设备移动"""
        model = self.factory.create_model("BaseLSTM")
        
        # 检查初始设备
        self.assertEqual(next(model.parameters()).device, torch.device('cpu'))
        
        # 移动到GPU（如果可用）
        if torch.cuda.is_available():
            model = model.cuda()
            self.assertEqual(next(model.parameters()).device, torch.device('cuda'))
    
    def test_model_serialization(self):
        """测试模型序列化"""
        model = self.factory.create_model("BaseLSTM")
        
        # 保存模型
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(model.state_dict(), temp_path)
            
            # 加载模型
            new_model = self.factory.create_model("BaseLSTM")
            new_model.load_state_dict(torch.load(temp_path))
            
            # 检查参数是否相同
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestModelIntegration(unittest.TestCase):
    """测试模型集成"""
    
    def test_all_models_output_shape(self):
        """测试所有模型的输出形状一致性"""
        batch_size = 16
        sequence_length = 7
        num_features = 10
        output_size = 7
        
        # 创建模拟数据
        x = torch.randn(batch_size, sequence_length, num_features)
        
        hist_word_ids = torch.randint(0, 100, (batch_size, sequence_length))
        hist_num_feat = torch.randn(batch_size, sequence_length, num_features)
        curr_word_id = torch.randint(0, 100, (batch_size,))
        curr_num_feat = torch.randn(batch_size, num_features)
        
        # BaseLSTM
        base_model = BaseLSTM(num_features, 32, output_size)
        output1 = base_model(x)
        
        # ImprovedBiLSTM
        emb_matrix = np.random.randn(100, 50)
        improved_model = ImprovedBiLSTM(emb_matrix, num_features, 32, output_size)
        output2, _ = improved_model(hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        # Transformer
        transformer_model = TransformerModel(emb_matrix, num_features, 32, 
                                           nhead=4, num_layers=2, output_size=output_size)
        output3 = transformer_model(hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
        
        # 检查所有输出形状相同
        self.assertEqual(output1.shape, (batch_size, output_size))
        self.assertEqual(output2.shape, (batch_size, output_size))
        self.assertEqual(output3.shape, (batch_size, output_size))
    
    def test_probability_distribution_property(self):
        """测试所有模型输出都是概率分布"""
        batch_size = 4
        sequence_length = 7
        num_features = 5
        output_size = 3
        
        x = torch.randn(batch_size, sequence_length, num_features)
        hist_word_ids = torch.randint(0, 20, (batch_size, sequence_length))
        hist_num_feat = torch.randn(batch_size, sequence_length, num_features)
        curr_word_id = torch.randint(0, 20, (batch_size,))
        curr_num_feat = torch.randn(batch_size, num_features)
        
        # 测试每个模型
        models = []
        
        # BaseLSTM
        models.append(('BaseLSTM', BaseLSTM(num_features, 16, output_size)))
        
        # ImprovedBiLSTM
        emb_matrix = np.random.randn(20, 10)
        models.append(('ImprovedBiLSTM', 
                      ImprovedBiLSTM(emb_matrix, num_features, 16, output_size)))
        
        # Transformer
        models.append(('Transformer',
                      TransformerModel(emb_matrix, num_features, 16, 
                                     nhead=2, num_layers=1, output_size=output_size)))
        
        for name, model in models:
            model.eval()
            
            if name == 'BaseLSTM':
                output = model(x)
            elif name == 'ImprovedBiLSTM':
                output, _ = model(hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
            else:  # Transformer
                output = model(hist_word_ids, hist_num_feat, curr_word_id, curr_num_feat)
            
            # 检查每行和为1
            for i in range(batch_size):
                row_sum = output[i].sum().item()
                self.assertAlmostEqual(row_sum, 1.0, places=5, 
                                     msg=f"{name} 第{i}行和为{row_sum}")
            
            # 检查所有值在[0,1]范围内
            self.assertTrue(torch.all(output >= 0), 
                          f"{name} 有负值")
            self.assertTrue(torch.all(output <= 1), 
                          f"{name} 有大于1的值")


if __name__ == '__main__':
    unittest.main()
