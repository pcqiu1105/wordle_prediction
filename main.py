#!/usr/bin/env python3
"""
Wordle玩家表现分布预测主程序
"""

import argparse
import yaml
import torch
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelFactory
from src.train import Trainer
from src.evaluate import Evaluator
from src.visualize import Visualizer
from src.utils import set_seed, load_config


def main():
    parser = argparse.ArgumentParser(description="Wordle预测模型")
    parser.add_argument("--mode", type=str, default="train_eval", 
                       choices=["train", "eval", "visualize", "train_eval"],
                       help="运行模式")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    if args.mode in ["train", "train_eval"]:
        print("=" * 60)
        print("训练阶段")
        print("=" * 60)
        
        # 数据预处理
        preprocessor = DataPreprocessor(config)
        data = preprocessor.load_and_preprocess()
        
        # 特征工程
        engineer = FeatureEngineer(config)
        features = engineer.create_features(data)
        
        # 划分数据集
        train_data, val_data, test_data = engineer.split_data(features)
        
        # 构建数据加载器
        train_loader, val_loader, test_loader = engineer.create_dataloaders(
            train_data, val_data, test_data
        )
        
        # 创建模型
        model_factory = ModelFactory(config)
        model = model_factory.create_model("ImprovedBiLSTM", use_embeddings=True)
        
        # 训练
        trainer = Trainer(config)
        trainer.train(model, train_loader, val_loader)
        
        print("✅ 训练完成")
    
    if args.mode in ["eval", "train_eval"]:
        print("\n" + "=" * 60)
        print("评估阶段")
        print("=" * 60)
        
        # 加载最佳模型
        model_path = f"{config['paths']['models']}/best_model.pth"
        model = model_factory.load_model(model_path)
        
        # 评估
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(model, test_loader)
        
        print(f"测试结果: {metrics}")
        
        # 保存结果
        evaluator.save_results(metrics)
        
        print("✅ 评估完成")
    
    if args.mode == "visualize":
        print("\n" + "=" * 60)
        print("可视化阶段")
        print("=" * 60)
        
        visualizer = Visualizer(config)
        
        # 加载数据
        results = visualizer.load_results()
        
        # 生成可视化图表
        visualizer.plot_model_comparison(results)
        visualizer.plot_ablation_study(results)
        visualizer.plot_training_curves()
        visualizer.plot_predictions()
        
        print("✅ 可视化完成")


if __name__ == "__main__":
    main()
