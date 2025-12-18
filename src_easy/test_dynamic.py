#!/usr/bin/env python3
"""
测试动态融合probe的简单脚本
只测试新的probe方法性能，保留原来的测试框架
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dynamic_probe import run_dynamic_probe_pipeline, DynamicFusionProbe, DynamicProbeDataset
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from typing import List, Optional



def evaluate_uncertainty(model_path: str, test_data, num_samples: int = 100):
    """评估Dirichlet模型的不确定性"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint['metadata']
    probe_type = metadata.get('probe_type', 'softmax')

    if probe_type != "dirichlet":
        print("Uncertainty evaluation is only available for Dirichlet probes")
        return None

    model = DynamicFusionProbe(
        metadata['input_dim'],
        metadata['num_layers'],
        metadata['output_dim'],
        probe_type=probe_type,
        mlp_hidden_dims=metadata.get('mlp_hidden_dims', None),
        dropout=metadata.get('dropout', 0.1)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建测试数据集
    test_dataset = DynamicProbeDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_uncertainties = []
    all_predictions = []
    all_labels = []

    print(f"Evaluating uncertainty with {num_samples} samples per prediction...")

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_size = batch_features.size(0)

            # 多次采样计算不确定性
            batch_predictions = []
            batch_uncertainties = []

            for _ in range(num_samples):
                model.train()  # 开启训练模式进行采样
                logits, uncertainty = model(batch_features, return_uncertainty=True)
                predictions = torch.sigmoid(logits).squeeze(-1)

                batch_predictions.append(predictions.cpu().numpy())
                batch_uncertainties.append(uncertainty.cpu().numpy())

            # 计算预测的方差作为认知不确定性
            batch_predictions = np.array(batch_predictions)  # [num_samples, batch_size]
            prediction_variance = np.var(batch_predictions, axis=0)  # [batch_size]
            prediction_mean = np.mean(batch_predictions, axis=0)  # [batch_size]

            # 平均的熵作为总体不确定性
            batch_uncertainties = np.array(batch_uncertainties)  # [num_samples, batch_size]
            avg_uncertainty = np.mean(batch_uncertainties, axis=0)  # [batch_size]

            all_predictions.extend(prediction_mean)
            all_uncertainties.extend(avg_uncertainty)
            all_labels.extend(batch_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)

    # 计算不确定性指标
    binary_predictions = (all_predictions > 0.5).astype(int)
    correct_mask = (binary_predictions == all_labels)

    # 正确和错误预测的不确定性分布
    correct_uncertainty = all_uncertainties[correct_mask]
    incorrect_uncertainty = all_uncertainties[~correct_mask]

    uncertainty_stats = {
        'mean_uncertainty': float(np.mean(all_uncertainties)),
        'std_uncertainty': float(np.std(all_uncertainties)),
        'correct_mean_uncertainty': float(np.mean(correct_uncertainty)) if len(correct_uncertainty) > 0 else 0.0,
        'incorrect_mean_uncertainty': float(np.mean(incorrect_uncertainty)) if len(incorrect_uncertainty) > 0 else 0.0,
        'uncertainty_accuracy_correlation': float(np.corrcoef(all_uncertainties, correct_mask.astype(float))[0, 1])
    }

    print(f"🔍 Uncertainty Analysis:")
    print(f"   Mean uncertainty: {uncertainty_stats['mean_uncertainty']:.4f}")
    print(f"   Uncertainty std: {uncertainty_stats['std_uncertainty']:.4f}")
    print(f"   Correct predictions uncertainty: {uncertainty_stats['correct_mean_uncertainty']:.4f}")
    print(f"   Incorrect predictions uncertainty: {uncertainty_stats['incorrect_mean_uncertainty']:.4f}")
    print(f"   Uncertainty-accuracy correlation: {uncertainty_stats['uncertainty_accuracy_correlation']:.4f}")

    return uncertainty_stats

def test_mixed_datasets(test_configs, probe_type, mlp_hidden_dims: List[int] = None,
                       dropout: float = 0.1, save_dir: str = None):
    """测试混合数据集的函数"""
    print(f"🌟 Training on MIXED datasets:")
    
    try:
        # 收集所有数据
        all_data = []
        dataset_info = []
        
        print(f"🔄 Loading {len(test_configs)} datasets...")
        
        for config in test_configs:
            task = config["task"]
            hidden_states_file = config["hidden_states_file"]
            
            if not os.path.exists(hidden_states_file):
                print(f"❌ File not found: {hidden_states_file}")
                continue
            
            print(f"   📁 Loading {task}...")
            data = torch.load(hidden_states_file, map_location="cpu", weights_only=False)
            all_data.extend(data)
            dataset_info.append(f"{task}: {len(data)} samples")
            print(f"     ✅ Loaded {len(data)} samples")
        
        if not all_data:
            print("❌ No valid data loaded!")
            return None
        
        print(f"\n🔗 Combined total: {len(all_data)} samples")
        
        # 创建混合任务名
        mixed_task_name = "mixed_" + "_".join([config["task"] for config in test_configs])
        
        # 创建临时文件保存混合数据
        mixed_file = f"../temp_mixed_data_{probe_type}.pt"
        os.makedirs(os.path.dirname(os.path.abspath(mixed_file)), exist_ok=True)
        torch.save(all_data, mixed_file)
        
        print(f"💾 Saved mixed data to: {mixed_file}")
        print(f"🚀 Training {probe_type} probe on mixed dataset...")

        # 使用现有的测试函数
        results = test_dynamic_probe_on_task(mixed_task_name, mixed_file, probe_type,
                                            mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                            save_dir=save_dir)

        # 清理临时文件
        try:
            os.remove(mixed_file)
            print(f"🗑️ Cleaned up temporary file")
        except:
            pass

        if results:
            # 添加数据集信息到结果中
            results['dataset_info'] = dataset_info
            results['total_samples'] = len(all_data)
            print(f"✅ Mixed training completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Mixed dataset training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_dynamic_probe_on_task(task: str, hidden_states_file: str, probe_type: str = "softmax",
                              mlp_hidden_dims: List[int] = None, dropout: float = 0.1,
                              save_dir: str = None):
    """测试动态probe在特定任务上的性能"""

    print(f"=" * 60)
    print(f"Testing Dynamic Fusion Probe ({probe_type}) on task: {task}")
    print(f"=" * 60)

    if not os.path.exists(hidden_states_file):
        print(f"❌ Hidden states file not found: {hidden_states_file}")
        return None

    try:
        # 运行动态probe训练和测试
        results = run_dynamic_probe_pipeline(
            task=task,
            hidden_states_file=hidden_states_file,
            save_dir=save_dir or "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/test",
            probe_type=probe_type,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

        # 输出结果
        training_results = results['training_results']
        test_results = results['test_results']

        print(f"\n📊 Training Results:")
        print(f"   Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"   Final layer weights: {training_results['final_layer_weights']}")

        if probe_type == "dirichlet":
            print(f"   Final concentration: {training_results['final_concentration']}")
            print(f"   Global concentration (β₀): {training_results['final_global_concentration']:.4f}")

        print(f"\n📊 Test Results:")
        print(f"   Test accuracy: {test_results['accuracy']:.4f}")

        if probe_type == "dirichlet":
            print(f"   Final concentration: {test_results['concentration']}")
            print(f"   Global concentration (β₀): {test_results['global_concentration']:.4f}")

        print(f"\n💾 Model saved to: {results['model_path']}")

        # 为Dirichlet模型评估不确定性
        if probe_type == "dirichlet":
            print(f"\n🔍 Evaluating uncertainty for Dirichlet model...")
            # 加载测试数据进行不确定性评估
            data = torch.load(hidden_states_file, map_location="cpu",weights_only= False)
            split_idx = int(len(data) * 0.8)
            val_data = data[split_idx:]  # 使用验证集评估不确定性

            uncertainty_stats = evaluate_uncertainty(results['model_path'], val_data, num_samples=50)
            if uncertainty_stats:
                results['uncertainty_stats'] = uncertainty_stats

        return results

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

def test_mixed_datasets_with_sampling(test_configs, probe_type, max_samples=None,
                                     mlp_hidden_dims: List[int] = None, dropout: float = 0.1,
                                     save_dir: str = None):
    """测试混合数据集的函数，支持采样限制"""
    print(f"🌟 Training on MIXED datasets (max_samples: {max_samples}):")
    
    try:
        # 收集所有数据
        all_data = []
        dataset_info = []
        
        print(f"🔄 Loading {len(test_configs)} datasets...")
        
        for config in test_configs:
            task = config["task"]
            hidden_states_file = config["hidden_states_file"]
            
            if not os.path.exists(hidden_states_file):
                print(f"❌ File not found: {hidden_states_file}")
                continue
            
            print(f"   📁 Loading {task}...")
            data = torch.load(hidden_states_file, map_location="cpu", weights_only=False)
            all_data.extend(data)
            dataset_info.append(f"{task}: {len(data)} samples")
            print(f"     ✅ Loaded {len(data)} samples")
        
        if not all_data:
            print("❌ No valid data loaded!")
            return None
        
        # 如果指定了max_samples，进行随机采样
        if max_samples and len(all_data) > max_samples:
            print(f"🎯 Sampling {max_samples} from {len(all_data)} total samples")
            import random
            random.shuffle(all_data)
            all_data = all_data[:max_samples]
        
        print(f"\n🔗 Final dataset size: {len(all_data)} samples")
        
        # 创建混合任务名，包含样本数信息
        sample_suffix = f"_{max_samples}samples" if max_samples else "_allsamples"
        mixed_task_name = "mixed_" + "_".join([config["task"] for config in test_configs]) 
        
        # 创建临时文件保存混合数据
        mixed_file = f"../temp_mixed_data_{probe_type}{sample_suffix}.pt"
        os.makedirs(os.path.dirname(os.path.abspath(mixed_file)), exist_ok=True)
        torch.save(all_data, mixed_file)
        
        print(f"💾 Saved mixed data to: {mixed_file}")
        print(f"🚀 Training {probe_type} probe on mixed dataset...")

        # 使用现有的测试函数
        results = test_dynamic_probe_on_task(mixed_task_name, mixed_file, probe_type,
                                            mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                            save_dir=save_dir)

        # 清理临时文件
        try:
            os.remove(mixed_file)
            print(f"🗑️ Cleaned up temporary file")
        except:
            pass

        if results:
            # 添加数据集信息到结果中
            results['dataset_info'] = dataset_info
            results['total_samples'] = len(all_data)
            results['max_samples'] = max_samples
            print(f"✅ Mixed training completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Mixed dataset training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main_with_sampling(datasets=None, probe_types=None, max_samples=None,
                      mlp_hidden_dims: List[int] = None, dropout: float = 0.1,
                      save_dir: str = None):
    """主测试函数 - 支持不同采样大小

    Args:
        datasets: 数据集名称列表，如 ["alpaca_5k", "mmlu_train", "big_math"]
        probe_types: probe类型列表，如 ["softmax", "dirichlet"]
        max_samples: 最大采样数，如 12000
        mlp_hidden_dims: MLP隐藏层维度列表，如 [64, 128]
        dropout: Dropout比率
        save_dir: 保存目录路径
    """

    # 数据集映射表
    dataset_map = {
        "alpaca_5k": {
            "task": "alpaca_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_alpaca_5k_train.pt"
        },
        "mmlu_train": {
            "task": "mmlu_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_mmlu_train.pt"
        },
        "big_math": {
            "task": "big_math_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_big_math_5k_train.pt"
        }
    }

    # 使用传入的参数或默认值
    if datasets is None:
        datasets = ["alpaca_5k", "mmlu_train", "big_math"]

    if probe_types is None:
        probe_types = ["softmax", "dirichlet"]

    # 根据数据集名称构建test_configs
    test_configs = []
    for dataset_name in datasets:
        if dataset_name in dataset_map:
            test_configs.append(dataset_map[dataset_name])
        else:
            print(f"⚠️  Unknown dataset: {dataset_name}, skipping...")

    if not test_configs:
        print("❌ No valid datasets specified!")
        return

    # 定义不同的采样大小
    sample_sizes = [max_samples] if max_samples else [None]  # None表示使用全部数据
    
    all_results = {}

    for probe_type in probe_types:
        print(f"\n{'='*80}")
        print(f"🔬 Testing {probe_type.upper()} probe type with different sample sizes")
        print(f"{'='*80}")
        
        all_results[probe_type] = {}
        
        for sample_size in sample_sizes:
            size_label = f"{sample_size}samples" if sample_size else "allsamples"
            print(f"\n🎯 Training with sample size: {size_label}")
            print(f"{'-'*60}")

            # 测试混合数据集
            mixed_results = test_mixed_datasets_with_sampling(test_configs, probe_type, sample_size,
                                                             mlp_hidden_dims=mlp_hidden_dims,
                                                             dropout=dropout, save_dir=save_dir)
            
            if mixed_results:
                mixed_summary = {
                    "accuracy": mixed_results['test_results']['accuracy'],
                    "best_val_loss": mixed_results['training_results']['best_val_loss'],
                    "layer_weights": mixed_results['test_results']['layer_weights'].tolist(),
                    "total_samples": mixed_results['total_samples'],
                    "max_samples": mixed_results['max_samples']
                }
                
                if probe_type == "dirichlet":
                    mixed_summary.update({
                        "concentration": mixed_results['test_results']['concentration'].tolist(),
                        "global_concentration": mixed_results['test_results']['global_concentration']
                    })
                    
                    if 'uncertainty_stats' in mixed_results:
                        mixed_summary['uncertainty_stats'] = mixed_results['uncertainty_stats']
                
                # 使用样本大小作为键
                all_results[probe_type][f"mixed_{size_label}"] = mixed_summary
                
                print(f"✅ Mixed training completed - Accuracy: {mixed_results['test_results']['accuracy']:.4f}")
                print(f"   Sample size: {mixed_results['total_samples']}")
                print(f"   Best val loss: {mixed_results['training_results']['best_val_loss']:.4f}")
                
                if probe_type == "dirichlet":
                    print(f"   Global concentration: {mixed_results['test_results']['global_concentration']:.4f}")
                    if 'uncertainty_stats' in mixed_results:
                        print(f"   Mean uncertainty: {mixed_results['uncertainty_stats']['mean_uncertainty']:.4f}")
            
            print(f"\n{'-'*60}")

    # 保存所有测试结果
    if all_results:
       

        # 打印总结
        print(f"\n📋 Summary of Dynamic Probe Performance by Sample Size:")
        for probe_type, probe_results in all_results.items():
            print(f"\n{probe_type.upper()} Probe:")
            for size_key, result in probe_results.items():
                accuracy = result['accuracy']
                samples = result['total_samples']
                val_loss = result['best_val_loss']
                
                print(f"   {size_key}: Accuracy = {accuracy:.4f}, Samples = {samples}, Val Loss = {val_loss:.4f}")
                
                if probe_type == "dirichlet":
                    global_conc = result['global_concentration']
                    print(f"       Global concentration (β₀) = {global_conc:.4f}")
                    
                    if 'uncertainty_stats' in result:
                        mean_unc = result['uncertainty_stats']['mean_uncertainty']
                        print(f"       Mean uncertainty = {mean_unc:.4f}")

        # 生成性能对比表
        print(f"\n📊 Performance Comparison Table:")
        print(f"{'Probe Type':<12} {'Sample Size':<15} {'Accuracy':<10} {'Val Loss':<10} {'Samples':<8}")
        print(f"{'-'*65}")
        
        for probe_type in probe_types:
            for size_key, result in all_results[probe_type].items():
                accuracy = result['accuracy']
                val_loss = result['best_val_loss']
                samples = result['total_samples']
                
                print(f"{probe_type:<12} {size_key:<15} {accuracy:.4f}     {val_loss:.4f}     {samples:<8}")


def main_single_datasets_with_sampling():
    """训练单个数据集的不同采样大小"""
    
    # 定义测试任务
    test_configs = [
        {
            "task": "alpaca_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_alpaca_5k_train.pt"
        },
        {
            "task": "mmlu_train", 
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_mmlu_train.pt"
        },
        {
            "task": "big_math_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Llama-3.1-8B-Instruct_big_math_5k_train.pt"
        }
    ]
    
    probe_types = ["softmax", "dirichlet"]
    sample_sizes = [None]
    
    all_results = {}
    
    for probe_type in probe_types:
        all_results[probe_type] = {}
        
        for config in test_configs:
            task = config["task"]
            hidden_states_file = config["hidden_states_file"]
            
            print(f"\n🔬 Testing {probe_type.upper()} on {task}")
            
            for sample_size in sample_sizes:
                size_label = f"{sample_size}samples" if sample_size else "allsamples"
                
                # 加载并采样数据
                if os.path.exists(hidden_states_file):
                    data = torch.load(hidden_states_file, map_location="cpu", weights_only=False)
                    
                    if sample_size and len(data) > sample_size:
                        import random
                        random.shuffle(data)
                        data = data[:sample_size]
                    
                    # 创建临时文件
                    temp_file = f"../temp_{task}_{size_label}.pt"
                    torch.save(data, temp_file)
                    
                    # 测试
                    task_with_size = f"{task}_{size_label}"
                    results = test_dynamic_probe_on_task(task_with_size, temp_file, probe_type)
                    
                    if results:
                        result_key = f"{task}_{size_label}"
                        all_results[probe_type][result_key] = {
                            "accuracy": results['test_results']['accuracy'],
                            "best_val_loss": results['training_results']['best_val_loss'],
                            "total_samples": len(data)
                        }
                        
                        print(f"   {size_label}: Accuracy = {results['test_results']['accuracy']:.4f}")
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    return all_results



def test_mixed_datasets_for_leave_one_category(test_configs, probe_type, custom_save_name):
    """为留一类实验专门设计的混合数据集测试函数"""
    
    print(f"🌟 Training mixed datasets for leave-one-category experiment:")
    
    try:
        # 收集所有数据
        all_data = []
        dataset_info = []
        
        print(f"🔄 Loading {len(test_configs)} datasets...")
        
        for config in test_configs:
            task = config["task"]
            hidden_states_file = config["hidden_states_file"]
            
            if not os.path.exists(hidden_states_file):
                print(f"❌ File not found: {hidden_states_file}")
                continue
            
            print(f"   📁 Loading {task}...")
            data = torch.load(hidden_states_file, map_location="cpu", weights_only=False)
            all_data.extend(data)
            dataset_info.append(f"{task}: {len(data)} samples")
            print(f"     ✅ Loaded {len(data)} samples")
        
        if not all_data:
            print("❌ No valid data loaded!")
            return None
        
        print(f"\n🔗 Combined total: {len(all_data)} samples")
        
        # 创建混合任务名
        mixed_task_name = f"mmlu_pro_{custom_save_name}"
        
        # 创建临时文件保存混合数据
        mixed_file = f"../temp_mixed_data_{custom_save_name}_{probe_type}.pt"
        os.makedirs(os.path.dirname(os.path.abspath(mixed_file)), exist_ok=True)
        torch.save(all_data, mixed_file)
        
        print(f"💾 Saved mixed data to: {mixed_file}")
        print(f"🚀 Training {probe_type} probe on mixed dataset...")
        
        # 使用专门的保存目录
        save_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/mmlu_pro"
        results = run_dynamic_probe_pipeline(
            task=mixed_task_name,
            hidden_states_file=mixed_file,
            save_dir=save_dir,
            probe_type=probe_type
        )
        
        # 清理临时文件
        try:
            os.remove(mixed_file)
            print(f"🗑️ Cleaned up temporary file")
        except:
            pass
        
        if results:
            # 添加数据集信息到结果中
            results['dataset_info'] = dataset_info
            results['total_samples'] = len(all_data)
            print(f"✅ Mixed training completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Mixed dataset training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main_leave_one_category_mmlu_pro():
    """三缺一（MMLU-Pro 按大类留一训练）的动态probe实验"""
    
    # 定义MMLU-Pro的四个大类
    categories = {
        "stem": ["chemistry", "computer_science", "engineering", "math", "physics"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["economics", "psychology", "other"],
        "other_disciplines": ["business", "health", "medicine"]
    }
    
    # 基础文件路径模板
    base_path = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs"
    
    probe_types = ["softmax", "dirichlet"]
    
    all_results = {}
    
    print("\n================ LEAVE-ONE-CATEGORY (MMLU-Pro) DYNAMIC PROBE TRAINING ================")
    
    for probe_type in probe_types:
        all_results[probe_type] = {}
        
        print(f"\n🔬 Testing {probe_type.upper()} probe with leave-one-category strategy")
        
        for leave_out in categories.keys():
            include_categories = [c for c in categories.keys() if c != leave_out]
            
            # 构建训练任务配置列表
            train_configs = []
            total_tasks = 0
            
            for cat in include_categories:
                for subject in categories[cat]:
                    task_name = f"mmlu_pro_{subject}"
                    hidden_states_file = f"{base_path}/mmlu_pro/Llama-3.1-8B-Instruct_{task_name}.pt"
                    
                    # 检查文件是否存在
                    if os.path.exists(hidden_states_file):
                        train_configs.append({
                            "task": task_name,
                            "hidden_states_file": hidden_states_file
                        })
                        total_tasks += 1
                    else:
                        print(f"⚠️  File not found: {hidden_states_file}")
            
            print("\n" + "=" * 80)
            print(f"🚀 Training with three categories, leaving out: {leave_out}")
            print(f"📚 Using subjects from: {', '.join(include_categories)}")
            print(f"🧩 Total valid tasks: {total_tasks}")
            
            if not train_configs:
                print("❌ No valid training data found for this configuration!")
                continue
            
            # 训练混合数据集
            custom_save_name = f"leaveout_{leave_out.lower()}"
            mixed_results = test_mixed_datasets_for_leave_one_category(
                train_configs, 
                probe_type, 
                custom_save_name
            )
            
            if mixed_results:
                result_key = f"leaveout_{leave_out.lower()}"
                all_results[probe_type][result_key] = {
                    "accuracy": mixed_results['test_results']['accuracy'],
                    "best_val_loss": mixed_results['training_results']['best_val_loss'],
                    "layer_weights": mixed_results['test_results']['layer_weights'].tolist(),
                    "total_samples": mixed_results['total_samples'],
                    "leave_out_category": leave_out,
                    "include_categories": include_categories,
                    "total_tasks": total_tasks
                }
                
                if probe_type == "dirichlet":
                    result_key_data = all_results[probe_type][result_key]
                    result_key_data.update({
                        "concentration": mixed_results['test_results']['concentration'].tolist(),
                        "global_concentration": mixed_results['test_results']['global_concentration']
                    })
                    
                    if 'uncertainty_stats' in mixed_results:
                        result_key_data['uncertainty_stats'] = mixed_results['uncertainty_stats']
                
                print(f"✅ Leave-one-category training completed - Accuracy: {mixed_results['test_results']['accuracy']:.4f}")
                print(f"   Left out: {leave_out}")
                print(f"   Sample size: {mixed_results['total_samples']}")
                print(f"   Best val loss: {mixed_results['training_results']['best_val_loss']:.4f}")
                
                if probe_type == "dirichlet":
                    print(f"   Global concentration: {mixed_results['test_results']['global_concentration']:.4f}")
                    if 'uncertainty_stats' in mixed_results:
                        print(f"   Mean uncertainty: {mixed_results['uncertainty_stats']['mean_uncertainty']:.4f}")
            else:
                print(f"❌ Failed to train for leave-out category: {leave_out}")
    
    # 保存结果
    if all_results:
        results_file = "../results/dynamic_probe_leave_one_category_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📄 Leave-one-category results saved to: {results_file}")
        
        # 打印总结
        print(f"\n📋 Summary of Leave-One-Category Dynamic Probe Performance:")
        for probe_type, probe_results in all_results.items():
            print(f"\n{probe_type.upper()} Probe:")
            for result_key, result in probe_results.items():
                accuracy = result['accuracy']
                samples = result['total_samples']
                val_loss = result['best_val_loss']
                leave_out = result['leave_out_category']
                
                print(f"   {result_key}: Accuracy = {accuracy:.4f}, Samples = {samples}, Val Loss = {val_loss:.4f}")
                print(f"       Left out: {leave_out}")
                
                if probe_type == "dirichlet":
                    global_conc = result['global_concentration']
                    print(f"       Global concentration (β₀) = {global_conc:.4f}")
                    
                    if 'uncertainty_stats' in result:
                        mean_unc = result['uncertainty_stats']['mean_uncertainty']
                        print(f"       Mean uncertainty = {mean_unc:.4f}")
        
        # 生成性能对比表
        print(f"\n📊 Leave-One-Category Performance Comparison:")
        print(f"{'Probe Type':<12} {'Left Out':<18} {'Accuracy':<10} {'Val Loss':<10} {'Samples':<8}")
        print(f"{'-'*70}")
        
        for probe_type in probe_types:
            if probe_type in all_results:
                for result_key, result in all_results[probe_type].items():
                    accuracy = result['accuracy']
                    val_loss = result['best_val_loss']
                    samples = result['total_samples']
                    leave_out = result['leave_out_category']
                    
                    print(f"{probe_type:<12} {leave_out:<18} {accuracy:.4f}     {val_loss:.4f}     {samples:<8}")
    
    print("\n✅ Completed leave-one-category training for all four choices.")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dynamic Fusion Probe Training and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python test_dynamic.py --datasets alpaca_5k mmlu_train --probe_types softmax

  # 使用 MLP 结构和自定义配置
  python test_dynamic.py \\
    --datasets alpaca_5k mmlu_train big_math \\
    --probe_types softmax dirichlet \\
    --max_samples 12000 \\
    --mlp_hidden_dims 64 128 \\
    --dropout 0.5 \\
    --save_dir custom/save/path

  # 单数据集训练
  python test_dynamic.py --datasets alpaca_5k --probe_types dirichlet --max_samples 5000
        """
    )

    parser.add_argument('--datasets', type=str, nargs='+',
                       default=["alpaca_5k", "mmlu_train", "big_math"],
                       help='数据集名称列表（空格分隔），可选: alpaca_5k, mmlu_train, big_math')

    parser.add_argument('--probe_types', type=str, nargs='+',
                       default=["softmax", "dirichlet"],
                       help='Probe 类型列表（空格分隔），可选: softmax, dirichlet')

    parser.add_argument('--max_samples', type=int, default=12000,
                       help='最大采样数量（默认: 12000）')

    parser.add_argument('--mlp_hidden_dims', type=int, nargs='*', default=None,
                       help='MLP 隐藏层维度列表（空格分隔），例如: 64 128。留空表示单层线性分类器')

    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout 比率（默认: 0.1）')

    parser.add_argument('--save_dir', type=str,
                       default="/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/",
                       help='模型和历史保存目录')

    args = parser.parse_args()

    print(f"📋 Running with parameters:")
    print(f"   Datasets: {args.datasets}")
    print(f"   Probe types: {args.probe_types}")
    print(f"   Max samples: {args.max_samples}")
    print(f"   MLP hidden dims: {args.mlp_hidden_dims}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Save directory: {args.save_dir}")
    print()

    # 运行混合数据集的采样实验
    main_with_sampling(
        datasets=args.datasets,
        probe_types=args.probe_types,
        max_samples=args.max_samples,
        mlp_hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout,
        save_dir=args.save_dir
    )

    # main_leave_one_category_mmlu_pro()

    # main_single_datasets_with_sampling()
 
    