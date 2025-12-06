"""
Big-Math 难度分层抽样脚本
目标: 构造一个能拉开 7B 和 GPT-4o 差距的 benchmark
"""

from datasets import load_dataset
import pandas as pd
from collections import Counter
import random

# 设置随机种子保证可复现
random.seed(42)

def select_by_difficulty(ds, low, high, name=""):
    """根据 solve_rate 筛选难度区间"""
    filtered = ds.filter(
        lambda ex: low <= ex["llama8b_solve_rate"] < high
    )
    print(f"{name}: solve_rate [{low}, {high}) -> {len(filtered)} 题")
    return filtered

def stratified_sample(ds, n_samples, stratify_by="source"):
    """分层采样,保证不同source/domain的均衡性"""
    df = ds.to_pandas()
    
    # 统计各source分布
    source_counts = Counter(df[stratify_by])
    print(f"\n原始分布 ({stratify_by}):")
    for src, cnt in source_counts.most_common(10):
        print(f"  {src}: {cnt}")
    
    # 计算每个source应该抽多少题(按比例)
    total = len(df)
    samples_per_source = {
        src: max(1, int(n_samples * cnt / total)) 
        for src, cnt in source_counts.items()
    }
    
    # 微调确保总数正确
    diff = n_samples - sum(samples_per_source.values())
    if diff > 0:
        # 从大类中补充
        largest_sources = [src for src, _ in source_counts.most_common(abs(diff))]
        for src in largest_sources:
            samples_per_source[src] += 1
    elif diff < 0:
        # 从大类中削减
        largest_sources = [src for src, _ in source_counts.most_common(abs(diff))]
        for src in largest_sources:
            samples_per_source[src] = max(1, samples_per_source[src] - 1)
    
    # 分层抽样
    sampled_dfs = []
    for src, n in samples_per_source.items():
        src_df = df[df[stratify_by] == src]
        if len(src_df) >= n:
            sampled = src_df.sample(n=n, random_state=42)
        else:
            sampled = src_df  # 如果不够就全取
        sampled_dfs.append(sampled)
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n抽样后分布 ({stratify_by}):")
    sampled_counts = Counter(result_df[stratify_by])
    for src, cnt in sampled_counts.most_common(10):
        print(f"  {src}: {cnt}")
    
    return result_df

def main(total_samples=10000, train_ratio=0.8):
    """
    Args:
        total_samples: 总共需要的样本数量，默认 10000
        train_ratio: 训练集比例，默认 0.8 (80% train, 20% test)
    """
    print("=" * 60)
    print("开始加载 Big-Math-RL-Verified-Processed 数据集...")
    print("=" * 60)
    print(f"目标样本数: {total_samples}")
    print(f"Train/Test 划分比例: {train_ratio:.0%} / {1-train_ratio:.0%}")

    # 设置输出目录
    output_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/data"
    
    # 加载完整数据集
    ds = load_dataset(
        "open-r1/Big-Math-RL-Verified-Processed",
        "all", 
        split="train"
    )
    
    print(f"\n总题数: {len(ds)}")
    print(f"字段: {ds.features.keys()}")
    
    # 查看solve_rate分布
    solve_rates = [ex["llama8b_solve_rate"] for ex in ds]
    print(f"\nsolve_rate 统计:")
    print(f"  最小值: {min(solve_rates):.3f}")
    print(f"  最大值: {max(solve_rates):.3f}")
    print(f"  平均值: {sum(solve_rates)/len(solve_rates):.3f}")
    
    # 过滤掉MATH数据源
    print("\n" + "=" * 60)
    print("过滤MATH数据源...")
    print("=" * 60)
    original_count = len(ds)
    ds = ds.filter(lambda ex: ex.get("source", "").lower() != "math")
    print(f"过滤前: {original_count} 题")
    print(f"过滤后: {len(ds)} 题")
    print(f"已移除: {original_count - len(ds)} 题")
    
    # 查看剩余的source分布
    remaining_sources = Counter([ex.get("source", "unknown") for ex in ds])
    print(f"\n剩余数据源分布:")
    for src, cnt in remaining_sources.most_common(10):
        print(f"  {src}: {cnt}")
    
    print("\n" + "=" * 60)
    print("开始按难度分层抽样...")
    print("=" * 60)

    # 难度分层比例配置 (保持原始比例关系)
    # 目标: 让 Llama-3.1-8B 达到约50%通过率, GPT-4o 达到约75%通过率
    # 原始配置 (10k总量):
    # - 简单题 (0.8-1.0): 30%
    # - 中等偏简单 (0.6-0.8): 30%
    # - 中等偏难 (0.4-0.6): 20%
    # - 难题 (0.2-0.4): 14%
    # - 极难 (<0.2): 1.5%
    # 注: 剩余 4.5% 用于微调确保总数正确

    difficulty_ratios = {
        "easy": 0.30,           # 简单题
        "medium_easy": 0.30,    # 中等偏简单
        "medium_hard": 0.20,    # 中等偏难
        "hard": 0.14,           # 难题
        "very_hard": 0.015      # 极难题
    }

    # 计算每个难度层的目标样本数
    n_easy = int(total_samples * difficulty_ratios["easy"])
    n_medium_easy = int(total_samples * difficulty_ratios["medium_easy"])
    n_medium_hard = int(total_samples * difficulty_ratios["medium_hard"])
    n_hard = int(total_samples * difficulty_ratios["hard"])
    n_very_hard = int(total_samples * difficulty_ratios["very_hard"])

    # 微调确保总数正确
    current_total = n_easy + n_medium_easy + n_medium_hard + n_hard + n_very_hard
    diff = total_samples - current_total
    if diff > 0:
        # 补充到最大的类别（简单题）
        n_easy += diff
    elif diff < 0:
        # 从最大的类别削减
        n_easy += diff  # diff是负数

    print(f"\n难度层划分 (总计 {total_samples} 题):")
    print(f"  - 简单题 (0.8-1.0): {n_easy} 题 ({n_easy/total_samples*100:.1f}%)")
    print(f"  - 中等偏简单 (0.6-0.8): {n_medium_easy} 题 ({n_medium_easy/total_samples*100:.1f}%)")
    print(f"  - 中等偏难 (0.4-0.6): {n_medium_hard} 题 ({n_medium_hard/total_samples*100:.1f}%)")
    print(f"  - 难题 (0.2-0.4): {n_hard} 题 ({n_hard/total_samples*100:.1f}%)")
    print(f"  - 极难 (<0.2): {n_very_hard} 题 ({n_very_hard/total_samples*100:.1f}%)")
    
    print("\n1. 筛选简单题 (solve_rate: 0.8-1.0)")
    easy = select_by_difficulty(ds, 0.8, 1.01, "简单")
    
    print("\n2. 筛选中等偏简单题 (solve_rate: 0.6-0.8)")
    medium_easy = select_by_difficulty(ds, 0.6, 0.8, "中等偏简单")
    
    print("\n3. 筛选中等偏难题 (solve_rate: 0.4-0.6)")
    medium_hard = select_by_difficulty(ds, 0.4, 0.6, "中等偏难")
    
    print("\n4. 筛选难题 (solve_rate: 0.2-0.4)")
    hard = select_by_difficulty(ds, 0.2, 0.4, "难")
    
    print("\n5. 筛选极难题 (solve_rate: 0.0-0.2)")
    very_hard = select_by_difficulty(ds, 0.0, 0.2, "极难")
    
    # 分层抽样
    print("\n" + "=" * 60)
    print("执行分层抽样 (按 source 均衡)...")
    print("=" * 60)

    print(f"\n--- 简单题层 (目标: {n_easy}题) ---")
    easy_sampled = stratified_sample(easy, n_easy, "source")

    print(f"\n--- 中等偏简单层 (目标: {n_medium_easy}题) ---")
    medium_easy_sampled = stratified_sample(medium_easy, n_medium_easy, "source")

    print(f"\n--- 中等偏难层 (目标: {n_medium_hard}题) ---")
    medium_hard_sampled = stratified_sample(medium_hard, n_medium_hard, "source")

    print(f"\n--- 难题层 (目标: {n_hard}题) ---")
    hard_sampled = stratified_sample(hard, n_hard, "source")

    print(f"\n--- 极难层 (目标: {n_very_hard}题) ---")
    very_hard_sampled = stratified_sample(very_hard, n_very_hard, "source")
    
    # 合并所有层
    final_df = pd.concat([
        easy_sampled.assign(difficulty_tier="easy"),
        medium_easy_sampled.assign(difficulty_tier="medium_easy"),
        medium_hard_sampled.assign(difficulty_tier="medium_hard"),
        hard_sampled.assign(difficulty_tier="hard"),
        very_hard_sampled.assign(difficulty_tier="very_hard")
    ], ignore_index=True)
    
    # 重命名字段: problem -> instruction, solution -> response
    # 检查原始字段名
    print(f"\n原始字段: {list(final_df.columns)}")
    
    # 创建新的DataFrame,只保留需要的字段
    output_df = pd.DataFrame()
    
    # 根据Big-Math数据集的实际字段名映射
    # 通常是 'problem' 或 'question' 字段作为题目
    # 'solution' 或 'answer' 字段作为答案
    if 'problem' in final_df.columns:
        output_df['instruction'] = final_df['problem']
    elif 'question' in final_df.columns:
        output_df['instruction'] = final_df['question']
    else:
        # 如果都没有,打印所有字段帮助调试
        print("警告: 未找到problem或question字段")
        print(f"可用字段: {list(final_df.columns)}")
        # 暂时使用第一个文本字段
        text_cols = [col for col in final_df.columns if final_df[col].dtype == 'object']
        if text_cols:
            output_df['instruction'] = final_df[text_cols[0]]
    
    if 'solution' in final_df.columns:
        output_df['response'] = final_df['solution']
    elif 'answer' in final_df.columns:
        output_df['response'] = final_df['answer']
    else:
        print("警告: 未找到solution或answer字段")
        # 暂时使用第二个文本字段
        text_cols = [col for col in final_df.columns if final_df[col].dtype == 'object']
        if len(text_cols) > 1:
            output_df['response'] = final_df[text_cols[1]]
    
    # 保留一些元数据字段
    if 'llama8b_solve_rate' in final_df.columns:
        output_df['solve_rate'] = final_df['llama8b_solve_rate']
    output_df['difficulty_tier'] = final_df['difficulty_tier']
    if 'source' in final_df.columns:
        output_df['source'] = final_df['source']
    if 'domain' in final_df.columns:
        output_df['domain'] = final_df['domain']
    
    print("\n" + "=" * 60)
    print("最终 Benchmark 统计")
    print("=" * 60)
    print(f"总题数: {len(output_df)}")
    print(f"\n各难度层分布:")
    print(output_df['difficulty_tier'].value_counts())
    
    if 'source' in output_df.columns:
        print(f"\n各 source 分布:")
        print(output_df['source'].value_counts())
    
    if 'domain' in output_df.columns:
        print(f"\n各 domain 分布:")
        print(output_df['domain'].value_counts())
    
    # 保存结果
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 按难度层分层划分 train/test (80%/20%)
    print("\n" + "=" * 60)
    print("划分 Train/Test 集...")
    print("=" * 60)
    
    train_dfs = []
    test_dfs = []
    
    for tier in output_df['difficulty_tier'].unique():
        tier_df = output_df[output_df['difficulty_tier'] == tier]
        
        # 打乱顺序
        tier_df = tier_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 计算划分点 (使用传入的 train_ratio)
        n_total = len(tier_df)
        n_train = int(n_total * train_ratio)
        
        tier_train = tier_df.iloc[:n_train]
        tier_test = tier_df.iloc[n_train:]
        
        train_dfs.append(tier_train)
        test_dfs.append(tier_test)
        
        print(f"{tier}: {n_total}题 -> train {len(tier_train)}题, test {len(tier_test)}题")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # 打乱最终顺序
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n最终划分:")
    print(f"Train: {len(train_df)}题")
    print(f"Test: {len(test_df)}题")
    
    print(f"\nTrain集难度分布:")
    print(train_df['difficulty_tier'].value_counts().sort_index())
    
    print(f"\nTest集难度分布:")
    print(test_df['difficulty_tier'].value_counts().sort_index())
    
    # 动态生成文件名 (根据实际样本数)
    n_train_samples = len(train_df)
    n_test_samples = len(test_df)
    n_total_samples = len(output_df)

    # 保存train集
    train_file = os.path.join(output_dir, f"big_math_train_{n_train_samples//1000}k.jsonl")
    train_df.to_json(train_file, orient='records', lines=True, force_ascii=False)
    print(f"\n✓ Train数据已保存至: {train_file}")

    # 保存test集
    test_file = os.path.join(output_dir, f"big_math_test_{n_test_samples//1000}k.jsonl")
    test_df.to_json(test_file, orient='records', lines=True, force_ascii=False)
    print(f"✓ Test数据已保存至: {test_file}")

    # 同时保存完整数据(可选,用于参考)
    output_file = os.path.join(output_dir, f"big_math_{n_total_samples//1000}k.jsonl")
    output_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"✓ 完整数据已保存至: {output_file}")
    
    # 生成统计报告
    report_file = os.path.join(output_dir, "big_math_sampling_report.txt")
    with open(report_file, 'w') as f:
        f.write("Big-Math Benchmark 抽样报告 (10k, Train/Test划分)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("总体统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"总题数: {len(output_df)}\n")
        f.write(f"Train集: {len(train_df)}题 (80%)\n")
        f.write(f"Test集: {len(test_df)}题 (20%)\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("完整数据集难度分布\n")
        f.write("=" * 60 + "\n")
        f.write(str(output_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Train集难度分布\n")
        f.write("=" * 60 + "\n")
        f.write(str(train_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Test集难度分布\n")
        f.write("=" * 60 + "\n")
        f.write(str(test_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        if 'source' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Source分布 (完整数据)\n")
            f.write("=" * 60 + "\n")
            f.write(str(output_df['source'].value_counts()) + "\n\n")
        
        if 'domain' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Domain分布 (完整数据)\n")
            f.write("=" * 60 + "\n")
            f.write(str(output_df['domain'].value_counts()) + "\n\n")
        
        if 'solve_rate' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Solve Rate 统计\n")
            f.write("=" * 60 + "\n")
            f.write(f"完整数据:\n")
            f.write(f"  最小值: {output_df['solve_rate'].min():.3f}\n")
            f.write(f"  最大值: {output_df['solve_rate'].max():.3f}\n")
            f.write(f"  平均值: {output_df['solve_rate'].mean():.3f}\n")
            f.write(f"  中位数: {output_df['solve_rate'].median():.3f}\n\n")
            
            f.write(f"Train集:\n")
            f.write(f"  最小值: {train_df['solve_rate'].min():.3f}\n")
            f.write(f"  最大值: {train_df['solve_rate'].max():.3f}\n")
            f.write(f"  平均值: {train_df['solve_rate'].mean():.3f}\n")
            f.write(f"  中位数: {train_df['solve_rate'].median():.3f}\n\n")
            
            f.write(f"Test集:\n")
            f.write(f"  最小值: {test_df['solve_rate'].min():.3f}\n")
            f.write(f"  最大值: {test_df['solve_rate'].max():.3f}\n")
            f.write(f"  平均值: {test_df['solve_rate'].mean():.3f}\n")
            f.write(f"  中位数: {test_df['solve_rate'].median():.3f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("预期模型性能\n")
        f.write("=" * 60 + "\n")
        f.write("基于难度分布的预期准确率:\n\n")
        f.write("Llama-3.1-8B:\n")
        f.write("  - 简单题 (0.8-1.0): ~70%\n")
        f.write("  - 中等偏简单 (0.6-0.8): ~50%\n")
        f.write("  - 中等偏难 (0.4-0.6): ~32%\n")
        f.write("  - 难题 (0.2-0.4): ~18%\n")
        f.write("  - 极难 (<0.2): ~8%\n")
        f.write("  - 整体预期: ~45-50%\n\n")
        f.write("GPT-4o:\n")
        f.write("  - 简单题 (0.8-1.0): ~90%\n")
        f.write("  - 中等偏简单 (0.6-0.8): ~80%\n")
        f.write("  - 中等偏难 (0.4-0.6): ~65%\n")
        f.write("  - 难题 (0.2-0.4): ~45%\n")
        f.write("  - 极难 (<0.2): ~25%\n")
        f.write("  - 整体预期: ~70-75%\n\n")
        f.write("预期差距: ~25-30%\n")
    
    print(f"✓ 统计报告已保存至: {report_file}")
    
    print("\n" + "=" * 60)
    print("✓ 全部完成!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Big-Math 难度分层抽样脚本")
    parser.add_argument(
        "--total_samples",
        type=int,
        default=10000,
        help="总共需要的样本数量 (默认: 10000)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8, 即 80%% train, 20%% test)"
    )

    args = parser.parse_args()

    # 验证参数
    if args.total_samples <= 0:
        raise ValueError("total_samples 必须大于 0")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio 必须在 (0, 1) 范围内")

    main(total_samples=args.total_samples, train_ratio=args.train_ratio)