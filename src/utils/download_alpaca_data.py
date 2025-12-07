#!/usr/bin/env python3
"""
从 Hugging Face 下载原始 Alpaca 数据集并提取指定数量的数据
使用方法:
    python download_alpaca_data.py 10  # 下载 10k 条数据
    python download_alpaca_data.py 20  # 下载 20k 条数据
"""
import json
import argparse
from pathlib import Path
import sys


def download_and_extract_alpaca(num_k: int):
    """
    从 Hugging Face 下载 Alpaca 数据集并提取指定数量的数据

    Args:
        num_k: 要提取的数据数量（以千为单位）
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        sys.exit(1)

    num_samples = num_k * 1000

    print(f"正在从 Hugging Face 下载 Alpaca 数据集...")
    print(f"目标: 提取 {num_samples} 条数据")

    try:
        # 下载 alpaca 数据集
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

        total_available = len(dataset)
        print(f"数据集总共有 {total_available} 条数据")

        if num_samples > total_available:
            print(f"警告: 请求的数据量 ({num_samples}) 超过了可用数据量 ({total_available})")
            print(f"将提取所有 {total_available} 条数据")
            num_samples = total_available

        # 提取指定数量的数据
        extracted_data = []
        for i in range(num_samples):
            item = dataset[i]

            # 转换为目标格式
            converted_item = {
                "instruction": item.get("instruction", ""),
                "response": item.get("output", "")
            }

            # 如果有 input 字段且不为空，将其添加到 instruction 中
            if item.get("input", "").strip():
                converted_item["instruction"] = f"{item['instruction']}\n{item['input']}"

            extracted_data.append(converted_item)

        # 保存数据
        script_dir = Path(__file__).parent
        output_file = script_dir.parent / "data" / f"alpaca_{num_k}k.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in extracted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n成功提取 {len(extracted_data)} 条数据")
        print(f"保存到: {output_file}")

        # 显示统计信息
        print(f"\n数据统计:")
        print(f"  总条数: {len(extracted_data)}")
        print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 显示前3个样本
        print(f"\n前3个样本预览:")
        for i, item in enumerate(extracted_data[:3], 1):
            print(f"\n[{i}]")
            print(f"  instruction: {item['instruction'][:100]}...")
            print(f"  response: {item['response'][:100]}...")

        return True

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='从 Hugging Face 下载 Alpaca 数据集并提取指定数量的数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_alpaca_data.py 10   # 下载 10k 条数据
  python download_alpaca_data.py 20   # 下载 20k 条数据
  python download_alpaca_data.py 52   # 下载全部数据（约52k条）
        """
    )
    parser.add_argument(
        'num_k',
        type=int,
        help='要提取的数据数量（以千为单位，例如: 10 表示 10k 条数据）'
    )

    args = parser.parse_args()

    # 验证输入
    if args.num_k <= 0:
        print("错误: 数据数量必须大于 0")
        sys.exit(1)

    # 执行下载和提取
    success = download_and_extract_alpaca(args.num_k)

    if success:
        print("\n✓ 完成!")
    else:
        print("\n✗ 失败")
        sys.exit(1)


if __name__ == '__main__':
    main()

