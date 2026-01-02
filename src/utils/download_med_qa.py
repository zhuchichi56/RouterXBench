#!/usr/bin/env python3
"""
从 Hugging Face 下载 MedQA 数据集并转换为指定格式
使用方法:
    python download_med_qa.py
"""
import json
import argparse
from pathlib import Path
import sys


def download_and_extract_medqa(num_samples: int = 1000):
    """
    从 Hugging Face 下载 MedQA 数据集并转换为指定格式

    Args:
        num_samples: 要提取的数据数量，默认 1000
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        sys.exit(1)

    print(f"正在加载 MedQA 数据集（更兼容的加载方式）...")
    print(f"目标: 提取 {num_samples} 条数据")

    try:
        # 优先尝试从本地 parquet 文件加载（如果在本机上存在）
        local_parquet_dir = Path("/data1/wwx/02.dataset/med_qa/med_qa_en_source")
        split_name = "test"
        if local_parquet_dir.exists():
            split_file_map = {
                "train": "train-00000-of-00001.parquet",
                "test": "test-00000-of-00001.parquet",
                "validation": "validation-00000-of-00001.parquet"
            }
            file_path = local_parquet_dir / split_file_map.get(split_name, "test-00000-of-00001.parquet")
            print(f"从本地 parquet 文件加载: {file_path}")
            dataset = load_dataset("parquet", data_files=str(file_path), split="train")
        else:
            # 回退：尝试从 Hugging Face Hub 加载（如果可用）
            print("本地 parquet 文件未找到，尝试从 Hugging Face Hub 加载（config: med_qa_en_bigbio_qa）")
            dataset = load_dataset("med_qa", "med_qa_en_bigbio_qa", split=split_name)

        total_available = len(dataset)
        print(f"数据集总共有 {total_available} 条数据")

        if num_samples > total_available:
            print(f"警告: 请求的数据量 ({num_samples}) 超过了可用数据量 ({total_available})")
            print(f"将提取所有 {total_available} 条数据")
            num_samples = total_available

        # 提取并转换数据（兼容多种字段格式）
        converted_data = []
        for i in range(num_samples):
            item = dataset[i]

            # 兼容不同 schema 的字段名
            question = item.get("question") or item.get("query") or item.get("text") or ""

            # 可能的选项字段: 'options' (list of dicts or list of strings) 或 'choices' (list of strings)
            raw_options = item.get("options") if item.get("options") is not None else item.get("choices")
            choices = []
            option_labels = []

            if raw_options is None:
                raw_options = []

            if isinstance(raw_options, list) and len(raw_options) > 0 and isinstance(raw_options[0], dict):
                # 每个选项是 dict，尝试读取 'value'/'text' 和 'key'
                for opt in raw_options:
                    val = opt.get("value") or opt.get("text") or ""
                    key = opt.get("key")
                    choices.append(val)
                    option_labels.append(key)
            elif isinstance(raw_options, list):
                # 列表/字符串
                choices = [str(x) for x in raw_options]
                option_labels = [None] * len(choices)
            else:
                # 兼容旧格式
                choices = []
                option_labels = []

            # 生成默认标签 A, B, C...
            alpha_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for idx in range(len(choices)):
                if not option_labels[idx]:
                    option_labels[idx] = alpha_labels[idx] if idx < len(alpha_labels) else str(idx)

            # 查找正确答案（多种可能字段）
            correct_idx = None
            # 优先查看 answer_idx（常见于本项目 parquet）
            answer_idx = item.get("answer_idx")
            answer_field = item.get("answer")

            def label_to_index(lbl):
                if lbl is None:
                    return None
                lbls = str(lbl).strip()
                # 直接数字
                if lbls.isdigit():
                    return int(lbls)
                # 匹配字母标签
                for j, lab in enumerate(option_labels):
                    if lab and lbls.lower() == str(lab).lower():
                        return j
                return None

            if isinstance(answer_idx, int):
                correct_idx = answer_idx
            elif isinstance(answer_idx, str):
                correct_idx = label_to_index(answer_idx)
            elif isinstance(answer_idx, list) and len(answer_idx) > 0:
                first = answer_idx[0]
                if isinstance(first, int):
                    correct_idx = first
                else:
                    correct_idx = label_to_index(first)
            elif isinstance(answer_field, list) and len(answer_field) > 0:
                first = answer_field[0]
                if isinstance(first, int):
                    correct_idx = first
                else:
                    correct_idx = label_to_index(first)
            elif isinstance(answer_field, (int, str)):
                correct_idx = label_to_index(answer_field)

            if correct_idx is None or correct_idx < 0 or correct_idx >= len(choices):
                print(f"警告: 第 {i} 条数据没有可用或有效的答案索引，跳过")
                continue

            options_text = [f"{option_labels[idx]}. {choices[idx]}" for idx in range(len(choices))]

            # 构建 instruction
            instruction = (
                f"The following is a multiple choice question about medical knowledge. "
                f"Please select the correct answer.\n\n"
                f"Question: {question}\n\n"
                f"Options:\n" + "\n".join(options_text) + "\n\n"
                f"Please choose the correct answer from {', '.join(option_labels[:len(choices)])}."
            )

            correct_answer = option_labels[correct_idx]
            correct_text = choices[correct_idx]
            response = f"{correct_answer}. {correct_text}"

            converted_item = {
                "instruction": instruction,
                "response": response
            }

            converted_data.append(converted_item)

        # 保存数据
        script_dir = Path(__file__).parent
        output_file = script_dir.parent / "data" / "med_qa_1k.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n成功提取 {len(converted_data)} 条数据")
        print(f"保存到: {output_file}")

        # 显示统计信息
        print(f"\n数据统计:")
        print(f"  总条数: {len(converted_data)}")
        print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 显示前2个样本
        print(f"\n前2个样本预览:")
        for i, item in enumerate(converted_data[:2], 1):
            print(f"\n[{i}]")
            print(f"  instruction: {item['instruction'][:200]}...")
            print(f"  response: {item['response']}")

        return True

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='从 Hugging Face 下载 MedQA 数据集并转换为指定格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_med_qa.py           # 下载 1000 条数据（默认）
  python download_med_qa.py -n 2000   # 下载 2000 条数据
        """
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=1000,
        help='要提取的数据数量（默认: 1000）'
    )

    args = parser.parse_args()

    # 验证输入
    if args.num_samples <= 0:
        print("错误: 数据数量必须大于 0")
        sys.exit(1)

    # 执行下载和提取
    success = download_and_extract_medqa(args.num_samples)

    if success:
        print("\n✓ 完成!")
    else:
        print("\n✗ 失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
