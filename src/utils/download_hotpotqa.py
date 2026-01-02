"""
下载并处理 HotpotQA 数据集（适合测试搜索Agent）

HotpotQA 特点：
- 需要多跳推理（从多个来源综合信息）
- 问题设计为需要搜索才能回答
- 有明确的答案和支持事实

使用示例：
    python src/utils/download_hotpotqa.py --num_samples 500
"""

import json
from pathlib import Path
from datasets import load_dataset
import argparse


def download_and_process_hotpotqa(num_samples: int = 500, split: str = "validation"):
    """
    下载并处理 HotpotQA 数据集

    Args:
        num_samples: 要处理的样本数量
        split: 数据集分割 ("train" 或 "validation")
    """
    print(f"正在下载 HotpotQA 数据集 ({split} split)...")

    try:
        # 加载 HotpotQA 数据集
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

        print(f"数据集加载成功，共 {len(dataset)} 条数据")
        print(f"将处理前 {num_samples} 条数据")

        # 准备输出
        output_data = []

        for idx, item in enumerate(dataset):
            if idx >= num_samples:
                break

            # HotpotQA 格式：
            # - question: 问题
            # - answer: 答案
            # - type: 问题类型 (comparison 或 bridge)
            # - level: 难度级别 (easy, medium, hard)
            # - supporting_facts: 支持性事实

            question = item['question']
            answer = item['answer']
            question_type = item['type']
            level = item['level']

            # 格式化为统一格式
            formatted_item = {
                "id": f"hotpotqa_{idx}",
                "instruction": f"Answer the following question. You may need to search for information from multiple sources.\n\nQuestion: {question}",
                "response": answer,
                "metadata": {
                    "type": question_type,
                    "level": level,
                    "dataset": "hotpotqa"
                }
            }

            output_data.append(formatted_item)

            # 每100条打印进度
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1} 条数据...")

        # 保存到文件
        script_dir = Path(__file__).parent
        output_file = script_dir.parent / "data" / f"hotpotqa_{num_samples}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n✅ 成功！数据已保存到: {output_file}")
        print(f"   总条数: {len(output_data)}")

        # 打印几个示例
        print("\n示例问题：")
        for i in range(min(3, len(output_data))):
            item = output_data[i]
            print(f"\n{i+1}. {item['instruction'][:150]}...")
            print(f"   答案: {item['response']}")
            print(f"   类型: {item['metadata']['type']}, 难度: {item['metadata']['level']}")

        return True

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='下载并处理 HotpotQA 数据集（适合测试搜索Agent）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_hotpotqa.py              # 下载 500 条验证集数据（默认）
  python download_hotpotqa.py -n 1000      # 下载 1000 条数据
  python download_hotpotqa.py -s train     # 下载训练集数据
        """
    )

    parser.add_argument(
        '-n', '--num_samples',
        type=int,
        default=500,
        help='要下载的样本数量（默认: 500）'
    )

    parser.add_argument(
        '-s', '--split',
        type=str,
        default='validation',
        choices=['train', 'validation'],
        help='数据集分割（默认: validation）'
    )

    args = parser.parse_args()

    print("="*60)
    print("HotpotQA 数据集下载工具")
    print("="*60)
    print(f"配置:")
    print(f"  样本数量: {args.num_samples}")
    print(f"  数据分割: {args.split}")
    print("="*60)

    success = download_and_process_hotpotqa(args.num_samples, args.split)

    if success:
        print("\n" + "="*60)
        print("✅ 下载完成！")
        print("="*60)
        print("\n下一步:")
        print("1. 启动 vLLM 服务器")
        print("2. 运行搜索Agent测试:")
        print(f"""
python src/agent.py \\
    --agent_name /data1/wwx/models/Qwen3-8B \\
    --dataset hotpotqa_{args.num_samples} \\
    --agent_tools DuckDuckGoSearchTool \\
    --template_type search_guided \\
    --max_steps 15 \\
    --concurrent_limit 20 \\
    --use_openai_server \\
    --api_base "http://localhost:8002/v1"
        """)
    else:
        print("\n❌ 下载失败，请检查错误信息")
