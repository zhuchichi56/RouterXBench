import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from tqdm.auto import tqdm

def write_jsonl(records: Iterable[Dict[str, Any]], file_name: str) -> None:
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_jsonl(file_name: str) -> List[Dict[str, Any]]:
    path = Path(file_name)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_endpoints():
    gpt_4o = [
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://conversationhubwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinsouthcentralus.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        # {
        #     "endpoints": "https://conversationhubwestus.openai.azure.com/",
        #     "speed": 450,
        #     "model": "gpt-4o-global"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
    ]

    gpt_4o_mini = [
        # https://conversationhubeastus.openai.azure.com/
        # https://conversationhubeastus2.openai.azure.com/
        # https://conversationhubnorthcentralus.openai.azure.com/
        # https://conversationhubsouthcentralus.openai.azure.com/
        # https://conversationhubswedencentral.openai.azure.com/
        # https://conversationhubwestus.openai.azure.com/
        # https://readineastus.openai.azure.com/
        # https://readineastus2.openai.azure.com/
        # https://readinnorthcentralus.openai.azure.com/
        # https://readinwestus.openai.azure.com/
        # https://malicata-azure-ai-foundry.cognitiveservices.azure.com/
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://malicata-azure-ai-foundry.cognitiveservices.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },

    ]

    gpt_4_turbo = [
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://readinswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
    ]

    gpt_4_1 = [
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
    ]

    gpt_5 = [
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-5-global"
        },
    ]

    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-4.1": gpt_4_1,
        "gpt-5": gpt_5,
    }


@dataclass
class Stats:
    ok: int = 0
    api_errors: int = 0

    def postfix(self) -> Dict[str, int]:
        return {"ok": self.ok, "api_err": self.api_errors}


def refresh_progress(progress, stats: Stats) -> None:
    if progress:
        progress.update(1)
        progress.set_postfix(stats.postfix())


def select_endpoint(model_name: str) -> Dict[str, Any]:
    azure_endpoints = get_endpoints()
    entries = azure_endpoints[model_name]
    candidates = [e for e in entries if e.get("speed", 0) > 0 and e.get("endpoints")]
    weights = [e["speed"] for e in candidates]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen


def get_client(
    model_name: str = "gpt-4o",
    tenant_id: str = "72f988bf-86f1-41af-91ab-2d7cd011db47",
    api_version: str = "2024-12-01-preview",
    max_retries: int = 5,
    credential: Optional[AzureCliCredential] = None,
) -> Tuple[AzureOpenAI, str]:
    credential = credential or AzureCliCredential(tenant_id=tenant_id)
    azure_ad_token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )
    selected = select_endpoint(model_name)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoints"],
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version,
        max_retries=max_retries,
    )
    return client, selected["model"]


def get_response(
    client: AzureOpenAI,
    resolved_model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> str:
    if "gpt-5" in resolved_model:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
        )
    else:
        response = client.chat.completions.create(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
        )
    content = response.choices[0].message.content
    return content


def build_messages(instruction: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    return messages


def load_dataset_with_indices(input_file: str) -> List[Dict[str, Any]]:
    dataset = read_jsonl(input_file)
    for idx, item in enumerate(dataset):
        assert "instruction" in item
        assert "response" in item
        item["index"] = idx
    return dataset


def merge_existing_results(
    dataset: List[Dict[str, Any]],
    output_file: str,
) -> Tuple[Set[int], Stats]:
    if not Path(output_file).exists():
        return set(), Stats()

    existing_entries = read_jsonl(output_file)
    processed_indices: Set[int] = set()
    stats = Stats()

    for entry in existing_entries:
        idx = entry.get("index")
        if idx is None or not isinstance(idx, int):
            continue
        if 0 <= idx < len(dataset):
            dataset[idx].update(entry)
            processed_indices.add(idx)
            if entry.get("large_response"):
                stats.ok += 1
    return processed_indices, stats


async def async_write_jsonl(dataset: List[Dict[str, Any]], output_file: str, write_lock: asyncio.Lock) -> None:
    async with write_lock:
        await asyncio.to_thread(write_jsonl, dataset, output_file)


async def process_single_item(
    idx: int,
    dataset: List[Dict[str, Any]],
    model_name: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    credential: AzureCliCredential,
    semaphore: asyncio.Semaphore,
    output_file: str,
    write_lock: asyncio.Lock,
    progress,
    stats: Stats,
) -> None:
    async with semaphore:
        record = dataset[idx]
        instruction = record.get("instruction")
        if not instruction:
            print(f"[SKIP] index {idx}: missing instruction")
            refresh_progress(progress, stats)
            return

        success = False
        api_called = False
        try:
            client, resolved_model = get_client(model_name=model_name, credential=credential)
            messages = build_messages(instruction, system_prompt)
            api_called = True
            response = await asyncio.to_thread(
                get_response,
                client,
                resolved_model,
                messages,
                max_tokens,
                temperature,
                top_p,
            )
            record["large_response"] = response
            record.pop("last_error", None)
            record.pop("last_error_type", None)
            success = True
        except Exception as exc:  # noqa: BLE001
            if api_called:
                print(f"[API ERROR] index {idx}: {exc}")
                stats.api_errors += 1
                refresh_progress(progress, stats)
                return
            raise

        await async_write_jsonl(dataset, output_file, write_lock)
        if success:
            stats.ok += 1
        refresh_progress(progress, stats)


async def run_inference(
    input_file: str,
    output_file: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 1.0,
    system_prompt: Optional[str] = "You are a helpful AI.",
    concurrency: int = 150,
) -> None:
    dataset = load_dataset_with_indices(input_file)

    existing_indices, stats = merge_existing_results(dataset, output_file)

    pending_indices = [idx for idx in range(len(dataset)) if idx not in existing_indices or not dataset[idx].get("large_response")]
    if not pending_indices:
        if tqdm:
            bar = tqdm(total=0, desc="Processing", unit="item")
            bar.set_postfix(stats.postfix())
            bar.close()
        print("All records already processed.")
        return

    total_tasks = len(pending_indices)

    bar = tqdm(total=total_tasks, desc="Processing", unit="item") if tqdm else None
    if bar:
        bar.set_postfix(stats.postfix())

    semaphore = asyncio.Semaphore(min(concurrency, total_tasks))
    write_lock = asyncio.Lock()
    credential = AzureCliCredential()

    tasks = [
        asyncio.create_task(
            process_single_item(
                idx,
                dataset,
                model_name,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                credential,
                semaphore,
                output_file,
                write_lock,
                bar,
                stats,
            )
        )
        for idx in pending_indices
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        if bar:
            bar.close()


def main(
    input_file: str,
    output_file: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 1.0,
    system_prompt: Optional[str] = "You are a helpful AI.",
    concurrency: int = 150,
) -> None:
    asyncio.run(
        run_inference(
            input_file=input_file,
            output_file=output_file,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            concurrency=concurrency,
        )
    )

def check_api_error_indices(input_file: str, output_file: str) -> None:
    dataset = load_dataset_with_indices(input_file)
    existing_indices, stats = merge_existing_results(dataset, output_file)
    missing_indices = [idx for idx in range(len(dataset)) if idx not in existing_indices or not dataset[idx].get("large_response")]
    print(f"Total items: {len(dataset)}")
    print(f"Total processed items: {len(existing_indices)}")
    print(f"Total missing items: {len(missing_indices)}")
    if missing_indices:
        print(f"Missing indices: {missing_indices}")



if __name__ == "__main__":
    input_files_normal = [
        "src/data/alpaca_5k_test.jsonl",
        "src/data/magpie_5k_test.jsonl",
    ]

    input_files_box = [
        "src/data/numina_cot_5k_test.jsonl",
        "src/data/mmlu_test.jsonl",
        "src/data/mmlu_train.jsonl",
        "src/data/mmlu_pro/biology_converted.jsonl",
        "src/data/mmlu_pro/business_converted.jsonl",
        "src/data/mmlu_pro/chemistry_converted.jsonl",
        "src/data/mmlu_pro/computer_science_converted.jsonl",
        "src/data/mmlu_pro/economics_converted.jsonl",
        "src/data/mmlu_pro/engineering_converted.jsonl",
        "src/data/mmlu_pro/health_converted.jsonl",
        "src/data/mmlu_pro/history_converted.jsonl",
        "src/data/mmlu_pro/law_converted.jsonl",
        "src/data/mmlu_pro/math_converted.jsonl",
        "src/data/mmlu_pro/other_converted.jsonl",
        "src/data/mmlu_pro/philosophy_converted.jsonl",
        "src/data/mmlu_pro/physics_converted.jsonl",
        "src/data/mmlu_pro/psychology_converted.jsonl",
        "src/data/math.jsonl",
        "src/data/big_math_5k_train.jsonl",
    ]

    data = [
        {
            "input_files": input_files_normal,
            "output_root": "outputs_normal_gpt5",
            "system_prompt": "You are a helpful AI.",
        },
        {
            "input_files": input_files_box,
            "output_root": "outputs_box_gpt5",
            "system_prompt": "Please reason step by step, and put your final answer within \\boxed{{}}",
        },
    ]


    for item in data:
        input_files = item["input_files"]
        output_root = item["output_root"]
        system_prompt = item["system_prompt"]

        for input_file in input_files:
            print(f"\n\n#### Processing {input_file} ####")
            for run in range(1):
                output_file = os.path.join(output_root, input_file.replace(".jsonl", f"_run{run}.jsonl"))
                main(
                    input_file=input_file,
                    output_file=output_file,
                    # model_name="gpt-4o",
                    # max_tokens=2048,
                    # temperature=0.7,
                    # top_p=1.0,
                    model_name="gpt-5",
                    system_prompt=system_prompt,
                    concurrency=150,
                )