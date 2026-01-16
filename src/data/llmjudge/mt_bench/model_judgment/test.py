import json

def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        data = []
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line {line_num}: {line}")
                print(f"Error message: {e}")
                continue  # Skip the line with error
        return data

# Example: Replace with your actual data file path
# data = load_jsonl("/path/to/your/data/gpt-4_single.jsonl")
# print(data[:2])