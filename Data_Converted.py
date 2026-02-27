import json
import csv
import os
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))

input_files = [
    os.path.join(base_dir, "pulmobench_test.jsonl"),
    os.path.join(base_dir, "pulmobench_train.jsonl"),
    os.path.join(base_dir, "pulmobench_val.jsonl")
]

output_file = os.path.join(base_dir, "pulmobench_combined.csv")

def count_lines(files):
    total = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total

total_lines = count_lines(input_files)

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = None
    with tqdm(total=total_lines, desc="Processing", unit="lines") as pbar:
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        pbar.update(1)
                        continue
                    data = json.loads(line)
                    if isinstance(data.get("gold_differential"), list):
                        data["gold_differential"] = ",".join(data["gold_differential"])
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                        writer.writeheader()
                    writer.writerow(data)
                    pbar.update(1)

print("Done. Output written to", output_file)