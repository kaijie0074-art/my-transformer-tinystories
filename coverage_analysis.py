#!/usr/bin/env python3
"""按实际配置和显式分叉种子重放采样，统计训练 token 位置覆盖。"""

import argparse
import json
from pathlib import Path

import numpy as np


def union_length(starts, width):
    if len(starts) == 0:
        return 0
    starts = np.sort(starts)
    left = int(starts[0])
    right = left + width
    total = 0
    for value in starts[1:]:
        value = int(value)
        if value > right:
            total += right - left
            left, right = value, value + width
        else:
            right = max(right, value + width)
    return total + right - left


def analyze(run_dir: Path):
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    config_event = next(event for event in events if event["event"] == "config")
    cfg = config_event["config"]
    parent = config_event["parent_checkpoint"]
    if parent and parent.get("sampling_rng") != "new_seed":
        raise ValueError("该分叉继承父随机状态，不能使用种子直接重放")
    train_events = [event for event in events if event["event"] == "train"]
    steps = [event["step"] for event in train_events]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("训练日志步数不连续，不能可靠重放")
    manifest = json.loads((Path(cfg["data_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    pool = cfg["pool_tokens"] or manifest["train"]["tokens"]
    width = cfg["context_length"]
    draws = len(steps) * cfg["grad_accum"]
    starts = np.empty(draws * cfg["batch_size"], dtype=np.int64)
    rng = np.random.default_rng(cfg["seed"])
    for draw in range(draws):
        offset = draw * cfg["batch_size"]
        starts[offset:offset + cfg["batch_size"]] = rng.integers(
            0, pool - width, size=cfg["batch_size"]
        )
    unique = union_length(starts, width)
    processed = len(starts) * width
    if train_events and train_events[-1]["tokens"] != processed:
        raise ValueError("采样重放与日志中的 token 数不一致")
    return {
        "run": str(run_dir.resolve()),
        "stage_steps": len(steps),
        "seed": cfg["seed"],
        "training_pool_token_positions": pool,
        "processed_input_token_positions": processed,
        "distinct_input_token_positions": unique,
        "repeated_input_token_positions": processed - unique,
        "coverage_fraction": unique / pool,
        "sampling": "按 NumPy 生成器及已记录的显式种子逐微批次重放窗口起点",
        "note": "位置数不是去重的故事数或不同词语数。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
