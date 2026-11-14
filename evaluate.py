#!/usr/bin/env python3
"""在与检查点绑定的验证集上计算总 token 与正文 token 的平均 NLL。"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from generate import load_model
from train import digest, now


def evaluate(model, tokens, batch_size: int, max_batches: int = 0):
    if len(tokens) < 2:
        raise ValueError("验证集至少需要两个 token")
    length = model.config.context_length
    capacity = batch_size * length
    total_nll = content_nll = 0.0
    total_count = content_count = batches = 0
    offset = 0
    started = time.perf_counter()
    while offset < len(tokens) - 1 and (max_batches == 0 or batches < max_batches):
        count = min(capacity, len(tokens) - 1 - offset)
        x = np.zeros(capacity, dtype=np.int32)
        y = np.zeros(capacity, dtype=np.int32)
        x[:count] = tokens[offset:offset + count]
        y[:count] = tokens[offset + 1:offset + count + 1]
        valid = np.arange(capacity) < count
        content = valid & (y >= 4)
        losses = nn.losses.cross_entropy(
            model(mx.array(x.reshape(batch_size, length))),
            mx.array(y.reshape(batch_size, length)), reduction="none"
        ).reshape(-1)
        total_sum = mx.sum(losses * mx.array(valid.astype(np.float32)))
        content_sum = mx.sum(losses * mx.array(content.astype(np.float32)))
        mx.eval(total_sum, content_sum)
        total_nll += float(total_sum.item())
        content_nll += float(content_sum.item())
        total_count += count
        content_count += int(content.sum())
        offset += count
        batches += 1
        if batches % 100 == 0:
            print(f"已评估 {total_count:,} 个目标 token", flush=True)
    if content_count == 0:
        raise ValueError("验证集没有正文目标 token")
    return {
        "all_token_nll": total_nll / total_count,
        "content_token_nll": content_nll / content_count,
        "all_target_tokens": total_count,
        "content_target_tokens": content_count,
        "batches": batches,
        "seconds": time.perf_counter() - started,
        "complete": offset == len(tokens) - 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="0 表示完整验证集")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_batches < 0:
        parser.error("batch-size 必须为正，max-batches 必须非负")
    tokenizer_path = args.data_dir / "tokenizer.json"
    validation_path = args.data_dir / "val.bin"
    model, _, info = load_model(args.checkpoint, tokenizer_path)
    validation_hash = digest(validation_path)
    if validation_hash != info["sources"]["validation_sha256"]:
        raise ValueError("验证文件与检查点记录的数据不一致")
    tokens = np.memmap(validation_path, dtype="<u2", mode="r")
    result = evaluate(model, tokens, args.batch_size, args.max_batches)
    report = {
        "time": now(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_step": info["step"],
        "validation_sha256": validation_hash,
        "context_length": model.config.context_length,
        "batch_size": args.batch_size,
        "device": str(mx.default_device()),
        "description": "验证流连续分块，预测下一 token；正文 NLL 排除 PAD/BOS/EOS/UNK 目标。",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
