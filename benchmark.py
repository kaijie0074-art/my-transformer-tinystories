#!/usr/bin/env python3
"""比较自回归推理中的完整输出投影与最后位置投影。"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from generate import load_model
from train import digest, now


def logits(model, ids, last_only):
    tokens = mx.array([ids[-model.config.context_length:]], dtype=mx.int32)
    out = model(tokens, last_token_only=last_only)
    result = out[:, -1, :]
    mx.eval(result)
    return result


def greedy_tokens(model, prefix, count, last_only):
    ids = list(prefix)
    added = []
    for _ in range(count):
        token = int(mx.argmax(logits(model, ids, last_only), axis=-1).item())
        ids.append(token)
        added.append(token)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation-bin", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.context <= 0 or args.new_tokens <= 0 or args.warmup < 0 or args.trials <= 0:
        parser.error("context、新 token 数和测量次数必须为正，预热次数必须非负")
    model, tokenizer, info = load_model(args.checkpoint, args.tokenizer)
    if args.context > model.config.context_length:
        parser.error("context 不能超过模型最大上下文长度")
    validation_sha256 = digest(args.validation_bin)
    if validation_sha256 != info["sources"]["validation_sha256"]:
        raise ValueError("验证数据与检查点训练时记录的数据不一致")
    prefix = np.fromfile(args.validation_bin, dtype="<u2", count=args.context).astype(np.int32).tolist()
    if len(prefix) != args.context:
        parser.error("验证数据不足以提供指定长度的上下文")
    baseline = np.asarray(logits(model, prefix, False))
    optimized = np.asarray(logits(model, prefix, True))
    max_abs = float(np.max(np.abs(baseline - optimized)))
    close = bool(np.allclose(baseline, optimized, rtol=1e-4, atol=1e-4))
    baseline_ids = greedy_tokens(model, prefix, args.new_tokens, False)
    optimized_ids = greedy_tokens(model, prefix, args.new_tokens, True)
    for _ in range(args.warmup):
        greedy_tokens(model, prefix, args.new_tokens, False)
        greedy_tokens(model, prefix, args.new_tokens, True)
    old_times, new_times = [], []
    for trial in range(args.trials):
        # 交替先后次序，减少固定顺序引入的缓存与温度偏差。
        for last_only in ((False, True) if trial % 2 == 0 else (True, False)):
            started = time.perf_counter()
            greedy_tokens(model, prefix, args.new_tokens, last_only)
            (new_times if last_only else old_times).append(time.perf_counter() - started)
    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)
    report = {
        "time": now(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_step": info["step"],
        "parameter_count": model.count_parameters(),
        "device": str(mx.default_device()),
        "validation_bin": str(args.validation_bin.resolve()),
        "validation_sha256": validation_sha256,
        "prefix_offset_tokens": 0,
        "context": args.context,
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "trials": args.trials,
        "comparison": "greedy generation: full output projection versus last-position output projection",
        "max_abs_logits_difference": max_abs,
        "logits_allclose_rtol_atol_1e-4": close,
        "identical_generated_token_ids": baseline_ids == optimized_ids,
        "baseline_token_ids": baseline_ids,
        "optimized_token_ids": optimized_ids,
        "baseline_median_seconds": old_median,
        "optimized_median_seconds": new_median,
        "baseline_tokens_per_second": args.new_tokens / old_median,
        "optimized_tokens_per_second": args.new_tokens / new_median,
        "speed_ratio": old_median / new_median,
        "note": "端到端生成循环计时；不含模型加载、分词和文本解码。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
