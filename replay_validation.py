#!/usr/bin/env python3
"""按训练日志的固定采样口径重算 checkpoint 的验证 loss。"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from generate import load_model
from train import batch, digest, loss_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="包含匹配的 tokenizer.json 和 val.bin")
    args = parser.parse_args()
    if not mx.is_available(mx.gpu):
        raise RuntimeError("MLX GPU 不可用")
    mx.set_default_device(mx.gpu)
    model, _, info = load_model(args.checkpoint, args.data_dir / "tokenizer.json")
    validation_path = args.data_dir / "val.bin"
    validation_sha256 = digest(validation_path)
    if validation_sha256 != info["sources"]["validation_sha256"]:
        raise ValueError("验证文件与检查点记录的数据不一致")
    cfg = info["config"]
    validation = np.memmap(validation_path, dtype="<u2", mode="r")
    rng = np.random.default_rng(cfg["eval_seed"])
    losses = []
    for _ in range(cfg["eval_batches"]):
        x, y = batch(validation, cfg, rng)
        losses.append(loss_fn(model, x, y))
    mx.eval(losses)
    result = {
        "checkpoint_step": info["step"],
        "validation_sha256": validation_sha256,
        "eval_seed": cfg["eval_seed"],
        "eval_batches": cfg["eval_batches"],
        "batch_size": cfg["batch_size"],
        "context_length": cfg["context_length"],
        "target_tokens": cfg["eval_batches"] * cfg["batch_size"] * cfg["context_length"],
        "loss": sum(float(value.item()) for value in losses) / len(losses),
        "device": str(mx.default_device()),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
