#!/usr/bin/env python3
"""从随机初始化训练 MLX GPT，并以无 pickle 的单文件检查点恢复。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np
from tokenizers import Tokenizer

from model import GPT, GPTConfig


DEFAULTS = {
    "steps": 1000, "batch_size": 4, "grad_accum": 1, "seed": 42, "eval_seed": 314159,
    "context_length": 512, "n_layer": 8, "n_head": 8, "n_embd": 512,
    "ffn_hidden": 3072, "learning_rate": 3e-4, "warmup_steps": 100,
    "schedule": "cosine", "join_warmup_steps": 0, "eval_every": 100, "eval_batches": 10,
    "checkpoint_every": 100, "pool_tokens": 0, "weight_decay": 0.01,
}
MODEL_FIELDS = ("context_length", "n_layer", "n_head", "n_embd", "ffn_hidden")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path, help="同一 run 的 .safetensors 检查点")
    group.add_argument("--fork-from", type=Path, help="父 run 的 .safetensors 检查点")
    for name, default in DEFAULTS.items():
        typ = type(default)
        options = {"type": typ, "default": None}
        if name == "schedule":
            options["choices"] = ("cosine", "constant")
        p.add_argument("--" + name.replace("_", "-"), **options)
    return p.parse_args()


def read_checkpoint(path):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    if not isinstance(metadata, dict) or "checkpoint" not in metadata:
        raise ValueError("检查点缺少元数据")
    info = json.loads(metadata["checkpoint"])
    if info.get("format_version") != 1:
        raise ValueError("不支持的检查点版本")
    return arrays, info


def make_config(args, parent):
    saved = parent["config"] if parent else {}
    cfg = {}
    for name, default in DEFAULTS.items():
        value = getattr(args, name)
        previous = saved.get(name, default)
        if args.resume and value is not None and value != previous:
            safe_join = (
                name == "join_warmup_steps"
                and previous == 0
                and 0 < value < saved["steps"]
                and parent["step"] <= value
                and saved["warmup_steps"] == 0
                and saved["schedule"] == "cosine"
            )
            if not safe_join:
                raise ValueError(f"续训参数 {name} 与原训练配置不一致")
        cfg[name] = value if value is not None else previous
    if not parent and args.warmup_steps is None:
        cfg["warmup_steps"] = min(cfg["warmup_steps"], cfg["steps"])
    data_dir = str(args.data_dir.expanduser().resolve()) if args.data_dir else saved.get("data_dir")
    if not data_dir:
        raise ValueError("新训练必须指定 --data-dir")
    if args.resume and data_dir != saved["data_dir"]:
        raise ValueError("续训数据目录与原训练配置不一致")
    cfg["data_dir"] = data_dir
    for name in ("steps", "batch_size", "grad_accum", "context_length", "n_layer", "n_head", "n_embd", "ffn_hidden", "eval_every", "eval_batches", "checkpoint_every"):
        if cfg[name] <= 0:
            raise ValueError(f"{name} 必须为正数")
    for name in ("seed", "eval_seed"):
        if cfg[name] < 0 or cfg[name] >= 2**32:
            raise ValueError(f"{name} 必须在 [0, 2**32) 范围内")
    if cfg["schedule"] not in ("cosine", "constant"):
        raise ValueError("不支持的学习率计划")
    if cfg["pool_tokens"] < 0 or not 0 <= cfg["warmup_steps"] <= cfg["steps"]:
        raise ValueError("pool_tokens 或 warmup_steps 无效")
    if not 0 <= cfg["join_warmup_steps"] < cfg["steps"]:
        raise ValueError("join_warmup_steps 必须小于总步数且非负")
    if cfg["join_warmup_steps"] and (cfg["warmup_steps"] != 0 or cfg["schedule"] != "cosine"):
        raise ValueError("接入预热曲线仅支持无预热的余弦日程")
    if not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0 or not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("学习率或权重衰减无效")
    if cfg["n_embd"] % cfg["n_head"]:
        raise ValueError("n_embd 必须能被 n_head 整除")
    if args.fork_from and any(cfg[k] != saved[k] for k in MODEL_FIELDS):
        raise ValueError("fork 的模型结构必须与父检查点一致")
    return cfg


def load_data(cfg):
    root = Path(cfg["data_dir"])
    manifest_path = root / "manifest.json"
    tokenizer_path = root / "tokenizer.json"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("dtype") not in ("uint16", "uint16_le", "<u2", "little-endian uint16"):
        raise ValueError("数据 dtype 必须为小端 uint16")
    vocab = manifest["vocab_size"]
    if not isinstance(vocab, int) or not 4 <= vocab <= 65536:
        raise ValueError("词表大小无效")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab:
        raise ValueError("tokenizer 和 manifest 词表大小不一致")
    # 用 ID 验证保留 token，而不是依赖其具体拼写。
    if any(tokenizer.id_to_token(i) is None for i in range(vocab)):
        raise ValueError("tokenizer 的 token ID 必须连续")
    sources = {"manifest_sha256": digest(manifest_path), "tokenizer_sha256": digest(tokenizer_path),
               "dataset_revision": manifest.get("dataset_revision"), "vocab_size": vocab}
    streams = {}
    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        path = root / filename
        entry = manifest[split]
        size = path.stat().st_size
        if size % 2 or size != int(entry["tokens"]) * 2 or size < (cfg["context_length"] + 1) * 2:
            raise ValueError(f"{filename} 的大小或 token 数无效")
        sha = digest(path)
        if sha != entry["sha256"]:
            raise ValueError(f"{filename} 的 SHA-256 不匹配")
        data = np.memmap(path, mode="r", dtype="<u2")
        for start in range(0, len(data), 4 * 1024 * 1024):
            if int(data[start:start + 4 * 1024 * 1024].max()) >= vocab:
                raise ValueError(f"{filename} 含词表外 token")
        sources[split + "_sha256"] = sha
        streams[split] = data
    pool = cfg["pool_tokens"] or len(streams["train"])
    if pool > len(streams["train"]) or pool <= cfg["context_length"]:
        raise ValueError("pool_tokens 超出训练流或不足一个训练样本")
    streams["train"] = streams["train"][:pool]
    return streams, sources


def batch(stream, cfg, rng):
    length = cfg["context_length"]
    starts = rng.integers(0, len(stream) - length, size=cfg["batch_size"])
    indices = starts[:, None] + np.arange(length + 1)[None, :]
    tokens = mx.array(np.asarray(stream[indices], dtype=np.int32))
    return tokens[:, :-1], tokens[:, 1:]


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def lr_for_step(cfg, step):
    join = cfg.get("join_warmup_steps", 0)
    if join and step > join:
        progress = (step - join - 1) / (cfg["steps"] - join)
        return cfg["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))
    if cfg["warmup_steps"] and step <= cfg["warmup_steps"]:
        return cfg["learning_rate"] * step / cfg["warmup_steps"]
    if cfg["schedule"] == "constant":
        return cfg["learning_rate"]
    remaining = cfg["steps"] - cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"] - 1) / max(remaining, 1)
    return cfg["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))


def write_event(path, event):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(run_dir, model, optimizer, rng, step, cfg, sources, parent):
    arrays = {"model." + k: v for k, v in tree_flatten(model.parameters())}
    arrays.update({"optimizer." + k: v for k, v in tree_flatten(optimizer.state)})
    arrays.update({"rng." + str(i): v for i, v in enumerate(mx.random.state)})
    mx.eval(*arrays.values())
    info = {"format_version": 1, "step": step, "config": cfg, "sources": sources,
            "numpy_rng_state": rng.bit_generator.state, "mlx_rng_count": len(mx.random.state),
            "parent_checkpoint": parent, "saved_at": now()}
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".safetensors", dir=run_dir)
    os.close(fd)
    try:
        mx.save_safetensors(temp, arrays, metadata={"checkpoint": json.dumps(info, sort_keys=True)})
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        # 每次写入独立文件；即使从同一步重启也不覆盖旧检查点。
        filename = f"checkpoint-{step:08d}-{os.urandom(6).hex()}.safetensors"
        destination = run_dir / filename
        os.replace(temp, destination)
        atomic_json(run_dir / "latest.json", {"checkpoint": filename, "step": step})
        return destination
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def restore(arrays, info, model, optimizer, rng):
    model_arrays = {k[6:]: v for k, v in arrays.items() if k.startswith("model.")}
    optim_arrays = {k[10:]: v for k, v in arrays.items() if k.startswith("optimizer.")}
    expected_model = dict(tree_flatten(model.parameters()))
    optimizer.init(model.trainable_parameters())
    expected_optim = dict(tree_flatten(optimizer.state))
    for actual, expected, label in ((model_arrays, expected_model, "模型"), (optim_arrays, expected_optim, "优化器")):
        if actual.keys() != expected.keys() or any(actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype for k in expected):
            raise ValueError(f"检查点{label}状态与配置不匹配")
    model.update(tree_unflatten(model_arrays), strict=True)
    optimizer.state = tree_unflatten(optim_arrays)
    rng.bit_generator.state = info["numpy_rng_state"]
    count = info["mlx_rng_count"]
    if count != len(mx.random.state) or any(f"rng.{i}" not in arrays for i in range(count)):
        raise ValueError("检查点 MLX 随机状态不完整")
    mx.random.state[:] = [arrays[f"rng.{i}"] for i in range(count)]
    mx.eval(model.parameters(), optimizer.state, mx.random.state)


def trim_log(path, step):
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    for line in lines:
        event = json.loads(line)
        if event.get("step", 0) <= step:
            kept.append(line)
    if len(kept) == len(lines):
        return None
    backup = path.with_name(f"metrics-before-resume-{step:08d}-{os.urandom(6).hex()}.jsonl")
    os.link(path, backup)
    fd, temp = tempfile.mkstemp(prefix=".metrics-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in kept))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return backup


def main():
    args = parse_args()
    source_path = args.resume or args.fork_from
    arrays, info = read_checkpoint(source_path) if source_path else (None, None)
    cfg = make_config(args, info)
    run_dir = args.run_dir.expanduser().resolve()
    if args.resume:
        if source_path.resolve().parent != run_dir or not run_dir.is_dir():
            raise ValueError("--resume 必须指向同一 run 目录内的检查点")
    elif run_dir.exists():
        raise ValueError("run 目录已存在；为避免覆盖请指定新目录")
    if not mx.is_available(mx.gpu):
        raise RuntimeError("MLX GPU 不可用")
    mx.set_default_device(mx.gpu)
    streams, sources = load_data(cfg)
    if info and sources != info["sources"]:
        raise ValueError("数据来源或哈希与检查点不一致")
    model_cfg = GPTConfig(vocab_size=sources["vocab_size"], **{k: cfg[k] for k in MODEL_FIELDS})
    mx.random.seed(cfg["seed"])
    model = GPT(model_cfg)
    optimizer = optim.AdamW(learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    if info:
        restore(arrays, info, model, optimizer, rng)
        if args.fork_from and args.seed is not None:
            # 显式为新分叉提供种子时重置训练采样；验证采样由 eval_seed 独立控制。
            rng = np.random.default_rng(cfg["seed"])
    else:
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
    step = info["step"] if args.resume else 0
    if not 0 <= step <= cfg["steps"]:
        raise ValueError("检查点 step 超出原训练计划")
    parent = info.get("parent_checkpoint") if args.resume else (
        {"path": str(source_path.resolve()), "sha256": digest(source_path), "step": info["step"],
         "sampling_rng": "new_seed" if args.seed is not None else "parent_state"}
        if args.fork_from else None)
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    log = run_dir / "metrics.jsonl"
    schedule_changed = False
    if args.resume:
        archived_log = trim_log(log, step)
        resume_event = {"event": "resume", "time": now(), "step": step,
                        "checkpoint": str(source_path.resolve()),
                        "previous_log": str(archived_log) if archived_log else None}
        previous_join = info["config"].get("join_warmup_steps", 0)
        if cfg["join_warmup_steps"] != previous_join:
            schedule_changed = True
            transition_checkpoint = save_checkpoint(run_dir, model, optimizer, rng, step, cfg, sources, parent)
            resume_event["schedule_change"] = {
                "join_warmup_steps_before": previous_join,
                "join_warmup_steps_after": cfg["join_warmup_steps"],
                "first_affected_step": cfg["join_warmup_steps"] + 1,
            }
            resume_event["schedule_change"]["checkpoint"] = str(transition_checkpoint.resolve())
        write_event(log, resume_event)
    else:
        write_event(log, {"event": "config", "time": now(), "step": 0, "config": cfg,
                          "sources": sources, "parent_checkpoint": parent})
    stop = {"signal": None}
    def request_stop(signum, _frame):
        stop["signal"] = signum
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    tokens_per_step = cfg["batch_size"] * cfg["grad_accum"] * cfg["context_length"]
    value_and_grad = nn.value_and_grad(model, loss_fn)
    last_saved_step = step if schedule_changed else None
    print(f"训练目录：{run_dir}；参数量：{model.count_parameters():,}；目标步数：{cfg['steps']}", flush=True)
    try:
        if not args.resume:
            save_checkpoint(run_dir, model, optimizer, rng, step, cfg, sources, parent)
            last_saved_step = step
            print(f"已保存第 {step} 步初始状态", flush=True)
        while step < cfg["steps"] and stop["signal"] is None:
            started = time.monotonic()
            model.train()
            grads = None
            losses = []
            for _ in range(cfg["grad_accum"]):
                x, y = batch(streams["train"], cfg, rng)
                loss, grad = value_and_grad(model, x, y)
                # 每个微批次先完成计算，避免四个前向/反向计算图同时驻留。
                mx.eval(loss, grad)
                losses.append(loss)
                grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                mx.eval(grads)
            grads = tree_map(lambda g: g / cfg["grad_accum"], grads)
            rate = lr_for_step(cfg, step + 1)
            optimizer.learning_rate = rate
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, losses)
            step += 1
            elapsed = time.monotonic() - started
            train_loss = sum(float(x.item()) for x in losses) / len(losses)
            write_event(log, {"event": "train", "time": now(), "step": step, "loss": train_loss,
                              "learning_rate": rate, "tokens": step * tokens_per_step,
                              "step_tokens": tokens_per_step, "seconds": elapsed,
                              "tokens_per_second": tokens_per_step / elapsed,
                              "mlx_active_bytes": mx.get_active_memory(),
                              "mlx_peak_bytes": mx.get_peak_memory()})
            if step == 1 or step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                print(f"step={step}/{cfg['steps']} train_loss={train_loss:.4f} "
                      f"lr={rate:.3g} tok/s={tokens_per_step / elapsed:.0f} "
                      f"mlx_peak_gib={mx.get_peak_memory() / 1024**3:.2f}", flush=True)
            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                model.eval()
                val_rng = np.random.default_rng(cfg["eval_seed"])
                started_val = time.monotonic()
                vals = []
                for _ in range(cfg["eval_batches"]):
                    x, y = batch(streams["validation"], cfg, val_rng)
                    vals.append(loss_fn(model, x, y))
                mx.eval(vals)
                val_loss = sum(float(v.item()) for v in vals) / len(vals)
                write_event(log, {"event": "val", "time": now(), "step": step,
                                  "loss": val_loss,
                                  "tokens": cfg["eval_batches"] * cfg["batch_size"] * cfg["context_length"],
                                  "train_tokens": step * tokens_per_step, "seconds": time.monotonic() - started_val})
                print(f"step={step}/{cfg['steps']} val_loss={val_loss:.4f}", flush=True)
            if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"] or stop["signal"] is not None:
                save_checkpoint(run_dir, model, optimizer, rng, step, cfg, sources, parent)
                last_saved_step = step
        if stop["signal"] is not None and last_saved_step != step:
            save_checkpoint(run_dir, model, optimizer, rng, step, cfg, sources, parent)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if stop["signal"] is not None:
        print(f"收到信号 {stop['signal']}，已保存第 {step} 步检查点", file=sys.stderr)
        return 128 + stop["signal"]
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        sys.exit(1)
