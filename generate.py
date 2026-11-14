#!/usr/bin/env python3
"""加载本项目的完整训练检查点，按给定提示续写故事。"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train import MODEL_FIELDS, digest, read_checkpoint


def load_model(checkpoint_path: Path, tokenizer_path: Path):
    arrays, info = read_checkpoint(checkpoint_path)
    if digest(tokenizer_path) != info["sources"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer 与训练检查点不匹配")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # tokenizers 不会把此运行时选项写入 tokenizer.json，加载后需重新开启。
    tokenizer.encode_special_tokens = True
    vocab_size = info["sources"]["vocab_size"]
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError("Tokenizer 词表大小与检查点不匹配")
    config = GPTConfig(vocab_size=vocab_size, **{k: info["config"][k] for k in MODEL_FIELDS})
    model = GPT(config)
    model_arrays = {key[6:]: value for key, value in arrays.items() if key.startswith("model.")}
    model.update(tree_unflatten(model_arrays), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer, info


def sample_next(logits: mx.array, temperature: float, top_k: int) -> int:
    if temperature == 0:
        result = mx.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        if 0 < top_k < logits.shape[-1]:
            threshold = mx.sort(logits, axis=-1)[..., -top_k, None]
            logits = mx.where(logits < threshold, -float("inf"), logits)
        result = mx.random.categorical(logits, axis=-1)
    return int(result.item())


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
             top_k: int, seed: int, stop_on_eos: bool = True):
    mx.random.seed(seed)
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    if bos is None or eos is None:
        raise ValueError("Tokenizer 缺少 BOS 或 EOS")
    ids = [bos] + tokenizer.encode(prompt, add_special_tokens=False).ids
    generated = []
    stopped_on_eos = False
    for _ in range(max_new_tokens):
        window = mx.array([ids[-model.config.context_length:]], dtype=mx.int32)
        logits = model(window, last_token_only=True)[:, -1, :]
        token = sample_next(logits, temperature, top_k)
        ids.append(token)
        generated.append(token)
        if stop_on_eos and token == eos:
            stopped_on_eos = True
            break
    return {
        "prompt": prompt,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "generated_tokens": len(generated),
        "stopped_on_eos": stopped_on_eos,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path, help="默认读取 runs/main/latest.json")
    parser.add_argument("--tokenizer", type=Path,
                        default=project_dir / "artifacts" / "data-clean" / "tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.temperature < 0 or args.top_k < 0 or args.samples <= 0:
        parser.error("token 数、温度和 top-k 必须非负；样本数必须为正")
    if not 0 <= args.seed < 2**32 or args.seed + args.samples > 2**32:
        parser.error("seed 和样本数必须在随机种子有效范围内")
    if args.checkpoint is None:
        latest_path = project_dir / "runs" / "main" / "latest.json"
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            args.checkpoint = latest_path.parent / latest["checkpoint"]
        except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            parser.error(f"无法读取默认 checkpoint {latest_path}：{error}；可用 --checkpoint 指定")
    model, tokenizer, _ = load_model(args.checkpoint, args.tokenizer)
    for index in range(args.samples):
        result = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                          args.top_k, args.seed + index, not args.ignore_eos)
        if index:
            print()
        print(result["text"])


if __name__ == "__main__":
    main()
