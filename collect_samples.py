#!/usr/bin/env python3
"""用固定提示和种子记录训练检查点的故事续写样例。"""

import argparse
import json
from pathlib import Path

from generate import generate, load_model
from train import digest, now


PROMPTS = [
    "Once upon a time, a little bird found a key. ",
    "Lily found a tiny door under the old tree. ",
    "A little boy lost his favorite toy. ",
    "A small dog was afraid of the rain. ",
    "The little fox wanted to help her friend. ",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0 or args.top_k < 0:
        parser.error("输出 token 数必须为正，temperature 和 top-k 必须非负")
    if not 0 <= args.seed < 2**32 - len(PROMPTS):
        parser.error("seed 超出有效范围")
    model, tokenizer, info = load_model(args.checkpoint, args.data_dir / "tokenizer.json")
    samples = [generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature,
                        args.top_k, args.seed + index) for index, prompt in enumerate(PROMPTS)]
    result = {"time": now(), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": digest(args.checkpoint), "step": info["step"],
              "temperature": args.temperature, "top_k": args.top_k,
              "max_new_tokens": args.max_new_tokens, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
