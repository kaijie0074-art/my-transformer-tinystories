#!/usr/bin/env python3
"""训练字节级 BPE，并将故事流式编码为训练所需的 uint16 数据。"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


DATASET_NAME = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
OUTPUT_FILES = ("tokenizer.json", "train.bin", "val.bin", "manifest.json")


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def jsonl_stories(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 无效: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_number}: 每行必须是含字符串 text 字段的 JSON 对象")
            yield record["text"]


def dataset_stories(dataset: Iterable) -> Iterator[str]:
    for index, record in enumerate(dataset, 1):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"数据集第 {index} 条记录缺少字符串 text 字段")
        yield record["text"]


def limited(stories: Iterable[str], maximum: int | None) -> Iterator[str]:
    return iter(stories) if maximum is None else itertools.islice(stories, maximum)


def batches(stories: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    iterator = iter(stories)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def build_tokenizer(stories: Iterable[str], vocab_size: int, batch_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    count = 0

    def training_batches() -> Iterator[list[str]]:
        nonlocal count
        for batch in batches(stories, batch_size):
            count += len(batch)
            yield batch

    tokenizer.train_from_iterator(training_batches(), trainer=trainer)
    if count == 0:
        raise ValueError("训练集没有故事，无法训练 tokenizer")
    for expected, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"特殊 token {token} 的 ID 不是 {expected}")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    # 原文里出现特殊 token 的字面量时按普通文本编码；后处理仍负责添加 BOS/EOS。
    tokenizer.encode_special_tokens = True
    return tokenizer


def encode_split(
    tokenizer: Tokenizer, stories: Iterable[str], destination: Path, batch_size: int
) -> dict:
    digest = hashlib.sha256()
    story_count = 0
    token_count = 0
    with destination.open("xb") as output:
        for batch in batches(stories, batch_size):
            encoded = tokenizer.encode_batch(batch, add_special_tokens=True)
            chunk = bytearray()
            for encoding in encoded:
                ids = encoding.ids
                if any(token_id < 0 or token_id > 65535 for token_id in ids):
                    raise ValueError("token ID 超出 uint16 范围")
                chunk.extend(np.asarray(ids, dtype="<u2").tobytes())
                token_count += len(ids)
            output.write(chunk)
            digest.update(chunk)
            story_count += len(encoded)
    if story_count == 0:
        raise ValueError(f"数据集为空: {destination.name}")
    return {"stories": story_count, "tokens": token_count, "sha256": digest.hexdigest()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf-cache"), help="Hugging Face datasets 缓存目录")
    parser.add_argument(
        "--input-jsonl", type=Path, nargs=2, metavar=("TRAIN", "VALIDATION"),
        help="离线输入：分别指定 train 和 validation 的 JSONL 文件，每行一个含 text 字段的对象",
    )
    parser.add_argument("--max-train-stories", type=positive_int)
    parser.add_argument("--max-val-stories", type=positive_int)
    parser.add_argument("--vocab-size", type=positive_int, default=10000)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    args = parser.parse_args()
    if not 260 <= args.vocab_size <= 65536:
        parser.error("--vocab-size 必须在 260 到 65536 之间，容纳四个特殊 token 和完整字节字母表")
    return args


def main() -> None:
    args = parse_args()
    if args.input_jsonl and args.input_jsonl[0].samefile(args.input_jsonl[1]):
        raise ValueError("训练集和验证集不能指向同一个文件")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [str(output_dir / name) for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("不会覆盖已有输出: " + ", ".join(existing))
    staging = Path(tempfile.mkdtemp(prefix=".prepare-data-", dir=output_dir))
    try:
        if args.input_jsonl:
            train_path, validation_path = args.input_jsonl

            def train_stories() -> Iterator[str]:
                return limited(jsonl_stories(train_path), args.max_train_stories)

            def validation_stories() -> Iterator[str]:
                return limited(jsonl_stories(validation_path), args.max_val_stories)

            source = {"type": "jsonl", "train": str(train_path), "validation": str(validation_path)}
            dataset_revision = None
        else:
            dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(args.cache_dir))

            def train_stories() -> Iterator[str]:
                return limited(dataset_stories(dataset["train"]), args.max_train_stories)

            def validation_stories() -> Iterator[str]:
                return limited(dataset_stories(dataset["validation"]), args.max_val_stories)

            source = {"type": "huggingface", "dataset": DATASET_NAME,
                      "url": f"https://huggingface.co/datasets/{DATASET_NAME}",
                      "license": "cdla-sharing-1.0"}
            dataset_revision = DATASET_REVISION

        tokenizer = build_tokenizer(train_stories(), args.vocab_size, args.batch_size)
        actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if actual_vocab_size > 65536:
            raise ValueError("实际词汇表超过 uint16 可表示的大小")
        tokenizer.save(str(staging / "tokenizer.json"))
        train = encode_split(tokenizer, train_stories(), staging / "train.bin", args.batch_size)
        validation = encode_split(tokenizer, validation_stories(), staging / "val.bin", args.batch_size)
        manifest = {
            "vocab_size": actual_vocab_size,
            "dataset_revision": dataset_revision,
            "dtype": "<u2",
            "tokenizer_version": version("tokenizers"),
            "source": source,
            "train": train,
            "validation": validation,
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
        # 硬链接以排他方式发布；manifest 最后发布，已有文件绝不会被替换。
        for name in OUTPUT_FILES:
            os.link(staging / name, output_dir / name)
    except Exception:
        print(f"数据准备失败；临时文件保留在 {staging}，已有的部分输出不会自动删除。", file=sys.stderr)
        raise
    for name in OUTPUT_FILES:
        (staging / name).unlink()
    staging.rmdir()
    print(f"数据准备完成：{output_dir}（词汇量 {actual_vocab_size}）")


if __name__ == "__main__":
    main()
