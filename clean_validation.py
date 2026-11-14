#!/usr/bin/env python3
"""保留原始训练数据，排除与训练故事文本完全相同的验证故事。"""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import pyarrow.compute as pc
from datasets import load_dataset
from tokenizers import Tokenizer

from prepare_data import DATASET_NAME, DATASET_REVISION, OUTPUT_FILES, encode_split
from train import digest


def remove_exact_overlap(train_table, validation_table):
    """以完整原始文本相等为准；不把近似文本判为重复。"""
    validation_column = validation_table.column("text").combine_chunks()
    matched_texts = set()
    matching_train_rows = 0
    for batch in train_table.to_batches(max_chunksize=50000):
        texts = batch.column("text")
        selected = pc.filter(texts, pc.is_in(texts, value_set=validation_column))
        matching_train_rows += len(selected)
        matched_texts.update(selected.to_pylist())
    validation = validation_column.to_pylist()
    clean = [text for text in validation if text not in matched_texts]
    return clean, {
        "method": "exact raw text equality",
        "original_validation_stories": len(validation),
        "removed_validation_stories": len(validation) - len(clean),
        "remaining_validation_stories": len(clean),
        "matching_train_stories": matching_train_rows,
        "distinct_shared_stories": len(matched_texts),
        "remaining_distinct_validation_stories": len(set(clean)),
        "near_duplicates_checked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data-clean"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf-cache"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须为正")
    source = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise ValueError("输出目录必须与原始数据目录不同")
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("仅支持匹配固定版本的完整 TinyStories 数据")
    if digest(source / "train.bin") != original["train"]["sha256"]:
        raise ValueError("原始训练文件 SHA-256 不匹配")
    if digest(source / "val.bin") != original["validation"]["sha256"]:
        raise ValueError("原始验证文件 SHA-256 不匹配")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    if tokenizer.get_vocab_size(with_added_tokens=True) != original["vocab_size"]:
        raise ValueError("Tokenizer 词表与清单不一致")
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(args.cache_dir))
    if len(dataset["train"]) != original["train"]["stories"] or len(dataset["validation"]) != original["validation"]["stories"]:
        raise ValueError("清单不是该版本的完整原始训练集和验证集")
    clean, stats = remove_exact_overlap(dataset["train"].data.table, dataset["validation"].data.table)
    if not clean:
        raise ValueError("过滤后验证集为空")
    output.mkdir(parents=True, exist_ok=True)
    existing = [str(output / name) for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError("不会覆盖已有输出: " + ", ".join(existing))
    staging = Path(tempfile.mkdtemp(prefix=".clean-validation-", dir=output))
    try:
        # 同一磁盘上复用不可变的训练文件和 tokenizer，不重写或复制大文件。
        os.link(source / "train.bin", staging / "train.bin")
        os.link(source / "tokenizer.json", staging / "tokenizer.json")
        validation = encode_split(tokenizer, iter(clean), staging / "val.bin", args.batch_size)
        manifest = copy.deepcopy(original)
        manifest["validation"] = validation
        manifest["validation_filter"] = stats
        with (staging / "manifest.json").open("x", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        for name in OUTPUT_FILES:
            os.link(staging / name, output / name)
    except Exception:
        print(f"验证集准备失败，临时文件保留：{staging}")
        raise
    for name in OUTPUT_FILES:
        (staging / name).unlink()
    staging.rmdir()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"新的验证集：{output}")


if __name__ == "__main__":
    main()
