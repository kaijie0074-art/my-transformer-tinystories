#!/usr/bin/env python3
"""从同一训练检查点执行预注册的数据覆盖成对实验。"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from train import digest, now, read_checkpoint


ROOT = Path(__file__).resolve().parent


def run_command(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("执行：", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_new(path: Path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def verify_pair(small_dir: Path, large_dir: Path):
    initial_paths = []
    for directory in (small_dir, large_dir):
        matches = list(directory.glob("checkpoint-00000000-*.safetensors"))
        if len(matches) != 1:
            raise ValueError(f"{directory} 的初始检查点数量不是 1")
        initial_paths.append(matches[0])
    (small_arrays, small_info), (large_arrays, large_info) = [
        read_checkpoint(path) for path in initial_paths
    ]
    small_cfg = {k: v for k, v in small_info["config"].items() if k != "pool_tokens"}
    large_cfg = {k: v for k, v in large_info["config"].items() if k != "pool_tokens"}
    if small_cfg != large_cfg or small_info["sources"] != large_info["sources"]:
        raise ValueError("成对实验存在数据池大小以外的配置差异")
    if small_info["parent_checkpoint"] != large_info["parent_checkpoint"]:
        raise ValueError("成对实验的父检查点不同")
    if small_info["numpy_rng_state"] != large_info["numpy_rng_state"]:
        raise ValueError("成对实验的训练采样随机状态不同")
    if small_arrays.keys() != large_arrays.keys():
        raise ValueError("成对实验的初始状态字段不同")
    for key in small_arrays:
        a, b = small_arrays[key], large_arrays[key]
        if a.shape != b.shape or a.dtype != b.dtype or not bool(mx.array_equal(a, b).item()):
            raise ValueError(f"成对实验的初始张量不同：{key}")
    return {
        "same_initial_model_optimizer_and_mlx_rng": True,
        "same_training_sampler_rng": True,
        "only_config_difference": "pool_tokens",
        "small_initial_checkpoint": str(initial_paths[0]),
        "large_initial_checkpoint": str(initial_paths[1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--small-pool", type=int, default=1000000)
    parser.add_argument("--large-pool", type=int, default=10000000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.00006)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--full-eval-batch-size", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.small_pool < args.large_pool or args.steps <= 0:
        parser.error("需要 0 < small-pool < large-pool，且 steps 为正")
    if args.batch_size <= 0 or args.grad_accum <= 0 or args.learning_rate <= 0:
        parser.error("batch-size、grad-accum 和 learning-rate 必须为正")
    if args.eval_every <= 0 or args.eval_batches <= 0 or args.full_eval_batch_size <= 0:
        parser.error("评估参数必须为正")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds 必须互不相同且在 [0, 2**32) 范围内")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    data_dir = args.data_dir.expanduser().resolve(strict=True)
    root = args.output_root.expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.large_pool > manifest["train"]["tokens"]:
        raise ValueError("数据池超过训练数据总长度")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": now(), "parent_checkpoint": str(checkpoint),
        "parent_sha256": digest(checkpoint), "data_dir": str(data_dir),
        "small_pool": args.small_pool, "large_pool": args.large_pool,
        "steps": args.steps, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate, "seeds": args.seeds,
        "eval_every": args.eval_every, "eval_batches": args.eval_batches,
        "full_eval_batch_size": args.full_eval_batch_size,
        "description": "每对相同父模型、优化器和训练采样种子，只改变可采样的嵌套数据池大小。",
    }
    write_new(root / "plan.json", plan)
    run_command([ROOT / "evaluate.py", "--checkpoint", checkpoint, "--data-dir", data_dir,
                 "--batch-size", args.full_eval_batch_size, "--output", root / "parent-evaluation.json"])
    pairs = []
    for pair_index, seed in enumerate(args.seeds):
        variants = [("small", args.small_pool), ("large", args.large_pool)]
        if pair_index % 2:
            variants.reverse()
        measurements = {}
        for label, pool in variants:
            run_dir = root / f"{label}-seed-{seed}"
            run_command([ROOT / "train.py", "--data-dir", data_dir, "--run-dir", run_dir,
                         "--fork-from", checkpoint, "--steps", args.steps, "--pool-tokens", pool,
                         "--seed", seed, "--batch-size", args.batch_size, "--grad-accum", args.grad_accum,
                         "--learning-rate", args.learning_rate, "--warmup-steps", 0, "--schedule", "constant",
                         "--eval-every", args.eval_every, "--eval-batches", args.eval_batches,
                         "--checkpoint-every", args.steps])
            latest = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
            if latest["step"] != args.steps:
                raise ValueError(f"{run_dir} 未完成预期步数")
            final_checkpoint = run_dir / latest["checkpoint"]
            run_command([ROOT / "coverage_analysis.py", "--run-dir", run_dir,
                         "--output", run_dir / "coverage.json"])
            run_command([ROOT / "evaluate.py", "--checkpoint", final_checkpoint, "--data-dir", data_dir,
                         "--batch-size", args.full_eval_batch_size, "--output", run_dir / "evaluation.json"])
            evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            coverage = json.loads((run_dir / "coverage.json").read_text(encoding="utf-8"))
            if not evaluation["complete"]:
                raise ValueError(f"{run_dir} 验证不完整")
            measurements[label] = {"run_dir": str(run_dir), "checkpoint": str(final_checkpoint),
                                   "evaluation": evaluation, "coverage": coverage}
        small, large = measurements["small"], measurements["large"]
        if small["coverage"]["processed_input_token_positions"] != large["coverage"]["processed_input_token_positions"]:
            raise ValueError("成对实验处理的 token 预算不同")
        if small["evaluation"]["validation_sha256"] != large["evaluation"]["validation_sha256"]:
            raise ValueError("成对实验的验证集不同")
        verification = verify_pair(Path(small["run_dir"]), Path(large["run_dir"]))
        delta = small["evaluation"]["content_token_nll"] - large["evaluation"]["content_token_nll"]
        pairs.append({"seed": seed, "measurements": measurements, "verification": verification,
                      "small_minus_large_content_nll": delta})
        write_new(root / f"pair-{seed}.json", pairs[-1])
    differences = [pair["small_minus_large_content_nll"] for pair in pairs]
    summary = {
        "completed_at": now(), "plan": plan, "pairs": pairs,
        "mean_small_minus_large_content_nll": statistics.mean(differences),
        "min_small_minus_large_content_nll": min(differences),
        "max_small_minus_large_content_nll": max(differences),
        "interpretation": "差值为小池正文 NLL 减大池正文 NLL；正值表示本次大池更低。三对共享同一父模型，不能代表不同预训练初始化。",
    }
    write_new(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
