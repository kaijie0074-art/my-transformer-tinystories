#!/usr/bin/env python3
"""核验完整实验后生成两份中文报告及对应 Word 文档。"""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"非法数值：{x}")))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_info(path):
    # safetensors 的前 8 字节是 JSON header 长度；无需载入数百 MB 张量。
    with path.open("rb") as source:
        size_bytes = source.read(8)
        require(len(size_bytes) == 8, f"检查点头部不完整：{path}")
        size = struct.unpack("<Q", size_bytes)[0]
        require(0 < size <= min(path.stat().st_size - 8, 100_000_000), f"检查点头部长度无效：{path}")
        header = json.loads(source.read(size))
    return json.loads(header["__metadata__"]["checkpoint"])


def events(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def latest(directory):
    pointer = load(directory / "latest.json")
    name = pointer["checkpoint"]
    require(isinstance(name, str) and Path(name).name == name, f"检查点路径无效：{directory}")
    path = directory / name
    info = checkpoint_info(path)
    require(info["step"] == pointer["step"], f"检查点与指针步数不一致：{directory}")
    return path, info


def check_log(directory, info):
    rows = events(directory / "metrics.jsonl")
    configs = [row for row in rows if row["event"] == "config"]
    require(len(configs) == 1, f"配置日志缺失或重复：{directory}")
    record = configs[0]
    cfg = info["config"]
    require(record["config"] == cfg and record["sources"] == info["sources"] and record["parent_checkpoint"] == info["parent_checkpoint"], f"配置日志与检查点不一致：{directory}")
    trains = [row for row in rows if row["event"] == "train"]
    require([row["step"] for row in trains] == list(range(1, cfg["steps"] + 1)), f"训练步骤不完整或不连续：{directory}")
    budget = cfg["batch_size"] * cfg["grad_accum"] * cfg["context_length"]
    require(all(row["step_tokens"] == budget and row["tokens"] == row["step"] * budget and math.isfinite(row["loss"]) for row in trains), f"训练 token 预算或 loss 无效：{directory}")
    vals = [row for row in rows if row["event"] == "val"]
    expected = sorted(set(range(cfg["eval_every"], cfg["steps"] + 1, cfg["eval_every"])) | {cfg["steps"]})
    require([row["step"] for row in vals] == expected and all(math.isfinite(row["loss"]) and row["tokens"] == cfg["eval_batches"] * cfg["batch_size"] * cfg["context_length"] for row in vals), f"训练中验证里程碑不完整：{directory}")
    return trains, vals


def check_eval(evaluation, checkpoint_hash, step, val_hash, val_tokens, context):
    require(evaluation["complete"] is True and evaluation["checkpoint_sha256"] == checkpoint_hash and evaluation["checkpoint_step"] == step, "全量验证不完整或检查点不一致")
    require(evaluation["validation_sha256"] == val_hash and evaluation["all_target_tokens"] == val_tokens - 1 and evaluation["context_length"] == context, "全量验证数据或目标 token 数不一致")
    require(0 < evaluation["content_target_tokens"] <= evaluation["all_target_tokens"] and all(math.isfinite(evaluation[k]) for k in ("all_token_nll", "content_token_nll")), "验证指标无效")


def link(path, label):
    return f"[{label}]({path.resolve().as_uri()})"


def fmt(value):
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "experiment-root", "benchmark", "samples", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, data, experiment = (p.expanduser().resolve(strict=True) for p in (args.run_dir, args.data_dir, args.experiment_root))
    benchmark_path, samples_path = (p.expanduser().resolve(strict=True) for p in (args.benchmark, args.samples))
    manifest_path = data / "manifest.json"
    manifest = load(manifest_path)
    checkpoint, info = latest(run)
    cfg, sources = info["config"], info["sources"]
    require(info["step"] == cfg["steps"] and info["parent_checkpoint"] is None, "主训练未完成计划步数或并非随机初始化")
    require(Path(cfg["data_dir"]).resolve() == data, "主训练数据目录不一致")
    require(sources["manifest_sha256"] == sha256(manifest_path) and sources["tokenizer_sha256"] == sha256(data / "tokenizer.json"), "清单或 tokenizer 散列不一致")
    require(sources["dataset_revision"] == manifest["dataset_revision"] and sources["vocab_size"] == manifest["vocab_size"], "数据集版本或词表不一致")
    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        entry = manifest[split]
        require(sources[split + "_sha256"] == entry["sha256"] == sha256(data / filename) and (data / filename).stat().st_size == entry["tokens"] * 2, f"{split} 数据与清单不一致")
    filt = manifest["validation_filter"]
    require(filt["removed_validation_stories"] == 6601 and filt["method"] == "exact raw text equality" and filt["original_validation_stories"] - filt["removed_validation_stories"] == manifest["validation"]["stories"], "验证过滤证据不一致")
    train, vals = check_log(run, info)
    config_event = next(row for row in events(run / "metrics.jsonl") if row["event"] == "config")
    elapsed = (datetime.fromisoformat(info["saved_at"]) - datetime.fromisoformat(config_event["time"])).total_seconds()
    require(elapsed >= 0, "训练开始与保存时间不一致")
    peak_gib = max(row["mlx_peak_bytes"] for row in train) / 1024**3
    checkpoint_hash = sha256(checkpoint)
    bench, samples = load(benchmark_path), load(samples_path)
    require(bench["checkpoint_sha256"] == samples["checkpoint_sha256"] == checkpoint_hash and bench["checkpoint_step"] == samples["step"] == info["step"], "基准或样例不是最终检查点")
    require(bench["validation_sha256"] == sources["validation_sha256"] and
            isinstance(bench["logits_allclose_rtol_atol_1e-4"], bool) and
            bench["identical_generated_token_ids"] == (bench["baseline_token_ids"] == bench["optimized_token_ids"]),
            "推理基准数据或结果记录不一致")
    require(bench["trials"] > 0 and bench["new_tokens"] > 0 and bench["baseline_median_seconds"] > 0 and bench["optimized_median_seconds"] > 0, "推理基准参数无效")
    prompts = ["Once upon a time, a little bird found a key. ", "Lily found a tiny door under the old tree. ", "A little boy lost his favorite toy. ", "A small dog was afraid of the rain. ", "The little fox wanted to help her friend. "]
    require([s["prompt"] for s in samples["samples"]] == prompts and all(isinstance(s["text"], str) and 0 <= s["generated_tokens"] <= samples["max_new_tokens"] for s in samples["samples"]), "固定提示样例不完整")
    summary = load(experiment / "summary.json")
    plan = load(experiment / "plan.json")
    require(summary["plan"] == plan and plan["parent_sha256"] == checkpoint_hash and Path(plan["data_dir"]).resolve() == data, "实验计划与共同父检查点或数据源不一致")
    require(plan["seeds"] == [2027, 2028, 2029] and plan["small_pool"] == 1_000_000 and plan["large_pool"] == 10_000_000 and plan["steps"] == 500, "实验参数不符合预注册")
    parent_eval = load(experiment / "parent-evaluation.json")
    check_eval(parent_eval, checkpoint_hash, info["step"], sources["validation_sha256"], manifest["validation"]["tokens"], cfg["context_length"])
    pairs = summary["pairs"]
    require(len(pairs) == 3 and [p["seed"] for p in pairs] == plan["seeds"], "三对实验不完整")
    differences = []
    curves = {}
    for pair in pairs:
        seed = pair["seed"]
        require(load(experiment / f"pair-{seed}.json") == pair, f"种子 {seed} 的成对结果不一致")
        verification = pair["verification"]
        require(verification["same_initial_model_optimizer_and_mlx_rng"] is True and verification["same_training_sampler_rng"] is True and verification["only_config_difference"] == "pool_tokens", f"种子 {seed} 的配对校验失败")
        pair_configs = []
        for label in ("small", "large"):
            measurement = pair["measurements"][label]
            directory = experiment / f"{label}-seed-{seed}"
            require(Path(measurement["run_dir"]).resolve() == directory.resolve(), "实验目录不一致")
            final, child = latest(directory)
            ccfg = child["config"]
            pair_configs.append({k: v for k, v in ccfg.items() if k != "pool_tokens"})
            parent = child["parent_checkpoint"]
            require(child["step"] == ccfg["steps"] == plan["steps"] and parent["sha256"] == checkpoint_hash and parent["step"] == info["step"] and parent["sampling_rng"] == "new_seed" and child["sources"] == sources, "分叉检查点步数、父模型或数据源不一致")
            require(ccfg["seed"] == seed and ccfg["pool_tokens"] == plan[label + "_pool"] and ccfg["batch_size"] == plan["batch_size"] and ccfg["grad_accum"] == plan["grad_accum"] and ccfg["context_length"] == cfg["context_length"] and ccfg["learning_rate"] == plan["learning_rate"] and ccfg["schedule"] == "constant", "分叉训练配置不一致")
            child_train, child_val = check_log(directory, child)
            curves[(seed, label)] = {row["step"]: row["loss"] for row in child_val}
            coverage, evaluation = measurement["coverage"], measurement["evaluation"]
            require(load(directory / "coverage.json") == coverage and load(directory / "evaluation.json") == evaluation, "实验汇总与原始结果不一致")
            budget = plan["steps"] * plan["batch_size"] * plan["grad_accum"] * cfg["context_length"]
            require(coverage["seed"] == seed and coverage["stage_steps"] == plan["steps"] and coverage["training_pool_token_positions"] == ccfg["pool_tokens"] and coverage["processed_input_token_positions"] == budget == child_train[-1]["tokens"] and 0 <= coverage["distinct_input_token_positions"] <= min(budget, ccfg["pool_tokens"]) and coverage["repeated_input_token_positions"] == budget - coverage["distinct_input_token_positions"], "采样预算或覆盖数据不一致")
            require(Path(measurement["checkpoint"]).resolve() == final.resolve(), "实验最终检查点路径不一致")
            check_eval(evaluation, sha256(final), plan["steps"], sources["validation_sha256"], manifest["validation"]["tokens"], cfg["context_length"])
            initial = Path(verification[label + "_initial_checkpoint"])
            initial_info = checkpoint_info(initial)
            require(initial.resolve().parent == directory.resolve() and initial_info["step"] == 0 and initial_info["parent_checkpoint"]["sha256"] == checkpoint_hash and initial_info["config"] == ccfg and initial_info["sources"] == sources, "配对初始状态与实验不一致")
        require(pair_configs[0] == pair_configs[1], "成对训练配置除数据池外存在差异")
        difference = pair["measurements"]["small"]["evaluation"]["content_token_nll"] - pair["measurements"]["large"]["evaluation"]["content_token_nll"]
        require(math.isclose(difference, pair["small_minus_large_content_nll"], rel_tol=1e-12, abs_tol=1e-12), "成对差值不一致")
        differences.append(difference)
    for key, value in (("mean_small_minus_large_content_nll", sum(differences) / 3), ("min_small_minus_large_content_nll", min(differences)), ("max_small_minus_large_content_nll", max(differences))):
        require(math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12), "实验汇总差值不一致")
    prereg = ROOT / "experiments" / "训练动态预注册.md"
    history = ROOT / "work" / "wayfinder-map.md"
    require(prereg.is_file() and history.is_file(), "缺少预注册或过程证据")
    history_text = history.read_text(encoding="utf-8")
    require(all(marker in history_text for marker in ("socksio", "180 秒", "batch=1", "batch=5", "6,601", "不完全相同")), "过程记录缺少报告所需历史证据")
    source_url = manifest["source"]["url"]
    budget = cfg["batch_size"] * cfg["grad_accum"] * cfg["context_length"]
    experiment_budget = plan["steps"] * plan["batch_size"] * plan["grad_accum"] * cfg["context_length"]
    milestones = sorted(set([0, len(vals) // 2, len(vals) - 1]))
    val_table = "\n".join(f"| {vals[i]['step']} | {fmt(vals[i]['loss'])} | {vals[i]['tokens']:,} |" for i in milestones)
    step_sets = [set(curves[(seed, label)]) for seed in plan["seeds"] for label in ("small", "large")]
    require(all(steps == step_sets[0] for steps in step_sets), "成对实验的验证时间点不一致")
    curve_rows = []
    for step in sorted(step_sets[0]):
        small_mean = sum(curves[(seed, "small")][step] for seed in plan["seeds"]) / len(plan["seeds"])
        large_mean = sum(curves[(seed, "large")][step] for seed in plan["seeds"]) / len(plan["seeds"])
        curve_rows.append(f"| {step} | {fmt(small_mean)} | {fmt(large_mean)} | {fmt(small_mean - large_mean)} |")
    if all(value > 0 for value in differences):
        outcome = "三对最终正文 NLL 均为大池较低，支持这个预算和数据池条件下的大池假设。"
    elif all(value < 0 for value in differences):
        outcome = "三对最终正文 NLL 均为小池较低，与我最初预期的方向相反。"
    else:
        outcome = "三对最终正文 NLL 的方向不完全一致；仅凭这组结果不能稳定断定哪种数据池更优。"
    benchmark_outcome = (
        "logits 在设定容差内一致，生成 token ID 也一致。"
        if bench["logits_allclose_rtol_atol_1e-4"] and bench["identical_generated_token_ids"]
        else "至少一项数值或生成结果不一致；这次对照不能证明优化路径可直接替换基线。"
    )
    sample_text = "\n\n".join(f"### 样例 {i}（种子 {s['seed']}）\n\n提示：`{s['prompt']}`\n\n续写结果（含提示）：\n\n> " + s['text'].replace('\n', '\n> ') + f"\n\n生成 {s['generated_tokens']} token；遇 EOS 停止：{s['stopped_on_eos']}。" for i, s in enumerate(samples["samples"], 1))
    date = datetime.now().strftime("%Y%m%d")
    a = f"""# 任务 A：本机真实训练与推理

本项目独立实现模型和训练流程，使用 AI 协助分析与实现；不将外部作者的历史过程当作本次经历。计算依赖 [MLX](https://github.com/ml-explore/mlx)，Byte-level BPE 使用 [tokenizers](https://github.com/huggingface/tokenizers)，不声称自行发明两者。数据来自 [TinyStories]({source_url})（revision `{manifest['dataset_revision']}`，许可 `{manifest['source']['license']}`）；见 {link(manifest_path, '数据清单')}、{link(run / 'metrics.jsonl', '训练日志')}、{link(history, '过程记录')}。

## 结构与从 token 到更新

词表 {sources['vocab_size']:,}，上下文 {cfg['context_length']}，{cfg['n_layer']} 层，{cfg['n_head']} 个注意力头，嵌入维度 {cfg['n_embd']}，前馈隐藏维度 {cfg['ffn_hidden']}，参数量 {bench['parameter_count']:,}。模型包含 token/位置嵌入、因果多头注意力、RMSNorm、GELU 前馈、残差和输出投影。

1. **Token：**训练集拟合的 BPE 将文本编码为 ID，每篇故事加一次 BOS/EOS 并写入 uint16 流；验证文本不参与 tokenizer 训练。采样窗口含 {cfg['context_length'] + 1} 个 token：前面作输入、右移一位作目标。
2. **Embedding：**token 和位置分别查表并相加，微批次隐藏状态形状为 `[{cfg['batch_size']}, {cfg['context_length']}, {cfg['n_embd']}]`。
3. **Attention：**每层将隐藏状态投影成 Q、K、V，拆为 {cfg['n_head']} 个头、每头 {cfg['n_embd'] // cfg['n_head']} 维。`softmax(QKᵀ / √head_dim + causal_mask) V` 聚合历史信息；因果掩码禁止读取未来位置。
4. **MLP：**每个位置的表示经过 `{cfg['n_embd']} → {cfg['ffn_hidden']} → {cfg['n_embd']}` 前馈网络及 GELU。注意力与 MLP 均使用 pre-RMSNorm 和残差连接。
5. **Logits：**最终 RMSNorm 与独立输出头为每个位置产生 {sources['vocab_size']:,} 个分数，微批次张量为 `[{cfg['batch_size']}, {cfg['context_length']}, {sources['vocab_size']}]`。
6. **Loss、backward、optimizer：**交叉熵比较 logits 与右移目标，自动求导得到梯度；每微批次 {cfg['batch_size']} 条，累积 {cfg['grad_accum']} 次并取平均，再按学习率计划执行一次 AdamW 更新。每次更新处理 {budget:,} 个输入 token 位置；处理量不等于不重复数据量。

## 真实训练与验证

随机初始化训练完成 {info['step']:,}/{cfg['steps']:,} 次更新，累计处理 {train[-1]['tokens']:,} 个输入 token 位置；训练 loss 首步 {fmt(train[0]['loss'])}、末步 {fmt(train[-1]['loss'])}，两者为各自随机批次，不是同一样本的严格对照。学习率峰值设置 {cfg['learning_rate']}，预热 {cfg['warmup_steps']} 步，计划 {cfg['schedule']}，weight decay {cfg['weight_decay']}。以下训练中验证是固定种子抽样，每次 {cfg['eval_batches']} 批，**不是全量验证**：

从配置记录到最终 checkpoint 保存的墙钟时间为 {elapsed / 3600:.2f} 小时（包含这段期间的验证、检查点和可能的中断等待）；训练日志记录的 MLX 峰值为 {peak_gib:.2f} GiB，**不是整机内存峰值**。

| 更新步 | 抽样验证 loss | 目标 token 数 |
|---:|---:|---:|
{val_table}

最终模型独立全量验证：正文 NLL {fmt(parent_eval['content_token_nll'])}，全部目标 NLL {fmt(parent_eval['all_token_nll'])}，共 {parent_eval['all_target_tokens']:,} 个目标，其中正文 {parent_eval['content_target_tokens']:,} 个；记录见 {link(experiment / 'parent-evaluation.json', '全量验证')}。正文 NLL 排除 PAD/BOS/EOS/UNK 目标，不能与上述抽样 loss 混为一谈。最终检查点 SHA-256：`{checkpoint_hash}`。

## 数据泄漏与真实问题

官方 train/validation 原文存在 6,601 篇精确重合，正式验证剔除这些文本：原验证 {filt['original_validation_stories']:,} 篇，保留 {manifest['validation']['stories']:,} 篇；训练 {manifest['train']['stories']:,} 篇。保留原始文件；仅检查全文精确相等，近似重复未排查，不能称绝对独立。20 步预检使用未过滤验证集，不能作为严格泛化证据。

首次下载出现 `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`；在独立环境安装 `socksio==1.0.0` 后同一下载命令成功。完整尺寸模型合成数据预检首次 180 秒未完成，且未保留逐步输出；随后对惰性梯度累积显式求值，分别用 batch=1、batch=5 完成 3 步合成预检。前后输入及环境不完全相同，不能据此证明超时唯一原因；MLX 峰值不等于系统总内存。以上取自 {link(history, '本项目过程记录')}。

## 最终检查点推理优化

{link(benchmark_path, '基准原始记录')}：设备 `{bench['device']}`，上下文 {bench['context']}，贪心新增 {bench['new_tokens']} token，预热 {bench['warmup']} 次，交替测量 {bench['trials']} 次。全位置输出投影中位 {fmt(bench['baseline_median_seconds'])} 秒，末位置投影 {fmt(bench['optimized_median_seconds'])} 秒，速度比 {fmt(bench['speed_ratio'])}；分别 {fmt(bench['baseline_tokens_per_second'])} 与 {fmt(bench['optimized_tokens_per_second'])} token/s。logits 最大绝对差 {bench['max_abs_logits_difference']:.8g}。{benchmark_outcome}计时是生成循环，不含加载、分词、解码；这不是服务端并发吞吐或全系统延迟。基准记录未测量其他进程并发及资源争用，不能假设独占或据此外推生产并发；历史预检基准曾与训练并发，不能替代本次最终结果。

## 固定提示真实样例

{link(samples_path, '样例原始记录')}；temperature={samples['temperature']}，top-k={samples['top_k']}，最多新增 {samples['max_new_tokens']} token。样例仅展示个案，不是质量评估。

{sample_text}

## 技术答辩与复核问题

- **为何看不到未来？**注意力的因果掩码让位置 `t` 只读取 `≤t` 的位置；输入 `tokens[:-1]` 和目标 `tokens[1:]` 实现下一 token 预测。模型测试改变后缀并检查前缀 logits 不变。
- **为什么累积梯度？**分 {cfg['grad_accum']} 个微批次计算并平均，可以在较低单次激活内存下形成有效批次；由于浮点运算顺序等原因，不保证与一次大批次逐位一致。
- **为什么只投影最后位置不改变注意力计算？**它只截取最终隐藏状态再计算 LM head，Transformer 前面的上下文仍需处理；训练 loss 仍需要所有位置的 logits。
- **验证集绝对没有泄漏吗？**只能确认已排除原文完全相等的故事；近似重复未检查。全量 NLL 逐位置累计，正文指标排除特殊 token 目标。
- **续训如何验证？**检查点记录模型、AdamW、采样 RNG、MLX RNG、步数和计划；本项目的集成测试将中断恢复后的状态与不中断运行逐张量比较。
"""
    rows = []
    for pair in pairs:
        small, large = pair['measurements']['small'], pair['measurements']['large']
        sc, lc = small['coverage'], large['coverage']
        rows.append(f"| {pair['seed']} | {fmt(small['evaluation']['content_token_nll'])} | {fmt(large['evaluation']['content_token_nll'])} | {fmt(pair['small_minus_large_content_nll'])} | {sc['distinct_input_token_positions']:,} / {sc['repeated_input_token_positions']:,} / {sc['coverage_fraction']:.2%} | {lc['distinct_input_token_positions']:,} / {lc['repeated_input_token_positions']:,} / {lc['coverage_fraction']:.2%} |")
    b = f"""# 任务 B：数据覆盖与训练动态

## 预注册问题与假设

依据 {link(prereg, '实验前预注册')}：从同一训练 checkpoint 出发，固定后续处理 token 预算，仅改变嵌套训练数据池大小，更大覆盖是否让独立验证 loss 降低得更快？假设是大池更有利于降低验证 loss，小池可能更容易重复局部文本；这是待检验的预期，不是质量结论。数据来自 [TinyStories]({source_url})，固定 revision `{manifest['dataset_revision']}`，详见 {link(manifest_path, '清单')}。

## 控制与测量

三对均从同一随机初始化后训练得到的最终 checkpoint 分叉（SHA-256 `{checkpoint_hash}`），并非三次独立预训练。每对控制相同模型与优化器初态、MLX 随机状态、训练采样随机状态、验证种子、数据源及后续训练配置，仅数据池大小不同；种子分别为 {', '.join(map(str, plan['seeds']))}。小池为训练 token 流前 {plan['small_pool']:,} 个位置，大池前 {plan['large_pool']:,} 个位置并包含小池；每组新增 {plan['steps']} 更新，每次 {plan['batch_size']} × {plan['grad_accum']} × {cfg['context_length']}，共 {experiment_budget:,} 个输入 token 位置。位置重复不是重复故事数。分叉前训练可能已接触池内样本。

配对校验记录于 {link(experiment / 'summary.json', '实验汇总')}；所有组在同一去精确重合验证流上完成全量评估。共同父模型正文 NLL {fmt(parent_eval['content_token_nll'])}。正文 NLL 排除特殊 token；差值定义为小池减大池，正值表示大池更低，负值表示小池更低。下表覆盖列依次为不同输入位置 / 重复输入位置 / 池覆盖率；覆盖由实际配置与随机种子重放采样得出。

| 种子 | 小池正文 NLL | 大池正文 NLL | 小减大 | 小池覆盖 | 大池覆盖 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

三对差值均值 {fmt(summary['mean_small_minus_large_content_nll'])}，范围 [{fmt(summary['min_small_minus_large_content_nll'])}, {fmt(summary['max_small_minus_large_content_nll'])}]。{outcome}这只描述本次共同父模型和预算下的最终验证结果，不能据此断言故事质量。

### 不同训练阶段

下面按相同的训练中固定验证样本，列出三对的平均总 token loss；该口径是抽样验证，与上表的全量正文 NLL 不同：

| 后续更新步数 | 小池抽样验证 loss 均值 | 大池抽样验证 loss 均值 | 小减大 |
|---:|---:|---:|---:|
{chr(10).join(curve_rows)}

## 失败检查与仍未知的问题

在正式对照前，分叉随机种子曾被父 checkpoint 的随机状态覆盖，而验证采样也跟随训练种子。用小规模测试重现后，我让显式分叉种子只控制训练采样，验证使用独立固定种子。正式结果逐对检查模型、优化器和随机状态，以及相同 token 预算与验证数据；这些检查没有证明数据池文本分布完全相同。

如果结果与预期冲突，仍应检查池内分布及评估口径，再在本条件下修正假设。三对共享单一预训练 checkpoint，仅反映后续采样变化，不能估计不同初始化的全部不确定性；训练后期、嵌套数据池、单一数据集和模型规模限制外推。近似重复、故事级覆盖、样本质量、其他预算及训练早期效应尚未由这些结果确定。

## 技术复核问题

- 初始模型、优化器和随机状态如何逐张量核验？为何不同池的采样位置仍会不同？
- 位置覆盖如何由窗口并集计算？为什么不能当作不同故事或语义多样性？
- 全量验证的正文目标数、上下文边界与 NLL 分母如何复现？
- 若要判断下降速度或不同初始化的稳健性，还需要哪些额外测量和独立运行？
"""
    output = args.output_dir.expanduser().resolve()
    names = [f"任务A-本机真实训练与推理-{date}", f"任务B-数据覆盖训练动态-{date}"]
    paths = [output / (name + ext) for name in names for ext in (".md", ".docx")]
    require(not any(path.exists() for path in paths), "目标报告已存在，禁止覆盖")
    require(shutil.which("pandoc") is not None, "未找到 pandoc，无法生成 Word")
    output.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with tempfile.TemporaryDirectory(prefix=".report-stage-", dir=output) as temp:
            stage = Path(temp)
            for name, content in zip(names, (a, b)):
                markdown = stage / (name + ".md")
                markdown.write_text(content, encoding="utf-8")
                subprocess.run(["pandoc", str(markdown), "-o", str(stage / (name + ".docx"))], check=True)
            # 硬链接是原子且不覆盖已有目标；任一发布失败时撤销本次文件。
            for destination in paths:
                os.link(stage / destination.name, destination)
                created.append(destination)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    for path in paths:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"报告生成失败：{exc}", file=sys.stderr)
        sys.exit(1)
