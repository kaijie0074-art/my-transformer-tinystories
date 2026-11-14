# 验证数据来源、修改与许可

`val.bin` 和 `tokenizer.json` 是为复核本项目实验提供的数据快照，来源为公开数据集 [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，固定 revision 为 `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。原数据集卡片标注的许可证为 [Community Data License Agreement – Sharing, Version 1.0（CDLA-Sharing-1.0）](https://cdla.dev/sharing-1-0/)。此处的数据快照按该协议提供；本项目代码没有因此被授予同一许可证。

**修改说明：** 从原始验证划分中排除了与训练集故事原文完全相同的 6,601 篇；未检查近似重复。剩余验证故事使用在 TinyStories 训练集上训练的 Byte-level BPE 编码，按小端 `uint16` token ID 存入 `val.bin`，故事边界带 BOS/EOS 标记。`tokenizer.json` 保存这一 tokenizer；`manifest.json` 记录数据版本、处理方式和文件散列。`train.bin` 不包含在此目录。

`A-metrics.jsonl`、`B-metrics.jsonl` 与推理结果 JSON 是项目实验记录的副本，不是验证数据文件：它们保留实验事件和数值，仅将本机项目绝对路径前缀替换为 `$PROJECT_ROOT`；原始文件留在本机。
