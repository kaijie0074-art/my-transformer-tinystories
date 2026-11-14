# 本机 Transformer 语言模型实验

本项目使用公开的 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 和独立编写的 MLX Transformer 代码，从随机初始化训练英文短故事语言模型。数据集版本固定为 `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，数据集许可为 CDLA-Sharing-1.0。

模型代码和训练入口没有加载 Checo 的模型权重。Transformer 是公开架构；GPU、自动求导与优化器由 Apple MLX 提供。Tokenizer 是在本机训练的 Byte-level BPE，BPE 训练算法使用 `tokenizers` 库，而不是本项目重新实现该库。

## 文件与数据

- `prepare_data.py`：下载数据、训练 BPE、按故事分块编码和生成校验清单。
- `clean_validation.py`：保留训练集，排除验证集中与训练原文完全相同的故事。
- `model.py`：嵌入、因果注意力、前馈网络和输出投影。
- `train.py`：随机初始化训练、验证、持久指标、完整 checkpoint 和实验分叉。
- `generate.py`：使用本项目 checkpoint 和 tokenizer 生成文本。
- `evaluate.py`：在与 checkpoint 绑定的验证集上计算全部 token 与正文 token 的 NLL。
- `benchmark.py`：比较完整输出投影和最后位置投影的真实模型生成循环。
- `replay_validation.py`：复现训练日志中固定随机采样的验证 loss。
- `tests/`：数据、模型、训练状态恢复、生成及评估的回归检查。
- `artifacts/data/`：原始公开划分的编码数据；`artifacts/data-clean/`：同一训练数据与去除精确重复后的验证集。
- `runs/`：各自独立的训练配置、指标和 checkpoint。
- `experiments/`：实验预注册与实际实验记录。
- `review-data/`：与检查点匹配的 tokenizer、去重验证 token 流，以及路径匿名化后的 A/B 日志和推理实验结果。
- [任务 A：模型原理与错误复盘](experiments/任务A-模型原理与错误复盘-20261115.md)；[任务 B：学习率预热技术报告](experiments/学习率预热-技术报告-20261114.md)。

## 安装环境

需要 Apple Silicon、支持 MLX 的 macOS、Python 3.12 和 `uv`。在项目目录执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

如在使用 SOCKS 代理的环境运行 Hugging Face 客户端，依赖中的 `socksio` 为其提供支持；项目不会修改系统代理设置。

## 数据准备

```bash
HF_HOME="$PWD/artifacts/hf" .venv/bin/python -u prepare_data.py \
  --output-dir artifacts/data --cache-dir artifacts/hf-cache \
  --vocab-size 10000 --batch-size 256
```

Tokenizer 仅在训练集上训练，验证集不参与。故事以 `[BOS] + 正文 + [EOS]` 存储一次，保留空白和换行。`manifest.json` 记录 split 数量、token 数量、文件 SHA-256 和来源；文件为小端 `uint16`。脚本不会覆盖已有输出。

该版本官方验证集有 6,601 篇与训练集原文完全相同的故事。正式实验先构造独立的精确去重版本，原文件不会被覆盖；近似重复未检查：

```bash
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 .venv/bin/python -u clean_validation.py \
  --data-dir artifacts/data --output-dir artifacts/data-clean --cache-dir artifacts/hf-cache
```

运行时使用 `tokenizers.Tokenizer` 重新加载 `tokenizer.json` 并需要编码原文时，应设置 `tokenizer.encode_special_tokens = True`，以便文本里字面的 `[BOS]` 等字符串作为普通文本编码。该库不会把这个运行时选项持久化到 JSON；项目生成入口已设置此选项。

## 训练

示例，首次训练务必使用新的 `--run-dir`；参数需根据本机内存和目标实验调整：

```bash
caffeinate -i .venv/bin/python -u train.py \
  --data-dir artifacts/data-clean --run-dir runs/example \
  --steps 1000 --batch-size 5 --grad-accum 8 \
  --context-length 512 --warmup-steps 100 --learning-rate 0.0006 \
  --eval-every 100 --eval-batches 10 --checkpoint-every 500 --seed 2026
```

训练不加载外部预训练权重。`metrics.jsonl` 保存训练和验证记录；每个 checkpoint 包含模型、AdamW、数据采样 RNG、MLX RNG、步数、训练配置和数据来源散列。`latest.json` 指向最新文件。续训需指定同一目录内的 checkpoint：

```bash
.venv/bin/python -u train.py --run-dir runs/example \
  --resume runs/example/具体检查点文件.safetensors
```

`--steps` 是该次训练计划的总步数；resume 不能悄悄更改原计划。不同条件的实验使用新目录和 `--fork-from`，并记录父 checkpoint。分叉时显式给出 `--seed` 会重置训练采样随机数；不传则沿用父状态。验证样本使用独立的 `--eval-seed`，便于跨训练种子保持同一验证口径。恢复旧 checkpoint 时，如现有日志包含更晚的记录，原日志会另存备份再继续，不会静默丢弃。

实验需要将无预热余弦日程在指定步数后接入对应的预热余弦尾段时，可以在该步或之前的 checkpoint 上显式传入 `--join-warmup-steps N`。例如设置为 500 时，前 500 步仍使用原无预热余弦曲线，第 501 步开始采用“预热 500 步”的余弦尾段；B 组本身不会因此获得预热。接入可能产生学习率突变。该选项只允许从无预热余弦日程在切换点或之前启用；程序会另存完整状态与新配置，并在日志记录变更，后续正常 resume 可沿用该计划。

训练过程建议接入电源、保持设备通风和可用磁盘空间；`caffeinate -i` 只在命令运行期间阻止空闲休眠，不保证合盖时继续运行。

A 组主训练和 B 组学习率对照都已完成 10,000 步；最终模型和结果见上面的任务说明。早期自动推理基准及其他后续流程曾取消；之后已单独完成正式模型的推理优化实验，结果见 `review-data/inference-final.json`。

## 从 GitHub 复核已训练模型

仓库为私有仓库，访问者须先获得仓库读取权限并登录 GitHub CLI。A、B 的完整最终 checkpoint 位于 Release `v1.0.0`，每个约 505 MB；它们包含优化器和随机状态，原始元数据保留了训练机器的本地路径。下载不会覆盖本地已存在的文件：

```bash
mkdir -p checkpoints
gh release download v1.0.0 --repo kaijie0074-art/my-transformer-tinystories \
  --pattern 'checkpoint-00010000-79a9b244cba6.safetensors' --dir checkpoints
gh release download v1.0.0 --repo kaijie0074-art/my-transformer-tinystories \
  --pattern 'checkpoint-00010000-63e5f5be0f19.safetensors' --dir checkpoints
shasum -a 256 -c review-data/checkpoints.sha256
```

下载安装本项目环境后，可用 A 组权重和随仓库提供的原始 tokenizer 生成文本：

```bash
.venv/bin/python generate.py \
  --checkpoint checkpoints/checkpoint-00010000-79a9b244cba6.safetensors \
  --tokenizer review-data/tokenizer.json \
  --prompt 'Once upon a time, a little bird found a key. ' \
  --temperature 0.7 --top-k 50 --max-new-tokens 80 --seed 42
```

`review-data/` 保留与检查点匹配的 tokenizer 和去重验证 token 流，不包含完整训练集。其来源、处理方式和 CDLA-Sharing-1.0 数据许可见 [review-data/NOTICE.md](review-data/NOTICE.md)。A/B 指标与推理结果是路径匿名化的快照；原始实验数值和哈希字段未改变。

训练日志中的 validation loss 是固定 seed 抽样的 25,600 个目标 token 位置。下面分别重算两组最终 checkpoint 的同一验证口径，无需重新训练：

```bash
.venv/bin/python replay_validation.py \
  --checkpoint checkpoints/checkpoint-00010000-79a9b244cba6.safetensors --data-dir review-data
.venv/bin/python replay_validation.py \
  --checkpoint checkpoints/checkpoint-00010000-63e5f5be0f19.safetensors --data-dir review-data
```

重算结果应分别接近 A 的 `1.428094828129` 和 B 的 `1.919713675976`；数值在不同硬件或数值核上可能略有差异。`evaluate.py` 使用连续分块评估整个验证流，口径不同，不能直接当作上述固定抽样结果。

要重新测量推理优化，可运行下列命令；输出文件必须是不存在的新路径，避免覆盖已有结果：

```bash
.venv/bin/python benchmark.py \
  --checkpoint checkpoints/checkpoint-00010000-79a9b244cba6.safetensors \
  --tokenizer review-data/tokenizer.json --validation-bin review-data/val.bin \
  --context 128 --new-tokens 64 --warmup 2 --trials 10 --output runs/inference-review.json
```

复核这两项只需验证数据和最终 checkpoint；如果要从随机初始化重新训练，仍需按上面的“数据准备”步骤下载并编码完整训练集。Release 未包含实验用的初始 checkpoint，不能仅凭仓库和最终权重保证逐位复现原训练轨迹。

## 生成和验证

```bash
.venv/bin/python generate.py \
  --prompt 'Once upon a time, a little bird found a key. ' \
  --temperature 0.7 --top-k 50
```

终端默认直接显示完整故事（包含一次提示词），保留生成文本中的换行；多条样本之间用空行分隔。程序调用 `generate()` 时仍返回包含生成统计的字典。

默认使用 `runs/main/latest.json` 指向的 checkpoint、`artifacts/data-clean/tokenizer.json` 和随机种子 42，最多生成 80 个新 token；也可通过 `--checkpoint`、`--tokenizer`、`--seed` 和 `--max-new-tokens` 显式覆盖。默认路径相对于项目目录，与运行命令时的当前目录无关。

```bash
.venv/bin/python evaluate.py \
  --checkpoint runs/example/具体检查点文件.safetensors \
  --data-dir artifacts/data-clean --batch-size 5 --output runs/example/evaluation.json
```

`evaluate.py` 默认完整处理验证 token 流，分别报告所有目标 token 的 NLL，以及排除 PAD/BOS/EOS/UNK 目标后的正文 NLL。因果上下文每 `context_length` 个 token 重置；要比较两次结果，必须使用相同 tokenizer、验证数据、上下文长度和评估口径。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试涵盖 UTF-8 文本往返、数据完整性和隔离、因果遮罩、真实梯度更新、断点恢复、随机数分叉以及推理路径的输出一致性。测试数据是合成的，只用于检查程序行为；正式训练结果只能来自单独记录的真实数据训练。
