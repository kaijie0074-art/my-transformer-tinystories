#!/usr/bin/env python3
"""等待主训练结束，并在完整检查点上顺序执行后续实验与报告。"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
KIND = "my-transformer-complete-workflow-v1"
LOW_DISK = 5 * 1024**3
POLL_SECONDS = 60


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def atomic_status(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".workflow-status-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(state, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def status_update(path, state, phase, error=None):
    state.update(phase=phase, time=now(), error=error)
    atomic_status(path, state)


def process_info(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
        raise RuntimeError(f"无法检查进程 {pid}：{result.stderr.strip() or 'ps 未返回进程信息'}")
    match = re.match(r"^\s*(\S+\s+\S+\s+\d+\s+\S+\s+\d{4})\s+(.*)$", result.stdout.strip())
    if not match:
        raise RuntimeError(f"无法解析进程 {pid} 的 ps 输出：{result.stdout.strip()}")
    return match.group(1), match.group(2)


def process_cwd(pid):
    proc_path = Path(f"/proc/{pid}/cwd")
    if proc_path.exists():
        return proc_path.resolve(strict=True)
    result = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                            text=True, capture_output=True)
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return Path(line[1:]).resolve(strict=True)
    if process_info(pid) is None:
        return None
    raise RuntimeError(f"无法确认进程 {pid} 工作目录：{result.stderr.strip() or 'lsof 未返回 cwd'}")


def is_trainer(pid, run_dir, start_time=None):
    info = process_info(pid)
    if info is None:
        return False, None
    started, command = info
    if start_time is not None and started != start_time:
        return False, started
    if not re.search(r"(?:^|\s)(?:\S*/)?train\.py(?:\s|$)", command):
        return False, started
    match = re.search(r"(?:^|\s)--run-dir(?:=|\s+)(.*?)(?=\s+--[\w-]+(?:=|\s|$)|$)", command)
    if not match:
        return False, started
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    target = Path(value).expanduser()
    if not target.is_absolute():
        cwd = process_cwd(pid)
        if cwd is None:
            return False, started
        target = cwd / target
    return target.resolve() == run_dir, started


def training_config(run_dir, data_dir):
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            if event.get("event") == "config":
                config = event["config"]
                steps = config["steps"]
                if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
                    raise ValueError("训练配置中的目标步数无效")
                if Path(config["data_dir"]).resolve() != data_dir:
                    raise ValueError("指定数据目录与训练配置不一致")
                return event
    raise ValueError("训练日志中没有 config 事件")


def checkpoint_info(path):
    # safetensors 的元数据位于文件头，不必加载模型张量。
    with path.open("rb") as source:
        size_bytes = source.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"检查点头部不完整：{path}")
        size = int.from_bytes(size_bytes, "little")
        if size <= 0 or size > 100_000_000:
            raise ValueError(f"检查点头部大小无效：{path}")
        header = source.read(size)
        if len(header) != size:
            raise ValueError(f"检查点头部不完整：{path}")
    return json.loads(json.loads(header)["__metadata__"]["checkpoint"])


def final_checkpoint(run_dir, event):
    latest = read_json(run_dir / "latest.json")
    expected = event["config"]["steps"]
    if latest.get("step") != expected:
        raise ValueError(f"主训练未达到目标：latest 步数 {latest.get('step')}，目标 {expected}")
    name = latest.get("checkpoint")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("latest.json 中检查点文件名无效")
    path = (run_dir / name).resolve(strict=True)
    if path.parent != run_dir or not path.is_file():
        raise ValueError("最终检查点不在训练目录中或不是文件")
    info = checkpoint_info(path)
    if (info.get("format_version") != 1 or info.get("step") != expected
            or info.get("config") != event["config"] or info.get("sources") != event["sources"]):
        raise ValueError("最终检查点元数据与训练配置或步数不一致")
    return path


def file_records(paths):
    return [{"path": str(path.resolve(strict=True)), "sha256": digest(path)} for path in paths]


def verify_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("成功状态缺少产物记录")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"已记录产物不存在或已更改：{path}")


def validate_benchmark(path, checkpoint, checkpoint_hash, step, val_hash):
    data = read_json(path)
    if (data.get("checkpoint") != str(checkpoint) or data.get("checkpoint_sha256") != checkpoint_hash
            or data.get("checkpoint_step") != step or data.get("validation_sha256") != val_hash):
        raise ValueError(f"基准结果与最终检查点或验证集不一致：{path}")


def validate_samples(path, checkpoint, checkpoint_hash, step):
    data = read_json(path)
    if (data.get("checkpoint") != str(checkpoint) or data.get("checkpoint_sha256") != checkpoint_hash
            or data.get("step") != step or not isinstance(data.get("samples"), list)
            or len(data["samples"]) != 5):
        raise ValueError(f"生成样例与最终检查点不一致或不完整：{path}")


def validate_diversity(root, checkpoint, checkpoint_hash, data_dir, val_hash):
    summary = read_json(root / "summary.json")
    plan = summary["plan"]
    if (plan.get("parent_checkpoint") != str(checkpoint) or plan.get("parent_sha256") != checkpoint_hash
            or plan.get("data_dir") != str(data_dir) or plan.get("seeds") != [2027, 2028, 2029]
            or plan.get("steps") != 500 or plan.get("small_pool") != 1000000
            or plan.get("large_pool") != 10000000):
        raise ValueError("数据覆盖实验计划与预注册默认设置或最终检查点不一致")
    if read_json(root / "plan.json") != plan:
        raise ValueError("数据覆盖实验计划文件与汇总不一致")
    parent = read_json(root / "parent-evaluation.json")
    if (not parent.get("complete") or parent.get("checkpoint_sha256") != checkpoint_hash
            or parent.get("validation_sha256") != val_hash):
        raise ValueError("父检查点评估不完整或来源不一致")
    pairs = summary["pairs"]
    if len(pairs) != 3 or [pair.get("seed") for pair in pairs] != plan["seeds"]:
        raise ValueError("数据覆盖实验未完成三对预注册实验")
    paths = [root / "summary.json", root / "plan.json", root / "parent-evaluation.json"]
    for pair in pairs:
        pair_file = root / f"pair-{pair['seed']}.json"
        if read_json(pair_file) != pair:
            raise ValueError(f"成对实验结果不一致：{pair_file}")
        paths.append(pair_file)
        for label in ("small", "large"):
            run = root / f"{label}-seed-{pair['seed']}"
            measurement = pair["measurements"][label]
            latest = read_json(run / "latest.json")
            name = latest.get("checkpoint")
            if latest.get("step") != 500 or not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"成对实验检查点不完整：{run}")
            final = (run / name).resolve(strict=True)
            evaluation = read_json(run / "evaluation.json")
            coverage = read_json(run / "coverage.json")
            if (final.parent != run.resolve() or not final.is_file()
                    or measurement.get("checkpoint") != str(final)
                    or measurement.get("evaluation") != evaluation or measurement.get("coverage") != coverage
                    or not evaluation.get("complete") or evaluation.get("validation_sha256") != val_hash
                    or evaluation.get("checkpoint_sha256") != digest(final)):
                raise ValueError(f"成对实验产物不完整或不一致：{run}")
            paths.extend([run / "latest.json", run / "evaluation.json", run / "coverage.json"])
    return paths


def validate_reports(directory):
    paths = []
    for prefix in ("任务A-本机真实训练与推理-", "任务B-数据覆盖训练动态-"):
        for extension in ("md", "docx"):
            matches = list(directory.glob(f"{prefix}*.{extension}"))
            if len(matches) != 1 or not matches[0].is_file() or matches[0].stat().st_size == 0:
                raise ValueError(f"报告文件缺失、为空或不唯一：{prefix}*.{extension}")
            paths.append(matches[0])
    return paths


def run_step(name, command, targets, validator, status_path, state):
    previous = state["steps"].get(name)
    if previous and previous.get("success"):
        verify_records(previous.get("artifacts"))
        paths = validator()
        if {str(p.resolve()) for p in paths} != {item["path"] for item in previous["artifacts"]}:
            raise ValueError(f"{name} 产物路径与成功状态不一致")
        return
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"{name} 的目标已存在但无成功状态，拒绝覆盖：{target}")
    status_update(status_path, state, name)
    print("执行：", " ".join(map(str, command)), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stderr=subprocess.PIPE)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name} 退出码 {result.returncode}：{result.stderr.strip() or '子进程未提供标准错误'}")
    paths = validator()
    state["steps"][name] = {"success": True, "time": now(), "artifacts": file_records(paths)}
    status_update(status_path, state, name)


def main():
    print("后续流程已启动", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/main")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/data-clean")
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "runs/diversity")
    parser.add_argument("--status", type=Path, default=ROOT / "runs/workflow-status.json")
    args = parser.parse_args()
    if args.trainer_pid <= 0:
        parser.error("trainer-pid 必须为正整数")
    run_dir = args.run_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()
    status_path = args.status.expanduser().absolute()
    benchmark = ROOT / "experiments/inference-final.json"
    samples = ROOT / "experiments/main-samples.json"
    reports = ROOT / "reports"
    status_resolved = status_path.resolve()
    if (any(status_resolved == p or status_resolved.is_relative_to(p) for p in
            (run_dir, data_dir, experiment_root, reports))
            or status_resolved in (benchmark, samples) or status_path.is_symlink()):
        raise ValueError("状态路径与输入或结果路径冲突，或为符号链接")
    parameters = {"trainer_pid": args.trainer_pid, "run_dir": str(run_dir), "data_dir": str(data_dir),
                  "experiment_root": str(experiment_root), "benchmark": str(benchmark),
                  "samples": str(samples), "reports": str(reports)}
    if status_path.exists():
        state = read_json(status_path)
        if not isinstance(state, dict) or state.get("kind") != KIND or state.get("parameters") != parameters:
            raise ValueError(f"已有状态文件不属于本次流程，拒绝覆盖：{status_path}")
        if not isinstance(state.get("steps"), dict):
            raise ValueError("已有状态文件格式无效，拒绝覆盖")
    else:
        state = {"kind": KIND, "parameters": parameters, "steps": {}, "warnings": []}
    try:
        status_update(status_path, state, "等待训练")
        event = training_config(run_dir, data_dir)
        state["expected_steps"] = event["config"]["steps"]
        alive, started = is_trainer(args.trainer_pid, run_dir)
        print(f"训练进程检查：{'运行中' if alive else '已退出'}（PID {args.trainer_pid}）", flush=True)
        while alive:
            free = shutil.disk_usage(ROOT).free
            state["free_disk_bytes"] = free
            if free < LOW_DISK:
                warning = {"time": now(), "free_disk_bytes": free,
                           "message": "项目所在磁盘可用空间低于 5 GiB；仅告警，不停止训练。"}
                state["warnings"].append(warning)
                print(warning["message"], free, flush=True)
            status_update(status_path, state, "等待训练")
            time.sleep(POLL_SECONDS)
            alive, _ = is_trainer(args.trainer_pid, run_dir, started)
        checkpoint = final_checkpoint(run_dir, event)
        checkpoint_hash = digest(checkpoint)
        state["checkpoint"] = {"path": str(checkpoint), "sha256": checkpoint_hash,
                               "step": event["config"]["steps"]}
        sources = event["sources"]
        val_hash = digest(data_dir / "val.bin")
        if val_hash != sources["validation_sha256"] or digest(data_dir / "tokenizer.json") != sources["tokenizer_sha256"]:
            raise ValueError("验证集或 tokenizer 与训练记录的来源不一致")
        python = ROOT / ".venv/bin/python"
        executable = str(python) if python.is_file() else sys.executable
        step = event["config"]["steps"]
        run_step("推理基准", [executable, str(ROOT / "benchmark.py"), "--checkpoint", str(checkpoint),
                 "--tokenizer", str(data_dir / "tokenizer.json"), "--validation-bin", str(data_dir / "val.bin"),
                 "--output", str(benchmark)], [benchmark],
                 lambda: (validate_benchmark(benchmark, checkpoint, checkpoint_hash, step, val_hash), [benchmark])[1],
                 status_path, state)
        run_step("生成样例", [executable, str(ROOT / "collect_samples.py"), "--checkpoint", str(checkpoint),
                 "--data-dir", str(data_dir), "--output", str(samples)], [samples],
                 lambda: (validate_samples(samples, checkpoint, checkpoint_hash, step), [samples])[1],
                 status_path, state)
        run_step("数据覆盖实验", [executable, str(ROOT / "run_diversity.py"), "--checkpoint", str(checkpoint),
                 "--data-dir", str(data_dir), "--output-root", str(experiment_root)], [experiment_root],
                 lambda: validate_diversity(experiment_root, checkpoint, checkpoint_hash, data_dir, val_hash),
                 status_path, state)
        # 报告脚本自行拒绝覆盖；此处同时拒绝未经本流程成功记录的既有报告。
        existing_reports = list(reports.glob("任务A-本机真实训练与推理-*")) + list(reports.glob("任务B-数据覆盖训练动态-*")) if reports.exists() else []
        run_step("生成报告", [executable, str(ROOT / "render_reports.py"), "--run-dir", str(run_dir),
                 "--data-dir", str(data_dir), "--experiment-root", str(experiment_root),
                 "--benchmark", str(benchmark), "--samples", str(samples), "--output-dir", str(reports)],
                 existing_reports, lambda: validate_reports(reports), status_path, state)
        status_update(status_path, state, "完成")
        print("后续流程完成", flush=True)
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        try:
            state["failed_phase"] = state.get("phase")
            status_update(status_path, state, "失败", message)
        except Exception as status_exc:
            print(f"无法写入失败状态：{status_exc}", file=sys.stderr, flush=True)
        print(f"后续流程失败：{message}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"后续流程启动失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
