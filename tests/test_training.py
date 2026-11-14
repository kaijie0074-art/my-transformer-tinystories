import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from generate import generate, load_model, sample_next
from evaluate import evaluate
from render_reports import check_log, checkpoint_info
from train import batch, lr_for_step


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


class LearningRateScheduleTests(unittest.TestCase):
    def test_join_keeps_original_prefix_and_matches_warmup_tail(self):
        base = {"steps": 10000, "learning_rate": 0.0006, "warmup_steps": 0, "schedule": "cosine"}
        joined = {**base, "join_warmup_steps": 500}
        warmup = {**base, "warmup_steps": 500}
        for step in range(1, 501):
            self.assertEqual(lr_for_step(joined, step), lr_for_step(base, step))
        for step in range(501, 10001):
            self.assertEqual(lr_for_step(joined, step), lr_for_step(warmup, step))
        self.assertAlmostEqual(lr_for_step(joined, 500), 0.0005963212312, places=13)
        self.assertEqual(lr_for_step(joined, 501), 0.0006)
        self.assertGreater(lr_for_step(joined, 501), lr_for_step(joined, 500))


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        stories = [
            {"text": f"Story {i}. The little bird found a shiny stone in a garden and shared it."}
            for i in range(80)
        ]
        for filename, records in (("train.jsonl", stories), ("val.jsonl", stories[:12])):
            (cls.root / filename).write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
        cls.data = cls.root / "data"
        prep = run(ROOT / "prepare_data.py", "--input-jsonl", cls.root / "train.jsonl",
                   cls.root / "val.jsonl", "--output-dir", cls.data, "--vocab-size", 512)
        if prep.returncode:
            raise RuntimeError(prep.stderr)
        cls.original = cls.root / "original"
        train = run(ROOT / "train.py", "--data-dir", cls.data, "--run-dir", cls.original,
                    "--steps", 4, "--batch-size", 2, "--grad-accum", 2, "--seed", 321,
                    "--context-length", 8, "--n-layer", 1, "--n-head", 2,
                    "--n-embd", 16, "--ffn-hidden", 32, "--learning-rate", 0.003,
                    "--warmup-steps", 1, "--eval-every", 2, "--eval-batches", 2,
                    "--checkpoint-every", 2)
        if train.returncode:
            raise RuntimeError(f"短训练失败：{train.stdout}\n{train.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def checkpoint_at(self, directory, step):
        paths = list(directory.glob(f"checkpoint-{step:08d}-*.safetensors"))
        self.assertEqual(len(paths), 1)
        return paths[0]

    def assert_same_state(self, first, second):
        arrays_a, meta_a = mx.load(str(first), return_metadata=True)
        arrays_b, meta_b = mx.load(str(second), return_metadata=True)
        self.assertEqual(arrays_a.keys(), arrays_b.keys())
        for key in arrays_a:
            np.testing.assert_allclose(np.asarray(arrays_a[key]), np.asarray(arrays_b[key]),
                                       rtol=1e-6, atol=1e-6, err_msg=key)
        info_a = json.loads(meta_a["checkpoint"])
        info_b = json.loads(meta_b["checkpoint"])
        self.assertEqual(info_a["numpy_rng_state"], info_b["numpy_rng_state"])
        self.assertEqual(info_a["step"], info_b["step"])

    def test_resume_restores_complete_training_state(self):
        resumed = self.root / "resumed"
        shutil.copytree(self.original, resumed)
        result = run(ROOT / "train.py", "--resume", self.checkpoint_at(resumed, 2),
                     "--run-dir", resumed)
        self.assertEqual(result.returncode, 0, result.stderr)
        new_files = list(resumed.glob("checkpoint-00000004-*.safetensors"))
        original_file_name = self.checkpoint_at(self.original, 4).name
        added = [p for p in new_files if p.name != original_file_name]
        self.assertEqual(len(added), 1)
        self.assert_same_state(self.checkpoint_at(self.original, 4), added[0])
        events = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([e["step"] for e in events if e["event"] == "train"], [1, 2, 3, 4])
        resume_event = next(event for event in events if event["event"] == "resume")
        archived = Path(resume_event["previous_log"])
        self.assertEqual(archived.read_text(), (self.original / "metrics.jsonl").read_text())

    def test_join_resume_preserves_state_and_rejects_retroactive_change(self):
        parent = self.checkpoint_at(self.original, 0)
        joined = self.root / "join-uninterrupted"
        resumed = self.root / "join-resumed"
        common = ("--fork-from", parent, "--steps", 4, "--warmup-steps", 0)
        for directory, extra in ((joined, ("--join-warmup-steps", 2)), (resumed, ())):
            result = run(ROOT / "train.py", *common, "--run-dir", directory, *extra)
            self.assertEqual(result.returncode, 0, result.stderr)

        log = resumed / "metrics.jsonl"
        before = log.read_text()
        latest_before = (resumed / "latest.json").read_text()
        rejected = run(ROOT / "train.py", "--resume", self.checkpoint_at(resumed, 4),
                       "--run-dir", resumed, "--join-warmup-steps", 2)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("续训参数 join_warmup_steps", rejected.stderr)
        self.assertEqual(log.read_text(), before)
        self.assertEqual((resumed / "latest.json").read_text(), latest_before)

        checkpoint = self.checkpoint_at(resumed, 2)
        result = run(ROOT / "train.py", "--resume", checkpoint, "--run-dir", resumed,
                     "--join-warmup-steps", 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line) for line in log.read_text().splitlines()]
        resume_event = next(event for event in events if event["event"] == "resume")
        change = resume_event["schedule_change"]
        self.assertEqual(change["join_warmup_steps_before"], 0)
        self.assertEqual(change["join_warmup_steps_after"], 2)
        self.assertEqual(change["first_affected_step"], 3)
        transition = Path(change["checkpoint"])
        self.assertNotEqual(transition, checkpoint)
        self.assert_same_state(checkpoint, transition)
        transition_info = json.loads(mx.load(str(transition), return_metadata=True)[1]["checkpoint"])
        self.assertEqual(transition_info["config"]["join_warmup_steps"], 2)
        self.assertEqual([event["step"] for event in events if event["event"] == "train"], [1, 2, 3, 4])
        latest = json.loads((resumed / "latest.json").read_text())["checkpoint"]
        self.assert_same_state(self.checkpoint_at(joined, 4), resumed / latest)

    def test_equal_forks_are_reproducible(self):
        parent = self.checkpoint_at(self.original, 2)
        results = []
        for name in ("fork-a", "fork-b"):
            target = self.root / name
            result = run(ROOT / "train.py", "--fork-from", parent, "--run-dir", target,
                         "--steps", 2, "--pool-tokens", 256,
                         "--warmup-steps", 0, "--schedule", "constant",
                         "--eval-every", 1, "--checkpoint-every", 2)
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(self.checkpoint_at(target, 2))
        self.assert_same_state(*results)

    def test_distinct_fork_seeds_change_training_rng_but_not_validation_seed(self):
        parent = self.checkpoint_at(self.original, 2)
        checkpoints = []
        for seed in (101, 202):
            target = self.root / f"fork-seed-{seed}"
            result = run(ROOT / "train.py", "--fork-from", parent, "--run-dir", target,
                         "--steps", 1, "--seed", seed, "--pool-tokens", 256,
                         "--warmup-steps", 0, "--schedule", "constant",
                         "--eval-every", 1, "--checkpoint-every", 1)
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoints.append(self.checkpoint_at(target, 1))
        info = [json.loads(mx.load(str(path), return_metadata=True)[1]["checkpoint"])
                for path in checkpoints]
        self.assertNotEqual(info[0]["numpy_rng_state"], info[1]["numpy_rng_state"])
        self.assertEqual(info[0]["config"]["eval_seed"], info[1]["config"]["eval_seed"])
        validation = np.memmap(self.data / "val.bin", dtype="<u2", mode="r")
        batches = [batch(validation, row["config"], np.random.default_rng(row["config"]["eval_seed"]))
                   for row in info]
        np.testing.assert_array_equal(np.asarray(batches[0][0]), np.asarray(batches[1][0]))
        np.testing.assert_array_equal(np.asarray(batches[0][1]), np.asarray(batches[1][1]))

    def test_checkpoint_generates_reproducibly(self):
        model, tokenizer, _ = load_model(self.checkpoint_at(self.original, 4), self.data / "tokenizer.json")
        first = generate(model, tokenizer, "The bird", 8, 0.7, 1, 42)
        second = generate(model, tokenizer, "The bird", 8, 0.7, 1, 42)
        self.assertEqual(first, second)
        self.assertTrue(first["text"].startswith("The bird"))
        self.assertGreater(first["generated_tokens"], 0)
        self.assertLessEqual(first["generated_tokens"], 8)
        logits = mx.array([[0.0, 10.0, -2.0]])
        self.assertEqual(sample_next(logits, temperature=0, top_k=0), 1)
        self.assertEqual(sample_next(logits, temperature=0.7, top_k=1), 1)
        literal = generate(model, tokenizer, "The [EOS] bird", 0, 0, 1, 42)
        self.assertEqual(literal["text"], "The [EOS] bird")

    def test_generate_cli_displays_text_and_real_newlines(self):
        prompt = "The bird\nfound a key."
        args = (ROOT / "generate.py", "--checkpoint", self.checkpoint_at(self.original, 4),
                "--tokenizer", self.data / "tokenizer.json", "--prompt", prompt)
        single = run(*args, "--max-new-tokens", 0)
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertEqual(single.stdout, prompt + "\n")
        multiple = run(*args, "--max-new-tokens", 0, "--samples", 2)
        self.assertEqual(multiple.returncode, 0, multiple.stderr)
        self.assertEqual(multiple.stdout, prompt + "\n\n" + prompt + "\n")
        generated = run(*args, "--max-new-tokens", 8, "--temperature", 0)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        self.assertTrue(generated.stdout.startswith(prompt))
        self.assertNotIn('"generated_tokens":', generated.stdout)

    def test_generate_cli_uses_project_defaults_and_allows_overrides(self):
        project = self.root / "cli-defaults"
        project.mkdir()
        for name in ("generate.py", "model.py", "train.py"):
            shutil.copy2(ROOT / name, project / name)
        run_dir = project / "runs" / "main"
        run_dir.mkdir(parents=True)
        checkpoint = self.checkpoint_at(self.original, 4)
        shutil.copy2(checkpoint, run_dir / checkpoint.name)
        latest = run_dir / "latest.json"
        latest.write_text(json.dumps({"checkpoint": checkpoint.name}), encoding="utf-8")
        tokenizer_dir = project / "artifacts" / "data-clean"
        tokenizer_dir.mkdir(parents=True)
        tokenizer = self.data / "tokenizer.json"
        shutil.copy2(tokenizer, tokenizer_dir / "tokenizer.json")

        command = [sys.executable, str(project / "generate.py"), "--prompt", "The bird",
                   "--temperature", "0.7", "--top-k", "10", "--max-new-tokens", "8"]
        default = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertTrue(default.stdout.startswith("The bird"))

        latest.write_text("invalid JSON", encoding="utf-8")
        explicit = subprocess.run(command + ["--checkpoint", str(checkpoint),
                                  "--tokenizer", str(tokenizer), "--seed", "42"],
                                  cwd=self.root, capture_output=True, text=True)
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(default.stdout, explicit.stdout)
        invalid = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("无法读取默认 checkpoint", invalid.stderr)

    def test_benchmark_compares_greedy_token_sequences(self):
        output = self.root / "benchmark.json"
        result = run(ROOT / "benchmark.py", "--checkpoint", self.checkpoint_at(self.original, 4),
                     "--tokenizer", self.data / "tokenizer.json", "--validation-bin", self.data / "val.bin",
                     "--context", 8, "--new-tokens", 3, "--warmup", 0, "--trials", 1, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text())
        self.assertTrue(report["logits_allclose_rtol_atol_1e-4"])
        self.assertEqual(report["baseline_token_ids"], report["optimized_token_ids"])
        self.assertEqual(len(report["baseline_token_ids"]), 3)

    def test_benchmark_rejects_wrong_validation_input(self):
        payload = bytearray((self.data / "val.bin").read_bytes())
        payload[0] ^= 1
        other = self.root / "other-validation.bin"
        other.write_bytes(payload)
        output = self.root / "wrong-benchmark.json"
        result = run(ROOT / "benchmark.py", "--checkpoint", self.checkpoint_at(self.original, 4),
                     "--tokenizer", self.data / "tokenizer.json", "--validation-bin", other,
                     "--context", 8, "--new-tokens", 3, "--warmup", 0, "--trials", 1, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_full_validation_counts_and_next_token_loss(self):
        model, _, _ = load_model(self.checkpoint_at(self.original, 4), self.data / "tokenizer.json")
        tokens = np.fromfile(self.data / "val.bin", dtype="<u2").astype(np.int32)
        result = evaluate(model, tokens, batch_size=3)
        losses, targets = [], []
        for offset in range(0, len(tokens) - 1, model.config.context_length):
            end = min(offset + model.config.context_length, len(tokens) - 1)
            x = mx.array(tokens[offset:end][None, :], dtype=mx.int32)
            y = mx.array(tokens[offset + 1:end + 1][None, :], dtype=mx.int32)
            row = nn.losses.cross_entropy(model(x), y, reduction="none")
            mx.eval(row)
            losses.extend(np.asarray(row).reshape(-1).tolist())
            targets.extend(tokens[offset + 1:end + 1].tolist())
        losses = np.asarray(losses)
        targets = np.asarray(targets)
        self.assertTrue(result["complete"])
        self.assertEqual(result["all_target_tokens"], len(tokens) - 1)
        self.assertEqual(result["content_target_tokens"], int((targets >= 4).sum()))
        self.assertAlmostEqual(result["all_token_nll"], float(losses.mean()), places=5)
        self.assertAlmostEqual(result["content_token_nll"], float(losses[targets >= 4].mean()), places=5)

    def test_replay_validation_matches_log_and_rejects_wrong_data(self):
        checkpoint = self.checkpoint_at(self.original, 4)
        result = run(ROOT / "replay_validation.py", "--checkpoint", checkpoint, "--data-dir", self.data)
        self.assertEqual(result.returncode, 0, result.stderr)
        replay = json.loads(result.stdout)
        events = [json.loads(line) for line in (self.original / "metrics.jsonl").read_text().splitlines()]
        logged = next(event["loss"] for event in events if event["event"] == "val" and event["step"] == 4)
        self.assertAlmostEqual(replay["loss"], logged, places=6)
        self.assertEqual(replay["target_tokens"], 2 * 2 * 8)

        wrong = self.root / "wrong-replay-data"
        wrong.mkdir()
        shutil.copy2(self.data / "tokenizer.json", wrong / "tokenizer.json")
        payload = bytearray((self.data / "val.bin").read_bytes())
        payload[0] ^= 1
        (wrong / "val.bin").write_bytes(payload)
        rejected = run(ROOT / "replay_validation.py", "--checkpoint", checkpoint, "--data-dir", wrong)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("验证文件与检查点记录的数据不一致", rejected.stderr)

    def test_paired_experiment_uses_same_budget_and_full_validation(self):
        root = self.root / "paired-experiment"
        result = run(ROOT / "run_diversity.py", "--checkpoint", self.checkpoint_at(self.original, 4),
                     "--data-dir", self.data, "--output-root", root, "--small-pool", 256,
                     "--large-pool", 512, "--steps", 1, "--batch-size", 2, "--grad-accum", 1,
                     "--learning-rate", 0.001, "--seeds", 77, "--eval-every", 1,
                     "--eval-batches", 1, "--full-eval-batch-size", 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((root / "summary.json").read_text())
        pair = summary["pairs"][0]
        small = pair["measurements"]["small"]
        large = pair["measurements"]["large"]
        self.assertEqual(small["coverage"]["processed_input_token_positions"],
                         large["coverage"]["processed_input_token_positions"])
        self.assertEqual(small["coverage"]["training_pool_token_positions"], 256)
        self.assertEqual(large["coverage"]["training_pool_token_positions"], 512)
        self.assertTrue(small["evaluation"]["complete"])
        self.assertTrue(large["evaluation"]["complete"])
        self.assertTrue(pair["verification"]["same_initial_model_optimizer_and_mlx_rng"])
        self.assertTrue(pair["verification"]["same_training_sampler_rng"])
        self.assertEqual(pair["verification"]["only_config_difference"], "pool_tokens")
        self.assertAlmostEqual(pair["small_minus_large_content_nll"],
                               small["evaluation"]["content_token_nll"] - large["evaluation"]["content_token_nll"])

    def test_report_rejects_missing_training_step(self):
        copy = self.root / "incomplete-log"
        shutil.copytree(self.original, copy)
        info = checkpoint_info(self.checkpoint_at(copy, 4))
        complete, _ = check_log(copy, info)
        self.assertEqual(len(complete), 4)
        path = copy / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows
                                if not (row["event"] == "train" and row["step"] == 3)))
        with self.assertRaises(ValueError):
            check_log(copy, info)

    def test_workflow_refuses_incomplete_main_training(self):
        incomplete = self.root / "workflow-incomplete"
        shutil.copytree(self.original, incomplete)
        checkpoint = self.checkpoint_at(incomplete, 2)
        (incomplete / "latest.json").write_text(
            json.dumps({"checkpoint": checkpoint.name, "step": 2}), encoding="utf-8"
        )
        status = self.root / "workflow-status.json"
        experiment = self.root / "workflow-experiment"
        result = run(ROOT / "complete_workflow.py", "--trainer-pid", 999999,
                     "--run-dir", incomplete, "--data-dir", self.data,
                     "--experiment-root", experiment, "--status", status)
        self.assertNotEqual(result.returncode, 0)
        state = json.loads(status.read_text())
        self.assertEqual(state["phase"], "失败")
        self.assertIn("未达到目标", state["error"])
        self.assertFalse(experiment.exists())

    def test_resume_rejects_incompatible_configuration(self):
        resumed = self.root / "invalid-resume"
        shutil.copytree(self.original, resumed)
        result = run(ROOT / "train.py", "--resume", self.checkpoint_at(resumed, 2),
                     "--run-dir", resumed, "--learning-rate", 0.02)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(list(resumed.glob("checkpoint-*.safetensors"))), 3)


if __name__ == "__main__":
    unittest.main()
