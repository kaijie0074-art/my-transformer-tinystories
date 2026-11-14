import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "prepare_data.py"


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = [
            "A tiny bird found a blue ribbon.",
            "The ribbon fell into the water.\nThe bird called for help.",
            "Café, garçon, and the little cat.",
            "A star shines over the house.",
            "The sign literally says [PAD] [BOS] [EOS] and [UNK].",
        ]
        self.val = ["没有见过的字 🐱", "The green fox listened quietly."]
        for name, stories in (("train.jsonl", self.train), ("val.jsonl", self.val)):
            (self.root / name).write_text(
                "".join(json.dumps({"text": s}, ensure_ascii=False) + "\n" for s in stories),
                encoding="utf-8",
            )
        self.out = self.root / "output"

    def prepare(self, *additional):
        return subprocess.run(
            [sys.executable, str(PREPARE), "--input-jsonl",
             str(self.root / "train.jsonl"), str(self.root / "val.jsonl"),
             "--output-dir", str(self.out), "--vocab-size", "512", "--batch-size", "2",
             *additional],
            cwd=ROOT, capture_output=True, text=True,
        )

    def test_round_trip_special_tokens_split_and_hash(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        tokenizer = Tokenizer.from_file(str(self.out / "tokenizer.json"))
        tokenizer.encode_special_tokens = True
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dtype"], "<u2")
        self.assertEqual(manifest["train"]["stories"], len(self.train))
        self.assertEqual(manifest["validation"]["stories"], len(self.val))
        for key, filename, stories in (
            ("train", "train.bin", self.train),
            ("validation", "val.bin", self.val),
        ):
            payload = (self.out / filename).read_bytes()
            actual = np.frombuffer(payload, dtype="<u2").tolist()
            expected = []
            for story in stories:
                ids = tokenizer.encode(story).ids
                self.assertEqual(ids[0], 1)
                self.assertEqual(ids[-1], 2)
                self.assertTrue(all(token_id not in {0, 1, 2, 3} for token_id in ids[1:-1]))
                self.assertNotIn(3, ids)
                self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), story)
                expected.extend(ids)
            self.assertEqual(actual, expected)
            self.assertEqual(manifest[key]["tokens"], len(expected))
            self.assertEqual(manifest[key]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_existing_data_is_not_overwritten(self):
        first = self.prepare()
        self.assertEqual(first.returncode, 0, first.stderr)
        before = hashlib.sha256((self.out / "train.bin").read_bytes()).hexdigest()
        second = self.prepare()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(hashlib.sha256((self.out / "train.bin").read_bytes()).hexdigest(), before)

    def test_rejects_vocabulary_without_byte_alphabet(self):
        result = self.prepare("--vocab-size", "259")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.out / "manifest.json").exists())

    def test_rejects_train_validation_alias(self):
        alias = self.root / "alias.jsonl"
        alias.symlink_to(self.root / "train.jsonl")
        result = subprocess.run(
            [sys.executable, str(PREPARE), "--input-jsonl", str(self.root / "train.jsonl"),
             str(alias), "--output-dir", str(self.out), "--vocab-size", "512"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
