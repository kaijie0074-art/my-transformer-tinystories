import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coverage_analysis import analyze, union_length


class CoverageTests(unittest.TestCase):
    def test_interval_union_counts_overlap_once(self):
        self.assertEqual(union_length(np.array([], dtype=np.int64), 4), 0)
        self.assertEqual(union_length(np.array([9, 0, 3, 3]), 4), 11)

    def test_replayed_positions_match_explicit_sample_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            run = root / "run"
            data.mkdir()
            run.mkdir()
            (data / "manifest.json").write_text(json.dumps({"train": {"tokens": 25}}))
            cfg = {"data_dir": str(data), "seed": 52, "pool_tokens": 25,
                   "context_length": 4, "grad_accum": 2, "batch_size": 2}
            events = [{"event": "config", "config": cfg, "parent_checkpoint": {"sampling_rng": "new_seed"}},
                      {"event": "train", "step": 1, "tokens": 16},
                      {"event": "train", "step": 2, "tokens": 32}]
            (run / "metrics.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            rng = np.random.default_rng(cfg["seed"])
            positions = set()
            for _ in range(4):
                for start in rng.integers(0, 21, size=2):
                    positions.update(range(int(start), int(start) + 4))
            result = analyze(run)
            self.assertEqual(result["processed_input_token_positions"], 32)
            self.assertEqual(result["distinct_input_token_positions"], len(positions))
            self.assertEqual(result["repeated_input_token_positions"], 32 - len(positions))


if __name__ == "__main__":
    unittest.main()
