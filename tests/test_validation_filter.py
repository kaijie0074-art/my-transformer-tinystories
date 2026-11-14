import unittest

import pyarrow as pa

from clean_validation import remove_exact_overlap


class ValidationFilterTests(unittest.TestCase):
    def test_exact_overlap_is_removed_without_fuzzy_matching(self):
        train = pa.table({"text": ["first story", "shared story", "shared story"]})
        validation = pa.table({"text": ["shared story", "new story", "shared story", "shared story "]})
        clean, counts = remove_exact_overlap(train, validation)
        self.assertEqual(clean, ["new story", "shared story "])
        self.assertEqual(counts["original_validation_stories"], 4)
        self.assertEqual(counts["removed_validation_stories"], 2)
        self.assertEqual(counts["matching_train_stories"], 2)
        self.assertEqual(counts["distinct_shared_stories"], 1)
        self.assertFalse(counts["near_duplicates_checked"])

    def test_disjoint_sets_remain_unchanged(self):
        train = pa.table({"text": ["train only"]})
        validation = pa.table({"text": ["validation only", "another validation"]})
        clean, counts = remove_exact_overlap(train, validation)
        self.assertEqual(clean, ["validation only", "another validation"])
        self.assertEqual(counts["removed_validation_stories"], 0)


if __name__ == "__main__":
    unittest.main()
