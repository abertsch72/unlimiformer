import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.tokenized_inputs import preprocess_tokenized_inputs


class TokenizedInputTests(unittest.TestCase):
    def test_truncates_ids_and_masks_and_marks_long_evaluation_rows(self):
        result = preprocess_tokenized_inputs(
            {
                "input": [[1, 2, 3, 4], [5, 6]],
                "attention_mask": [[1, 1, 1, 1], [1, 1]],
            },
            input_column="input",
            max_source_length=3,
            assign_zero_to_too_long_val_examples=True,
        )

        self.assertEqual(result["input_ids"], [[1, 2, 3], [5, 6]])
        self.assertEqual(result["attention_mask"], [[1, 1, 1], [1, 1]])
        self.assertEqual(result["not_valid_for_eval"], [True, False])

    def test_builds_a_mask_when_pre_tokenized_data_does_not_include_one(self):
        result = preprocess_tokenized_inputs(
            {"input": [[1, 2], [3]]},
            input_column="input",
            max_source_length=None,
            assign_zero_to_too_long_val_examples=False,
        )

        self.assertEqual(result["attention_mask"], [[1, 1], [1]])
        self.assertEqual(result["not_valid_for_eval"], [False, False])

    def test_rejects_mismatched_input_and_attention_mask_lengths(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            preprocess_tokenized_inputs(
                {"input": [[1, 2]], "attention_mask": [[1]]},
                input_column="input",
                max_source_length=None,
                assign_zero_to_too_long_val_examples=False,
            )


if __name__ == "__main__":
    unittest.main()
