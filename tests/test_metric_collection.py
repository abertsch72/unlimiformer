import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def import_metrics_module():
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_metric = lambda *args, **kwargs: None
    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = lambda *args, **kwargs: None

    previous_module = sys.modules.pop("metrics.metrics", None)
    with patch.dict(sys.modules, {"datasets": datasets_stub, "huggingface_hub": hub_stub}):
        module = importlib.import_module("metrics.metrics")
    sys.modules["metrics.metrics"] = module
    return module, previous_module


class MetricCollectionTests(unittest.TestCase):
    def test_raw_metric_still_gets_character_length_summary(self):
        metrics_module, previous_module = import_metrics_module()

        def restore_module():
            sys.modules.pop("metrics.metrics", None)
            if previous_module is not None:
                sys.modules["metrics.metrics"] = previous_module

        self.addCleanup(restore_module)

        class RawMetric(metrics_module.Metric):
            def _compute_metrics(self, id_to_pred, id_to_labels):
                return {"count": len(id_to_pred)}

        class Tokenizer:
            pad_token_id = 0

            def decode(self, token_ids, **kwargs):
                return "decoded prediction"

        tokenizer = Tokenizer()
        collection = metrics_module.MetricCollection(
            [RawMetric(tokenizer=tokenizer)], tokenizer=tokenizer
        )
        result = collection({0: [1, 2]}, {0: [3]})

        self.assertEqual(result["mean_prediction_length_characters"], 18)


if __name__ == "__main__":
    unittest.main()
