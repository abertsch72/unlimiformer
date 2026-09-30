import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def import_unlimiformer_with_transformers_stub():
    transformer_stub = types.ModuleType("transformers")
    model_names = (
        "BartModel", "BartForConditionalGeneration", "T5Model",
        "T5ForConditionalGeneration", "LEDModel", "LEDForConditionalGeneration",
        "AutoModelForCausalLM", "AutoModelForSeq2SeqLM", "LlamaModel", "LlamaForCausalLM",
    )
    for name in model_names:
        setattr(transformer_stub, name, type(name, (), {}))
    transformer_stub.MODEL_WITH_LM_HEAD_MAPPING = {}
    transformer_stub.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = {}
    sys.modules.pop("unlimiformer", None)
    with patch.dict(sys.modules, {"transformers": transformer_stub}):
        return importlib.import_module("unlimiformer")


class DatastoreReconstructionTests(unittest.TestCase):
    def test_import_does_not_require_optional_faiss(self):
        import index_building

        missing_faiss = ModuleNotFoundError("No module named 'faiss'")
        with patch.object(index_building, "faiss", None), patch.object(
            index_building, "_faiss_import_error", missing_faiss
        ):
            with self.assertRaisesRegex(ImportError, "Faiss is required when using the datastore"):
                index_building.Datastore(dim=2)

    def test_search_and_reconstruct_calls_the_ivf_index(self):
        from index_building import Datastore

        class FakeIndex:
            def __init__(self):
                self.query = None

            def search_and_reconstruct(self, query, k):
                self.query = query
                return "scores", "indices", "vectors"

        datastore = Datastore.__new__(Datastore)
        datastore.dimension = 2
        datastore.use_flat_index = False
        datastore.index = FakeIndex()

        result = datastore.search_and_reconstruct(torch.ones((1, 2), dtype=torch.float16), 1)

        self.assertEqual(result, ("scores", "indices", "vectors"))
        self.assertEqual(datastore.index.query.dtype, torch.float32)
        self.assertEqual(datastore.index.query.device.type, "cpu")

    def test_flat_reconstruction_looks_up_the_selected_vectors(self):
        from index_building import Datastore

        datastore = Datastore.__new__(Datastore)
        datastore.dimension = 2
        datastore.use_flat_index = True
        datastore.keys = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        datastore.search = lambda queries, k: ("scores", torch.tensor([[2, 0]]))

        scores, indices, vectors = datastore.search_and_reconstruct(torch.ones((1, 2)), 2)

        self.assertEqual(scores, "scores")
        self.assertTrue(torch.equal(indices, torch.tensor([[2, 0]])))
        self.assertTrue(torch.equal(vectors, torch.tensor([[[5., 6.], [1., 2.]]])))

    def test_flat_index_keeps_source_vectors_when_reconstruction_is_enabled(self):
        import index_building

        keys = torch.tensor([[1., 2.], [3., 4.]])
        with patch.object(index_building, "faiss", types.SimpleNamespace()):
            datastore = index_building.Datastore(dim=2, use_flat_index=True)
            datastore.train_index(keys, keep_keys=False)

        self.assertIs(datastore.keys, keys)

    def test_ivf_reconstruction_releases_keys_and_clamps_with_index_size(self):
        import index_building

        class FakeIndex:
            def __init__(self, *args):
                self.ntotal = 0
                self.nprobe = 0

            def train(self, keys):
                self.training_rows = keys.shape[0]

            def add(self, keys):
                self.ntotal += keys.shape[0]

            def search(self, queries, k):
                values = torch.tensor([[0, self.ntotal - 1, -1]])[:, :k]
                return torch.ones_like(values, dtype=torch.float32), values

        fake_faiss = types.SimpleNamespace(
            IndexFlatIP=lambda dim: FakeIndex(),
            IndexIVFPQ=lambda quantizer, dim, ncentroids, size, bits: FakeIndex(),
        )
        with patch.object(index_building, "faiss", fake_faiss):
            datastore = index_building.Datastore(dim=64)
            datastore.train_index(torch.ones((256, 64)), keep_keys=False)
            self.assertIsNone(datastore.keys)

            _, values = datastore.search(torch.ones((1, 64)), k=3)

        self.assertTrue(torch.equal(values, torch.tensor([[0, 255, 0]])))

    def test_llama_reconstruction_uses_saved_sequence_length_after_release(self):
        unlimiformer = import_unlimiformer_with_transformers_stub()

        class FakeRotaryEmbedding:
            seq_len = None

            def __call__(self, values, seq_len):
                self.seq_len = seq_len
                cos = torch.zeros((1, 1, seq_len, values.shape[-1]))
                sin = torch.zeros_like(cos)
                cos[..., 0] = 1
                return cos, sin

        rotary = FakeRotaryEmbedding()
        attention = types.SimpleNamespace(rotary_emb=rotary)
        model = types.SimpleNamespace(
            base_model=types.SimpleNamespace(
                layers=[types.SimpleNamespace(self_attn=attention)]
            )
        )
        wrapper = unlimiformer.UnlimiformerLLaMa.__new__(unlimiformer.UnlimiformerLLaMa)
        wrapper.model = model
        wrapper.num_heads = 1
        wrapper.hidden_states = [None]
        wrapper.datastore_sequence_length = 3
        wrapper.prompt_input_ids = torch.ones((1, 3), dtype=torch.long)
        wrapper.actual_model_window_size = 3
        projection = torch.nn.Linear(2, 2, bias=False)
        projection.weight.data.copy_(torch.eye(2))

        keys, values = wrapper.post_process_retrieved(
            torch.ones((1, 1, 1, 1, 2)),
            projection,
            projection,
            torch.tensor([[[[1]]]]),
        )

        self.assertEqual(rotary.seq_len, 3)
        self.assertEqual(tuple(keys.shape), (1, 1, 1, 1, 2))
        self.assertEqual(tuple(values.shape), (1, 1, 1, 1, 2))

    def test_reconstruction_mode_still_trains_index_with_input_vectors(self):
        unlimiformer = import_unlimiformer_with_transformers_stub()

        class FakeModel:
            config = types.SimpleNamespace(is_encoder_decoder=True, hidden_size=2, num_hidden_layers=1)

            def __call__(self, *args, **kwargs):
                return None

        class RecordingDatastoreBatch:
            instances = []

            def __init__(self, **kwargs):
                self.keys = None
                self.instances.append(self)

            def train_index(self, keys, keep_keys=True):
                self.keys = keys
                self.keep_keys = keep_keys

        wrapper = unlimiformer.Unlimiformer.__new__(unlimiformer.Unlimiformer)
        wrapper.use_datastore = True
        wrapper.is_encoder_decoder = True
        wrapper.flat_index = False
        wrapper.gpu_index = False
        wrapper.index_devices = [torch.device("cpu")]
        wrapper.datastore_device = torch.device("cpu")
        wrapper.gpu_datastore = False
        wrapper.reconstruct_embeddings = True
        wrapper.test_datastore = False
        wrapper.model = FakeModel()
        wrapper.device = torch.device("cpu")
        wrapper.layer_begin = 0
        wrapper.layer_end = None
        wrapper.original_decoder_layer_cross_attn_forward_funcs = []
        wrapper.activation_capturer = [types.SimpleNamespace(captured=torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]]))]
        wrapper.save_heatmap = False
        wrapper.verbose = False
        wrapper.window_indices = lambda seq_len: [(0, seq_len, 0, seq_len)]
        RecordingDatastoreBatch.instances.clear()

        with patch.object(unlimiformer, "DatastoreBatch", RecordingDatastoreBatch):
            wrapper.reset_memory(
                torch.ones((1, 3), dtype=torch.long),
                torch.ones((1, 3), dtype=torch.long),
            )

        trained_keys = RecordingDatastoreBatch.instances[0].keys
        self.assertIsInstance(trained_keys, torch.Tensor)
        self.assertEqual(tuple(trained_keys.shape), (1, 3, 2))
        self.assertTrue(torch.equal(trained_keys, wrapper.activation_capturer[0].captured))
        self.assertFalse(RecordingDatastoreBatch.instances[0].keep_keys)
        self.assertEqual(wrapper.hidden_states, [None])
        self.assertEqual(wrapper.datastore_sequence_length, 3)


if __name__ == "__main__":
    unittest.main()
