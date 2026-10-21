import sys
import unittest
from pathlib import Path

import torch
from transformers import BartConfig, BartForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlimiformer import Unlimiformer


class ChunkStrideTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(0)
        return BartForConditionalGeneration(BartConfig(
            vocab_size=32, d_model=8, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_position_embeddings=32, _attn_implementation="eager",
        ))

    def test_nonpositive_stride_is_rejected_before_hooks(self):
        for chunk_size, overlap in [(32, 1), (32, 1.5), (0, 0)]:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                model = self.make_model()
                original_train, original_eval = model.train.__func__, model.eval.__func__
                with self.assertRaisesRegex(ValueError, "positive chunk stride"):
                    Unlimiformer.convert_model(
                        model, model_encoder_max_len=chunk_size, chunk_overlap=overlap,
                        gpu_datastore=False, gpu_index=False,
                    )
                self.assertIs(model.train.__func__, original_train)
                self.assertIs(model.eval.__func__, original_eval)

    def test_positive_strides_still_encode_long_inputs(self):
        torch.set_num_threads(1)
        for chunk_size, overlap in [(32, 0), (32, 0.5), (31, 1)]:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                model = Unlimiformer.convert_model(
                    self.make_model(), model_encoder_max_len=chunk_size,
                    chunk_overlap=overlap, gpu_datastore=False, gpu_index=False,
                )
                model.eval()
                inputs = torch.arange(65).remainder(28).add(4).unsqueeze(0)
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=2, num_beams=1, do_sample=False,
                    )
                self.assertEqual(output.shape[0], 1)
                self.assertEqual(model.unlimiformer.window_indices(65)[-1][1], 65)


if __name__ == "__main__":
    unittest.main()
