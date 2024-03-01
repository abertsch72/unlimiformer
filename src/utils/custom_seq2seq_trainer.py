import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from transformers.debug_utils import DebugOption
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import speed_metrics

from transformers.utils import logging
from transformers import Seq2SeqTrainer, is_torch_tpu_available

import gc

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


from utils.decoding import decode

logger = logging.get_logger(__name__)


def _clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# This custom trainer is based on the trainer defined in https://github.com/huggingface/transformers/compare/main...eladsegal:public-transformers:scrolls
class CustomTrainer(Seq2SeqTrainer):
    def __init__(
        self, *args, untokenized_eval_dataset=None, data_args=None, output_dir: Optional[str] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._untokenized_eval_dataset = untokenized_eval_dataset
        self._max_length = data_args.val_max_target_length
        self._num_beams = data_args.num_beams
        self._output_dir = output_dir
        self._data_args = data_args
        self.mock_predictions_to_assign_zero_metric_score = self.tokenizer.encode("TOO_MANY_INPUT_TOKENS",return_tensors="np")[0]

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to ret`urn the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not ("labels" in inputs or 'decoder_input_ids' in inputs):
            if model.training:
                logger.warning('When computing loss, must give labels or decoder_input_ids. '
                           'If you only perform prediction, you can safely ignore this message')
            # This is an issue here because the input may be longer than the max-output length of the model,
            # and if nothing was given it will shift the input and use it to compute loss (and later discard it).
            # This may cause an indexing error when absolute embeddings are used (CUDA device side assert)
            inputs['decoder_input_ids'] = inputs['input_ids'][:,:2].clone()  # dummy outputs

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # --------------------- addition compared to the source file --------------------
        if 'prefix_length' in inputs:
            gen_kwargs['prefix_length'] = inputs['prefix_length']
        _clean_memory()
        # ------------------------------------------------------------------------------

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

         # Uri: to make sure we use cache even during mid-training evaluation, where this is disabled in general:
        gen_kwargs['use_cache'] = True
        
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # --------------------- addition compared to the source file --------------------
        _clean_memory()
        # ------------------------------------------------------------------------------
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        if has_labels:  # changed the order of the if's here because there is no point going through the model if there are no labels to compute the loss on..
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        else:
            loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    @property
    def _restart_generator(self):
        if getattr(self, '_is_restart_generator', False):
            self._is_restart_generator = False
            return True
        return False

    def set_restart_generator(self):
        self._is_restart_generator = True

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        sampler = super()._get_train_sampler()
        try:
            if self._restart_generator:
                sampler.generator.manual_seed(self._initial_seed)
            else:
                self._initial_seed = sampler.generator.initial_seed()
        except Exception as e:
            logger.warning(f'Cannot save or set the seed of the generator: {e}')
        return sampler

    def _post_process_function(self, untokenized_eval_dataset, predictions):
        id_to_prediction = {}
        id_to_label_ids = defaultdict(list)

        assert len(untokenized_eval_dataset) == len(self.eval_dataset)

        for i, (instance, not_valid_for_eval) in enumerate(zip(untokenized_eval_dataset, self.eval_dataset["not_valid_for_eval"])):
            if not_valid_for_eval:
                id_to_prediction[instance["id"]] = self.mock_predictions_to_assign_zero_metric_score
            else:
                id_to_prediction[instance["id"]] = predictions[i]

            if "outputs" in instance:
                id_to_label_ids[instance["id"]] = instance["outputs"]
            else:
                id_to_label_ids[instance["id"]].append(instance["output"])

        return id_to_prediction, id_to_label_ids

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        untokenized_eval_dataset: Optional[Dataset] = None,
            **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # ----------------------------------- Added -----------------------------------
        untokenized_eval_dataset = (
            self._untokenized_eval_dataset if untokenized_eval_dataset is None else untokenized_eval_dataset
        )
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        # -----------------------------------------------------------------------------

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=None,  # MODIFIED since we need the predictions
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # ----------------------------------- Added -----------------------------------
            # revert the compute metrics back
            self.compute_metrics = compute_metrics
            # -----------------------------------------------------------------------------

        # ----------------------------------- Added -----------------------------------
        # compute our metrics
        if output.predictions is not None:
            eval_preds = self._post_process_function(untokenized_eval_dataset, output.predictions)

            if self._output_dir is not None and self.is_world_process_zero():
                predictions = decode(eval_preds[0], self.tokenizer, self._data_args)
                output_prediction_file = os.path.join(
                    self._output_dir, f"generated_predictions_eval_{self.state.global_step}.json"
                )
                with open(output_prediction_file, "w") as writer:
                    json.dump(predictions, writer, indent=4)

                output_labels_file = os.path.join(
                    self._output_dir, f"eval_labels.json"
                )
                if not os.path.isfile(output_labels_file):
                    with open(output_labels_file, "w") as writer:
                        json.dump(eval_preds[1], writer, indent=4)

            if self.compute_metrics is not None:
                output.metrics.update(self.compute_metrics(*eval_preds))

            # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)
        # -----------------------------------------------------------------------------

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
