#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from scipy.stats import boltzmann
import logging
import os
import sys

import numpy as np
from unlimiformer import Unlimiformer
from random_training_unlimiformer import RandomTrainingUnlimiformer

import nltk

# we import the logging frameworks before any other import to make sure all monkey patching for the logging are active
# from sled import SledConfig

import wandb
import torch

sys.path.insert(0, os.path.dirname(__file__))  # seq2seq package path
sys.path.insert(0, os.getcwd())

from dataclasses import dataclass, field
from typing import List, Optional
import json
from copy import deepcopy
import torch.nn.functional as F

import datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed, WEIGHTS_NAME,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import DataCollatorForSeq2Seq

from datasets import load_dataset

# noinspection PyUnresolvedReferences
# import sled  # *** required so that SledModels will be registered for the AutoClasses ***

from utils.config import handle_args_to_ignore
from utils.decoding import decode
from metrics import load_metric
from utils.duplicates import drop_duplicates_in_input
from utils.override_training_args import TrainingOverridesArguments
from utils.custom_seq2seq_trainer import CustomTrainer
from utils.custom_hf_argument_parser import CustomHfArgumentParser
from metrics.metrics import HFMetricWrapper, MetricCollection

logger = logging.getLogger('sled')

PREFIX_DOC_SEP = '\n\n'

DEBUG = os.environ.get('DEBUG', 'false').lower() in {'1', 'true', 'yes'}  # If set, will set some configuration to help debug
if DEBUG:
    assert not torch.cuda.is_available() or torch.cuda.device_count() == 1


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    drop_duplicates_in_eval: bool = field(
        default=True,
    )

    def __post_init__(self):
        pass
    


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library) or name of the file in src/data."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    metric_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The name of the metric to use (from src/metrics)."},
    )
    input_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    input_prefix_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the input prefix (e.g. questions), when those exist."},
    )
    output_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    eval_max_source_length: Optional[int] = field(
        default=None,
        metadata={"help": "if None, will be same as max_source_length"},
    )
    max_prefix_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "The maximum total input_prefix sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded from the left "
                    "(only used if prefixes are not merged)."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Defining the data_dir of the dataset configuration."},
    )
    download_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defining the download_mode when loading the dataset. Options are `reuse_dataset_if_exists` (default), `reuse_cache_if_exists` and `force_redownload`."
        },
    )
    evaluate_on_training_data: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate on training data or not, to make sure the model can overfit."},
    )
    folder_suffix: str = field(
        default="",
        metadata={"help": "args to be suffixes for the output folder of the run"},
    )
    preprocess_only: bool = field(
        default=False,
        metadata={"help": "Preprocess only: Don't start training, just do the things before"},
    )
    assign_zero_to_too_long_val_examples: bool = field(
        default=False,
        metadata={
            "help": "If true, all sequences longer then max_source_length will be assign a score of 0 in the metric evaluation"
        },
    )
    shared_storage: bool = field(
        default=True,
        metadata={"help": "Whether nodes share the same storage"},
    )
    trim_very_long_strings: bool = field(
        default=False,
        metadata={"help": "Whether to trim very long strings before tokenizing them"},
    )
    pad_prefix: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad the prefix if it exists to max_prefix_length. "
                    "Note - important if you are using a SLED model on an input that contains an input_prefix"
        },
    )
    test_start_ind: Optional[int] = field(
        default=None,
        metadata={"help": "if given, uses the test set starting from this index"},
    )
    test_end_ind: Optional[int] = field(
        default=None,
        metadata={"help": "if given, uses the test set ending at this index"},
    )
    # Uri:
    patience: Optional[int] = field(
        default=None,
    )
    length_penalty: Optional[float] = field(
        default=None,
    )
    extra_metrics: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The name of the metric to use (from src/metrics)."},
    )
    chunked_training_size: Optional[int] = field(
        default=None,
    )
    oracle_training: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, train on the input sentences that provide the highest ROUGE score with the labels"}
    )
    oracle_merge: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, merge the oracle dataset and the standard training dataset"}
    )
    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.pad_prefix and self.max_prefix_length == 0:
            raise ValueError('When padding prefix, you must set a max_prefix_length')
        assert self.max_prefix_length == 0 or self.max_prefix_length <= 0.5*self.max_source_length,\
            'If max_prefix_length is given, it must be much shorter than the total input'
        # Uri: 
        if self.eval_max_source_length is None:
            self.eval_max_source_length = self.max_source_length


@dataclass
class UnlimiformerArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    test_unlimiformer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use KNN."
        },
    )
    unlimiformer_verbose: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to print KNN intermediate predictions (mostly for debugging)."
        },
    )
    layer_begin: Optional[int] = field(
        default=0,
        metadata={"help": "The layer to begin applying KNN to. KNN will be applied to layers[knn_layer_begin:layer_end]. "
                          "By default, it will be applied to all layers: [0:None]]"}, 
    )
    layer_end: Optional[int] = field(
        default=None,
        metadata={"help": "The layer to end applying KNN to. KNN will be applied to layers[knn_layer_begin:layer_end]. "
                          "By default, it will be applied to all layers: [0:None]]"}, 
    )
    unlimiformer_chunk_overlap: Optional[float] = field(
        default=0.5,
        metadata={"help": "The fraction of overlap between input chunks"},
    )
    unlimiformer_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "The size of each input chunk"},
    )
    unlimiformer_head_num: Optional[int] = field(
        default=None,
        metadata={"help": "The head to apply KNN to (if None, apply to all heads)"},
    )
    unlimiformer_exclude: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, prioritize the inputs that are **not** in the standard attention window."
        },
    )
    random_unlimiformer_training: Optional[bool] = field(
        default=False,
    )
    unlimiformer_training: Optional[bool] = field(
        default=False,
    )
    use_datastore: Optional[bool] = field(default=False)
    flat_index: Optional[bool] = field(default=False)
    test_datastore: Optional[bool] = field(default=False)
    reconstruct_embeddings: Optional[bool] = field(default=False)
    gpu_datastore: Optional[bool] = field(default=True)
    gpu_index: Optional[bool] = field(default=True)


def main():
    handle_args_to_ignore(sys.argv)  # Just for sweeps

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = CustomHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingOverridesArguments, UnlimiformerArguments))
    model_args, data_args, training_args, unlimiformer_args = parser.parse_dictionary_and_args()
    
    set_up_logging(training_args)
    logger.info(f"Training Arguments: {training_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Unlimiformer Arguments: {unlimiformer_args}")


    # Added to avoid wandb.errors.UsageError: Error communicating with wandb process
    wandb.init(settings=wandb.Settings(start_method="fork"), name=training_args.output_dir)

    # Used to find missing dependencies early on
    load_metric(data_args.metric_names, **locals())
    load_extra_metrics(data_args.extra_metrics)

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = _detect_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    seq2seq_dataset = _get_dataset(data_args, model_args, training_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_name = None
    if model_args.config_name:
        config_name = model_args.config_name
    else:
        if os.path.isfile(model_args.model_name_or_path):
            config_name = os.path.dirname(model_args.model_name_or_path)
        else:
            config_name = model_args.model_name_or_path

    config_overrides = {}
    if training_args.gradient_checkpointing is not None:
        config_overrides["gradient_checkpointing"] = training_args.gradient_checkpointing

    config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=training_args.use_auth_token,
        **config_overrides
    )
    # override for sled models to make sure we are explicit in our request
    # if isinstance(config, SledConfig) and (not data_args.pad_prefix or data_args.max_prefix_length == 0):
    #     logger.warning('Setting prepend_prefix to False if using a SLED model, as the input does not have a prefix or '
    #                    'pad_prefix is False (all prefixes must be of the same length for SLED). If you do not use SLED '
    #                    'or finetune on a dataset with no prefixes, ignore this warning')
    #     config.prepend_prefix = False

    if model_args.model_name_or_path is None:
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0 and training_args.fp16_padding:
            config.vocab_size += 8 - (config.vocab_size % 8)

    tokenizer_name = None
    if model_args.tokenizer_name:
        tokenizer_name = model_args.tokenizer_name
    else:
        if os.path.isfile(model_args.model_name_or_path):
            tokenizer_name = os.path.dirname(model_args.model_name_or_path)
        else:
            tokenizer_name = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=training_args.use_auth_token,
    )
    if model_args.model_name_or_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=training_args.use_auth_token,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_config(
            config,
        )
    if unlimiformer_args.test_unlimiformer:
        unlimiformer_kwargs = {
            'layer_begin': unlimiformer_args.layer_begin, 
            'layer_end': unlimiformer_args.layer_end,
            'unlimiformer_head_num': unlimiformer_args.unlimiformer_head_num, 
            'exclude_attention': unlimiformer_args.unlimiformer_exclude, 
            'chunk_overlap': unlimiformer_args.unlimiformer_chunk_overlap,
            'model_encoder_max_len': unlimiformer_args.unlimiformer_chunk_size,
            'verbose': unlimiformer_args.unlimiformer_verbose, 'tokenizer': tokenizer,
            'unlimiformer_training': unlimiformer_args.unlimiformer_training,
            'use_datastore': unlimiformer_args.use_datastore,
            'flat_index': unlimiformer_args.flat_index,
            'test_datastore': unlimiformer_args.test_datastore,
            'reconstruct_embeddings': unlimiformer_args.reconstruct_embeddings,
            'gpu_datastore': unlimiformer_args.gpu_datastore,
            'gpu_index': unlimiformer_args.gpu_index
        }
        if unlimiformer_args.random_unlimiformer_training:
            model = RandomTrainingUnlimiformer.convert_model(model, **unlimiformer_kwargs)
        else:
            model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)

    model.config.use_cache = True
    if training_args.gradient_checkpointing and getattr(model.config, 'use_cache', False) and training_args.do_train:
        logger.warning('Cannot use cache in models when using gradient checkpointing. turning it off')
        model.config.use_cache = False

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = seq2seq_dataset["train"].column_names
    elif training_args.do_eval:
        column_names = seq2seq_dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = seq2seq_dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.input_column is None:
        input_column = "input"
    else:
        input_column = data_args.input_column
        if input_column not in column_names:
            raise ValueError(
                f"--input_column' value '{data_args.input_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.input_prefix_column is None:
        input_prefix_column = "input_prefix"
    else:
        input_prefix_column = data_args.input_prefix_column
        if input_prefix_column not in column_names:
            raise ValueError(
                f"--input_prefix_column' value '{data_args.input_prefix_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.output_column is None:
        output_column = "output"
    else:
        output_column = data_args.output_column
        if output_column not in column_names:
            raise ValueError(
                f"--output_column' value '{data_args.output_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function_kwargs_fn():
        return {
            "tokenizer": deepcopy(tokenizer),
            "prefix": prefix,
            "input_column": input_column,
            "input_prefix_column": input_prefix_column,
            "output_column": output_column,
            "max_source_length": data_args.max_source_length,
            "max_prefix_length": data_args.max_prefix_length,
            "max_target_length": max_target_length,
            "prefix_sep": PREFIX_DOC_SEP,
            "padding": padding,
            "ignore_pad_token_for_loss": data_args.ignore_pad_token_for_loss,
            "assign_zero_to_too_long_val_examples": data_args.assign_zero_to_too_long_val_examples,
            "trim_very_long_strings": data_args.trim_very_long_strings,
            "pad_prefix": data_args.pad_prefix
        }

    if training_args.do_train:
        if "train" not in seq2seq_dataset:
            raise ValueError("--do_train requires a train dataset")
        logger.info("")
        logger.info("Training examples before tokenization:")
        if input_prefix_column in column_names:
            logger.info(f"input_prefix #0: {seq2seq_dataset['train'][0][input_prefix_column]}")
        # logger.info(f"input #0: {seq2seq_dataset['train'][0]['input']}")
        # logger.info(f"output #0: {seq2seq_dataset['train'][0]['output']}")
        if input_prefix_column in column_names:
            logger.info(f"input_prefix #1: {seq2seq_dataset['train'][1][input_prefix_column]}")
        # logger.info(f"input #1: {seq2seq_dataset['train'][1]['input']}")
        # logger.info(f"output #1: {seq2seq_dataset['train'][1]['output']}")
        logger.info("")
        untokenized_train_dataset = seq2seq_dataset["train"]
        if data_args.max_train_samples is not None:
            untokenized_train_dataset = untokenized_train_dataset.select(range(data_args.max_train_samples))

        if DEBUG:
            # In debug mode, we want ot recreate the data
            data_args.shared_storage = False
            data_args.overwrite_cache = True
        with training_args.main_process_first(
            local=not data_args.shared_storage, desc="train dataset map pre-processing"
            ):

            if data_args.oracle_training:
                logger.info("Using oracle training")
                oracle_processed_dir = f'oracle_input_{data_args.dataset_config_name}'
                if os.path.isdir(oracle_processed_dir):
                    logger.info(f"Using oracle training from {oracle_processed_dir}")
                    oracle_training_set = datasets.load_from_disk(oracle_processed_dir)
                else:
                    rouge_scorer = datasets.load_metric('rouge')
                    oracle_training_set = untokenized_train_dataset.map(
                        extract_oracle_sent_batch,
                        fn_kwargs={'max_length': data_args.max_source_length,
                                'tokenizer': tokenizer,
                                'rouge_scorer': rouge_scorer},
                        batched=True,
                        batch_size=1,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Extracting oracle sentences from every training example",
                    )
                    oracle_training_set.save_to_disk(oracle_processed_dir)
                
                
                if data_args.oracle_merge:
                    untokenized_train_dataset = datasets.concatenate_datasets([untokenized_train_dataset, oracle_training_set])
                    untokenized_train_dataset = untokenized_train_dataset.shuffle(seed=training_args.seed)
                else:
                    untokenized_train_dataset = oracle_training_set

            train_dataset = untokenized_train_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs_fn(),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=untokenized_train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

            if data_args.chunked_training_size is not None:
                train_dataset = train_dataset.map(
                    chunk_dataset_function,
                    fn_kwargs={'chunk_size': data_args.chunked_training_size},
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Chunking train dataset source",
                )
                train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        preprocess_function_kwargs = preprocess_function_kwargs_fn()
        preprocess_function_kwargs["max_target_length"] = max_target_length
        preprocess_function_kwargs['max_source_length'] = data_args.eval_max_source_length
        if "validation" not in seq2seq_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        logger.info("")
        logger.info("Validation examples before tokenization:")
        if input_prefix_column in column_names:
            logger.info(f"input_prefix #0: {seq2seq_dataset['validation'][0][input_prefix_column]}")
        # logger.info(f"input #0: {seq2seq_dataset['validation'][0]['input']}")
        # logger.info(f"output #0: {seq2seq_dataset['validation'][0]['output']}")
        if input_prefix_column in column_names:
            logger.info(f"input_prefix #1: {seq2seq_dataset['validation'][1][input_prefix_column]}")
        # logger.info(f"input #1: {seq2seq_dataset['validation'][1]['input']}")
        # logger.info(f"output #1: {seq2seq_dataset['validation'][1]['output']}")
        logger.info("")
        untokenized_eval_dataset = seq2seq_dataset["validation"]
        if data_args.max_eval_samples is not None:
            untokenized_eval_dataset = untokenized_eval_dataset.select(range(data_args.max_eval_samples))
        if model_args.drop_duplicates_in_eval is True:
            untokenized_eval_dataset = drop_duplicates_in_input(untokenized_eval_dataset)
        untokenized_eval_dataset_orig = untokenized_eval_dataset
        assert training_args.eval_fraction > 0
        n = len(untokenized_eval_dataset)
        training_args.eval_fraction = min(training_args.eval_fraction, n)
        if training_args.eval_fraction != 1:
            if training_args.eval_fraction > 1:
                assert training_args.eval_fraction == int(training_args.eval_fraction)
                logger.info(f'using predetermined absolute samples from eval set ({training_args.eval_fraction} )')
                training_args.eval_fraction = training_args.eval_fraction / n
            indices = np.random.permutation(n)[:int(np.ceil(max(1, training_args.eval_fraction * n)))]
            untokenized_eval_dataset = type(untokenized_eval_dataset).from_dict(untokenized_eval_dataset[indices])
            logger.info(f'During training, will only use {training_args.eval_fraction:.3%} samples of the eval set '
                        f'which amounts to {len(untokenized_eval_dataset)} out of {n} samples')

        eval_dataset = process_eval_set(data_args, preprocess_function_kwargs, training_args, untokenized_eval_dataset)
        eval_dataset_orig = eval_dataset
        if training_args.eval_fraction < 1:
            eval_dataset_orig = process_eval_set(data_args, preprocess_function_kwargs, training_args,
                                                 untokenized_eval_dataset_orig)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        preprocess_function_kwargs = preprocess_function_kwargs_fn()
        preprocess_function_kwargs["max_target_length"] = max_target_length
        preprocess_function_kwargs['max_source_length'] = data_args.eval_max_source_length
        if "test" not in seq2seq_dataset:
            raise ValueError("--do_predict requires a test dataset")
        untokenized_predict_dataset = seq2seq_dataset["test"]
        if data_args.max_predict_samples is not None:
            untokenized_predict_dataset = untokenized_predict_dataset.select(range(data_args.max_predict_samples))
        if model_args.drop_duplicates_in_eval is True:
            untokenized_predict_dataset = drop_duplicates_in_input(untokenized_predict_dataset)

        if output_column in untokenized_predict_dataset.column_names:
            untokenized_predict_dataset = untokenized_predict_dataset.remove_columns(output_column)

        if data_args.test_start_ind is not None:
            sind =  data_args.test_start_ind
            eind = -1 if data_args.test_end_ind is None else data_args.test_end_ind
            logger.info(f'Using only a subset of the test dataset [{sind}, {eind}]')
            untokenized_predict_dataset = type(untokenized_predict_dataset).from_dict(untokenized_predict_dataset[sind:eind])

        with training_args.main_process_first(
            local=not data_args.shared_storage, desc="prediction dataset map pre-processing"
        ):
            predict_dataset = untokenized_predict_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=untokenized_predict_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    if data_args.preprocess_only:
        logger.info(f"With --preprocess_only, exiting after preprocess_on the data")
        exit()

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    pad_to = 8 if training_args.fp16 and training_args.fp16_padding else None


    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to,
    )

    # Metric
    compute_metrics = load_metric(data_args.metric_names, **locals())
    compute_metrics = load_extra_metrics(data_args.extra_metrics, compute_metrics)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        untokenized_eval_dataset=untokenized_eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        output_dir=training_args.output_dir,
        data_args=data_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
    )

    # setup_cometml_trainer_callback(trainer)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint  # look for checkpoints in the outdir

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info('Done training')
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if training_args.eval_fraction < 1:
            logger.info('setting the eval set back to the full one')
            trainer.eval_dataset = eval_dataset_orig
            trainer._untokenized_eval_dataset = untokenized_eval_dataset_orig

        metrics = trainer.evaluate(metric_key_prefix="eval", use_cache=True, length_penalty=data_args.length_penalty)
        logger.info('Done evaluating')

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        trainer.args.predict_with_generate = True # during prediction, we don't have labels

        # load last (and best) model, or the one specified if any
        logger.info("*** Loading model weights before the prediction ***")
        last_checkpoint = model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else _detect_last_checkpoint(training_args)
        if last_checkpoint is not None and os.path.isdir(last_checkpoint):
            logger.info(f'Loading weights from {last_checkpoint} for the prediction')
            state_dict = torch.load(os.path.join(last_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            # trainer._load_state_dict_in_model(state_dict)
            # release memory
            del state_dict
            logger.info("*** Done loading weights ***")
        elif training_args.do_train:
            raise ValueError('Could not find a model to load for prediction')
        else:
            logger.info(f'Using {model_args.model_name_or_path} as the model for the prediction')

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", use_cache=True)
        logger.info('Done predicting')

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                id_to_prediction = {}
                for i, instance in enumerate(untokenized_predict_dataset):
                    id_to_prediction[instance["id"]] = predict_results.predictions[i]
                predictions = decode(id_to_prediction, tokenizer, data_args)
                output_name = "generated_predictions.json"
                if data_args.test_start_ind is not None:
                    output_name = f"generated_predictions_{data_args.test_start_ind}_{data_args.test_end_ind}.json"
                output_prediction_file = os.path.join(training_args.output_dir, output_name)
                with open(output_prediction_file, "w") as writer:
                    json.dump(predictions, writer, indent=4)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results

def _detect_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        if not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)

            if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
    return last_checkpoint

def process_eval_set(data_args, preprocess_function_kwargs, training_args, untokenized_eval_dataset):
    with training_args.main_process_first(
            local=not data_args.shared_storage, desc="validation dataset map pre-processing"
    ):
        eval_dataset = untokenized_eval_dataset.map(
            preprocess_function,
            fn_kwargs=preprocess_function_kwargs,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=untokenized_eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    return eval_dataset


def _get_dataset(data_args, model_args, training_args):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `input_column` and `output_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = None
    if data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
    # Downloading and loading a dataset from the hub/local script.
    seq2seq_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        ignore_verifications=True,
        cache_dir=model_args.cache_dir,
        data_dir=data_args.data_dir,
        data_files=data_files,
        download_mode=data_args.download_mode,
        use_auth_token=training_args.use_auth_token
    )
    if training_args.do_train:
        training_args.apply_overrides(len(seq2seq_dataset['train']))
    if data_args.evaluate_on_training_data:
        seq2seq_dataset["validation"] = seq2seq_dataset["train"]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    return seq2seq_dataset


def set_up_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def extract_oracle_sent_batch(examples, max_length, tokenizer, rouge_scorer):
    items = examples.data.items()
    keys = [item[0] for item in items]
    values = [item[1] for item in items]
    extracted = {k: [] for k in keys}
    input_str = 'input'

    for ex in zip(*values):
        ex = dict(zip(keys, ex))
        ex_input = ex[input_str]
        extracted_input = extract_oracle_sentences(ex_input, ex['output'], max_length, tokenizer, rouge_scorer)
        extracted[input_str].append(extracted_input)
        for k in set(keys) - {input_str}:
            extracted[k].append(ex[k])
    return extracted

def extract_oracle_sentences(input_sequence, output, max_length, tokenizer, rouge_scorer, criterion='rouge/geometric_mean'):
    sentences = nltk.sent_tokenize(input_sequence)
    selected_mask = [False for _ in sentences]

    max_rouge = 0.0
    joined_selection = ''
    counter = 0
    while len(tokenizer(joined_selection)) < max_length and counter < 100:
        cur_max_rouge = max_rouge
        max_index = -1
        
        cur_candidate_indices = []
        cur_candidates = []
        for i in range(len(sentences)):
            if selected_mask[i]:
                # We already selected this sentence
                continue
            candidate_mask = list(selected_mask)
            candidate_mask[i] = True
            candidate_prediction = ' '.join(sent for sent, mask in zip(sentences, candidate_mask) if mask)
            cur_candidates.append(candidate_prediction)
            cur_candidate_indices.append(i)
        
        rouge = rouge_scorer.compute(predictions=cur_candidates, references=[[output]] * len(cur_candidates), use_aggregator=False)
        aggregated_rouge_types = [s1.fmeasure * s2.fmeasure * sL.fmeasure for s1, s2, sL in zip(rouge['rouge1'], rouge['rouge2'], rouge['rougeLsum'])]
        max_index = np.argmax(aggregated_rouge_types)
        cur_max_rouge = aggregated_rouge_types[max_index]
        
        if max_rouge >= cur_max_rouge:
            # No sentence improves the score
            break
        
        selected_mask[cur_candidate_indices[max_index]] = True
        max_rouge = cur_max_rouge
        joined_selection = ' '.join(sent for sent, mask in zip(sentences, selected_mask) if mask)
        counter += 1
    
    return joined_selection        
    

def chunk_dataset_function(examples, chunk_size):
    input_ids_str = 'input_ids'
    attention_mask_str = 'attention_mask'
    items = examples.data.items()
    keys = [item[0] for item in items]
    values = [item[1] for item in items]
    chunked = {k: [] for k in keys}
    for ex in zip(*values):
        ex = dict(zip(keys, ex))
        for i in range(0, len(ex[input_ids_str]), chunk_size):
            chunked_input_ids_st = ex[input_ids_str][i:i + chunk_size]
            chunked_attention_mask = ex[attention_mask_str][i:i + chunk_size]

            if sum(chunked_attention_mask) < 10:
                continue
            chunked[input_ids_str].append(chunked_input_ids_st)
            chunked[attention_mask_str].append(chunked_attention_mask)
            for k in set(keys) - {input_ids_str, attention_mask_str}:
                chunked[k].append(ex[k])
    return chunked

    

def preprocess_function(
    examples,
    tokenizer,
    prefix,
    input_column,
    input_prefix_column,
    output_column,
    max_source_length,
    max_prefix_length,
    max_target_length,
    prefix_sep,
    padding,
    ignore_pad_token_for_loss,
    assign_zero_to_too_long_val_examples,
    trim_very_long_strings,
    pad_prefix
):
    if not isinstance(examples[input_column][0], str):
        model_inputs = _preprocess_tokenized_inputs()
    else:
        model_inputs = _preprocess_raw_inputs(assign_zero_to_too_long_val_examples, examples, input_column, input_prefix_column,
                                              max_source_length, padding, prefix, tokenizer, trim_very_long_strings, max_prefix_length,
                                              prefix_sep, pad_prefix)

    _preprocess_targets(examples, ignore_pad_token_for_loss, max_target_length, model_inputs, output_column, padding, tokenizer)
    model_inputs["length"] = [len(x) for x in model_inputs["input_ids"]]
    return model_inputs


def _preprocess_raw_inputs(assign_zero_to_too_long_val_examples, examples, input_column, input_prefix_column,
                           max_source_length, padding, prefix, tokenizer, trim_very_long_strings, max_prefix_length,
                           prefix_sep, pad_prefix):
    inputs = examples[input_column]

    # the given prefix is what used in models like T5 (e.g. "summarize: ")
    # if prefix exists, it is added to the input_prefixes
    if input_prefix_column in examples.keys():
        input_prefixes = [inp + prefix_sep for inp in examples[input_prefix_column]]
        if prefix != "":
            input_prefixes = [prefix + inp for inp in input_prefixes]
    elif prefix != "":
        inputs = [prefix + inp for inp in inputs]

    # tokenize the input prefix if it exists
    model_prefix_inputs = None
    if input_prefix_column in examples.keys():
        if trim_very_long_strings:
            input_prefixes = [inp[: max_prefix_length * 7] for inp in input_prefixes]
        if pad_prefix:
            model_prefix_inputs = tokenizer(input_prefixes, max_length=max_prefix_length, padding='max_length', truncation=True)
        else:
            # for led, we do not pad the prefix
            model_prefix_inputs = tokenizer(input_prefixes, max_length=max_source_length, padding='do_not_pad', truncation=True)

    if trim_very_long_strings:
        inputs = [inp[: max_source_length * 7] for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    if max_source_length is not None and assign_zero_to_too_long_val_examples:
        model_inputs_untrimmed = tokenizer(inputs)
        model_inputs["not_valid_for_eval"] = [
            len(token_ids) > max_source_length for token_ids in model_inputs_untrimmed["input_ids"]
        ]
    else:
        model_inputs["not_valid_for_eval"] = [False] * len(model_inputs["input_ids"])

    # now, combine the concat prefix to the input, trimming it to max_source_length if given
    if model_prefix_inputs is not None:
        max_source_length = max_source_length or -1
        model_inputs['input_ids'] = [(inp1+inp2)[:max_source_length] for inp1, inp2
                                     in zip(model_prefix_inputs['input_ids'], model_inputs['input_ids'])]
        model_inputs['attention_mask'] = [(inp1+inp2)[:max_source_length] for inp1, inp2
                                          in zip(model_prefix_inputs['attention_mask'], model_inputs['attention_mask'])]
        # add prefix_length
        if pad_prefix:
            # no need to go over them as they will all be of the same length
            model_inputs['prefix_length'] = [max_prefix_length] * len(model_inputs['input_ids'])
        else:
            model_inputs['prefix_length'] = [len(inp) for inp in model_prefix_inputs['input_ids']]

    return model_inputs

def _preprocess_targets(examples, ignore_pad_token_for_loss, max_target_length, model_inputs, output_column, padding, tokenizer):
    targets = examples[output_column] if output_column in examples else None
    if targets is not None:
        if not isinstance(targets[0], str):
            if max_target_length is not None:
                targets = [target[:max_target_length] for target in targets]
            model_inputs["labels"] = targets
        else:
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]

def load_extra_metrics(metric_names, loaded_metrics=None):
    if loaded_metrics is None:
        loaded_metrics = MetricCollection([])
    if metric_names is not None:
        for metric_name in metric_names:
            if len(metric_name) > 0:
                loaded_metrics._metrics.append(HFMetricWrapper(metric_name))
    return loaded_metrics

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
