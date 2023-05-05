from unlimiformer import Unlimiformer
from random_training_unlimiformer import RandomTrainingUnlimiformer

from dataclasses import dataclass, field
from typing import List, Optional


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



# include these lines in your code somewhere before model training
def training_addin():
    if knn_args.test_unlimiformer:
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

