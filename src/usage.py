from attention_knn import AttentionKNNWrapper
from random_attention_knn import RandomAttentionKNNWrapper

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KNNArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    knn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use KNN."
        },
    )
    knn_verbose: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to print KNN intermediate predictions (mostly for debugging)."
        },
    )
    knn_heatmap: Optional[bool] = field(
        default=False
    )
    knn_layer_begin: Optional[int] = field(
        default=0,
        metadata={"help": "The layer to begin applying KNN to. KNN will be applied to layers[knn_layer_begin:knn_layer_end]. "
                          "By default, it will be applied to all layers: [0:None]]"}, 
    )
    knn_layer_end: Optional[int] = field(
        default=None,
        metadata={"help": "The layer to end applying KNN to. KNN will be applied to layers[knn_layer_begin:knn_layer_end]. "
                          "By default, it will be applied to all layers: [0:None]]"}, 
    )
    knn_chunk_overlap: Optional[float] = field(
        default=0.5,
        metadata={"help": "The fraction of overlap between input chunks"},
    )
    knn_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "The size of each input chunk"},
    )
    knn_head_num: Optional[int] = field(
        default=None,
        metadata={"help": "The head to apply KNN to (if None, apply to all heads)"},
    )
    knn_exclude: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, prioritize the inputs that are **not** in the standard attention window."
        },
    )
    knn_normalize: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, l2-normalize keys and queries before performing KNN search."
        },
    )
    random_knn_training: Optional[bool] = field(
        default=False,
    )
    random_knn_initial_inputs: Optional[bool] = field(
        default=True,
    )
    knn_training: Optional[bool] = field(
        default=False,
    )
    knn_use_pointers: Optional[bool] = field(default=False)
    use_datastore: Optional[bool] = field(default=False)
    flat_index: Optional[bool] = field(default=False)
    test_datastore: Optional[bool] = field(default=False)
    reconstruct_embeddings: Optional[bool] = field(default=False)
    gpu_datastore: Optional[bool] = field(default=True)
    gpu_index: Optional[bool] = field(default=True)



# include these lines in your code somewhere before model training
def training_addin():
    if knn_args.knn:
        knn_kwargs = {
            'knn_layer_begin': knn_args.knn_layer_begin, 
            'knn_layer_end': knn_args.knn_layer_end,
            'knn_head_num': knn_args.knn_head_num, 
            'normalize': knn_args.knn_normalize, 'exclude_attention': knn_args.knn_exclude, 
            'chunk_overlap': knn_args.knn_chunk_overlap,
            'use_pointers': knn_args.knn_use_pointers, 
            'model_encoder_max_len': knn_args.knn_chunk_size,
            'verbose': knn_args.knn_verbose, 'save_heatmap': knn_args.knn_heatmap, 'tokenizer': tokenizer,
            'knn_training': knn_args.knn_training,
            'use_datastore': knn_args.use_datastore,
            'flat_index': knn_args.flat_index,
            'test_datastore': knn_args.test_datastore,
            'reconstruct_embeddings': knn_args.reconstruct_embeddings,
            'gpu_datastore': knn_args.gpu_datastore,
            'gpu_index': knn_args.gpu_index
        }
        if knn_args.random_knn_training:
            knn_wrapper = RandomAttentionKNNWrapper(random_knn_initial_inputs=knn_args.random_knn_initial_inputs, 
                **knn_kwargs)
        else:
            knn_wrapper = AttentionKNNWrapper(**knn_kwargs)
        knn_wrapper.break_into(model)

# OPTIONALLY, to stop using Unlimiformer:
def breakout_addin():
    knn_wrapper.break_out(model)

