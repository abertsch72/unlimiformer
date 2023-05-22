# Unlimiformer
![image](https://user-images.githubusercontent.com/42593540/236538293-1d5fdfe3-3e34-4979-9611-a9c9f56e3a00.png)

This is the official implementation of the paper:

[Amanda Bertsch](https://www.cs.cmu.edu/~abertsch/), [Uri Alon](https://urialon.ml/), [Graham Neubig](http://www.phontron.com/), and [Matthew R. Gormley](http://www.cs.cmu.edu/~mgormley/):   
[Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://arxiv.org/abs/2305.01625)

Unlimiformer is a method for augmenting pretrained encoder-decoder models with retrieval-based attention, without changing the mathematical definition of attention. 
This allows the use of unlimited length inputs with any pretrained encoder-decoder!  
See also our [**Tweet**](https://twitter.com/abertsch72/status/1654110919977324545?s=20).

Unlimiformer can be used to improve performance of an already-trained model. For best results, the model can be trained with Unlimiformer training. 

Please let us know if anything here is not working as expected, and feel free to create [new issues](/issues) with any questions.

## Getting Started

### General Instructions
Copy the files from `src` into your source code folder.

You'll need to set values for the Unlimiformer-specific arguments outlined in [`usage.py`](blob/main/src/usage.py)-- you can add these arguments wherever you usually process hyperparameters. To use the model, you must set `test_unlimiformer=True`. For datastore usage, the model must be in evaluation model (e.g. call ```model.eval()``` before inference). 

[`inference-example.py`](blob/main/src/usage.py) outlines a minimal example for running a sequence through an Unlimiformer model, using the default arguments. 

[`run.py`](blob/main/src/run.py) is an example of a full training setup that integrates Unlimiformer, adopted from [SLED](https://github.com/Mivg/SLED). See full command lines below.

## Trained models
The following models from the paper are available on Hugging Face. Please note that you must add the Unlimiformer-specific files to your repository, and load these models with ```knn=True```. *If you download these models from Hugging Face, they may not use Unlimiformer by default!* 

### Table 3: low-cost training methods
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| GovReport | Baseline: BART-base  | [abertsch/bart-base-govreport](https://huggingface.co/abertsch/bart-base-govreport)  |
| GovReport  | BART-base + Unlimiformer early stopping  | [abertsch/unlimiformer-bart-govreport-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-govreport-earlyk) |
| SummScreen | Baseline: BART-base  | [abertsch/bart-base-summscreen](https://huggingface.co/abertsch/bart-base-summscreen) |
| SummScreen  | BART-base + Unlimiformer early stopping  | [abertsch/unlimiformer-bart-summscreen-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-summscreen-earlyk)  |


### Table 4: Long-range training methods
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| GovReport | BART + Unlimiformer (alternating training)  | [abertsch/unlimiformer-bart-govreport-alternating](https://huggingface.co/abertsch/unlimiformer-bart-govreport-alternating)  |
| SummScreen | BART + Unlimiformer (retrieval training)  | [abertsch/unlimiformer-bart-summscreen-retrieval](https://huggingface.co/abertsch/unlimiformer-bart-summscreen-retrieval) |

## Table 5: BookSum
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| BookSum | Baseline: BART-base  | [abertsch/bart-base-booksum](https://huggingface.co/abertsch/bart-base-booksum)  |
| BookSum  | BART-base + Unlimiformer early stopping  | [abertsch/unlimiformer-bart-booksum-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-booksum-earlyk) |
| Booksum  | BART-base + Unlimiformer (random-encoding training)  | [abertsch/unlimiformer-bart-booksum-random-encoding](https://huggingface.co/abertsch/unlimiformer-bart-booksum-random-encoding)  |
| Booksum  | BART-base + Unlimiformer (alternating training)  | [abertsch/unlimiformer-bart-booksum-alternating](https://huggingface.co/abertsch/unlimiformer-bart-booksum-alternating)  |

## Recommended settings

### To evaluate with Unlimiformer
At evaluation time, we recommend the default value for each setting. 

### To train with Unlimiformer
For an inexpensive method, we recommend training as usual and using Unlimiformer during early stopping. To do so, set ```knn=True``` and leave all other values at default.


For best performance, there are 3 expensive settings for training. The best one varies by dataset.
1. Set ```random_unlimiformer_training=True```: this is the *random-encoded training* setting from the paper
2. Set ```unlimiformer_training=True```: this is the *approximate-retrieval training* setting from the paper
3. Set ```random_unlimiformer_training=True``` AND ```unlimiformer_training=True```: this is the *alternating training* setting from the paper

See Table 5 in the paper for a more detailed breakdown of relative training costs. 

## Tips for very large inputs
### For training
* you may need to truncate your inputs at training time, e.g. to 8k or 16k tokens. You can use the full inputs at evaluation time
* you can also try splitting your inputs into 16k-token-chunks and training on each one as its own example
### For evaluation (including early stopping)
* if you're consistently running out of CUDA memory, set ```use_datastore=True``` to use a Faiss datastore to store hidden states.
* if you're still having issues, set ```gpu_datastore=False``` or ```gpu_index=False```, but note that this will degrade performance


## Citation
If you use our method or models, please cite [our paper](https://arxiv.org/abs/2305.01625):
```
@article{bertsch2023unlimiformer,
  title={Unlimiformer: Long-Range Transformers with Unlimited Length Input},
  author={Bertsch, Amanda and Alon, Uri and Neubig, Graham and Gormley, Matthew R},
  journal={arXiv preprint arXiv:2305.01625},
  year={2023}
}
```

If you have any questions on this work, please open a GitHub issue or email the authors at ```abertsch@cs.cmu.edu, ualon@cs.cmu.edu```

