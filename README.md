# Unlimiformer
![image](https://user-images.githubusercontent.com/42593540/236538293-1d5fdfe3-3e34-4979-9611-a9c9f56e3a00.png)

This is the official implementation of the paper Unlimiformer: Long-Range Transformers with Unlimited Length Input.

Unlimiformer is a method for augmenting pretrained encoder-decoder models with a type of retrieval-based attention. This allows the use of unlimited length inputs with any pretrained encoder-decoder!

Unlimiformer can be used to improve performance of an already-trained model. However, for best results, the model should be trained with Unlimiformer. 

## Getting Started
Paste the files from ```src``` into your source code folder.

You'll need to set values for the Unlimiformer-specific arguments outlined in ```usage.py```-- you can add these arguments wherever you usually process hyperparameters. To use the model, you must set ```test_unlimiformer=True```. For datastore usage, the model must be in evaluation model (e.g. call ```model.eval()``` before inference). 

```inference-example.py``` outlines a minimal example for running a sequence through an Unlimiformer model, using the default arguments. 

```run.py``` is an example of a full training setup that integrates Unlimiformer, adopted from [SLED](https://github.com/Mivg/SLED) -- this is likely more complex than you will need. 

## Trained models
Trained models from the paper will be made available on HuggingFace. Please note that you must add the Unlimiformer-specific files to your repository, and load these models with ```knn=True```. *If you download these models from Hugging Face, they may not use Unlimiformer by default!* 

### Table 3: low-cost training methods
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| GovReport | Baseline: BART-base  | <anonymized>  |
| GovReport  | BART-base + Unlimiformer early stopping  | <anonymized> |
| SummScreen | Baseline: BART-base  | <anonymized> |
| SummScreen  | BART-base + Unlimiformer early stopping  | <anonymized>  |


### Table 4: Long-range training methods
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| GovReport | BART + Unlimiformer (alternating training)  | <anonymized>  |
| SummScreen | BART + Unlimiformer (retrieval training)  | <anonymized> |

## Table 5: BookSum
| Dataset  |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| BookSum | Baseline: BART-base  | <anonymized>  |
| BookSum  | BART-base + Unlimiformer early stopping  | <anonymized> |
| Booksum  | BART-base + Unlimiformer (random-encoding training)  | <anonymized>  |
| Booksum  | BART-base + Unlimiformer (alternating training)  | <anonymized>  |


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


