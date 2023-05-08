from unlimiformer import Unlimiformer
from random_training_unlimiformer import RandomTrainingUnlimiformer
from usage import UnlimiformerArguments, training_addin

from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

# example using booksum
modelname = "abertsch/unlimiformer-bart-booksum-alternating"
dataset = load_dataset("abertsch/booksum-fullbooks", "validation")

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(modelname)

example_input = dataset['train'][0]['book']

example = tokenizer(example_input, truncation=False, return_tensors="pt")
truncated_example = tokenizer(example_input, truncation=True, max_length=1024, return_tensors="pt")

print(f"INPUT LENGTH (tokens): {example['input_ids'].shape[-1]}")


defaults = UnlimiformerArguments()
unlimiformer_kwargs = {
            'layer_begin': defaults.layer_begin, 
            'layer_end': defaults.layer_end,
            'unlimiformer_head_num': defaults.unlimiformer_head_num, 
            'exclude_attention': defaults.unlimiformer_exclude, 
            'chunk_overlap': defaults.unlimiformer_chunk_overlap,
            'model_encoder_max_len': defaults.unlimiformer_chunk_size,
            'verbose': defaults.unlimiformer_verbose, 'tokenizer': tokenizer,
            'unlimiformer_training': defaults.unlimiformer_training,
            'use_datastore': defaults.use_datastore,
            'flat_index': defaults.flat_index,
            'test_datastore': defaults.test_datastore,
            'reconstruct_embeddings': defaults.reconstruct_embeddings,
            'gpu_datastore': defaults.gpu_datastore,
            'gpu_index': defaults.gpu_index
}
#print(model.generate(**truncated_example, max_length=1024))
truncated_out = tokenizer.batch_decode(model.generate(**truncated_example, max_length=1024))
model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)

unlimiformer_out = tokenizer.decode(model.generate(**example, max_length=1024))

