from unlimiformer import Unlimiformer
from random_training_unlimiformer import RandomTrainingUnlimiformer
from usage import UnlimiformerArguments, training_addin

from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# example using govreport
modelname = "abertsch/unlimiformer-bart-govreport-alternating"
dataset = load_dataset("urialon/gov_report_validation")

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(modelname)

example_input = dataset['validation'][0]['input']

example = tokenizer(example_input, truncation=False, return_tensors="pt")
truncated_example = tokenizer(example_input, truncation=True, max_length=1024, return_tensors="pt")

example.to(device)
truncated_example.to(device)

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

model.to(device)
# the output of the model /without/ using unlimiformer 
truncated_out = tokenizer.batch_decode(model.generate(**truncated_example, max_length=512))

model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
model.eval()
model.to(device)

# the output of the model /with/ unlimiformer 
unlimiformer_out = tokenizer.batch_decode(model.generate(**example, max_length=512), ignore_special_tokens=True)[0]
print(unlimiformer_out)
