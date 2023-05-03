import contextlib
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from attention_knn import AttentionKNNWrapper

class RandomAttentionKNNWrapper(AttentionKNNWrapper):
    def __init__(self, random_knn_initial_inputs=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_hooks_injected = False
        self.random_knn_initial_inputs = random_knn_initial_inputs
        self.train_step = 0

    def pre_eval_hook(self):
        self.remove_training_hooks(self.model)
        self.inject_hooks(self.model)   
        self.original_model_eval_func()

    def pre_train_hook(self, mode=True):
        # mode=True means model.train() is called
        # mode=False means model.eval() is called
        torch.cuda.empty_cache()
        if mode is True:
            self.break_out(self.model)
            self.remove_training_hooks(self.model)
            if self.knn_training and self.train_step % 2 == 0:
                super().inject_training_hooks(self.model)
            else:
                self.inject_training_hooks(self.model)
            self.train_step += 1
        self.original_model_train_func(mode)
    
    # def remove_training_hooks(self, model):
    #     if not self.training_hooks_injected:
    #         return
    #     for h in self.hook_handles:
    #         h.remove()
    #     model.forward = self.original_forward_func

    #     decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
    #         model, self.knn_layer_begin, self.knn_layer_end)
    #     for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_self_attn_forward_funcs):
    #         decoder_layer.self_attn.forward = original_func
    #     for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_forward_funcs):
    #         decoder_layer.forward = original_func
    #     self.training_hooks_injected = False
    
    def inject_training_hooks(self, model):
        if self.training_hooks_injected:
            return
        # self.original_forward_func = model.forward
        model.forward = self.random_inputs_forward_hook

        decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
            model, self.knn_layer_begin, self.knn_layer_end)
        
        self.original_decoder_layer_self_attn_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            self.original_decoder_layer_self_attn_forward_funcs.append(decoder_layer.self_attn.forward)
            decoder_layer.self_attn.forward = self.create_self_attn_random_pre_forward_hook(decoder_layer.self_attn.forward)

        self.original_decoder_layer_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            self.original_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_decoder_layer_random_func(decoder_layer.forward, decoder_layer)

        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            self.original_decoder_layer_cross_attn_forward_funcs.append(decoder_layer.encoder_attn.forward)

        self.inject_hooks_for_unaffected_layers(model, decoder_layers_to_run)

        self.training_hooks_injected = True

    def create_self_attn_random_pre_forward_hook(self, original_self_attn_forward_func):
        def self_attention_pre_forward_hook(*args, **kwargs):
            kwargs['past_key_value'] = None
            return original_self_attn_forward_func(*args, **kwargs)
        
        return self_attention_pre_forward_hook

    def create_decoder_layer_random_func(self, decoder_layer_original_forward_func, decoder_layer):
        def checkpointed_decoder_layer(
                hidden_states: torch.Tensor,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=True):

            
            
            def sample_and_forward(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, past_key_value, 
                    output_attentions, use_cache, long_inputs, long_inputs_mask, rand_indices):
                
                sampled_input, _ = self.sample_long_input(long_inputs, long_inputs_mask, rand_indices)
                key, value = AttentionKNNWrapper.bart_create_key_value(sampled_input, decoder_layer)
                return decoder_layer_original_forward_func(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, (None, None, key, value), 
                    output_attentions, use_cache)

            with torch.no_grad():
                # This sampling must be done outside of the checkpoint, to ensure that the same sampling happens
                # both in "forward" and "backward" passes
                rand_indices = self.sample_random_indices()

            return torch.utils.checkpoint.checkpoint(
                sample_and_forward, hidden_states, attention_mask, 
                encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                cross_attn_layer_head_mask, None, 
                output_attentions, use_cache, self.long_inputs_encoded, self.long_inputs_mask, rand_indices)

        return checkpointed_decoder_layer

    def sample_random_indices(self):
        rand_indices_list = []
        seq_lens = self.long_inputs_mask.sum(-1).tolist()
        for seq_len in seq_lens:
            if seq_len < self.model_encoder_max_len:
                rand_indices = torch.arange(self.model_encoder_max_len).to(self.device)
                rand_indices_list.append(rand_indices)
                continue
            
            rand_indices = torch.torch.randperm(seq_len)[:self.model_encoder_max_len].to(self.device)
            if seq_len < self.model_encoder_max_len:
                padding = max(self.model_encoder_max_len - seq_len, 0)
                rand_indices = torch.cat([rand_indices, torch.arange(padding).to(self.device) + seq_len], axis=-1).to(self.device)
            rand_indices_list.append(rand_indices)
        rand_indices = torch.stack(rand_indices_list, dim=0)
        return rand_indices

    def random_inputs_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.model.base_model.decoder.gradient_checkpointing = False
        self.long_inputs_encoded, self.long_inputs_mask = self.chunked_encode_input(input_ids=input_ids, attention_mask=attention_mask)

        #  TODO: should the inputs be sampled or the truncated beginning?
        if self.random_knn_initial_inputs:
            encoded_inputs, encoded_inputs_mask = self.sample_long_input(self.long_inputs_encoded, self.long_inputs_mask)
        else:
            encoded_inputs = self.long_inputs_encoded[:, :self.model_encoder_max_len]
            encoded_inputs_mask = self.long_inputs_mask[:, :self.model_encoder_max_len]
        return self.original_forward_func(encoder_outputs=(encoded_inputs, ), labels=labels, attention_mask=encoded_inputs_mask, **kwargs)

    def sample_long_input(self, long_inputs_encoded, long_inputs_mask, random_indices=None):
        if long_inputs_mask.shape[-1] < self.model_encoder_max_len:
            return long_inputs_encoded, long_inputs_mask
        batch_size = long_inputs_encoded.shape[0]
        
        if random_indices is None:
            random_indices = self.sample_random_indices()
        random_mask = torch.zeros_like(long_inputs_mask).to(self.device) \
            .scatter_(dim=-1, index=random_indices, src=torch.ones_like(random_indices)).bool().to(self.device)
        sampled_input = long_inputs_encoded[random_mask].reshape(batch_size, self.model_encoder_max_len, -1).to(self.device)
        sampled_mask = long_inputs_mask[random_mask].reshape(batch_size, self.model_encoder_max_len).to(self.device)
        return sampled_input, sampled_mask

    def chunked_encode_input(self, input_ids, attention_mask):
        long_inputs_encoded = []
        long_inputs_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        self.is_input_encoding_pass = True
        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = input_ids[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            output = self.model.base_model.encoder(chunk, attention_mask=chunk_attention_mask, return_dict=True, output_hidden_states=True)
            encoder_last_hidden_state = output.last_hidden_state # (batch, time, dim)
            
            # list of (batch, head, chunked_time, dim)
            encoder_last_hidden_state = encoder_last_hidden_state[:, update_start_ind:update_end_ind] # (batch, chunked_time, dim)
            chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

            long_inputs_encoded.append(encoder_last_hidden_state) # (batch, chunked_source_len, dim)
            long_inputs_mask.append(chunk_attention_mask) # (batch, chunked_source_len)
        
        long_inputs_encoded = torch.cat(long_inputs_encoded, dim=1) # (batch, source_len, dim)
        long_inputs_mask = torch.cat(long_inputs_mask, dim=1) # (batch, source_len)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.model_encoder_max_len], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.model_encoder_max_len:], skip_special_tokens=True)}')
            print()
        return long_inputs_encoded, long_inputs_mask

        
