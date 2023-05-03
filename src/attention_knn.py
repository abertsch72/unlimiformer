import logging
import numpy as np
import torch
from torch import nn
from enum import Enum, auto

from index_building import Datastore, DatastoreBatch

logger = logging.getLogger('attention_knn')
logger.setLevel(20)
class AttentionKNNWrapper(object):
    def __init__(self, knn_layer_begin=-1, knn_layer_end=None,
            knn_head_num=None, normalize=False, 
            exclude_attention=False, use_pointers=False, 
            model_encoder_max_len=None,
            chunk_overlap=0,
            verbose=False, save_heatmap=False, 
            tokenizer=None, knn_training=False,
            use_datastore=False, 
            flat_index=False,
            test_datastore=False, reconstruct_embeddings=False, 
            gpu_datastore=False, gpu_index=False):
        self.knn_layer_begin = knn_layer_begin
        self.knn_layer_end = knn_layer_end
        self.specific_head = knn_head_num
        self.normalize = normalize
        self.exclude_attention = exclude_attention
        self.use_pointers = use_pointers
        self.model_encoder_max_len = model_encoder_max_len
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.save_heatmap = save_heatmap
        self.tokenizer = tokenizer
        self.knn_training = knn_training

        self.use_datastore = use_datastore
        self.flat_index = flat_index
        self.reconstruct_embeddings = reconstruct_embeddings
        self.gpu_datastore = gpu_datastore
        self.gpu_index = gpu_index
        self.test_datastore = test_datastore # flag for debugging

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []
        self.is_input_encoding_pass = False
        self.is_first_test_decoding_step = False
        self.model_window_size = None
        self.model_type = None
        self.prev_tokens = None
        self.last_beam_idx = None
        self.heatmap = None
        self.cur_decoder_layer_index = None
        self.datastore = None

    def break_into(self, model):
        self.model = model
        self.model_type = model.config.model_type
        self.is_encoder_decoder = model.config.is_encoder_decoder
        if self.model_encoder_max_len is None:
            self.model_encoder_max_len = AttentionKNNWrapper.model_window_size[model.config.model_type](model)
        self.window_margin = int(self.model_encoder_max_len * self.chunk_overlap / 2)
        self.num_heads = model.config.num_attention_heads
        if self.specific_head is None:
            self.head_nums = Ellipsis # torch.arange(0, self.num_heads, device=self.device)
        else:
            self.head_nums = self.specific_head
        # Save a reference to the wrapper
        model.knn_wrapper = self
        self.hooks_injected = False
        self.training_hooks_injected = False
        self.original_forward_func = model.forward

        # Activate AttentionKNN when calling model.eval(), deactivate for model.train()
        self.original_model_eval_func = model.eval
        model.eval = self.pre_eval_hook
        self.original_model_train_func = model.train
        model.train = self.pre_train_hook

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
            if self.knn_training:
                self.inject_training_hooks(self.model)
        self.original_model_train_func(mode)
        
    def inject_hooks(self, model):
        if self.hooks_injected:
            return
        # Inject our activation_capturer to capture the activations at every forward pass
        attention_layers_to_capture = AttentionKNNWrapper.model_attention_layer_to_capture[model.config.model_type](model, self.knn_layer_begin, self.knn_layer_end)
        self.activation_capturer = []
        for layer in attention_layers_to_capture:
            if type(layer) is list:
                layer_capturers = []
                for k_or_v in layer:
                    capturer = ActivationCapturer(k_or_v, capture_input=False)
                    layer_capturers.append(capturer)
                    self.register_hook(k_or_v, capturer)
                self.activation_capturer.append(layer_capturers)
            else:
                capturer = ActivationCapturer(layer, capture_input=False)
                self.register_hook(attention_layers_to_capture, capturer)
                self.activation_capturer.append(capturer)

        # Inject our main function after the main attention function
        attention_layers_to_run = attention_layers_to_capture
        if model.config.model_type in AttentionKNNWrapper.model_attention_op_to_run:
            attention_layers_to_run = AttentionKNNWrapper.model_attention_op_to_run[model.config.model_type](model, self.knn_layer_begin, self.knn_layer_end)
            for layer in attention_layers_to_run:
                self.register_hook(layer, self.attention_forward_hook)

        decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
            model, self.knn_layer_begin, self.knn_layer_end)
        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            self.original_decoder_layer_cross_attn_forward_funcs.append(decoder_layer.encoder_attn.forward)
            decoder_layer.encoder_attn.forward = self.create_cross_attn_pre_forward_hook(decoder_layer.encoder_attn.forward, decoder_layer, i)

        # Inject our hook function in the beginning of generation.
        # When the "model.generate()" will be called, it will first call our "reset_generation()" function, 
        # and only then call "model.generate()"
        self.original_generate_func = model.generate
        model.generate = self.pre_generate_hook

        model.forward = self.pre_forward_hook
        
        self.original_reorder_cache_func = model._reorder_cache
        model._reorder_cache = self.reorder_cache_hook
        self.hooks_injected = True

    def inject_training_hooks(self, model):
        if self.training_hooks_injected:
            return
        # self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
            model, self.knn_layer_begin, self.knn_layer_end)
        
        self.original_decoder_layer_self_attn_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            attention = AttentionKNNWrapper.model_self_attention[model.config.model_type](decoder_layer)
            self.original_decoder_layer_self_attn_forward_funcs.append(attention.forward)
            attention.forward = self.create_self_attn_pre_forward_hook(attention.forward)

        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            attention = AttentionKNNWrapper.model_cross_attention[model.config.model_type](decoder_layer)
            self.original_decoder_layer_cross_attn_forward_funcs.append(attention.forward)
            attention.forward = self.create_cross_attn_pre_forward_hook(attention.forward, decoder_layer, i)

        self.original_decoder_layer_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            self.original_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_decoder_layer_func(decoder_layer.forward, decoder_layer)

        self.inject_hooks_for_unaffected_layers(model, decoder_layers_to_run)

        attention_layers_to_run = AttentionKNNWrapper.model_attention_op_to_run[model.config.model_type](model, self.knn_layer_begin, self.knn_layer_end)
        for layer in attention_layers_to_run:
            self.register_hook(layer, self.train_attention_forward_hook)

        self.training_hooks_injected = True

    def inject_hooks_for_unaffected_layers(self, model, decoder_layers_to_run):
        self.original_non_injected_decoder_layer_forward_funcs = []
        non_injected_decoder_layers = [l for l in AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
                model, 0, None) if l not in decoder_layers_to_run]
        for decoder_layer in non_injected_decoder_layers:
            self.original_non_injected_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_noninjected_decoder_layer_func(decoder_layer.forward, decoder_layer)

    def create_self_attn_pre_forward_hook(self, original_self_attn_forward_func):
        def self_attention_pre_forward_hook(*args, **kwargs):
            kwargs['past_key_value'] = None
            return original_self_attn_forward_func(*args, **kwargs)
        
        return self_attention_pre_forward_hook

    def create_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
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

           
            def forward_with_all_keys(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, past_key_value, 
                    output_attentions, use_cache, long_inputs, long_inputs_mask):
                
                key, value = AttentionKNNWrapper.bart_create_key_value(long_inputs, decoder_layer)
                return decoder_layer_original_forward_func(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, (None, None, key, value), 
                    output_attentions, use_cache)

            return torch.utils.checkpoint.checkpoint(
                forward_with_all_keys, hidden_states, attention_mask, 
                encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                cross_attn_layer_head_mask, None, 
                output_attentions, use_cache, self.long_inputs_encoded, self.long_inputs_mask)

        return checkpointed_decoder_layer

    def create_noninjected_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
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

            return torch.utils.checkpoint.checkpoint(
                decoder_layer_original_forward_func, hidden_states, attention_mask, 
                encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                cross_attn_layer_head_mask, None, 
                output_attentions, use_cache)

        return checkpointed_decoder_layer    

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self, model):
        self.prompt_keys = []
        self.prompt_values = []
        self.prompt_attention_mask = []
        self.generated_input_ids = []
        torch.cuda.empty_cache()
        if not self.hooks_injected:
            return

        for h in self.hook_handles:
            h.remove()
        model.generate = self.original_generate_func
        model.forward = self.original_forward_func
        model._reorder_cache = self.original_reorder_cache_func

        decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
            model, self.knn_layer_begin, self.knn_layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            decoder_layer.encoder_attn.forward = original_func
        self.hooks_injected = False

    def remove_training_hooks(self, model):
        self.long_inputs_encoded, self.long_inputs_mask = None, None
        if not self.training_hooks_injected:
            return
        for h in self.hook_handles:
            h.remove()
        model.forward = self.original_forward_func

        decoder_layers_to_run = AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
            model, self.knn_layer_begin, self.knn_layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_self_attn_forward_funcs):
            decoder_layer.self_attn.forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            decoder_layer.encoder_attn.forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        non_injected_decoder_layers = [l for l in AttentionKNNWrapper.model_attention_layer_to_run[model.config.model_type](
                model, 0, None) if l not in decoder_layers_to_run]
        for decoder_layer, original_func in zip(non_injected_decoder_layers, self.original_non_injected_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        self.training_hooks_injected = False

    def reset_memory(self, input_ids, attention_mask):
        if self.use_datastore:
            self.datastore = DatastoreBatch(dim=self.model.config.hidden_size, batch_size=input_ids.shape[0], flat_index=self.flat_index, gpu_index=self.gpu_index)
            self.embeddings = []
            torch.cuda.empty_cache()
        self.prompt_input_ids = input_ids
        self.input_ids = torch.tensor([], dtype=torch.long, device=input_ids.device)
        self.prompt_keys, self.prompt_values = None, None
        self.prev_tokens = [None for _ in range(len(self.original_decoder_layer_cross_attn_forward_funcs))]
        self.last_beam_idx = None
        self.cur_layer_key_value_placeholder = None
        self.is_input_encoding_pass = True
        dummy_labels = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        if self.save_heatmap:
            if self.heatmap is not None:
                print(f'Generated: {self.tokenizer.decode(self.generated_input_ids[0])}')
                self.plot_heatmap(self.heatmap[0].detach().cpu().numpy())
            self.heatmap = torch.tensor([], dtype=torch.float, device=input_ids.device)
        self.generated_input_ids = torch.tensor([], dtype=torch.long, device=input_ids.device)

        self.prompt_keys = []
        self.prompt_values = []
        self.prompt_attention_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = input_ids[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            hidden_states = self.model(chunk, attention_mask=chunk_attention_mask, labels=dummy_labels, return_dict=True)
            last_hidden = hidden_states.encoder_last_hidden_state # (batch, chunked_source_len, dim)
            if self.use_datastore:
                to_add = last_hidden[:, update_start_ind:update_end_ind].detach()
                to_apply_mask = chunk_attention_mask[:, update_start_ind:update_end_ind]
                if not self.reconstruct_embeddings:
                    to_add_embeddings = to_add
                    if not self.gpu_datastore:
                        to_add_embeddings = to_add_embeddings.cpu()
                    self.embeddings.append(to_add_embeddings)
                # list of len batch, each item is (masked_time, dim)
                to_add = [key[mask.bool()] for key, mask in zip(to_add, to_apply_mask)]
                self.datastore.add_keys(to_add)
            if (not self.use_datastore) or self.test_datastore:
                layers_kv = [
                    AttentionKNNWrapper.model_key_value_process[self.model_type](
                    self.model, layer_capturer) # (batch, head, time, dim)
                    for layer_capturer in self.activation_capturer
                ] # list of pairs of (batch, head, time, dim)

                # list of (batch, head, chunked_time, dim)
                key = [layer[0][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                value = [layer[1][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

                key = torch.stack(key, dim=0) # (num_layers, batch, head, time, dim)
                value = torch.stack(value, dim=0) # (num_layers, batch, head, time, dim)

                self.prompt_keys.append(key) # (num_layers, batch, head, chunked_source_len, dim)
                self.prompt_values.append(value) # (num_layers, batch, head, chunked_source_len, dim)
                self.prompt_attention_mask.append(chunk_attention_mask) # (batch, chunked_source_len)
        
        if self.use_datastore:
            # keys are all in datastore already!
            if not self.reconstruct_embeddings:
                self.embeddings = torch.cat(self.embeddings, axis=1)
            self.datastore.train_index()
        if (not self.use_datastore) or self.test_datastore:
            self.prompt_keys = torch.cat(self.prompt_keys, dim=-2) # (num_layers, batch, head, source_len, dim)
            self.prompt_values = torch.cat(self.prompt_values, dim=-2) # (num_layers, batch, head, source_len, dim)
            self.prompt_attention_mask = torch.cat(self.prompt_attention_mask, dim=-1) # (batch, source_len)

        if self.normalize:
            self.prompt_keys = torch.nn.functional.normalize(self.prompt_keys, dim=-1)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.model_encoder_max_len], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.model_encoder_max_len:], skip_special_tokens=True)}')
            print()

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

    def window_indices(self, total_seq_len):
        # Copied from SLED (Ivgy et al., 2022)
        # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
        if total_seq_len <= self.model_encoder_max_len:
            return [(0, total_seq_len, 0, total_seq_len)]
        else:
            results = []
            stride = self.model_encoder_max_len - 2 * self.window_margin
            # if self.chunk_overlap == 0:
            #     stride = self.model_encoder_max_len
            context_start = update_start_ind = 0
            context_end = self.model_encoder_max_len
            update_end_ind = context_end - self.window_margin
            # first window always should update from the beginning
            results.append((context_start, context_end, update_start_ind, update_end_ind))  

            while context_end < total_seq_len:
                context_end = min(total_seq_len, context_end + stride)
                context_start = (
                    context_start + stride if context_end < total_seq_len else total_seq_len - self.model_encoder_max_len
                )
                update_start_ind = max(update_start_ind + stride, update_end_ind)
                # last window always should update until the end
                update_end_ind = (
                    min(total_seq_len, update_end_ind + stride) if context_end < total_seq_len else total_seq_len
                )

                cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                 update_end_ind - context_start

                results.append((cs, ce, us, ue))
            return results

    def pre_generate_hook(self, input_ids, **kwargs):
        self.reset_memory(input_ids, kwargs['attention_mask'])
        new_kwargs = kwargs
        if 'attention_mask' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
            new_kwargs['attention_mask'] = kwargs['attention_mask'][:, :self.model_encoder_max_len]
        new_kwargs['use_cache'] = True
        return self.original_generate_func(input_ids[:, :self.model_encoder_max_len], **new_kwargs)

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.model.base_model.decoder.gradient_checkpointing = False
        if not self.is_input_encoding_pass:
            if self.model.training:
                # self.reset_memory(input_ids, attention_mask)
                self.long_inputs_encoded, self.long_inputs_mask = self.chunked_encode_input(input_ids=input_ids, attention_mask=attention_mask)
                input_ids = input_ids[:, :self.model_encoder_max_len]
                labels = labels[:, :self.model_encoder_max_len] if labels is not None else None
                attention_mask = attention_mask[:, :self.model_encoder_max_len] if attention_mask is not None else None
            else:
                if kwargs.get('past_key_values') is None:
                    self.is_first_test_decoding_step = True

                if input_ids is not None:
                    self.input_ids = torch.cat([self.input_ids, input_ids[0]])
                if kwargs.get('decoder_input_ids') is not None:
                    self.generated_input_ids = torch.cat([self.generated_input_ids, kwargs['decoder_input_ids']], axis=-1)
            
        result = self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
        self.is_first_test_decoding_step = False
        return result

    def create_cross_attn_pre_forward_hook(self, original_cross_attn_forward_func, decoder_layer, i):
        def attention_pre_forward_hook(hidden_states, attention_mask=None, *args, **kwargs):
            self.cur_decoder_layer_index = i
            if kwargs.get('past_key_value') is not None:
                # it's a tuple, and we convert it to a list to be able to perform assignment 
                # and modify its items from our attention_forward_hook
                self.cur_layer_key_value_placeholder = \
                    kwargs['past_key_value'] = list(kwargs['past_key_value']) # (batch, head, time, attn_dim)

            if self.model.training:
                # from: (batch, tgt_len, dim) to: (batch * tgt_len, 1, dim)
                batch_size, tgt_len, dim = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, 1, hidden_states.shape[-1])
                # from: (batch, 1, tgt_len, dim) to: (batch * tgt_len, 1, 1, dim)
                attention_mask = attention_mask.reshape(-1, 1, 1, attention_mask.shape[-1])
                
                attn_output, attn_weights_reshaped, past_key_value = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                attn_output = attn_output.reshape(batch_size, tgt_len, dim)
                return attn_output, attn_weights_reshaped, past_key_value
            else:
                return original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
        
        return attention_pre_forward_hook

    def attention_forward_hook(self, module, input, output):
        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass or self.is_first_test_decoding_step:
            return
        with torch.no_grad():
            query = AttentionKNNWrapper.model_query_process[self.model_type](
                self.model, output)[:,-1] # (batch * beam, head, dim)
            query = query[:, self.head_nums] # (batch * beam, head, dim)
            if self.normalize:
                query = torch.nn.functional.normalize(query, dim=-1)

            if self.use_datastore:
                # query: (batch, beam, head, dim)
                # need to multiply by key vector
                # query.view(query.shape[0], query.shape[1] * query.shape[2])
                # k_proj in attention? 
                attention_layer_list = self.model_attention_layer_to_capture[self.model_type](self.model, self.knn_layer_begin, self.knn_layer_end)
                k_proj_layer = [layers[0] for layers in attention_layer_list][self.cur_decoder_layer_index]
                v_proj_layer = [layers[1] for layers in attention_layer_list][self.cur_decoder_layer_index]
                
                # modify query by k_projs 
                k_proj = k_proj_layer.weight
                k_proj = k_proj.view(1, self.num_heads, query.shape[-1], k_proj.shape[0]) # (1, num_heads, attn_dim, embed_dim)
                datastore_query = query.unsqueeze(-2) # (batch * beam, num_heads, 1, attn_dim)
                datastore_query = torch.matmul(datastore_query, k_proj) # (batch * beam, num_heads, 1, embed_dim)
                datastore_query = datastore_query.squeeze(-2)  # (batch * beam, num_heads, embed_dim)
                datastore_query = datastore_query.view((self.datastore.batch_size, -1, datastore_query.shape[2])) # (batch, beam * num_heads, embed_dim)
                # then search
                if self.reconstruct_embeddings:
                    # embeddings: (batch, beam * head, model_encoder_max_len, dim)
                    top_search_key_scores, top_search_key_indices, embeddings = self.datastore.search_and_reconstruct(datastore_query, k=self.model_encoder_max_len) 
                else:
                    top_search_key_scores, top_search_key_indices = self.datastore.search(datastore_query, k=self.model_encoder_max_len)
                    # self.embeddings: (batch,              src_len, dim)
                    # indices:         (batch, beam * head, model_encoder_max_len)
                    # embeddings: (batch, beam * head, model_encoder_max_len, dim)
                    embeddings = torch.take_along_dim(input=self.embeddings.unsqueeze(1), 
                        indices=top_search_key_indices.unsqueeze(-1).to(self.embeddings.device), dim=-2)
                    embeddings = embeddings.to(self.device)
                # (batch, beam, head, model_encoder_max_len)
                top_search_key_scores = top_search_key_scores.reshape((self.datastore.batch_size, -1, self.num_heads, self.model_encoder_max_len))
                top_search_key_indices = top_search_key_indices.reshape((self.datastore.batch_size, -1, self.num_heads, self.model_encoder_max_len))
                # embeddings: (batch, beam, head, model_encoder_max_len, dim)
                embeddings = embeddings.reshape((self.datastore.batch_size, -1, self.num_heads, self.model_encoder_max_len, embeddings.shape[-1]))
                                    
            # raw_values are actually token indices; need to look them up
            if (not self.use_datastore) or self.test_datastore:
                this_layer_prompt_keys = self.prompt_keys[self.cur_decoder_layer_index]
                this_layer_prompt_values = self.prompt_values[self.cur_decoder_layer_index]
                # query: (batch * beam, head, dim)
                batch_size = self.prompt_input_ids.shape[0]
                beam_size = query.shape[0] // batch_size
                # query: (batch, beam, head, dim)
                query = query.reshape(batch_size, beam_size, *query.shape[1:])
                # this_layer_prompt_keys: (batch, head, source_len, dim)
                # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
                # query.unsqueeze(-1):             (batch, beam, head, dim, 1)
                # attn_weights:  (batch, beam, head, source_len)
                attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums], query.unsqueeze(-1)).squeeze(-1) 
                # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
                prompt_attention_mask_to_add = (1 - self.prompt_attention_mask) * -1e9 # (batch, source_len)
                prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1)
                attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)
                if self.exclude_attention and attn_weights.shape[-1] > self.model_encoder_max_len:
                    attn_weights[..., :self.model_encoder_max_len] -= 1e9

                # target_keys, target_values, topk = self.get_target_slices(output)
                topk = min(self.model_encoder_max_len, attn_weights.shape[-1])
                top_key_scores, top_key_indices = torch.topk(attn_weights, k=topk, dim=-1, sorted=True) # (batch, beam, head, trunc_source)
                if self.save_heatmap:
                    # heatrow: (beam, heads, source_len)
                    heatrow = torch.zeros([top_key_indices.shape[1], top_key_indices.shape[2], this_layer_prompt_keys.shape[-2]], dtype=torch.float)
                    heatrow = heatrow.scatter(index=top_key_indices[0], src=torch.ones_like(top_key_scores[0]), dim=-1)
                    # heatrow = torch.nn.functional.softmax(heatrow, dim=-1)
                    # self.heatmap: (beam, heads, targets, source_len)
                    self.heatmap = torch.cat([self.heatmap, heatrow.unsqueeze(-2)], axis=-2)

            if self.test_datastore:
                assert top_key_indices.shape == top_search_key_indices.shape
                assert torch.mean((top_key_indices == top_search_key_indices).float()) > 0.99

            if self.use_pointers and self.prev_tokens[self.cur_decoder_layer_index] is not None:
                pointers = self.prev_tokens[self.cur_decoder_layer_index] + 1
                pointers[pointers >= this_layer_prompt_keys.shape[-2]] = -1
                # Take only pointers that pointed at tokens that we eventually generated
                # Can be possibly disabled, since we might want to attend to things and then generate something else
                previously_pointed_tokens = torch.take_along_dim(self.prompt_input_ids.unsqueeze(-2).unsqueeze(-2), 
                    indices=self.prev_tokens[self.cur_decoder_layer_index], dim=-1)
                previously_generated_tokens = self.generated_input_ids[:, -1].reshape(pointers.shape[:2])
                pointers[previously_pointed_tokens != previously_generated_tokens.unsqueeze(-1).unsqueeze(-1)] = -1
                # expensive, so commented out: 
                #   pointers_that_are_already_top_indices = (pointers.unsqueeze(-1) == top_key_indices.unsqueeze(-2)).any(-1)
                #   pointers[pointers_that_are_already_top_indices] = -1
                pointers_masked_first = pointers.sort(-1).values
                # pointers_masked_first = torch.nn.functional.pad(pointers_masked_first, 
                #     pad=[top_key_indices.shape[-1] - pointers_masked_first.shape[-1], 0], value=-1)
                top_key_indices = torch.where(pointers_masked_first > -1, pointers_masked_first, top_key_indices)

                self.prev_tokens[self.cur_decoder_layer_index] = top_key_indices
            
            if self.verbose:
                if self.is_encoder_decoder:
                    for i, beam in enumerate(self.generated_input_ids):
                        print(f'({i}) Generated: {self.tokenizer.decode(beam)}')
                else:
                    print(f'Generated: {self.tokenizer.decode(self.input_ids)}')
                print()
        
        if self.use_datastore:
            # k_proj_layer.weight, v_proj_layer.weight: (embed_dim, embed_dim)
            # embeddings: (batch, beam, head, encoder_len, embed_dim)
            embed_dim = embeddings.shape[-1]
            k_weight = k_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1) # (1, 1, heads, embed_dim, attn_dim)
            k_bias = k_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2)
            v_weight = v_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1)  # (1, heads, embed_dim, attn_dim)
            v_bias = v_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2)
            # new_keys, new_values: (batch, beam, head, encoder_len, attn_dim)
            new_keys = torch.matmul(embeddings, k_weight) + k_bias.unsqueeze(0) # (beam, head, encoder_len, embed_dim)
            new_values = torch.matmul(embeddings, v_weight) + v_bias.unsqueeze(0) # (beam, head, encoder_len, embed_dim)
        else:
            # this_layer_prompt_keys:   (batch,       head, source_len, dim)
            # top_key_indices:          (batch, beam, head, trunc_source)
            new_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            new_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)

        if self.test_datastore:
            correct_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            correct_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            assert correct_keys.shape == new_keys.shape
            assert correct_values.shape == new_values.shape
            assert torch.mean(torch.isclose(correct_keys, new_keys, rtol=1e-3, atol=1e-3).float()) > 0.99
            assert torch.mean(torch.isclose(correct_values, new_values, rtol=1e-3, atol=1e-3).float()) > 0.99

        self.cur_layer_key_value_placeholder[0] = new_keys.flatten(0, 1)
        self.cur_layer_key_value_placeholder[1] = new_values.flatten(0, 1)
        return

    def train_attention_forward_hook(self, module, input, output):
        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass or self.is_first_test_decoding_step:
            return
        this_layer_prompt_keys = self.cur_layer_key_value_placeholder[0]
        this_layer_prompt_values = self.cur_layer_key_value_placeholder[1]
        with torch.no_grad():
            query = AttentionKNNWrapper.model_query_process[self.model_type](
                self.model, output) # (batch * beam, tgt_len, head, dim)
            # query = query[:, :, self.head_nums] # (batch * beam, head, dim)
            if self.normalize:
                query = torch.nn.functional.normalize(query, dim=-1)

            # TODO: This matmul will be replaced by a kNN search for long inputs
            # query: (batch * beam, tgt_len, head, dim)
            batch_size = this_layer_prompt_keys.shape[0]
            tgt_len = query.shape[0] // batch_size
            # query: (batch, tgt, head, dim)
            query = query.reshape(batch_size, tgt_len, *query.shape[2:])
            # this_layer_prompt_keys: (batch, head, source_len, dim)
            # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
            # attn_weights:  (batch, tgt_len, head, 1, source_len)
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1).permute(0,1,2,4,3))
            attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1), query.unsqueeze(-1)) \
                .reshape(batch_size, tgt_len, query.shape[-2], 1, this_layer_prompt_keys.shape[-2])
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
            prompt_attention_mask_to_add = (1 - self.long_inputs_mask) * -1e9 # (batch, source_len)
            prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)

            # target_keys, target_values, topk = self.get_target_slices(output)
            topk = min(self.model_encoder_max_len, attn_weights.shape[-1])
            top_key_scores, top_key_indices = torch.topk(attn_weights, k=min(topk, attn_weights.shape[-1]), dim=-1, sorted=True) # (batch, beam, head, tgt, trunc_source)

                   
        # this_layer_prompt_keys:   (batch,          head,    source_len, dim)
        # top_key_indices:          (batch, tgt_len, head, 1, trunc_source)
        new_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        new_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        
        # (batch * beam, head, tgt_len, trunc_source, attn_dim)
        self.cur_layer_key_value_placeholder[0] = new_keys.flatten(0, 1)
        self.cur_layer_key_value_placeholder[1] = new_values.flatten(0, 1)
        return


    def reorder_cache_hook(self, past, beam_idx):
        self.last_beam_idx = beam_idx
        self.generated_input_ids = self.generated_input_ids[beam_idx]
        for i, layer_prev_tokens in enumerate(self.prev_tokens):
            if layer_prev_tokens is not None:
                self.prev_tokens[i] = layer_prev_tokens.flatten(0, 1)[beam_idx].reshape(layer_prev_tokens.shape)
        if self.save_heatmap and self.heatmap.numel() > 0:
            self.heatmap = self.heatmap[beam_idx]
        return self.original_reorder_cache_func(past, beam_idx)
        
    def bart_create_key_value(encoder_hidden_states, decoder_layer):
        # (batch, time, hidden_dim)
        attention = decoder_layer.encoder_attn
        # key, value: (batch, heads, time, attn_dim)
        key = attention.k_proj(encoder_hidden_states)
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = attention.v_proj(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # key, value: (batch, heads, time, attn_dim)
        return key, value 
    
    def bart_process_key_value(model, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value

    def t5_process_key_value(model, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value

        
    def bart_process_query(model, output):
        # (batch, time, heads, attn_dim)
        attention = model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query
    
    def t5_process_query(model, output):
        # (batch, time, heads, attn_dim)
        attention = model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query
    
    def gpt2_process_key_value(model, capturer):
        c_attn = capturer.captured
        attention = model.base_model.h[-1].attn
        _, key, value = c_attn.split(attention.split_size, dim=2)
        # query = attention._split_heads(query, attention.num_heads, attention.head_dim)
        key = attention._split_heads(key, attention.num_heads, attention.head_dim)
        value = attention._split_heads(value, attention.num_heads, attention.head_dim)
        return key, value
    
    def gpt2_process_query(model, output):
        c_attn = output
        attention = model.base_model.h[-1].attn
        query, _, _ = c_attn.split(attention.split_size, dim=2)
        query = attention._split_heads(query, attention.num_heads, attention.head_dim)
        return query

    def plot_heatmap(self, data, xticklabels='auto', yticklabels='auto'):
        # data: (heads, targets, source_len)
        import seaborn as sb
        import matplotlib.pyplot as plt
        # print('gat = np.array([')
        # for row in data[0]:
        #     rowstr = ', '.join([f'{x:.2f}' for x in row])
        #     print(f'    [{rowstr}],')
        # print(']')

        # sb.set(font_scale=1.5, rc={'text.usetex': True})
        for i in range(data.shape[0]):
            fig, axes = plt.subplots(1, 1, figsize=(40, 100))
            cur_ax = axes
            axes.set_title(f'Head #{i}, length: {data.shape[2]}, target length: {data.shape[1]}')
            cur_ax = axes
            # annot = [[x for x in row] for row in data]
            ax = sb.heatmap(data[i], annot=False, fmt='.2f',
                            xticklabels=512, yticklabels=yticklabels, ax=cur_ax)
            ax.xaxis.tick_top()
            plt.savefig(f'knns_head{i}.pdf')
            # plt.savefig('gat_s10_contrast.pdf')
            plt.show()
    
    model_key_value_process = {
        'bart': bart_process_key_value,
        't5': t5_process_key_value,
        'led': bart_process_key_value,
        'gpt2': gpt2_process_key_value,
    }

    model_query_process = {
        'bart': bart_process_query,
        't5': t5_process_query,
        'led': bart_process_query,
        'gpt2': gpt2_process_query,
    }
            
    model_attention_layer_to_capture = {
        'bart': lambda model, layer_begin, layer_end: [
                [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
                for layer in model.base_model.decoder.layers[layer_begin:layer_end]
            ],
        't5': lambda model, layer_begin, layer_end: [
                [layer.layer[1].EncDecAttention.k, layer.layer[1].EncDecAttention.v]
                for layer in model.base_model.decoder.layers[layer_begin:layer_end]
            ],
        'led': lambda model, layer_begin, layer_end: [
                [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
                for layer in model.base_model.decoder.layers[layer_begin:layer_end]
            ],
        'gpt2': lambda model, layer_begin, layer_end: 
            [layer.attn.c_attn 
                for layer in model.base_model.h[layer_begin:layer_end]]
    }

    model_attention_op_to_run = {
        'bart': lambda model, layer_begin, layer_end: [
            layer.encoder_attn.q_proj
                for layer in model.base_model.decoder.layers[layer_begin:layer_end]
        ],
        't5': lambda model, layer_begin, layer_end: [
            layer.layer[1].EncDecAttention.q
                for layer in model.base_model.decoder.block[layer_begin:layer_end]
        ],
        'led': lambda model, layer_begin, layer_end: [
            layer.encoder_attn.q_proj
                for layer in model.base_model.decoder.layers[layer_begin:layer_end]
        ],
    }

    model_attention_layer_to_run = {
        'bart': lambda model, layer_begin, layer_end: 
            model.base_model.decoder.layers[layer_begin:layer_end],
        't5': lambda model, layer_begin, layer_end: 
            model.base_model.decoder.block[layer_begin:layer_end],
        'led': lambda model, layer_begin, layer_end: 
            model.base_model.decoder.layers[layer_begin:layer_end],
    }

    model_self_attention = {
        'bart': lambda decoder_layer: 
            decoder_layer.self_attn,
        't5': lambda decoder_layer: 
            decoder_layer.layer[0],
        'led': lambda decoder_layer: 
            decoder_layer.self_attn,
    }

    model_cross_attention = {
        'bart': lambda decoder_layer: 
            decoder_layer.encoder_attn,
        't5': lambda decoder_layer: 
            decoder_layer.layer[1],
        'led': lambda decoder_layer: 
            decoder_layer.encoder_attn,
    }

    model_window_size = {
        'bart': lambda model: model.config.max_position_embeddings,
        't5': lambda model: model.config.n_positions,
        'led': lambda model: model.config.max_encoder_position_embeddings,
        'gpt2': lambda model: model.config.n_positions,
    }


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, module, input, output):
        self.captured = input if self.capture_input else output
        if not self.layer.training:
            self.captured = self.captured.detach()