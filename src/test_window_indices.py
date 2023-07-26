def window_indices(total_seq_len, model_encoder_max_len):
    # Copied from SLED (Ivgy et al., 2022)
    # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
    if total_seq_len <= model_encoder_max_len:
        return [(0, total_seq_len, 0, total_seq_len)]
    else:
        results = []
        # if self.chunk_overlap == 0:
        #     stride = self.model_encoder_max_len
        window_margin = int(model_encoder_max_len * 0.5 / 2)
        stride = model_encoder_max_len - 2 * window_margin
        context_start = update_start_ind = 0
        context_end = model_encoder_max_len
        
        update_end_ind = context_end
        
        # first window always should update from the beginning
        results.append((context_start, context_end, update_start_ind, update_end_ind))  

        while context_end < total_seq_len:
            context_end = min(total_seq_len, context_end + stride)
            context_start = (
                context_start + stride if context_end < total_seq_len else total_seq_len - model_encoder_max_len
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

if __name__ == "__main__":
    res = window_indices(100, 8)
    for r in res:
        a,b,c,d = r
        print(f'Encoding {a}:{b}, taking {a+c}:{a+d}')