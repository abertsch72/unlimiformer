from functools import partial
import numpy as np


def decode(id_to_something, tokenizer=None, data_args=None):
    decode_fn = None
    switch_case = None
    elem = next(iter(id_to_something.values()))
    if isinstance(elem, str):
        switch_case = -1
        decode_fn = lambda text: text.strip()
    elif isinstance(elem, list) and not isinstance(elem[0], int):
        if isinstance(elem[0], str):
            switch_case = 0
            decode_fn = lambda texts: [text.strip() for text in texts]
        else:
            switch_case = 1
            decode_fn = lambda token_ids_list: [
                text.strip()
                for text in partial(
                    tokenizer.batch_decode, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )(token_ids_list)
            ]
    else:
        switch_case = 2
        decode_fn = lambda token_ids: partial(
            tokenizer.decode, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )(token_ids).strip()

    id_to_text = {}
    for id_, something in id_to_something.items():
        if switch_case == -1 or switch_case == 0:
            obj_to_decode = something
        else:
            if data_args is None:
                data_args = {}
            if not isinstance(data_args, dict):
                data_args = vars(data_args)
            if data_args.get("ignore_pad_token_for_loss", True):
                # Replace -100 in the token_ids as we can't decode them.
                if switch_case == 1:
                    token_ids_list = something
                    for i in range(len(token_ids_list)):
                        token_ids_list[i] = _replace_padding(token_ids_list[i], tokenizer.pad_token_id)
                    obj_to_decode = token_ids_list
                elif switch_case == 2:
                    token_ids = something
                    token_ids = _replace_padding(token_ids, tokenizer.pad_token_id)
                    obj_to_decode = token_ids
            else:
                obj_to_decode = something

        id_to_text[id_] = decode_fn(obj_to_decode)

    return id_to_text


def _replace_padding(token_ids: np.array, pad_token_id):
    return np.where(token_ids != -100, token_ids, pad_token_id)
