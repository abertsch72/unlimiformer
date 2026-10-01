def preprocess_tokenized_inputs(
    examples,
    input_column,
    max_source_length,
    assign_zero_to_too_long_val_examples,
):
    source_ids = examples[input_column]
    too_long = [
        assign_zero_to_too_long_val_examples
        and max_source_length is not None
        and len(input_ids) > max_source_length
        for input_ids in source_ids
    ]
    if max_source_length is not None:
        source_ids = [input_ids[:max_source_length] for input_ids in source_ids]

    if "attention_mask" in examples:
        attention_mask = examples["attention_mask"]
        if max_source_length is not None:
            attention_mask = [mask[:max_source_length] for mask in attention_mask]
        if any(len(mask) != len(input_ids) for mask, input_ids in zip(attention_mask, source_ids)):
            raise ValueError("Tokenized input_ids and attention_mask must have matching lengths")
    else:
        attention_mask = [[1] * len(input_ids) for input_ids in source_ids]

    return {
        "input_ids": source_ids,
        "attention_mask": attention_mask,
        "not_valid_for_eval": too_long,
    }
