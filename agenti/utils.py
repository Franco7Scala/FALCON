from transformers import pipeline

def get_pipeline(task, model, tokenizer, max_new_tokens = 128, do_sample = False, return_full_text = False):

    return pipeline \
    (
        task = task, 
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = max_new_tokens,
        do_sample = do_sample,
        return_full_text = return_full_text
    )