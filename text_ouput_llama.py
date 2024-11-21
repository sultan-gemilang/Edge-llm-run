from transformers import AutoModelForCausalLM, LlamaTokenizer

from datetime import datetime

import torch
import numpy as np
import argparse

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_llm(model_name, dtype, accelerate_true):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        device_map= 'auto' if accelerate_true else None,
        offload_folder = './offload'
    )
    
    return model.eval()

def get_tokenizer(model_name, use_fast):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast
    )
    
    return tokenizer

def generate_text(model, tokenizer, promt, max_new_tokens, do_sample, dev):
    print(f'promt: {promt}')
    
    input_ids = tokenizer(promt, return_tensors='pt').to(dev)
    
    generate_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )
    
    output = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True
    )
    
    return output
    
def main():
    #arg parse placeholder
    seed = 0
    model_name = 'baffo32/decapoda-research-llama-7B-hf'
    token_gen_size = 100
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    promt = ['Hello World is a']
    
    print(f'model used\t{model_name}')
    print(f'device used\t{device}')
    
    set_seed(seed)
    
    print('\n-----Loading model-----')
    model = get_llm(
        model_name=model_name,
        dtype=torch.float16,
        accelerate_true=True
    )
    
    print('\n----Loading Tokenizer-----')
    tokenizer = get_tokenizer(
        model_name=model_name,
        use_fast=True
    )
    
    print('\n-----Generating Text-----')
    gen_time = datetime.now()
    output_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        promt=promt,
        max_new_tokens=token_gen_size,
        do_sample=True,
        dev=device
    )
    end_time = datetime.now()
    
    print(f'output_text:\n{output_text}')
    print(f'Gen time:\t{(end_time-gen_time).seconds}')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()