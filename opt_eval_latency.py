from transformers import AutoModelForCausalLM, GPT2Tokenizer

import torch

import numpy as np
import argparse
import subprocess

from datetime import datetime

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        offload_folder='./offload'
    )
    
    model.seqlen = model.config.max_position_embeddings
    return model

def get_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer

def generate_text(model, model_inputs, num_gen_token, do_sample):
    generate_ids = model.generate(
        **model_inputs,
        max_new_tokens = num_gen_token,
        do_sample = do_sample
    )
    
    return generate_ids

def main():
    # args parse var placeholder
    # seed = 0
    # model_name = 'baffo32/decapoda-research-llama-7B-hf'
    # token_gen_size = 100
    promt = "There are various way to compress LLM model, this can be done by reducing the number of parameters or by using the decompression scheme called 'Pulse' which is the equivalent of the first generation techniques, so its very important to minimize the number of parameters of LLM model. However, since LLM model can only compress LLM model's LLM model to the output, it is not possible"   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='OPT model used for inference')
    parser.add_argument('--seed', type=int, required=True, help='Sets seed fot repeatability')
    parser.add_argument('--token_size', type=int, required=True, help='Maximum generated tokens')
    parser.add_argument('--log', type=bool, default=False, help='Log the Jetson performance on csv file')
    args = parser.parse_args()
    
    if args.log:
        subprocess.Popen(['python3', './jtop_logger.py', '--file', f'{args.model_name}_log.csv'])
    else:
        pass
    
    print('\n-----Loading Model-----')
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_llm(args.model_name)
    model.eval()
    
    tokenizer = get_tokenizer(args.model_name)
    
    print(f'model used\t{args.model_name}')
    print(f'device used\t{device}')   
     
    
    #TGT & TPOT    
    print('\n-----TGT & TPOT-----')
    tgt_time = datetime.now()
    model_inputs = tokenizer(promt, return_tensors='pt').to(device)
    
    tpot_time = datetime.now()
    generate_ids = generate_text(
        model=model,
        model_inputs=model_inputs,
        num_gen_token=args.token_size,
        do_sample=True
    )
    tpot_end_time = datetime.now()
    
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    tgt_end_time = datetime.now()
    print('Done')
    
    
    #TTFT
    print('\n-----TTFT-----')
    ttft_time = datetime.now()
    _ = generate_text(
        model=model,
        model_inputs=model_inputs,
        num_gen_token=1,
        do_sample=False
    )
    ttft_end_time = datetime.now()
    print('Done')
    
    inputs_token_len = model_inputs.input_ids.size(dim=1) 
    gen_token_len = generate_ids.size(dim=1)
    
    # print(model_inputs)
    # print(generate_ids)
    
    print('\n-----Text Output----')
    print(f'\n{text}\n')
    
    tpot_delta  = round(((tpot_end_time - tpot_time).seconds * 1000) + ((tpot_end_time - tpot_time).microseconds / 1000))
    ttft_delta  = round(((ttft_end_time - ttft_time).seconds * 1000) + ((ttft_end_time - ttft_time).microseconds / 1000))
    tgt_delta   = round(((tgt_end_time - tgt_time).seconds * 1000) + ((tgt_end_time - tgt_time).microseconds / 1000))
    
    tpot = round(tpot_delta/(gen_token_len-inputs_token_len), 3)
    tps = round((gen_token_len-inputs_token_len)/tpot_delta*1000, 3)
    
    print('\n-----Latency Result----')
    print(f'Input token length\t{inputs_token_len}')
    print(f'Totalngth\t{gen_token_len}')
    print(f'Token Generated\t{gen_token_len-inputs_token_len} tokens')
    
    print()
    
    print(f'TGT\t-> {tgt_delta} ms')
    print(f'aTPOT\t-> {tpot} ms/tok | {gen_token_len-inputs_token_len} tokens')
    print(f'TpS\t-> {tps} tok/s | {gen_token_len-inputs_token_len} tokens')
    print(f'TTFT\t-> {ttft_delta} ms')
    

if __name__ == '__main__':
    main()
    