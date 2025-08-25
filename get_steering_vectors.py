import os.path
import random, os
import pandas as pd
import copy
import pickle
import numpy as np
import time
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
torch.cuda.empty_cache()  
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from typing import Dict, Union, List, Any
from tqdm import tqdm
import os
import re
import time

def initialize_model(device_map, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map, torch_dtype = torch.float16)
    model.eval()
    print("Finishing initialize_model")
    return model, tokenizer

def prepare_prompts(prompt_add, prompt_sub, tokenizer) -> tuple:
    def tlen(prompt): return len(tokenizer(prompt)["input_ids"])
    def pad_right(prompt, length): return prompt + " " * (length - tlen(prompt))
    l = max(tlen(prompt_add), tlen(prompt_sub))
    return pad_right(prompt_add, l), pad_right(prompt_sub, l)

def get_resid_pre(prompt, layer, model, tokenizer) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    resid_values = {}
    # In order to 直接拿到某層的中間輸出 
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        resid_values["resid"] = output.float().detach()
    target_block = model.model.layers[layer]
    hook = target_block.register_forward_hook(hook_fn)
    _ = model(input_ids)  # forward 運算時觸發 hook
    hook.remove()
    return resid_values["resid"]

def ave_hook(resid_pre, hook, act_diff, coeff):
    if resid_pre.shape[1] == 1: return
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) than prompt tokens ({ppos})!"
    resid_pre[:, :apos, :] += coeff * act_diff

def generate_actadd(model, tokenizer, layer: int, prompt_add: str, prompt_sub: str):
    prompt_add, prompt_sub = prepare_prompts(prompt_add, prompt_sub, tokenizer)
    act_add = get_resid_pre(prompt_add, layer, model, tokenizer)
    act_sub = get_resid_pre(prompt_sub, layer, model, tokenizer)
    print(f"act_add = {act_add}")
    print(f"act_sub = {act_sub}")
    act_diff = act_add - act_sub
    # ave_act_diff = act_diff/
    print(f"act_diff = {act_diff}")
    return {
        "prompt_add": prompt_add,
        "prompt_sub": prompt_sub,
        "act_diff": act_diff,
        "act_add": act_add,
        "act_sub": act_sub
    }

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def get_prompt_related_dict(model_name, device_map, prompt_add, prompt_sub, layer, seed = 88):
    seed_everything(seed) 
    model_zephyr_7b, tokenizer = initialize_model(device_map, model_name)
    prompt_dict = generate_actadd(model_zephyr_7b, tokenizer, layer = layer, prompt_add = prompt_add, prompt_sub = prompt_sub)
    return prompt_dict, model_zephyr_7b, tokenizer
