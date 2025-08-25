# required packages:
import os.path
import pickle
import numpy as np
import torch
import time
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
import glob
import csv
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from utils import parse_hidden_states
from get_steering_vectors import get_prompt_related_dict, initialize_model


def inner_product_unembed_diff(nopca_difference,model):
    inner = []
    unembedding = model.lm_head.weight
    device = unembedding.device
    dtype = unembedding.dtype
    difference_tensor = nopca_difference.to(device=device, dtype=dtype)
    inner = difference_tensor @ unembedding.T 
    print("inner:", inner)
    print("inner.shape:", inner.shape)
    return inner

def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data",type=int,default=500)
    parser.add_argument("--num_rounds",type=int,default=5)
    parser.add_argument("--model_name",type=str,default = "alignment-handbook/zephyr-7b-sft-full")
    parser.add_argument("--device_map",type=str,default = "cuda:0")
    parser.add_argument("--layer",type=int,default = 10)
    parser.add_argument("--prompt_add",type=str,default = "Nontoxic")
    parser.add_argument("--prompt_sub",type=str,default = "Toxic")
    parser.add_argument("--seed",type=int,default = 88)
    args = parser.parse_args()

    prompt_dict, model_zephyr_7b, tokenizer = get_prompt_related_dict(
                                                    model_name = args.model_name, 
                                                    device_map = args.device_map, 
                                                    prompt_add = args.prompt_add, 
                                                    prompt_sub = args.prompt_sub, 
                                                    layer = args.layer, 
                                                    seed = args.seed )
    diff_ave = prompt_dict["act_diff"].mean(dim=1)
    print(diff_ave.shape)
    inner_products_steering = inner_product_unembed_diff(diff_ave,model_zephyr_7b)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vocab = tokenizer.get_vocab()

    toxic_dict = {}
    with open('tokens.csv', mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            token = row[0]
            score = row[1]
            toxic_dict[vocab[token]] = [token,float(score)]
    # print(toxic_dict)

    keys = list(toxic_dict.keys())
    toxicity = 0
    for key in keys:
        toxicity += float(toxic_dict[key][1])*float(inner_products_steering[0][key].detach())
    print("toxicity:", toxicity)
    pass

if __name__ == '__main__':
    run()
    pass
