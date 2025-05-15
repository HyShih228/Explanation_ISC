# required packages:
import os.path
import random, os
import pandas as pd
import numpy as np
import time
import requests
import subprocess
import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import parse_hidden_states

def get_toxicity(sentence):
    payload = {
        "TextSegments": [{"Text": sentence}],
        "LanguageCode": "en"
    }

    command = [
        "aws", "comprehend", "detect-toxic-content",
        "--region", "us-east-1",
        "--cli-input-json", json.dumps(payload)  
    ]

    result = subprocess.run(
        command,
        input=json.dumps(payload),  
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            return float(response["ResultList"][0]["Toxicity"])
        except (KeyError, IndexError, ValueError) as e:
            print(f"Unexpected result for: {sentence}\n Error: {e}\n Raw: {result.stdout}")
            return 0.0
    else:
        print(f"AWS CLI error for: {sentence}\n {result.stderr}")
        return 0.0

def get_all_scores(Output,num_rounds,output_dir):
    toxicity = []
    for round in range(num_rounds):
        print("round ", round)
        print("\n")
        temp_record = []
        # run through each ans[round] and get toxicity_score to average 
        for text in Output:
            print(f"text[round] = {text[round]}")
            toxicity_score = get_toxicity(text[round])
            temp_record.append(toxicity_score)
        toxicity.append(temp_record)
        print(toxicity)

    toxicity_T = list(map(list, zip(*toxicity)))  # shape: (num_samples, num_rounds)

    with open(os.path.join(output_dir,'toxicity.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'Round {i+1}' for i in range(num_rounds)])
        writer.writerows(toxicity_T)
    print(f"Toxicity scores saved to {os.path.join(output_dir,'toxicity.csv')}")
    return toxicity

def plotwithstd(data,condition,output_dir):
    data = np.array(data)
    means = data.mean(axis=1)
    stds = data.std(axis=1)
    x =  np.array([0,1,2,3,4])

    plt.figure(figsize=(6, 4))  
    line_color = '#08519c' if condition == "Positive" else '#a50f15'
    plt.plot(x, means, label='Mean', color = line_color , linewidth=2)
    fill_color = '#377eb8' if condition == "Positive" else '#e41a1c'
    plt.fill_between(x, means - stds, means + stds, color=fill_color, alpha=0.2, label='±1 Std Dev')
    plt.xticks(x)
    plt.ylim([-0.05,1])
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Toxicity', fontsize=12)
    #plt.title('Mean with Standard Deviation Shading')
    plt.legend(loc='best', fontsize=10)
    #plt.grid(True)
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.savefig(os.path.join(output_dir,f"Toxicity_{condition}.png"), dpi=300, bbox_inches='tight')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data",type=int,default=500)
    parser.add_argument("--num_rounds",type=int,default=5)
    parser.add_argument("--output_dir",type=str,default="Positive_hidden_states")
    args = parser.parse_args()

    seed_everything(88) 
    length, result, last_hidden = parse_hidden_states(args.num_data, args.num_rounds, args.output_dir)
    
    save_path = os.path.join(args.output_dir,'toxicity.csv')
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping computation.")
        Toxicity = [[],[],[],[],[]]
        with open(save_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                row = np.array(row, dtype=float)
                for r in range(0,args.num_rounds):
                    Toxicity[r].append(row[r]) 
        for i in range(0,args.num_rounds):
            Toxicity[i] = Toxicity[i][:args.num_data]    

        print(Toxicity)
    else:
        Toxicity = get_all_scores(result,args.num_rounds,args.output_dir)
    
    plotwithstd(Toxicity ,"Positive",args.output_dir)
    print(f"(Positive) Mean for each round = {np.array(Toxicity).mean(axis=1)}")
    print(f"(Positive) Std. for rach round = {np.array(Toxicity).std(axis=1)}")
    pass

if __name__ == '__main__':
    run()
    pass
