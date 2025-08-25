# required packages:
import os.path
import pandas as pd
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
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from utils import parse_hidden_states
from get_steering_vectors import get_prompt_related_dict

def pca_process(pca,last_hidden,initial = 0):
    if len(last_hidden.shape) == 2:
        # shape: (N, hidden_dim)
        if initial == 0:
            return pca.fit_transform(last_hidden)
        else:
            return pca.transform(last_hidden)
        
    elif len(last_hidden.shape) == 3:
        flat_hidden = last_hidden.reshape((last_hidden.shape[0]*last_hidden.shape[1],last_hidden.shape[2]))
        if initial == 0:
            pca_results = pca.fit_transform(flat_hidden)
        else: 
            pca_results = pca.transform(flat_hidden)
        pca_results  = pca_results.reshape((last_hidden.shape[0],last_hidden.shape[1],pca_results.shape[-1]))
    else:
        raise ValueError("Unsupported shape for last_hidden")
    return pca_results

def plot_3dim_vector(input_points, condition, output_dir, k):
    for L in range(len(input_points)):
        # print(f"len(input_points) = {len(input_points)}")
        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, facecolors='none', edgecolors='grey', s=50, linewidths=1.5, label='Origin')
        vectors = np.array(input_points[L])
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norm_vectors  = normalize(vectors)
        # print(f"input_points[L] = {input_points[L]}")
        ax.view_init(elev=30, azim=80) #elev=30, azim=45
        origin = np.zeros((len(norm_vectors), 3))

        if condition[:8] == "Positive":
            if k == 1: 
                vector_color = 'seagreen' 
                line_width=1.2 
                Alpha = 0.6
                if L == 0:
                    arrow_length = 0.01
                else: 
                    arrow_length = 0.1
            elif k == 2:
                line_width = 0.4
                vector_color = '#A7C7E7'
                arrow_length = 0.1
                Alpha = 0.7
            else:
                line_width = 0.4
                vector_color = '#1F3A93'
                arrow_length = 0.1
                Alpha = 0.5
        elif condition[2:] == "compare_vec":
            line_width = 1
            vector_color = "#A54B0B"
            arrow_length = 0.1
            Alpha = 0.9
        else: 
            if k == 1: 
                vector_color = 'seagreen'
                line_width=1.2 
                Alpha = 0.6
                if L == 0:
                    arrow_length = 0.01
                else: 
                    arrow_length = 0.1
            elif k == 2:
                line_width = 0.4
                vector_color = '#F28B82'
                arrow_length = 0.1
                Alpha = 0.7
            else:
                line_width = 0.4
                vector_color = '#800020'
                arrow_length = 0.1
                Alpha = 0.4
        ax.quiver(
            origin[:, 0], origin[:, 1], origin[:, 2],
            norm_vectors[:, 0], norm_vectors[:, 1], norm_vectors[:, 2],
            color=vector_color, alpha=Alpha, linewidth=line_width, arrow_length_ratio = arrow_length
        )
        tick_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_zticks(tick_vals)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        margin_ratio = 1.1  
        for dim, set_lim in zip(range(3), [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            max_abs = np.max(np.abs(norm_vectors[:, dim]))
            lim = max_abs * margin_ratio
            set_lim(-lim, lim)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"test_3d_plot_{condition}{L}.png"), bbox_inches='tight', dpi=300)#, bbox_inches='tight'
    plt.close()

def analyze_vectors_pca(diff_ave, round1_ave, pca, condition, output_dir):
    vecs = np.vstack([diff_ave, round1_ave])
    compare_vec = pca_process(pca, vecs, initial=1)
    compare_vec = compare_vec[np.newaxis, :]

    plot_3dim_vector(compare_vec ,f"{condition}_compare_vec", output_dir, k=4)
    # P_compare_vec[0][0] = diff_ave, P_compare_vec[0][1] = P_round1_ave
    # print("diff_ave.shape =", diff_ave.shape)
    # print("round1_ave.shape =", round1_ave.shape)
    #sim = cosine_similarity(np.array(diff_ave), np.array([round1_ave]))
    sim = cosine_similarity(np.array([compare_vec[0][0]]), np.array([compare_vec[0][1]]))
    proj_coeff = np.dot(np.array(compare_vec[0][1]), np.array(compare_vec[0][0])) / np.dot(np.array(compare_vec[0][0]), np.array(compare_vec[0][0]))
    print("Cosine similarity:", sim[0][0])
    print("proj_coeff:", proj_coeff) #lambda

    proj = proj_coeff * np.array(compare_vec[0][0])
    rejection_vector = np.array(compare_vec[0][1]) - proj
    print("Rejection vector:", rejection_vector)

    round1_ave_norm = np.linalg.norm(np.array(compare_vec[0][1]))
    diff_ave_norm = np.linalg.norm(np.array(compare_vec[0][0]))
    rejection_norm = np.linalg.norm(rejection_vector)
    print(f"The norm of {condition}_round1_ave vector:", round1_ave_norm)
    print("The norm of diff_ave vector:", diff_ave_norm)
    print("The norm of rejection vector:", rejection_norm)

def analyze_vectors(diff_ave, round1_ave, pca, condition, output_dir):
    # print("diff_ave.shape =", diff_ave.shape)
    # print("round1_ave.shape =", round1_ave.shape)
    sim = cosine_similarity(np.array(diff_ave), np.array([round1_ave]))
    re_diff_ave = np.array(diff_ave).reshape(-1)
    proj_coeff = np.dot(np.array(round1_ave), re_diff_ave) / np.dot(re_diff_ave, re_diff_ave)
    print("Cosine similarity:", sim[0][0])
    print("proj_coeff:", proj_coeff) #lambda

    proj = proj_coeff * np.array(diff_ave)
    rejection_vector = np.array(round1_ave) - proj
    print("Rejection vector:", rejection_vector)
    round1_ave_norm = np.linalg.norm(np.array(round1_ave.astype(np.float64)))
    diff_ave_norm = np.linalg.norm(np.array(diff_ave))
    rejection_norm = np.linalg.norm(rejection_vector)
    print(f"The norm of {condition}_round1_ave vector:", round1_ave_norm)
    print("The norm of diff_ave vector:", diff_ave_norm)
    print("The norm of rejection vector:", rejection_norm)

def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data",type=int,default=500)
    parser.add_argument("--num_rounds",type=int,default=5)
    parser.add_argument("--output_dir1",type=str,default="Positive_hidden_states")
    parser.add_argument("--output_dir2",type=str,default="Negative_hidden_states")
    parser.add_argument("--model_name",type=str,default = "alignment-handbook/zephyr-7b-sft-full")
    parser.add_argument("--device_map",type=str,default = "cuda:0")
    parser.add_argument("--layer",type=int,default = 10)
    parser.add_argument("--prompt_add",type=str,default = "Nontoxic")
    parser.add_argument("--prompt_sub",type=str,default = "Toxic")
    parser.add_argument("--seed",type=int,default = 88)
    args = parser.parse_args()

    if os.path.exists("shared_pca.joblib"):
        pca = joblib.load("shared_pca.joblib")
        print("Use existed PCA.")
    else: 
        pca = PCA(n_components=3)
        joblib.dump(pca, "shared_pca.joblib")
        print("PCA saved.")

    prompt_dict, model_zephyr_7b, tokenizer = get_prompt_related_dict(
                            model_name = args.model_name, 
                            device_map = args.device_map, 
                            prompt_add = args.prompt_add, 
                            prompt_sub = args.prompt_sub, 
                            layer = args.layer, 
                            seed = args.seed
                    )
    
    diff_ave = prompt_dict["act_diff"].mean(dim=1)

    # Positive
    length1, result1, last_hidden1, last_hidden_mean1 = parse_hidden_states(args.num_data, args.num_rounds, args.output_dir1)
    pca_results1 = pca_process(pca,last_hidden1,0)
    P_diff = last_hidden_mean1[1,:] - last_hidden_mean1[0,:]
    print(f"shape = {last_hidden_mean1.shape}")

    # Negative
    length2, result2, last_hidden2, last_hidden_mean2 = parse_hidden_states(args.num_data, args.num_rounds, args.output_dir2)
    pca_results2 = pca_process(pca,last_hidden2,1)
    N_diff = last_hidden_mean2[1,:] - last_hidden_mean2[0,:]

    # Steering vector
    diff_ave = diff_ave.cpu().numpy()
    print(diff_ave)
    # P_round1_ave = last_hidden_mean1[0] 
    # N_round1_ave = last_hidden_mean2[0] 
    # print(P_round1_ave)
    # print(N_round1_ave)
    print(np.array_equal(P_diff,N_diff))
    print("------ Without PCA ------")
    print(f"For positive prompts:")
    analyze_vectors( diff_ave, P_diff, pca, "P", args.output_dir1)
    print(" ")
    print(f"For negative prompts: ")
    analyze_vectors( diff_ave, N_diff, pca, "N", args.output_dir2)
    print(" ")
    print("------ With PCA ------")
    print(f"For positive prompts:")
    analyze_vectors_pca( diff_ave, P_diff, pca, "P", args.output_dir1)
    print(" ")
    print(f"For negative prompts: ")
    analyze_vectors_pca(-diff_ave, N_diff, pca, "N", args.output_dir2)

    pass

if __name__ == '__main__':
    run()
    pass
