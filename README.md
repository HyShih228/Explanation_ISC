# Explanation_ISC
We provide an explanation of how a large language model (LLM) responds to instructions, with a particular focus on its intrinsic self-correction behavior. In the experimental section, we use the Zephyr-7B-SFT model on the RealToxicityPrompts dataset, and this repository contains the code used to conduct our experiments.

## Requirement
You need to install Python packages torch, transformers, scikit-learn, random, numpy, matplotlib, json, tqdm and so on. 

## Experiments
- get_hidden_states.py: Extracts the input length, generated response, and hidden states from the LLM.
- get_pca_results.py: Visualizes the differences in hidden states across rounds, corresponding to Figure 2.
- get_relation.py: Computes the inner product between the hidden state differences and the unembedding matrix across rounds, corresponding to Table 2.
- rounds_toxic.py: Sends generated responses to the AWS Perspective API to obtain toxicity scores, corresponding to Table 1 and Figure 1.
- utils.py: Contains parse_hidden_states functions for recording input lengths, responses, and hidden states from the LLM.
