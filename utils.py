import os.path
import pandas as pd
import pickle
import numpy as np

def parse_hidden_states(num_data, num_rounds, output_dir = "hidden_states"):
    length = []
    result = []
    last_hidden = []

    for d in range(num_data):
        length_temp =[]
        result_temp =[]
        last_hidden_temp =[]
        for ri in range(num_rounds):
            fname = os.path.join(output_dir,f"%s_%s.pkl"%(ri, str(d).zfill(5)))
            item = pickle.load(open(fname, "rb"))
            length_temp.append(item['length'])
            result_temp.append(item['result'])
            last_hidden_temp.append(item['last_hidden'])
        length.append(length_temp)
        result.append(result_temp)
        last_hidden.append(last_hidden_temp)

    return np.array(length), result, np.array(last_hidden)
