import json
import numpy as np
import os
root_dir = "./attack/Amoeba/"
if not os.path.exists(root_dir + "dataset/CICDOH20"):
    os.makedirs(root_dir + "dataset/CICDOH20")
if not os.path.exists(root_dir + "dataset/TIISSRC23"):
    os.makedirs(root_dir + "dataset/TIISSRC23")

datasets = ["CICDOH20", "TIISSRC23"]
datanames = ["all.json", "train.json", "test.json", "valid.json"]

for dataset in datasets:
    src_dir = root_dir + "../../dataset/" + dataset + "/json/"
    save_dir = root_dir + "/dataset/" + dataset + "/"
    
    for filename in datanames:
        datadir = src_dir + filename
        save_X_dir = save_dir + filename.split(".")[0] + "_X.npy"
        save_y_dir = save_dir + filename.split(".")[0] + "_y.npy"
        
        with open(datadir, "r") as f:
            data = json.load(f)
        X = []
        y = []
        for flow in data:
            processed_flow = []
            timestamp = flow["timestamp"]
            direction_length = flow["direction_length"]
            
            last_time = 0
            for i in range(len(direction_length)):
                d = direction_length[i]
                t = timestamp[i] - last_time
                last_time = timestamp[i]
                processed_flow.append([d, t])
            
            X.append(np.array(processed_flow))
            y.append(1 if flow["label"] == 3 else 0)
        X = np.array(X, dtype=object)
        y = np.array(y)
        np.save(save_X_dir, X)
        np.save(save_y_dir, y)