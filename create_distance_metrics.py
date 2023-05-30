import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy
from sklearn.metrics import pairwise_distances

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *

dataset_loc = "/store/datasets/AdversarialWeather"
dataset_type = "AdversarialWeather"
img_loc = "/store/datasets/AdversarialWeather"
clip_emb_loc = "/store/datasets/AdversarialWeather"
run_loc = "./runs/AdversarialWeather/run22"

def clip_obtain_embeddings(X_train,inds,train_embs,dataset_type):
        
    emb_size = train_embs[list(train_embs.keys())[0]].shape[1]
    embeddings = np.zeros((len(inds),emb_size))

    if dataset_type == "CIFAR10" or dataset_type == "MNIST":  
        for i in range(len(embeddings)):
            embeddings[i] = train_embs[inds[i]]
    elif dataset_type == "AdversarialWeather":
        for i in range(len(embeddings)):
            embeddings[i] = train_embs["/".join(X_train[inds[i]].split("/")[-4:])]
        
    elif dataset_type == "DeepDrive":
        for i in range(len(embeddings)):
            embeddings[i] = train_embs[X_train[inds[i]].split("/")[-1]]
        
    return embeddings

X_train,X_test,y_train,y_test = load_datasets(dataset_loc,dataset_type,img_loc=img_loc)

if dataset_type == "AdversarialWeather":
    embs = np.load(clip_emb_loc+"/clip_embs.npy",allow_pickle=True).item()
    train_embs = {"/".join(X_train[i].split("/")[-4:]): embs["/".join(X_train[i].split("/")[-4:])]
                           for i in  range(len(X_train))}
else:
    train_embs = np.load(clip_emb_loc+"/train_embs.npy",allow_pickle=True).item()


dataset_inds = []
sim_types = ["Distributed","Oracle","Interactive"]

for sim_type in sim_types:

    with open(run_loc+"/"+sim_type+"_dataset_ind.json") as f:
        dataset_ind = json.load(f)
    
    dataset_inds.append(dataset_ind)

sim_keys = list(dataset_inds[0].keys())
Values = [np.zeros((len(dataset_inds[0]),len(dataset_inds[0][sim_keys[0]]))) for i in range(len(sim_types))]

all_inds = [i for i in range(len(X_train))]
all_embs = clip_obtain_embeddings(X_train,all_inds,train_embs,dataset_type)
pbar = tqdm(total=len(sim_types)*len(sim_keys)*len(dataset_inds[0][sim_keys[0]]))

for i in range(len(sim_types)):
    for j in range(len(sim_keys)):
        for k in range(len(dataset_inds[i][sim_keys[j]])):

            dataset_ind = dataset_inds[i][sim_keys[j]][k].copy()

            dataset_embs = clip_obtain_embeddings(X_train,dataset_ind,train_embs,dataset_type)

            dist_w_all_training = pairwise_distances(all_embs,dataset_embs,metric='euclidean')

            M = 1/(1+0.01*dist_w_all_training)

            Values[i][j][k] = (np.max(M,axis=1)).sum()

            if k !=0:
                Values[i][j][k] -= Values[i][j][0]

            pbar.update(1)
        Values[i][j][0] = 0


plot_values(Values,sim_types,save_loc=run_loc+"/submodular_values.jpg",y_label="$f(\mathcal{D}_c^r)$ Value of Submodular Objective")

mean_distributed = np.mean(Values[0][:,-1])
mean_oracle = np.mean(Values[1][:,-1])
mean_interactive = np.mean(Values[2][:,-1])

# Write it to txt file 
with open(run_loc+"/submodular_values.txt","w") as f:
    f.write("Distributed: "+str(mean_distributed)+"\n")
    f.write("Oracle: "+str(mean_oracle)+"\n")
    f.write("Interactive: "+str(mean_interactive)+"\n")

# Additionally we will write the mean values of the accuracies

accs = []
for sim_type in sim_types:
    acc = np.load(run_loc+"/"+sim_type+"_acc.npy")
    accs.append(acc)

accs = [np.mean(accs[i][:,-1]) for i in range(len(accs))]

with open(run_loc+"/accs.txt","w") as f:
    for i in range(len(accs)):
        f.write(sim_types[i]+": "+str(accs[i])+"\n")