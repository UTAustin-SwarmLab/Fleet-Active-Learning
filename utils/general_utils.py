from re import VERBOSE
from tabnanny import verbose
from tkinter import N
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from iteround import saferound
import random
import seaborn as sns
import pandas as pd
import os
from statannot import add_stat_annotation
import sklearn.datasets as ds
import torch
import torchvision.datasets as datasets
import torchvision.transforms as trfm
import mosek
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
import shutil
import torchvision.models as vsmodels
import json
from PIL import Image
import cv2
import math
import matplotlib.patches
import pylab
import plotly.express as px
import gmplot
from utils.plotting_utils import *


# Combines multiple trials 
def combine_sims(run_ids,run_loc,target_run_loc,sim_types,name="run"):

    with open(run_loc+"/"+name+str(run_ids[0])+"/"+sim_types[0]+"_params.yml") as f:
        params = yaml.safe_load(f)

    Acc = dict()

    for sim_type in sim_types:

        params = dict()
        with open(run_loc+"/"+name+str(run_ids[0])+"/"+sim_type+"_params.yml") as f:
            params = yaml.safe_load(f)
        seeds = list()
        obs_ind = dict()
        dataset_ind = dict()
        Acc[sim_type] = np.zeros((len(run_ids),params["n_rounds"]+1))
        tot_sim = 0
        
        for i,run_i in enumerate(run_ids):
            

            with open(run_loc+"/"+name+str(run_i)+"/"+sim_type+"_params.yml") as f:
                new_params = yaml.safe_load(f)
            
            if new_params != params:
                print("Error in run"+str(run_i)+". Params don't match.")
                continue

            with open(run_loc+"/"+name+str(run_i)+"/"+sim_type+"_seeds.yml") as f:
                new_seed = yaml.safe_load(f)
            
            if any(s in seeds for s in new_seed):
                print("Error in run"+str(run_i)+". The sim seed is already added.")
                continue
            else:
                seeds += new_seed
            tot_sim += 1
            with open(run_loc+"/"+name+str(run_i)+"/"+sim_type+"_obs_ind.yml") as f:
                new_obs_ind = yaml.safe_load(f)
            obs_ind.update(new_obs_ind)

            with open(run_loc+"/"+name+str(run_i)+"/"+sim_type+"_dataset_ind.yml") as f:
                new_dataset_ind = yaml.safe_load(f)
            dataset_ind.update(new_dataset_ind)

            Acc[sim_type][i,:] = np.load(run_loc+"/"+name+str(run_i)+"/"+sim_type+"_acc.npy")
        
        params["n_sim"] = tot_sim
        
        with open(target_run_loc+'/'+sim_type+'_params.yml', 'w') as outfile:
            yaml.dump(params, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_obs_ind.yml', 'w') as outfile:
            yaml.dump(obs_ind, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_dataset_ind.yml', 'w') as outfile:
            yaml.dump(dataset_ind, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_seeds.yml', 'w') as outfile:
            yaml.dump(seeds, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_acc.npy', 'wb') as outfile:
            np.save(outfile, Acc[sim_type])

    Accs = []
    for sim_type in sim_types:
        Accs.append(Acc[sim_type])

    plot_accs(Accs,sim_types,target_run_loc+"/Accs.jpg")

# Creates new directories for simulations
def create_run_dir(run_loc,name="run"):

    run_i = 0
    if os.path.exists(run_loc):
        exps = os.listdir(run_loc)
    else:
        os.makedirs(run_loc)
        exps = []
    
    for i in range(len(exps)):
        if name+str(run_i) in exps:
            run_i += 1
        else:
            break

    os.makedirs(run_loc+"/"+name+str(run_i))
    return run_loc+"/"+name+str(run_i)

