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
from copy import deepcopy

from utils.model_utils import *
from utils.dataset_utils import *
from utils.coresets import *


class Sim:
    def __init__(self,params,sim_type,device,model):

        """
        :param params: Dictionary of simulation parameters
        :param sim_type: Type of simulation - Distributed, Oracle, Interactive
        :param device: Device to run simulations on (CPU or GPU)
        :param model: Model to use for simulation (CNN)
        """
        
        self.n_device = params["n_device"]
        self.n_sim = params["n_sim"]
        self.n_rounds = params["n_rounds"]
        self.n_epoch = params["n_epoch"]
        self.b_size = params["b_size"]
        self.n_iter = params["n_iter"]
        self.n_class = params["n_class"]
        self.test_b_size = params["test_b_size"]
        self.lr = params["lr"]
        self.n_size = params["n_size"]
        self.n_obs = params["n_obs"]
        self.n_cache = params["n_cache"]
        self.unc_type = params["unc_type"]
        self.sim_type = sim_type
        self.dataset_type = params["dataset_type"]
        self.dirichlet_alpha = params["dirichlet_alpha"]
        self.dirichlet = params["dirichlet"]
        self.dirichlet_base = params["dirichlet_base"]
        self.dirichlet_base_alpha = params["dirichlet_base_alpha"]

        self.params = params
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

        self.accs = np.zeros((self.n_sim,self.n_rounds+1))
        self.accs_matrix = np.zeros((self.n_sim,self.n_rounds+1,self.n_class,self.n_class))
        self.min_accs = np.zeros((self.n_sim,self.n_rounds+1)) 

        self.obs_ind = dict()
        self.dataset_ind = dict()

        self.seeds = list()
        self.base_seeds = list()
        
        self.sim_i = 0

        self.model = model

    # Creates observed indices from dataset based on given data distributions
    def create_base_inds(self,y,base_classes,sim_i,seed):
        
        self.base_seeds.append(seed)
        np.random.seed(seed)
        
        inds = list()
        if self.dirichlet_base:
            x_dist = np.random.dirichlet(np.repeat(self.dirichlet_base_alpha,len(base_classes)))
        else:
            x_dist = np.random.rand(len(base_classes))
        x_dist = x_dist / sum(x_dist)
        
        N_x = [0 for i in range(len(base_classes))]

        N_x_lims = np.zeros((len(base_classes),1),int)
        for i in range(len(base_classes)):
            N_x_lims[i] = torch.sum( y == base_classes[i]).item()

        n_lim = self.n_size
        args = [i for i in range(len(base_classes))]
        fulls = []
        while n_lim !=0:
            n_cache_cand = saferound(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]*n_lim/sum(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]),places=0)
            f = [arg for arg in args if (arg not in fulls)]
            for g,arg in enumerate(f): 
                if N_x_lims[arg]<=n_cache_cand[g]+N_x[arg]:
                    prev_nx = N_x[arg]
                    N_x[arg] = int(N_x_lims[arg])
                    fulls.append(arg)
                    n_lim = n_lim - int(N_x_lims[arg]-prev_nx)
                else:
                    N_x[arg] += int(n_cache_cand[g])
                    n_lim = n_lim - int(n_cache_cand[g]) 
              
        for i,c_i in enumerate(base_classes):
            ind_s = random.sample(np.argwhere(y == c_i).reshape(-1).tolist(), k = N_x[i])
            inds.extend(ind_s)
        self.dataset_ind[sim_i]= [inds]

    # Sets base indices to given indices
    def set_base_inds(self,inds,sim_i):
        self.dataset_ind[sim_i]= [inds]

    # Calculates dataset stats
    def dataset_stats(self,y_train):
        N_x = np.zeros((self.n_class,1),int)

        labels = y_train[self.dataset_ind[self.sim_seed][-1]]
        for i in range(self.n_class):
            N_x[i] += sum(labels==i).item()
        return N_x

    # Creates random data distributions for different classes
    def create_xdist(self,sim_i,obs_clss,y):
        
        np.random.seed(sim_i)

        x_dist = np.zeros((self.n_device,self.n_class))
        if self.dirichlet:
            for i in range(self.n_device):
                x_dist[i,:] = np.random.dirichlet(np.repeat(self.dirichlet_alpha,self.n_class))
        else:
            for i in range(self.n_device):
                for j in obs_clss[i]:
                    x_dist[i,j] = np.random.rand()
            
        x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

        N_x_lims = np.zeros((self.n_class,1),int)
        for i in range(self.n_class):
            N_x_lims[i] = torch.sum( y == i).item()

        N_x = np.zeros((self.n_device,self.n_class),dtype=int)
        for i in range(self.n_device):
            n_lim = self.n_obs
            args = [arg for arg in range(self.n_class)]
            fulls = []
            while n_lim!=0:
                n_cache_cand = saferound(x_dist[i,~np.isin(np.arange(len(x_dist[i])), fulls)]*n_lim/sum(x_dist[i,~np.isin(np.arange(len(x_dist[i])), fulls)]),places=0)
                f = [arg for arg in args if (arg not in fulls)]
                for g,arg in enumerate(f): 
                    if N_x_lims[arg]<=n_cache_cand[g]+N_x[i,arg]:
                        prev_nx = N_x[i,arg]
                        N_x[i,arg] = int(N_x_lims[arg])
                        fulls.append(arg)
                        n_lim = n_lim - int(N_x_lims[arg]-prev_nx)
                    else:
                        N_x[i,arg] += int(n_cache_cand[g])
                        n_lim = n_lim - int(n_cache_cand[g]) 
            x_dist[i] = N_x[i,:]
        
        x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

        return x_dist, N_x

    # Creates obs indics based on given distributions
    def create_obs_ind(self,N_x,y,sim_i):
    
        np.random.seed(sim_i)
        random.seed(sim_i)

        obs_ind = [[] for i in range(self.n_rounds)]

        for round_i in range(self.n_rounds):
            obs_ind[round_i] = [[] for i in range(self.n_device)]
            for dev_i in range(self.n_device):
                for i in range(self.n_class):
                    ind_s = random.sample(np.argwhere(y == i).tolist()[0], k = N_x[dev_i][i])
                    obs_ind[round_i][dev_i].extend(ind_s)
        
        return obs_ind

    # Creates dataset for intereference
    def create_dataset(self,X,y):
        if self.dataset_type == "MNIST":
            transform = trfm.Compose([
            trfm.ToTensor(),
            trfm.Normalize((0.1307), (0.3081))])
            dataset = MNISTDataset(X,y,transform)
        elif self.dataset_type == "CIFAR10":
            transform = trfm.Compose([
            trfm.ToTensor(),
            trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            dataset = CIFAR10Dataset(X,y,transform)
        elif self.dataset_type == "AdversarialWeather" or self.dataset_type == "DeepDrive":
            transform  = trfm.Compose([
            trfm.Resize(256),
            trfm.CenterCrop(224),
            trfm.ToTensor(),
            trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            dataset = AdversarialWeatherDataset(X,y,transform,self.params["cache_all"])
        return dataset

    # Creates dataset for training
    def create_traindataset(self,X,y):
        if self.dataset_type == "MNIST":
            transform = trfm.Compose([
            trfm.ToTensor(),
            trfm.Normalize((0.1307), (0.3081))])
            dataset = MNISTDataset(X,y,transform)
        elif self.dataset_type == "CIFAR10":
            transform = trfm.Compose([
            trfm.RandomCrop(32, padding=4),
            trfm.RandomHorizontalFlip(),
            trfm.ToTensor(),
            trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            dataset = CIFAR10Dataset(X,y,transform)
        elif self.dataset_type == "AdversarialWeather" or self.dataset_type == "DeepDrive":
            transform = trfm.Compose([
            trfm.Resize(256),
            trfm.RandomCrop(224),
            trfm.RandomHorizontalFlip(),
            trfm.ToTensor(),
            trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            dataset = AdversarialWeatherDataset(X,y,transform,self.params["cache_all"])

        return dataset

    def Entropy(self,p):
        return -np.sum(p*np.log(p+1e-12),axis=1)
    
    def LeastConfidence(self,p):
        return 1 - np.max(p,axis=1)
    
    def MarginSampling(self,p):
        return 1-(np.max(p,axis=1) - np.sort(p)[:,-2])

    def Random(self,p):
        return np.random.rand(p.shape[0])

    def unc_scores(self,p):
        if self.unc_type == "Entropy":
            return self.Entropy(p)
        elif self.unc_type == "LeastConfidence":
            return self.LeastConfidence(p)
        elif self.unc_type == "MarginSampling":
            return self.MarginSampling(p)
        elif self.unc_type == "Random":
            return self.Random(p)
        else:
            raise ValueError("Invalid Uncertainty Type")
    
    def obtain_logits(self,X_train,y_train,inds):

        self.model.eval()
        dataset = self.create_dataset(X_train[inds],y_train[inds])
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.test_b_size, shuffle=False)

        logits = np.zeros((len(inds),self.n_class))

        with torch.no_grad():
            ind = 0

            for x,_ in test_loader:

                x = x.to(self.device)
                m = torch.nn.Softmax(dim=1)
                out,_ = self.model(x)
                logits[ind:ind+len(x)] = m(out).cpu().detach().numpy()
                ind += len(x)

        return logits

    def coreset_obtain_embeddings(self,X_train,y_train,obs_inds,base_inds):

        self.model.eval()

        embeddings = np.zeros((self.n_device,len(obs_inds[0]),self.model.emb_size))
        base_embeddings = np.zeros((len(base_inds),self.model.emb_size))

        with torch.no_grad():
            
            for i in range(self.n_device):

                dataset_i = self.create_dataset(X_train[obs_inds[i]],y_train[obs_inds[i]])
                loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
                ind = 0
                for x,_ in loader:
                    x = x.to(self.device)
                    out,emb = self.model(x)
                    embeddings[i][ind:ind+len(x)] = emb.cpu().detach().numpy()
                    ind += len(x)

            dataset_i = self.create_dataset(X_train[base_inds],y_train[base_inds])
            loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
            ind = 0
            for x,_ in loader:
                x = x.to(self.device)
                out,emb = self.model(x)
                base_embeddings[ind:ind+len(x)] = emb.cpu().detach().numpy()
                ind += len(x)
        
        return embeddings, base_embeddings

    def badge_obtain_embeddings(self,X_train,y_train,obs_inds,base_inds):
        
        emb_size = self.model.emb_size
        self.model.eval()
        embeddings = np.zeros((self.n_device,len(obs_inds[0]),emb_size*self.n_class))
        base_embeddings = np.zeros((len(base_inds),emb_size*self.n_class))
        m = torch.nn.Softmax(dim=1)

        with torch.no_grad():

            for i in range(self.n_device):

                dataset_i = self.create_dataset(X_train[obs_inds[i]],y_train[obs_inds[i]])
                loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
                ind = 0
                for x,_ in loader:
                    x = x.to(self.device)
                    out,emb = self.model(x)
                    emb = emb.cpu().detach().numpy()
                    y_preds = torch.argmax(out,1).cpu().detach().numpy()
                    
                    batchProbs = m(out).cpu().detach().numpy()

                    for c in range(self.n_class):
                         embeddings[i][ind:ind+len(x),emb_size * c : emb_size * (c+1)] = deepcopy(emb) * (-1 * batchProbs[:,c].reshape(-1,1))
                         embeddings[i][ind:ind+len(x),emb_size * c : emb_size * (c+1)][y_preds == c] = deepcopy(emb)[y_preds == c] * (1 - batchProbs[:,c][y_preds == c].reshape(-1,1))

                    ind += len(x)
            
            dataset_i = self.create_dataset(X_train[base_inds],y_train[base_inds])
            loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
            ind = 0
            for x,_ in loader:
                x = x.to(self.device)
                out, emb = self.model(x)
                emb = emb.cpu().detach().numpy()
                y_preds = torch.argmax(out,1).cpu().detach().numpy()
                batchProbs = m(out).cpu().detach().numpy()

                for c in range(self.n_class):
                    base_embeddings[ind:ind+len(x),emb_size * c : emb_size * (c+1)] = deepcopy(emb) * (-1 * batchProbs[:,c].reshape(-1,1))
                    base_embeddings[ind:ind+len(x),emb_size * c : emb_size * (c+1)][y_preds == c] = deepcopy(emb)[y_preds == c] * (1 - batchProbs[:,c][y_preds == c].reshape(-1,1))

        return embeddings, base_embeddings

    def clip_obtain_embeddings(self,X_train,y_train,obs_inds,base_inds,train_embs):
        
        emb_size = 1024
        embeddings = np.zeros((self.n_device,len(obs_inds[0]),emb_size))
        base_embeddings = np.zeros((len(base_inds),emb_size))

        if self.dataset_type == "CIFAR10":  
            for i in range(self.n_device):
                for j in range(len(embeddings[0])):
                    embeddings[i][j] = train_embs[obs_inds[i][j]]
            for i in range(len(base_embeddings)):
                base_embeddings[i] = train_embs[base_inds[i]]
        elif self.dataset_type == "AdversarialWeather":
            for i in range(self.n_device):
                for j in range(len(embeddings[0])):
                    embeddings[i][j] = train_embs["/".join(X_train[obs_inds[i][j]].split("/")[-4:])]
            for i in range(len(base_embeddings)):
                base_embeddings[i] = train_embs["/".join(X_train[base_inds[i]].split("/")[-4:])]
        
        elif self.dataset_type == "DeepDrive":
            for i in range(self.n_device):
                for j in range(len(embeddings)):
                    embeddings[i][j] = train_embs[X_train[obs_inds[i][j]]]
            for i in range(len(base_embeddings)):
                base_embeddings[i] = train_embs[X_train[base_inds[i]]]
        
        return embeddings, base_embeddings

    def cache_inds(self,all_inds,obs_inds,unc_scores):
        cached_inds = list()
        for i in range(self.n_device):
            inds = obs_inds[i]
            locs = [all_inds.index(ind) for ind in inds]
            scores_i = unc_scores[locs]

            cached_inds.extend(np.array(inds)[np.argsort(scores_i)[-self.n_cache:]].tolist())
            
        return cached_inds

    def reset_model(self):
        if self.dataset_type == "MNIST" or self.dataset_type == "CIFAR10":
            self.model.apply(init_weights)
        elif self.dataset_type == "AdversarialWeather" or self.dataset_type == "DeepDrive":
            self.model.apply(init_weights)

    def sim_round(self,sim_i,sim_seed,X_train,y_train,testset,base_inds,obs_ind,train_embs):

        self.sim_i = sim_i
        self.sim_seed = sim_seed
        self.obs_ind[sim_seed] = obs_ind

        random.seed(sim_seed)
        np.random.seed(sim_seed)
        self.seeds.append(sim_seed)

        self.set_base_inds(base_inds[0],sim_seed)

        self.accs[sim_i,0] = test_model(self.model,testset)
        
        for round_i in range(self.n_rounds):

            if self.unc_type in ["Entropy","LeastConfidence","MarginSampling","Random"]:

                all_indices = list()
                for i in range(self.n_device):
                    all_indices.extend(obs_ind[round_i][i])
                
                all_indices = list(set(all_indices))

                logits = self.obtain_logits(X_train,y_train,all_indices)

                scores = self.unc_scores(logits)

                cached_inds = self.cache_inds(all_indices,obs_ind[round_i],scores)
            
            else:
                if self.unc_type == "coreset":
                    embeddings,base_embeddings = self.coreset_obtain_embeddings(X_train,y_train,obs_ind[round_i],self.dataset_ind[sim_seed][-1])
                elif self.unc_type == "badge":
                    embeddings,base_embeddings = self.badge_obtain_embeddings(X_train,y_train,obs_ind[round_i],self.dataset_ind[sim_seed][-1])
                elif self.unc_type == "clip":
                    embeddings,base_embeddings = self.clip_obtain_embeddings(X_train,y_train,obs_ind[round_i],self.dataset_ind[sim_seed][-1],train_embs)

                sampling_policy = kCenterGreedy(embeddings,base_embeddings,obs_ind[round_i],self.n_iter,self.n_cache)

                cached_inds = sampling_policy.sample_caches(self.sim_type)

            self.dataset_ind[sim_seed].append(self.dataset_ind[sim_seed][-1] + cached_inds)

            self.reset_model()
            trainset = self.create_traindataset(X_train[tuple([list(set(self.dataset_ind[self.sim_seed][-1]))])],y_train[tuple([list(set(self.dataset_ind[self.sim_seed][-1]))])])
            train_model(self.model,trainset,converge=self.params["converge"])

            self.accs[sim_i,round_i+1] = test_model(self.model,testset,self.test_b_size)

    def save_infos(self,save_loc,sim_type):

        with open(save_loc+'/'+sim_type+'_params.json', 'w') as outfile:
            json.dump(self.params, outfile)
    
        with open(save_loc+'/'+sim_type+'_dataset_ind.json', 'w') as outfile:
            json.dump(self.dataset_ind, outfile)

        with open(save_loc+'/'+sim_type+'_obs_ind.json', 'w') as outfile:
            json.dump(self.obs_ind, outfile)

        with open(save_loc+'/'+sim_type+'_seeds.json', 'w') as outfile:
            json.dump(self.seeds, outfile)

        with open(save_loc+'/'+sim_type+'_acc.npy', 'wb') as outfile:
            np.save(outfile, self.accs)
