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
from sklearn.metrics import pairwise_distances

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
            for dev_i in range(self.params["n_unique_device"]):
                for i in range(self.n_class):
                    ind_s = random.sample(np.argwhere(y == i).tolist()[0], k = N_x[dev_i][i])
                    obs_ind[round_i][dev_i*self.params["n_same_device"]].extend(ind_s)
                for same_i in range(1,self.params["n_same_device"]):
                    obs_ind[round_i][dev_i*self.params["n_same_device"]+same_i] = obs_ind[round_i][dev_i*self.params["n_same_device"]].copy()
        
        return obs_ind

    def create_det_obs_ind(self,y,sim_i):
    
        np.random.seed(sim_i)
        random.seed(sim_i)

        obs_ind = [[] for i in range(self.n_rounds)]

        for round_i in range(self.n_rounds):
            obs_ind[round_i] = [[] for i in range(self.n_device)]
            for dev_i in range(self.params["n_unique_device"]):
                obs_ind[round_i][dev_i*self.params["n_same_device"]] = random.sample([i for i in range(len(y))],k=self.n_obs)
                for same_i in range(1,self.params["n_same_device"]):
                    obs_ind[round_i][dev_i*self.params["n_same_device"]+same_i] = obs_ind[round_i][dev_i*self.params["n_same_device"]].copy()
        
        return obs_ind


    # Creates dataset for intereference
    def create_dataset(self,X,y):
        if self.dataset_type == "MNIST":
            transform = trfm.Compose([
            trfm.ToTensor(),
            trfm.Normalize((0.1307), (0.3081))])
            dataset = MNISTDataset(X,y,transform)
        elif self.dataset_type == "CIFAR10":
            if self.params["use_embeddings"]:
                dataset = EmbeddingDataset(X,y)
            else:
                transform = trfm.Compose([
                trfm.ToTensor(),
                trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                dataset = CIFAR10Dataset(X,y,transform)
        elif self.dataset_type == "AdversarialWeather" or self.dataset_type == "DeepDrive":
            if self.params["use_embeddings"]:
                dataset = EmbeddingDataset(X,y)
            else:
                transform  = trfm.Compose([
                trfm.Resize(256),
                trfm.CenterCrop(224),
                trfm.ToTensor(),
                trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
                dataset = AdversarialWeatherDataset(X,y,transform,self.params["cache_all"])
        elif self.dataset_type == "DeepDrive-Detection":
            transforms = []
            transforms.append(T.PILToTensor())
            transforms.append(T.ConvertImageDtype(torch.float))
            transform = T.Compose(transforms)
            dataset = DetectionDataset(X,y,transform)

        return dataset

    # Creates dataset for training
    def create_traindataset(self,X,y):
        if self.dataset_type == "MNIST":
            transform = trfm.Compose([
            trfm.ToTensor(),
            trfm.Normalize((0.1307), (0.3081))])
            dataset = MNISTDataset(X,y,transform)
        elif self.dataset_type == "CIFAR10":
            if self.params["use_embeddings"]:
                dataset = EmbeddingDataset(X,y)
            else:
                transform = trfm.Compose([
                trfm.RandomCrop(32, padding=4),
                trfm.RandomHorizontalFlip(),
                trfm.ToTensor(),
                trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                dataset = CIFAR10Dataset(X,y,transform)
        elif self.dataset_type == "AdversarialWeather" or self.dataset_type == "DeepDrive":
            if self.params["use_embeddings"]:
                dataset = EmbeddingDataset(X,y)
            else:
                transform = trfm.Compose([
                trfm.Resize(256),
                trfm.RandomCrop(224),
                trfm.RandomHorizontalFlip(),
                trfm.ToTensor(),
                trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
                dataset = AdversarialWeatherDataset(X,y,transform,self.params["cache_all"])
                dataset.set_use_cache(self.params["cache_in_first"])

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

    def coreset_obtain_embeddings(self,X_train,y_train,all_inds):

        self.model.eval()

        embeddings = np.zeros((len(all_inds),self.model.emb_size))

        with torch.no_grad():

            dataset_i = self.create_dataset(X_train[all_inds],y_train[all_inds])
            loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
            ind = 0
            for x,_ in loader:
                x = x.to(self.device)
                out,emb = self.model(x)
                embeddings[ind:ind+len(x)] = emb.cpu().detach().numpy()
                ind += len(x)
        
        return embeddings

    def badge_obtain_embeddings(self,X_train,y_train,all_inds):
        
        emb_size = self.model.emb_size
        self.model.eval()
        embeddings = np.zeros((len(all_inds),emb_size*self.n_class))
        m = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            
            dataset_i = self.create_dataset(X_train[all_inds],y_train[all_inds])
            loader = torch.utils.data.DataLoader(dataset_i, batch_size=self.test_b_size, shuffle=False)
            ind = 0
            for x,_ in loader:
                x = x.to(self.device)
                out, emb = self.model(x)
                emb = emb.cpu().detach().numpy()
                y_preds = torch.argmax(out,1).cpu().detach().numpy()
                batchProbs = m(out).cpu().detach().numpy()

                for c in range(self.n_class):
                    embeddings[ind:ind+len(x),emb_size * c : emb_size * (c+1)] = emb * (-1 * batchProbs[:,c].reshape(-1,1))
                    embeddings[ind:ind+len(x),emb_size * c : emb_size * (c+1)][y_preds == c] = emb[y_preds == c] * (1 - batchProbs[:,c][y_preds == c].reshape(-1,1))

        return embeddings

    def clip_obtain_embeddings(self,X_train,y_train,all_inds,train_embs):
        
        emb_size = train_embs[list(train_embs.keys())[0]].shape[1]
        embeddings = np.zeros((len(all_inds),emb_size))

        if self.dataset_type == "CIFAR10":  
            for i in range(len(embeddings)):
                embeddings[i] = train_embs[all_inds[i]]
        elif self.dataset_type == "AdversarialWeather":
            for i in range(len(embeddings)):
                embeddings[i] = train_embs["/".join(X_train[all_inds[i]].split("/")[-4:])]
        
        elif self.dataset_type == "DeepDrive" or self.dataset_type == "DeepDrive-Detection":
            for i in range(len(embeddings)):
                embeddings[i] = train_embs[X_train[all_inds[i]].split("/")[-1]]
        
        return embeddings

    def create_M_max_M(self,M,all_inds,obs_inds,base_inds):

        obs_ind = np.concatenate(obs_inds,axis=0)

        obs_mask = [all_inds.index(i) for i in obs_ind]

        base_mask = [all_inds.index(i) for i in base_inds]

        M_max = np.max(M[obs_mask][:,base_mask],axis=1).reshape(-1,1)

        M = M[np.ix_(obs_mask,obs_mask)]

        return M,M_max

    def map_embeddings_devices(self,embeddings,all_inds,obs_inds,base_inds):

        embs = np.zeros((self.n_device,len(obs_inds[0]),embeddings.shape[1]))
        base_embs = np.zeros((len(base_inds),embeddings.shape[1]))

        for i in range(self.n_device):
            
            for j in range(len(obs_inds[i])):

                embs[i][j] = embeddings[all_inds.index(obs_inds[i][j])]
            
        for j in range(len(base_inds)):
            base_embs[j] = embeddings[all_inds.index(base_inds[j])]

        return embs,base_embs

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
            if self.params["use_embeddings"]:
                self.model.apply(init_weights)
            else:
                self.model.fc = nn.Linear(self.model.emb_size,self.n_class).to(self.device)
                self.model.fc.apply(init_weights)
        if self.dataset_type == "DeepDrive-Detection":
            self.model = get_model("DeepDrive-Detection","DeepDrive-Detection",
                                self.model.device,self.model.b_size,self.model.n_epoch,self.model.lr,self.model.n_class)

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

                # Obtains unique indices for all devices
                all_indices = list()
                for i in range(self.n_device):
                    all_indices.extend(obs_ind[round_i][i])
                all_indices.extend(self.dataset_ind[sim_seed][-1])
                
                all_indices = list(set(all_indices))

                # Obtains embeddings for these unique indices

                if self.unc_type == "coreset":
                    embeddings = self.coreset_obtain_embeddings(X_train,y_train,all_indices)
                elif self.unc_type == "badge":
                    embeddings = self.badge_obtain_embeddings(X_train,y_train,all_indices)
                elif self.unc_type == "clip":
                    embeddings = self.clip_obtain_embeddings(X_train,y_train,all_indices,train_embs)

                distances = pairwise_distances(embeddings, metric="euclidean")
                M = 1/(1+0.01*distances)

                M,M_max = self.create_M_max_M(M,all_indices,obs_ind[round_i],self.dataset_ind[sim_seed][-1])
 
                if self.params["center_selection"]=="kcenter":
                    sampling_policy = kCenterGreedy(embeddings,base_embeddings,obs_ind[round_i],self.n_iter,self.n_cache)
                elif self.params["center_selection"]=="facility":
                    sampling_policy = FacilityLocation_with_M(M,M_max,obs_ind[round_i],self.n_cache)
                else:
                    raise ValueError("Invalid Center Selection Type")
                cached_inds = sampling_policy.sample_caches(self.sim_type)

            self.dataset_ind[sim_seed].append(self.dataset_ind[sim_seed][-1] + cached_inds)

            self.reset_model()
            trainset = self.create_traindataset(X_train[tuple([list(set(self.dataset_ind[self.sim_seed][-1]))])],y_train[tuple([list(set(self.dataset_ind[self.sim_seed][-1]))])])
            train_model(self.model,trainset,converge=self.params["converge"],only_final=self.params["train_only_final"])

            self.accs[sim_i,round_i+1] = test_model(self.model,testset,self.test_b_size,self.params["accuracy_type"])

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
    
    def save_params(self,save_loc,sim_type):

        with open(save_loc+'/'+sim_type+'_params.json', 'w') as outfile:
            json.dump(self.params, outfile)

class Sim_Detect(Sim):
    def __init__(self,params,sim_type,device,model,init_results):
        super().__init__(params,sim_type,device,model)
        self.precision = np.zeros((self.n_sim,2))
        self.recall = np.zeros((self.n_sim,2))
        self.map50 = np.zeros((self.n_sim,2))
        self.map50_95 = np.zeros((self.n_sim,2))
        self.fitness = np.zeros((self.n_sim,2))
        self.init_results = init_results
    
    def metrics_update(self,metrics,round_i):

        self.precision[self.sim_i,round_i+1] = metrics["metrics/precision(B)"]
        self.recall[self.sim_i,round_i+1] = metrics["metrics/recall(B)"]
        self.map50[self.sim_i,round_i+1] = metrics["metrics/mAP50(B)"]
        self.map50_95[self.sim_i,round_i+1] = metrics["metrics/mAP50-95(B)"]
        self.fitness[self.sim_i,round_i+1] = metrics["fitness"]

    def sim_round(self,sim_i,sim_seed,X_train,y_train,testset,base_inds,obs_ind,train_embs):

        self.sim_i = sim_i
        self.sim_seed = sim_seed
        self.obs_ind[sim_seed] = obs_ind

        random.seed(sim_seed)
        np.random.seed(sim_seed)
        self.seeds.append(sim_seed)

        self.set_base_inds(base_inds[0],sim_seed)

        self.metrics_update(self.init_results,-1)
        
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
                # Obtains unique indices for all devices
                all_indices = list()
                for i in range(self.n_device):
                    all_indices.extend(obs_ind[round_i][i])
                all_indices.extend(self.dataset_ind[sim_seed][-1])
                
                all_indices = list(set(all_indices))

                # Obtains embeddings for these unique indices

                if self.unc_type == "coreset":
                    embeddings = self.coreset_obtain_embeddings(X_train,y_train,all_indices)
                elif self.unc_type == "badge":
                    embeddings = self.badge_obtain_embeddings(X_train,y_train,all_indices)
                elif self.unc_type == "clip":
                    embeddings = self.clip_obtain_embeddings(X_train,y_train,all_indices,train_embs)

                distances = pairwise_distances(embeddings, metric="euclidean")
                M = 1/(1+0.01*distances)

                M,M_max = self.create_M_max_M(M,all_indices,obs_ind[round_i],self.dataset_ind[sim_seed][-1])
                                
                if self.params["center_selection"]=="kcenter":
                    sampling_policy = kCenterGreedy(embeddings,base_embeddings,obs_ind[round_i],self.n_iter,self.n_cache)
                elif self.params["center_selection"]=="facility":
                    sampling_policy = FacilityLocation_with_M(M,M_max,obs_ind[round_i],self.n_cache)
                else:
                    raise ValueError("Invalid Center Selection Type")
                cached_inds = sampling_policy.sample_caches(self.sim_type)

            self.dataset_ind[sim_seed].append(self.dataset_ind[sim_seed][-1] + cached_inds)

        trainset = create_detection_dataset(X_train[tuple([list(set(self.dataset_ind[self.sim_seed][-1]))])],testset)
        self.model.train(data=trainset,epochs=self.params["n_epoch"],save=False,device=self.device, val=False,pretrained=True,
        batch=self.params["b_size"],verbose=False,plots=False,cache=self.params["cache_all"],workers=self.params["n_workers"])
        metrics =  self.model.val(batch=self.params["test_b_size"],device=self.device)
        self.metrics_update(metrics.results_dict,round_i)

    def save_infos(self,save_loc,sim_type):

        with open(save_loc+'/'+sim_type+'_params.json', 'w') as outfile:
            json.dump(self.params, outfile)
    
        with open(save_loc+'/'+sim_type+'_dataset_ind.json', 'w') as outfile:
            json.dump(self.dataset_ind, outfile)

        with open(save_loc+'/'+sim_type+'_obs_ind.json', 'w') as outfile:
            json.dump(self.obs_ind, outfile)

        with open(save_loc+'/'+sim_type+'_seeds.json', 'w') as outfile:
            json.dump(self.seeds, outfile)

        with open(save_loc+'/'+sim_type+'_precision.npy', 'wb') as outfile:
            np.save(outfile, self.precision)
        
        with open(save_loc+'/'+sim_type+'_recall.npy', 'wb') as outfile:
            np.save(outfile, self.recall)
        
        with open(save_loc+'/'+sim_type+'_map50.npy', 'wb') as outfile:
            np.save(outfile, self.map50)
        
        with open(save_loc+'/'+sim_type+'_map50_95.npy', 'wb') as outfile:
            np.save(outfile, self.map50_95)
        
        with open(save_loc+'/'+sim_type+'_fitness.npy', 'wb') as outfile:
            np.save(outfile, self.fitness)
