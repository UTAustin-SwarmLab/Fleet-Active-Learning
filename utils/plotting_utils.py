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

# Creates L2-Norm Plots for the simulations
def plot_accs(accs: list,names: list,save_loc:str,linestyles=["--","-","-."]) -> None:

    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1,1),columns=["Accuracy"])
        data["Round"] = [i for i in range(accs[i].shape[1])]* accs[i].shape[0]
        sns.lineplot(data=data,x="Round",y="Accuracy",label=names[i],linestyle=linestyles[i],linewidth=3)

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $r$",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)

    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20,weight='bold'))
    plt.tight_layout()
    plt.legend([],[], frameon=False)
    plt.savefig(save_loc)

def plot_accs_wrt_n_samples(accs: list,names: list,n_samples: list,save_loc:str) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param n_samples: Number of samples used for each round
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(14,7),dpi=600)

    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1,1),columns=["Accuracy"])
        data["N. of Samples"] = n_samples* accs[i].shape[0]
        sns.lineplot(data=data,x="N. of Samples",y="Accuracy",label=names[i],linewidth=3)

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Number of Samples",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)

    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #plt.legend(prop=dict(size=16,weight='bold'),bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.legend([],[], frameon=False)
    plt.savefig(save_loc)

def make_legend(names,save_loc):

    
    acc_df = pd.DataFrame(np.concatenate([[0,1] for i in range(len(names))]),columns=["Acc"])

    n_sim = 2
    df_names = []
    for name in names:
        df_names.extend([name]*n_sim)
    acc_df["Policy"] = df_names

    #sns.set_theme(style="darkgrid")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7,7),dpi=600)
    order = names
    gg = sns.boxplot(data=acc_df,x="Policy",y="Acc",order=order)
    boxes = ax.findobj(matplotlib.patches.PathPatch)
    for i in range(len(names)):
        boxes[i].set_label(names[i])

    figLegend = pylab.figure(figsize = (11.5,0.5),dpi=600)

    pylab.figlegend(handles=boxes,loc='upper left', mode='expand', ncol=len(names), fontsize=26, borderaxespad=0, frameon=False,prop={'weight':'bold',"size":26})
    figLegend.savefig(save_loc)

# Creates L2-Norm Plots for the simulations
def plot_values(values: list,names: list,save_loc:str,y_label:str,linestyles=["--","-","-."]) -> None:

    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    for i in range(len(values)):
        data = pd.DataFrame(values[i].reshape(-1,1),columns=[y_label])
        data["Round"] = [i for i in range(values[i].shape[1])]* values[i].shape[0]
        sns.lineplot(data=data,x="Round",y=y_label,label=names[i],linewidth=3,linestyle=linestyles[i])

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $r$",fontweight="bold" ,fontsize=24)
    plt.ylabel(y_label,fontweight="bold" ,fontsize=24)

    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20,weight='bold'))
    plt.tight_layout()
    plt.legend([],[], frameon=False)
    plt.savefig(save_loc)




































    