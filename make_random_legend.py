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

fig = plt.figure(constrained_layout=False)
fig.set_size_inches(18.5, 10.5)
gs = fig.add_gridspec(1, 1)

acc = np.ones((7,2))

all_data_df = pd.DataFrame(acc,columns=["Iterations","Real Grad Norm"])
all_data_df["method"] = ["Distributed","Centralized","Interactive","Random","Entropy","Least Confidence","BvSB"]

labels = ["Distributed","Centralized","Interactive","Random","Entropy","Least Confidence","BvSB"]
ax00 = fig.add_subplot(gs[0])
markers =["P","o","s","D","^","v","X"]
linestyles=["--","-","-.",":","--","-","-."]
for i in range(7):
    sns.lineplot(x=[1],y=[2],marker=markers[i],markersize=9,linestyle=linestyles[i],ax=ax00,linewidth=3,label=labels[i])
#sns_plot = sns.lineplot(x="Iterations", y="Real Grad Norm", data=all_data_df, hue="method", style="method", markers=["P","o","s","D","^","v","X"],linestyles=["--","-","-.",":","--","-","-."], ax=ax00,linewidth=10)
handles, labels = ax00.get_legend_handles_labels()

figLegend = pylab.figure(figsize=(12, 0.3))
# print(ax00.get_legend_handles_labels())
pylab.figlegend(
     *ax00.get_legend_handles_labels(),
     loc="upper left",
     mode="expand",
     ncol=7,
     prop={'weight':'bold',"size":12},
     borderaxespad=0,
     frameon=False,
 )
figLegend.savefig(os.path.join("./", f"randomlengend.png"), dpi=600)