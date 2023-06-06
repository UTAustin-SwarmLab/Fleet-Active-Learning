import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *


def plot_tsne(train_embs,method_embs,save_loc):

    fig, ax = plt.subplots(figsize=(8,7),dpi=600)
    # plot the distribution of the data points with gray color

    sns.color_palette("tab10")

    data = np.concatenate((train_embs,method_embs[0],method_embs[1],method_embs[2]),axis=0)
    data = pd.DataFrame(data,columns=["x","y"])
    data["Cluster"] = ["Training Data"]*len(train_embs)+["Distributed"]*len(method_embs[0])+["Centralized"]*len(method_embs[1])+["Interactive"]*len(method_embs[2])

    #sns.scatterplot(x="x",y="y",data=data,hue="Cluster",palette=["gray","tab:blue","tab:orange","tab:green"],alpha=0.5,legend=False,ax=ax)
    
    plt.scatter(train_embs[:,0],train_embs[:,1],color="tab:gray",label="Training Data",alpha=0.1)

    plt.scatter(method_embs[0][:,0],method_embs[0][:,1],marker="x",label="Distributed",linewidths=3,alpha=1,color="tab:blue",s=200)
    plt.scatter(method_embs[1][:,0],method_embs[1][:,1],marker="+",label="Centralized",alpha=1,linewidths=3,color="tab:orange",s=200)
    plt.scatter(method_embs[2][:,0],method_embs[2][:,1],marker="o",facecolors='none',alpha=1,label="Interactive",linewidths=3,color="tab:green",s=200)
    

    plt.rcParams["font.size"]=15
    plt.rcParams["axes.linewidth"]=2
    plt.rcParams["legend.labelspacing"] = 0.4

    handles = [plt.scatter([],[],marker="o",color="tab:gray",label="Training Data",s=150), 
           plt.scatter([],[],marker="x",color="tab:blue",label="Distributed",s=200,linewidths=3), 
           plt.scatter([],[],marker="+",color="tab:orange",label="Centralized",s=200,linewidths=3),
           plt.scatter([],[],marker="o",facecolors='none',color="tab:green",s=200,label="Interactive",linewidths=3)]
    #legend = plt.legend(handles=handles,frameon=True,loc='lower left',bbox_to_anchor=(-0.1, 1.1))
    legend = plt.legend(handles=handles,frameon=True)

    for text in legend.get_texts():
        text.set_weight('bold')

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(save_loc)
    

dataset_loc = "/store/datasets/CIFAR10"
dataset_type = "CIFAR10"
img_loc = "/store/datasets/CIFAR10"
clip_emb_loc = "/store/datasets/CIFAR10"
run_loc = "./runs/CIFAR10/run10"

values_saved = False

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

sim_types = ["Distributed","Oracle","Interactive"]

#X_train,X_test,y_train,y_test = load_datasets(dataset_loc,dataset_type,img_loc=img_loc)

if dataset_type == "AdversarialWeather":
    embs = np.load(clip_emb_loc+"/clip_embs.npy",allow_pickle=True).item()
    train_embs = {"/".join(X_train[i].split("/")[-4:]): embs["/".join(X_train[i].split("/")[-4:])]
                        for i in  range(len(X_train))}
else:
    train_embs = np.load(clip_emb_loc+"/train_embs.npy",allow_pickle=True).item()


dataset_inds = []


for sim_type in sim_types:

    with open(run_loc+"/"+sim_type+"_dataset_ind.json") as f:
        dataset_ind = json.load(f)
    
    dataset_inds.append(dataset_ind)

sim_keys = list(dataset_inds[0].keys())

all_inds = [i for i in range(len(X_train))]
if os.path.exists(run_loc+"/embs_tsne.npy"):
    all_tsne = np.load(run_loc+"/embs_tsne.npy")
else:
    all_embs = clip_obtain_embeddings(X_train,all_inds,train_embs,dataset_type)
    all_tsne = TSNE(n_components=2,n_jobs=-1,init="pca",learning_rate="auto",verbose=1).fit_transform(all_embs)
    # Save the TSNE Embeddings

    np.save(run_loc+"/embs_tsne.npy",all_tsne)


dataset_tsnes = []

for i in range(len(sim_types)):
    dataset_ind = dataset_inds[i][sim_keys[0]][-1].copy()
    dataset_tsne = all_tsne[dataset_ind]
    dataset_tsnes.append(dataset_tsne)

plot_tsne(all_tsne,dataset_tsnes,run_loc+"/tsne.jpg")

"""
n_centers = 3
n_samples = 2000
n_iter = 3
n_cache = 3
k_factor = 5
# data set generate
X = np.random.rand(n_centers*n_samples,2)

mu_1 = np.array([0,-1])
mu_2 = np.array([np.sqrt(3)/2,1/2])
mu_3 = np.array([-np.sqrt(3)/2,1/2])

r_1 = np.array([[0.5,0],[0,1]])
r_2 = np.array([[1,1/np.sqrt(3)],[1/np.sqrt(3),1]])
r_3 = np.array([[1,-1/np.sqrt(3)],[-1/np.sqrt(3),1]])

X = np.zeros((n_centers*n_samples*k_factor,2))

X[:n_samples*k_factor] = np.tile(np.random.multivariate_normal(mu_1,r_1,n_samples),(k_factor,1))
X[n_samples*k_factor:2*n_samples*k_factor] = np.tile(np.random.multivariate_normal(mu_2,r_2,n_samples),(k_factor,1))
X[2*n_samples*k_factor:] = np.tile(np.random.multivariate_normal(mu_3,r_3,n_samples),(k_factor,1))


# Generate the random samples.
y = np.arange(n_centers*k_factor).repeat(n_samples)
#X, y = make_blobs(n_samples = n_centers*n_samples, n_features = 2, centers = n_centers, random_state = 2,cluster_std=4)


X[:,0] = (X[:,0]-np.min(X[:,0]))/(np.max(X[:,0])-np.min(X[:,0]))
X[:,1] = (X[:,1]-np.min(X[:,1]))/(np.max(X[:,1])-np.min(X[:,1]))

base_embeddings = np.zeros((1,2))+0.5
embeddings = np.zeros((n_centers,n_samples,2))
obs_inds = []


for i in range(n_centers):
    embeddings[i] = X[y==i]
    obs_inds.append(np.where(y==i)[0].tolist())

sampling_policy = kCenterGreedy(embeddings,base_embeddings,obs_inds,n_iter,n_cache)

dist_inds = sampling_policy.sample_caches("Distributed")
centr_inds = sampling_policy.sample_caches("Oracle")

X_dist = np.concatenate((X[dist_inds],base_embeddings),axis=0)

X_centr = np.concatenate((X[centr_inds],base_embeddings),axis=0)

# Now we plot the selected points and the original dataset
# The original dataset is colored by a cluster and the selected points are colored by a different color

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(7,7),dpi=600)
# plot the distribution of the data points with gray color
sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,palette="tab10",legend=False,ax=ax,alpha=0.2)

plt.scatter(X_dist[:,0],X_dist[:,1],marker="x",s=100,color="r",label="Distributed",linewidths=3)
plt.scatter(X_centr[:,0],X_centr[:,1],marker="+",s=100,color="b",label="Oracle",linewidths=3)

# scatter plot of blobs
plt.savefig("blobs.png")
"""

"""    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1,1),columns=["Accuracy"])
        data["Round"] = [i for i in range(accs[i].shape[1])]* accs[i].shape[0]
        sns.lineplot(data=data,x="Round",y="Accuracy",label=names[i],linewidth=3)

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $r$",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)

    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20,weight='bold'))
    plt.tight_layout()
    #plt.legend([],[], frameon=False)
    plt.savefig(save_loc)


# Creates a toy example
def make_toy_example(save_loc):

    
    

    plt.plot(N_cache[:,0],N_cache[:,1],color="k",linewidth=3,linestyle='--',alpha=0.4,label='_nolegend_')
    d_tar =  plt.scatter(D_target[:,0],D_target[:,1],marker="X",s=200,color="b",label="Target Distribution")
    plt.scatter(D_target[:,0],D_target[:,1],marker="X",s=500,color="b",label="D_target")
    plt.plot(D_target[:,0],D_target[:,1],linewidth=3,color="k",linestyle='--',alpha=0.4,label='_nolegend_')

    act_spc, = plt.plot(action_space_1[:,0],action_space_1[:,1],color="orange",linewidth=4,linestyle='-',marker="o",alpha=0.7,label="Robot 1 Action Space")
    act_spc1 = plt.Polygon(np.concatenate((action_space_1,D_0.reshape(-1,2))),alpha=0.4,color="orange",label='_nolegend_')
    plt.gca().add_patch(act_spc1)
    act_spc22, = plt.plot(action_space_2[:,0],action_space_2[:,1],color="purple",linewidth=4,linestyle='-',marker="o",alpha=0.7,label="Robot 2 Action Space")
    act_spc2 = plt.Polygon(np.concatenate((action_space_2,D_0.reshape(-1,2))),alpha=0.4,color="purple",label='_nolegend_')
    plt.gca().add_patch(act_spc2)

    plt.plot(action_space_2[:,0],action_space_2[:,1],color="purple",linewidth=4,linestyle='-',marker="o",alpha=0.7,label='_nolegend_')

    gred_acc = plt.arrow(D_0[0],D_0[1],greedy_action_1[1,0],greedy_action_1[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label="Greedy Action")
    plt.arrow(D_0[0],D_0[1],greedy_action_2[1,0],greedy_action_2[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label='_nolegend_')
    plt.arrow(greedy_action_1[1,0],greedy_action_1[1,1],greedy_action_2[1,0],greedy_action_2[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label='_nolegend_')

    orac_acc = plt.arrow(D_0[0],D_0[1],oracle_action_1[1,0],oracle_action_1[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label="Oracle Action")
    plt.arrow(D_0[0],D_0[1],oracle_action_2[1,0],oracle_action_2[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label='_nolegend_')
    plt.arrow(oracle_action_2[1,0],oracle_action_2[1,1],oracle_action_1[1,0],oracle_action_1[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label='_nolegend_')

    error_acc, = plt.plot((total_greedy_action[1,0],total_oracle_action[1,0]),(total_greedy_action[1,1],total_oracle_action[1,1]),color="r",linewidth=3,linestyle='-.',marker="X",label="Error")

    plt.scatter((total_greedy_action[1,0]),(total_greedy_action[1,1]),marker="X",s=500,color="r",label='_nolegend_')
    plt.scatter((total_oracle_action[1,0]),(total_oracle_action[1,1]),marker="X",s=500,color="g",label='_nolegend_')


    plt.ylim(np.min(D_target), np.max(D_target)*1.1)
    plt.xlim(np.min(D_target), np.max(D_target)*1.1)

    plt.ylabel("Num. Images Class 2",fontweight="bold" ,fontsize=26)
    plt.xlabel("Num. Images Class 1",fontweight="bold" ,fontsize=26)

    plt.rcParams["font.size"]=15
    plt.rcParams["axes.linewidth"]=2
    plt.rcParams["legend.labelspacing"] = 0.5
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(handles=[d_tar, act_spc,act_spc22,gred_acc,orac_acc,error_acc], loc='upper left' )
    plt.tight_layout()
    plt.savefig(save_loc)
""" 