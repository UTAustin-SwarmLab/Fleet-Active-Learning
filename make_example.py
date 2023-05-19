
from utils.coresets import *
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def generate_random_points_within_rotated_ellipse(a, b, center, num_points,rotation_angle=0):
    points = []
    while len(points) < num_points:
        # Generate random point within a unit circle
        theta = np.random.uniform(0, 2*np.pi)
        r = np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale the random point to fit within the ellipse
        x *= a
        y *= b
        
        # Rotate the point
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        point = np.dot(rotation_matrix, [x, y])
        points.append(point+center)
    return points



n_iter = 3
n_cache = 4
n_samples = 1000


#centers = [np.array([3,3]),np.array([1,0]),np.array([0,2])]
centers = [np.array([0,0.5]),np.array([0.5,0]),np.array([0,-0.5]),np.array([-0.5,0])]
a_s = [0.5,0.5,0.5,0.5]
b_s = [0.5,0.5,0.5,0.5]
n_repeat = 1
n_device = len(centers)
n_device_show = 4
rotation_angles = [np.pi/6,np.pi/2,-np.pi/6,-np.pi/2]

X = np.zeros((0,2))
y = np.zeros((0))
for j in range(n_repeat):
    for i in range(len(centers)):
        X = np.concatenate((X,generate_random_points_within_rotated_ellipse(a_s[i], b_s[i],centers[i], n_samples,rotation_angles[i])),axis=0)
        y = np.concatenate((y,np.ones((n_samples))*i+j*len(centers)),axis=0)

X_b = X.copy()
y_b = y.copy()

base_embeddings = np.zeros((1,2))
embeddings = np.zeros((n_device*n_repeat,n_samples,2))
obs_inds = []


for i in range(n_device*n_repeat):
    embeddings[i] = X[y==i]
    obs_inds.append(np.where(y==i)[0].tolist())

sampling_policy = kCenterGreedy(embeddings,base_embeddings,obs_inds,n_iter,n_cache)

dist_inds = sampling_policy.sample_caches("Distributed")
centr_inds = sampling_policy.sample_caches("Oracle")

X_dist = np.concatenate((X[dist_inds],base_embeddings),axis=0)
X_centr = np.concatenate((X[centr_inds],base_embeddings),axis=0)

# Now we plot the selected points and the original dataset
# The original dataset is colored by a cluster and the selected points are colored by a different color

fig, ax = plt.subplots(figsize=(7,7),dpi=600)
# plot the distribution of the data points with gray color

legends = ["Robot %d"%(i+1) for i in range(n_device_show)]

for i in range(n_device_show):
    plt.scatter(X[obs_inds[i],0],X[obs_inds[i],1],label=legends[i],alpha=0.3)

plt.scatter(X_dist[:,0],X_dist[:,1],marker="x",s=100,color="b",label="Distributed",linewidths=3)
plt.scatter(X_centr[:,0],X_centr[:,1],marker="o",s=100,color="k",facecolors='none',label="Oracle",linewidths=3)

#plt.legend(loc="lower left",fontsize=20)
plt.legend()
plt.ylabel("Feature 2",fontweight="bold" ,fontsize=26)
plt.xlabel("Feature 1",fontweight="bold" ,fontsize=26)

plt.rcParams["font.size"]=15
plt.rcParams["axes.linewidth"]=2
plt.rcParams["legend.labelspacing"] = 0.5
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.tight_layout()


# scatter plot of blobs
plt.savefig("blobs.png")

    

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