from utils.plotting_utils import *
import numpy as np


accs = np.load("runs/Stat_ML/run19/Accs.npy")
names = ["Random","Entropy","Least Confidence", "BvSB","BADGE","Coreset"]
n_cache = 20
n_size = 20
n_rounds = 10
n_samples = [i*n_cache + n_size for i in range(n_rounds+1)]
plot_accs_wrt_n_samples(accs,names,n_samples,"./Accs.jpg")
make_legend(names,"./Accs_legend.jpg")
