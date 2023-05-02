import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *


def run_sim(opt,device):

    X_train,X_test,y_train,y_test = load_CIFAR10_dataset(opt.dataset_loc)

    test_data = create_CIFAR10_datasets(X_test,y_test)

    params = dict()
   
    params["n_device"] = opt.n_device
    params["n_sim"] = 1
    params["n_rounds"] = opt.n_rounds
    params["n_epoch"] = opt.n_epoch
    params["b_size"] = opt.b_size
    params["n_iter"] = opt.n_iter
    
    params["n_class"] = opt.n_class
    params["test_b_size"] = opt.test_b_size
    params["lr"] = opt.lr
    params["n_size"] = opt.n_size
    params["n_obs"] = opt.n_obs
    params["n_cache"] = opt.n_cache
    params["unc_type"] = opt.unc_type


    base_classes =  [i for i in range(params["n_class"])]
    sim_bar = tqdm(range(opt.init_sim, opt.init_sim+opt.n_sim),total=opt.n_sim)

    for sim_i in sim_bar:

        run_i_loc = create_run_dir(opt.run_loc)

        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        simcoef_int = np.random.randint(low=1,high=100)
        simsum_int = np.random.randint(low=1,high=100)

        model = get_model("CIFAR10Classifier","CIFAR10",device,opt.b_size,opt.n_epoch,opt.lr)

        Unc_Model = CIFAR10_Sim(params,device,model)
        
        Unc_Model.create_base_inds(y_train,base_classes,sim_i,sim_i)

        base_inds = Unc_Model.dataset_ind[sim_i]

        initial_dataset = Unc_Model.create_traindataset(X_train[Unc_Model.dataset_ind[sim_i]],y_train[Unc_Model.dataset_ind[sim_i]])

        train_model(Unc_Model.model,initial_dataset)

        accs = test_model(Unc_Model.model,test_data)

        print("Test accuracy: ",accs)

        save_model(Unc_Model.model,run_i_loc,"init_model"+str(sim_i)+".pt")

        obs_classes = [base_classes for i in range(params["n_device"])]

        for trial_i in range(opt.init_trial,opt.init_trial+opt.n_trial):

            random.seed(trial_i*simcoef_int+simsum_int)
            torch.manual_seed(trial_i*simcoef_int+simsum_int)
            np.random.seed(trial_i*simcoef_int+simsum_int)

            trial_loc = create_run_dir(run_i_loc,"trial")

            Entropy_Model = CIFAR10_Sim(params,device,copy.deepcopy(Unc_Model.model))

            x_dist, N_x = Entropy_Model.create_xdist(trial_i*simcoef_int+simsum_int,obs_classes,y_train)
                
            obs_inds = Entropy_Model.create_obs_ind(N_x,y_train,trial_i*simcoef_int+simsum_int)

            Entropy_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,test_data,base_inds,obs_inds)

            Entropy_Model.save_infos(trial_loc,"Entropy")
        
            plot_accs([Entropy_Model.accs],["Entropy"],trial_loc+"/Accs.jpg")

        run_ids = [i for i in range(opt.n_trial)]

        combine_sims(run_ids,run_i_loc,run_i_loc,["Entropy"],name="trial")


if __name__ == "__main__":

    # Runs the simulation for the MNIST dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-loc", type=str,default="/store/datasets")
    parser.add_argument("--gpu-no", type=int,default=2)
    parser.add_argument("--n-device", type=int, default=3)
    parser.add_argument("--n-sim", type=int, default=2)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-epoch", type=int, default=200)
    parser.add_argument("--b-size", type=int, default=1000)
    parser.add_argument("--init-sim", type=int, default=0) 
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--n-class", type=int, default=10)
    parser.add_argument("--test-b-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-size", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-cache", type=int, default=50)
    parser.add_argument("--run-loc", type=str, default="./runs/CIFAR10")
    parser.add_argument("--n-trial",type=int, default=4)
    parser.add_argument("--init-trial",type=int, default=0)
    parser.add_argument("--unc-type",type=str, default="Entropy")
    
    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.gpu_no) if (torch.cuda.is_available()) else "cpu")
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.gpu_no) if torch.cuda.is_available() else 'CPU'))

    run_sim(opt,device)