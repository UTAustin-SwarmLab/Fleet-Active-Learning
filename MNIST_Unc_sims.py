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

    train_data,test_data = load_datasets(opt.dataset_loc,opt.dataset)

    test_data = create_datasets(test_data.data,test_data.targets,opt.dataset)

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
    params["dataset"] = opt.dataset


    base_classes =  [i for i in range(params["n_class"])]
    sim_bar = tqdm(range(opt.init_sim, opt.init_sim+opt.n_sim),total=opt.n_sim)

    for sim_i in sim_bar:

        run_i_loc = create_run_dir(opt.run_loc)

        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        simcoef_int = np.random.randint(low=1,high=100)
        simsum_int = np.random.randint(low=1,high=100)

        model = get_model(opt.dataset,opt.dataset,device,opt.b_size,opt.n_epoch,opt.lr)

        Sim_Model = Sim(params,device,model)
        
        Sim_Model.create_base_inds(train_data.targets,base_classes,sim_i,sim_i)

        base_inds = Sim_Model.dataset_ind[sim_i]

        initial_dataset = Sim_Model.create_traindataset(train_data.data[Sim_Model.dataset_ind[sim_i]],train_data.targets[Sim_Model.dataset_ind[sim_i]])

        train_model(Sim_Model.model,initial_dataset)

        accs = test_model(Sim_Model.model,test_data)

        print("Test accuracy: ",accs)

        save_model(Sim_Model.model,run_i_loc,"init_model"+str(sim_i)+".pt")

        obs_classes = [base_classes for i in range(params["n_device"])]

        for trial_i in range(opt.init_trial,opt.init_trial+opt.n_trial):

            random.seed(trial_i*simcoef_int+simsum_int)
            torch.manual_seed(trial_i*simcoef_int+simsum_int)
            np.random.seed(trial_i*simcoef_int+simsum_int)

            trial_loc = create_run_dir(run_i_loc,"trial")

            Distributed_Model = Sim(params,"Distributed",device,copy.deepcopy(Sim_Model.model))
            Oracle_Model = Sim(params,"Oracle",device,copy.deepcopy(Sim_Model.model))
            Interactive_Model = Sim(params,"Interactive",device,copy.deepcopy(Sim_Model.model))

            x_dist, N_x = Distributed_Model.create_xdist(trial_i*simcoef_int+simsum_int,obs_classes,train_data.targets)
                
            obs_inds = Distributed_Model.create_obs_ind(N_x,train_data.targets,trial_i*simcoef_int+simsum_int)

            Distributed_Model.sim_round(0,trial_i*simcoef_int+simsum_int,train_data,test_data,base_inds,obs_inds)
            Oracle_Model.sim_round(0,trial_i*simcoef_int+simsum_int,train_data,test_data,base_inds,obs_inds)
            Interactive_Model.sim_round(0,trial_i*simcoef_int+simsum_int,train_data,test_data,base_inds,obs_inds)
            
            Distributed_Model.save_infos(trial_loc,"Distributed")
            Oracle_Model.save_infos(trial_loc,"Oracle")
            Interactive_Model.save_infos(trial_loc,"Interactive")
        
            plot_accs([Distributed_Model.accs,Oracle_Model.accs,Interactive_Model.accs],["Distributed","Oracle","Interactive"],trial_loc+"/Accs.jpg")

        run_ids = [i for i in range(opt.n_trial)]

        combine_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")


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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-size", type=int, default=50)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-cache", type=int, default=4)
    parser.add_argument("--run-loc", type=str, default="./runs/MNIST")
    parser.add_argument("--n-trial",type=int, default=4)
    parser.add_argument("--init-trial",type=int, default=0)
    parser.add_argument("--unc-type",type=str, default="Entropy")
    parser.add_argument("--dataset",type=str, default="MNIST")
    
    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.gpu_no) if (torch.cuda.is_available()) else "cpu")
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.gpu_no) if torch.cuda.is_available() else 'CPU'))

    run_sim(opt,device)