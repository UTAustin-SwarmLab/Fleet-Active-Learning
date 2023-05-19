#!/home/ou5277/miniconda3/envs/adv_cv/bin/python3
import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy
import optuna
import signal
import os

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *


def run_sim(opt, device, trial=None):

    X_train,X_test,y_train,y_test = load_datasets(opt.dataset_loc,opt.dataset_type)

    n_class = len(np.unique(y_train))

    test_data = create_datasets(X_test,y_test,opt.dataset_type)

    params = dict()
   
    params["n_device"] = opt.n_device
    params["n_sim"] = 1
    params["n_rounds"] = opt.n_rounds
    params["n_epoch"] = opt.n_epoch
    params["b_size"] = opt.b_size
    params["n_iter"] = opt.n_iter
    
    params["n_class"] = n_class
    params["test_b_size"] = opt.test_b_size
    params["lr"] = opt.lr
    params["n_size"] = opt.n_size
    params["n_obs"] = opt.n_obs
    params["n_cache"] = opt.n_cache
    params["unc_type"] = opt.unc_type
    params["dataset_type"] = opt.dataset_type
    params["converge"] = opt.converge_train
    params["cache_all"] = opt.cache_all
    params["dirichlet_alpha"] = opt.dirichlet_alpha
    params["dirichlet"] = opt.dirichlet
    params["dirichlet_base"] = opt.dirichlet_base
    params["dirichlet_base_alpha"] = opt.dirichlet_base_alpha


    base_classes =  [i for i in range(params["n_class"])]
    sim_bar = tqdm(range(opt.init_sim, opt.init_sim+opt.n_sim),total=opt.n_sim)

    distributed_all_accs = np.zeros((opt.n_sim,opt.n_rounds+1,opt.n_trial))
    oracle_all_accs = np.zeros((opt.n_sim,opt.n_rounds+1,opt.n_trial))
    interactive_all_accs = np.zeros((opt.n_sim,opt.n_rounds+1,opt.n_trial))

    optuna_trial_path = opt.experiment_loc + f"/optuna-trial{trial.number}"
    os.makedirs(optuna_trial_path, exist_ok=True)

    for sim_i in sim_bar:
        sim_path = optuna_trial_path + f"/sim{sim_i}"
        os.makedirs(sim_path, exist_ok=True)

        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        simcoef_int = np.random.randint(low=1,high=100)
        simsum_int = np.random.randint(low=1,high=100)

        model = get_model(opt.dataset_type,opt.dataset_type,device,opt.b_size,opt.n_epoch,opt.lr,n_class)

        Unc_Model = Sim(params,"Base",device,model)
        
        Unc_Model.create_base_inds(y_train,base_classes,sim_i,sim_i)

        base_inds = Unc_Model.dataset_ind[sim_i]

        initial_dataset = Unc_Model.create_traindataset(X_train[tuple(Unc_Model.dataset_ind[sim_i])],y_train[tuple(Unc_Model.dataset_ind[sim_i])])

        train_model(Unc_Model.model,initial_dataset,converge=opt.converge_train)

        accs = test_model(Unc_Model.model,test_data,opt.test_b_size)

        print("Test accuracy: ",accs)

        save_model(Unc_Model.model, sim_path, "init_model" + str(sim_i) + ".pt")

        obs_classes = [base_classes for i in range(params["n_device"])]

        for trial_i in range(opt.init_trial,opt.init_trial+opt.n_trial):

            random.seed(trial_i*simcoef_int+simsum_int)
            torch.manual_seed(trial_i*simcoef_int+simsum_int)
            np.random.seed(trial_i*simcoef_int+simsum_int)

            trial_path = sim_path + f"/trial{trial_i}"
            os.makedirs(trial_path, exist_ok=True)

            Distributed_Model = Sim(params,"Distributed",device,copy.deepcopy(Unc_Model.model))
            Oracle_Model = Sim(params,"Oracle",device,copy.deepcopy(Unc_Model.model))
            Interactive_Model = Sim(params,"Interactive",device,copy.deepcopy(Unc_Model.model))

            x_dist, N_x = Distributed_Model.create_xdist(trial_i*simcoef_int+simsum_int,obs_classes,y_train)
                
            obs_inds = Distributed_Model.create_obs_ind(N_x,y_train,trial_i*simcoef_int+simsum_int)

            Distributed_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,test_data,base_inds,obs_inds)
            Oracle_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,test_data,base_inds,obs_inds)
            Interactive_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,test_data,base_inds,obs_inds)
            
            # Distributed_Model.save_infos(trial_path, "Distributed")
            # Oracle_Model.save_infos(trial_path, "Oracle")
            # Interactive_Model.save_infos(trial_path, "Interactive")
            Distributed_Model.save_params(trial_path, "Distributed")
            Oracle_Model.save_params(trial_path, "Oracle")
            Interactive_Model.save_params(trial_path, "Interactive")
        
            plot_accs([Distributed_Model.accs, Oracle_Model.accs, Interactive_Model.accs], ["Distributed", "Oracle", "Interactive"], trial_path + "/Accs.jpg")
            
            distributed_all_accs[sim_i-opt.init_sim,:,trial_i-opt.init_trial] = Distributed_Model.accs[0,:]
            oracle_all_accs[sim_i-opt.init_sim,:,trial_i-opt.init_trial] = Oracle_Model.accs[0,:]
            interactive_all_accs[sim_i-opt.init_sim,:,trial_i-opt.init_trial] = Interactive_Model.accs[0,:]

        run_ids = [i for i in range(opt.n_trial)]

        # combine_sims(run_ids, sim_path, sim_path, ["Distributed", "Oracle", "Interactive"], name="trial")

    
    return np.mean([-(interactive_all_accs-distributed_all_accs)+(interactive_all_accs-oracle_all_accs)**2])
        

trialed_params = []
def objective(trial):
    opt = copy.deepcopy(cli_opt)
    for key, value in vars(cli_opt).items():
        if type(value) == tuple:
            if "." in str(value):
                suggestion = trial.suggest_float(key, value[0], value[1])
            else:
                suggestion = trial.suggest_int(key, value[0], value[1])
            setattr(opt, key, suggestion)
    
    if trial.params in trialed_params:
        raise optuna.TrialPruned()
    else:
        trialed_params.append(trial.params)
    
    return run_sim(opt, device, trial)

def hparam(value):
    if "," in value:
        if "." in value:
            return tuple(map(float, value.split(",")))
        else:
            return tuple(map(int, value.split(",")))
    elif "." in value:
        return float(value)
    else:
        return int(value)

def pdb_handler(signal, frame):
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGUSR1, pdb_handler)
        print(f"Process ID: {os.getpid()}\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--process-id", type=int, default=0)
        parser.add_argument("--n-opt-trial", type=int, default=1)
        parser.add_argument("--coop", type=bool, default=True)
        parser.add_argument("--dataset-loc", type=str, default="./")
        parser.add_argument("--gpu-no", type=int,default=0)
        parser.add_argument("--n-device", type=hparam, default=20)
        parser.add_argument("--n-sim", type=hparam, default=3)
        parser.add_argument("--n-rounds", type=hparam, default=5)
        parser.add_argument("--n-epoch", type=hparam, default=200)
        parser.add_argument("--b-size", type=hparam, default=1000)
        parser.add_argument("--init-sim", type=hparam, default=0)
        parser.add_argument("--n-iter", type=hparam, default=3)
        parser.add_argument("--n-class", type=hparam, default=10)
        parser.add_argument("--test-b-size", type=hparam, default=1000)
        parser.add_argument("--lr", type=hparam, default=0.001)
        parser.add_argument("--n-size", type=hparam, default=30)
        parser.add_argument("--n-obs", type=hparam, default=1000)
        parser.add_argument("--n-cache", type=hparam, default=1)
        parser.add_argument("--run-loc", type=str, default="./old-runs/MNIST")
        parser.add_argument("--plot-loc", type=str, default="./plots/MNIST")
        parser.add_argument("--n-trial",type=hparam, default=3)
        parser.add_argument("--init-trial",type=hparam, default=0)
        parser.add_argument("--unc-type",type=str, default="badge")
        parser.add_argument("--dataset-type",type=str, default="MNIST")
        parser.add_argument("--converge-train",type=bool, default=True)
        parser.add_argument("--cache-all",type=bool, default=False)
        parser.add_argument("--dirichlet",type=bool, default=True)
        parser.add_argument("--dirichlet-base",type=bool, default=True)
        parser.add_argument("--dirichlet-alpha",type=hparam, default=1)
        parser.add_argument("--dirichlet-base-alpha",type=hparam, default=5)
        cli_opt = parser.parse_args()
        
        device = torch.device("cuda:"+str(cli_opt.gpu_no) if (torch.cuda.is_available()) else "cpu")
        print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(cli_opt.gpu_no) if torch.cuda.is_available() else 'CPU'))

        if cli_opt.process_id == 0:
            cli_opt.experiment_loc = create_run_dir(cli_opt.plot_loc, name="experiment")
        else:
            experiments = os.listdir(cli_opt.plot_loc)
            experiment_i = 0
            for i in range(len(experiments)):
                if f"experiment{experiment_i}" in experiments:
                    experiment_i += 1
                else:
                    break
            cli_opt.experiment_loc = f"{cli_opt.plot_loc}/experiment{experiment_i - 1}"

        if cli_opt.coop:
            study = optuna.create_study(study_name="sim", storage="mysql://onat:SwarmLab2022@localhost/optuna", load_if_exists=True)
        else:
            study = optuna.create_study()
        
        study.optimize(objective, n_trials=cli_opt.n_opt_trial)

        with open(f"{cli_opt.experiment_loc}/results/results{cli_opt.process_id}.log", "w") as results_log:
            for trial in sorted(filter(lambda trial: trial.state == optuna.trial.TrialState.COMPLETE, study.trials), key=lambda trial: trial.value):
                results_log.write(f"Trial number: {trial.number}\n")
                results_log.write(f"Trial params: {trial.params}\n")
                results_log.write(f"Trial value: {trial.value}\n")
                results_log.write("----------------------------------------------------------------\n")
                results_log.write("\n")

    except Exception as exception:
        with open(f"logs/err/err{cli_opt.process_id}.log", "w") as err_log:
            err_log.write(str(exception) + "\n")

        e = exception
        import pdb
        pdb.set_trace()
