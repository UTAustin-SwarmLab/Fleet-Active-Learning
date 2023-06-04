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

    X_train,X_test,y_train,y_test = load_datasets(opt.dataset_loc,opt.dataset_type,img_loc=opt.img_loc)

    if opt.unc_type == "clip":
        if opt.dataset_type == "AdversarialWeather":
            embs = np.load(opt.clip_emb_loc+"/clip_embs.npy",allow_pickle=True).item()
            train_embs = {"/".join(X_train[i].split("/")[-4:]): embs["/".join(X_train[i].split("/")[-4:])]
                           for i in  range(len(X_train))}
        else:
            train_embs = np.load(opt.clip_emb_loc+"/train_embs.npy",allow_pickle=True).item()
    else:
        train_embs = None
    
    if opt.use_embeddings:
        n_features = X_train.shape[1]
    else:
        n_features = 1000

    n_class = 10

    params = dict()

    params["n_device"] = opt.n_unique_device * opt.n_same_device
    params["n_unique_device"] = opt.n_unique_device
    params["n_same_device"] = opt.n_same_device
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
    params["cache_in_first"] = opt.cache_in_first
    params["train_only_final"] = opt.train_only_final
    params["use_embeddings"] = opt.use_embeddings
    params["n_features"] = n_features
    params["accuracy_type"] = opt.normalized_accuracy
    params["center_selection"] = opt.center_selection
    params["n_workers"] = opt.n_workers

    base_classes =  [i for i in range(params["n_class"])]
    pbar = tqdm(total=opt.n_sim*opt.n_trial)

    for sim_i in range(opt.init_sim, opt.init_sim+opt.n_sim):

        run_i_loc = create_run_dir(opt.run_loc)

        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        simcoef_int = np.random.randint(low=1,high=100)
        simsum_int = np.random.randint(low=1,high=100)

        model = get_model(opt.dataset_type,opt.dataset_type,device,opt.b_size,opt.n_epoch,opt.lr,n_class)

        Unc_Model = Sim_Detect(params,"Base",device,model,0)
        
        Unc_Model.create_base_inds(y_train,base_classes,sim_i,sim_i)

        base_inds = Unc_Model.dataset_ind[sim_i]

        initial_dataset = create_detection_dataset(X_train[tuple(Unc_Model.dataset_ind[sim_i])],X_test)

        pbar.set_description("Training initial model")
        train_model(Unc_Model.model,initial_dataset,converge=opt.converge_train,only_final=opt.train_only_final,silent=False,params=params,detection=True,device=device)

        results = test_model(Unc_Model.model,X_test,opt.test_b_size,detection=True)

        print("Initial mAP50: ",results["metrics/mAP50(B)"])

        obs_classes = [base_classes for i in range(params["n_device"])]

        for trial_i in range(opt.init_trial,opt.init_trial+opt.n_trial):

            random.seed(trial_i*simcoef_int+simsum_int)
            torch.manual_seed(trial_i*simcoef_int+simsum_int)
            np.random.seed(trial_i*simcoef_int+simsum_int)

            trial_loc = create_run_dir(run_i_loc,"trial")

            Distributed_Model = Sim_Detect(params,"Distributed-New",device,Unc_Model.model,results)
            Oracle_Model = Sim_Detect(params,"Oracle-New",device,Unc_Model.model,results)
            Interactive_Model = Sim_Detect(params,"Interactive-New",device,Unc_Model.model,results)

            x_dist, N_x = Distributed_Model.create_xdist(trial_i*simcoef_int+simsum_int,obs_classes,y_train)
                
            obs_inds = Distributed_Model.create_obs_ind(N_x,y_train,trial_i*simcoef_int+simsum_int)

            pbar.set_description("Running Distributed")
            Distributed_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,X_test,base_inds,obs_inds,train_embs)
            pbar.set_description("Running Oracle")
            Oracle_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,X_test,base_inds,obs_inds,train_embs)
            pbar.set_description("Running Interactive")
            Interactive_Model.sim_round(0,trial_i*simcoef_int+simsum_int,X_train,y_train,X_test,base_inds,obs_inds,train_embs)
            
            pbar.set_description("Saving results")
            Distributed_Model.save_infos(trial_loc,"Distributed")
            Oracle_Model.save_infos(trial_loc,"Oracle")
            Interactive_Model.save_infos(trial_loc,"Interactive")

            plot_values([Distributed_Model.precision,Oracle_Model.precision,Interactive_Model.precision],["Distributed","Oracle","Interactive"],trial_loc+"/Precision.jpg","Precision")
            plot_values([Distributed_Model.recall,Oracle_Model.recall,Interactive_Model.recall],["Distributed","Oracle","Interactive"],trial_loc+"/Recall.jpg","Recall")
            plot_values([Distributed_Model.map50,Oracle_Model.map50,Interactive_Model.map50],["Distributed","Oracle","Interactive"],trial_loc+"/mAP50.jpg","mAP50")
            plot_values([Distributed_Model.map50_95,Oracle_Model.map50_95,Interactive_Model.map50_95],["Distributed","Oracle","Interactive"],trial_loc+"/mAP50-95.jpg","mAP50-95")
            plot_values([Distributed_Model.fitness,Oracle_Model.fitness,Interactive_Model.fitness],["Distributed","Oracle","Interactive"],trial_loc+"/Fitness.jpg","Fitness")
            
            precision = [Distributed_Model.precision[:,0],Distributed_Model.precision[:,1],Oracle_Model.precision[:,1],Interactive_Model.precision[:,1]]
            recall = [Distributed_Model.recall[:,0],Distributed_Model.recall[:,1],Oracle_Model.recall[:,1],Interactive_Model.recall[:,1]]
            map50 = [Distributed_Model.map50[:,0],Distributed_Model.map50[:,1],Oracle_Model.map50[:,1],Interactive_Model.map50[:,1]]
            map50_95 = [Distributed_Model.map50_95[:,0],Distributed_Model.map50_95[:,1],Oracle_Model.map50_95[:,1],Interactive_Model.map50_95[:,1]]
            fitness = [Distributed_Model.fitness[:,0],Distributed_Model.fitness[:,1],Oracle_Model.fitness[:,1],Interactive_Model.fitness[:,1]]

            names = ["Initial","Distributed","Oracle","Interactive"]

            plot_boxplot_values(precision,names,trial_loc+"/Precision_box.jpg","Precision")
            plot_boxplot_values(recall,names,trial_loc+"/Recall_box.jpg","Recall")
            plot_boxplot_values(map50,names,trial_loc+"/mAP50_box.jpg","mAP50")
            plot_boxplot_values(map50_95,names,trial_loc+"/mAP50-95_box.jpg","mAP50-95")
            plot_boxplot_values(fitness,names,trial_loc+"/Fitness_box.jpg","Fitness")

            pbar.update(1)

        pbar.set_description("Combining results")
        run_ids = [i for i in range(opt.n_trial)]

        combine_det_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-loc", type=str,default="/store/datasets/bdd100k/labels/det_20")
    parser.add_argument("--img-loc", type=str,default="/store/datasets/bdd100k/images/100k")
    parser.add_argument("--clip-emb-loc", type=str, default= "/store/datasets/bdd100k/clip_embeddings")
    parser.add_argument("--emb-loc", type=str, default= "/store/datasets/CIFAR10/features/resnet50")
    parser.add_argument("--gpu-no", type=int,default=0)
    parser.add_argument("--n-unique-device", type=int, default=2)
    parser.add_argument("--n-same-device", type=int, default=2)
    parser.add_argument("--n-sim", type=int, default=1)
    parser.add_argument("--n-rounds", type=int, default=1)
    parser.add_argument("--n-epoch", type=int, default=20)
    parser.add_argument("--b-size", type=int, default=64)
    parser.add_argument("--init-sim", type=int, default=0) 
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--n-class", type=int, default=10)
    parser.add_argument("--test-b-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-size", type=int, default=20)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--n-cache", type=int, default=1)
    parser.add_argument("--run-loc", type=str, default="./runs/DeepDrive-Detection")
    parser.add_argument("--n-trial",type=int, default=2)
    parser.add_argument("--init-trial",type=int, default=0)
    parser.add_argument("--unc-type",type=str, default="clip")
    parser.add_argument("--dataset-type",type=str, default="DeepDrive-Detection")
    parser.add_argument("--use-embeddings",type=int, default=0)
    parser.add_argument("--converge-train",type=int, default=0)
    parser.add_argument("--cache-all",type=int, default=0)
    parser.add_argument("--dirichlet",type=int, default=1)
    parser.add_argument("--dirichlet-base",type=int, default=1)
    parser.add_argument("--dirichlet-alpha",type=float, default=1)
    parser.add_argument("--dirichlet-base-alpha",type=float, default=5)
    parser.add_argument("--cache-in-first",type=int, default=1)
    parser.add_argument("--train-only-final",type=int, default=1)
    parser.add_argument("--normalized-accuracy",type=int, default=0)
    parser.add_argument("--center-selection",type=str, default="facility")
    parser.add_argument("--n-workers", type=int, default=1)

    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.gpu_no) if (torch.cuda.is_available()) else "cpu")

    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.gpu_no) if torch.cuda.is_available() else 'CPU'))

    run_sim(opt,device)