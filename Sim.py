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


def run_sim(opt, device):
    # Benchmark Methods to compare our Interactive policy against
    benchmark_methods = [
        "Distributed",
        "Centralized",
        "Interactive",
        "Interactive",
        "Random",
        "Entropy",
        "LeastConfidence",
        "BvSB",
    ]

    # Load dataset and crate train and test splits
    X_train, X_test, y_train, y_test = load_datasets(
        opt.label_loc,
        opt.dataset_type,
        img_loc=opt.img_loc,
        emb_loc=opt.emb_loc,
        use_embs=opt.use_embeddings,
    )

    # If using clip embeddings, load them
    if opt.unc_type == "clip":
        clip_embs = get_clip_embeddings(
            X_train, opt.clip_emb_loc, opt.dataset_type, opt.label_loc, device
        )
    else:
        clip_embs = None

    # Number of classes in the dataset
    n_class = len(np.unique(y_train))

    # Create test dataset which will be used for all simulations
    test_data = create_datasets(
        X_test,
        y_test,
        opt.dataset_type,
        cache_in_first=opt.cache_in_first,
        use_embs=opt.use_embeddings,
    )

    # Number of features in the dataset
    n_features = X_train.shape[1] if opt.use_embeddings else 1000

    # Create dictionary of parameters
    params = vars(opt)

    # Add additional parameters
    params["n_total_device"] = opt.n_environment * opt.n_device_per_environment
    params["n_class"] = n_class
    params["n_features"] = n_features

    # Create a progress bar
    pbar = tqdm(total=opt.n_sim * opt.n_trial)

    # Run simulations
    for sim_i in range(
        opt.init_sim_for_model, opt.init_sim_for_model + opt.n_init_model
    ):
        # Create a directory to save results for each initial model
        run_i_loc = create_run_dir(opt.run_loc)

        # Set random seeds for reproducibility
        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        # Creating a random similarity coefficient and similarity sum to create a random integer for each simulation
        simcoef_int = np.random.randint(low=1, high=100)
        simsum_int = np.random.randint(low=1, high=100)

        # Obtain initial model for the simulation
        model = get_model(
            opt.dataset_type,
            opt.dataset_type,
            device,
            opt.b_size,
            opt.n_epoch,
            opt.lr,
            n_class,
            use_embs=opt.use_embeddings,
            n_features=n_features,
        )

        # Create a Sim object for the initial model
        Unc_Model = Sim(params, "Base", device, model)

        # Create initial training data indices
        Unc_Model.create_base_inds(y_train, sim_i, sim_i)
        base_inds = Unc_Model.dataset_ind[sim_i]

        # Create initial training dataset
        initial_dataset = Unc_Model.create_traindataset(
            X_train[tuple(Unc_Model.dataset_ind[sim_i])],
            y_train[tuple(Unc_Model.dataset_ind[sim_i])],
        )

        # Train initial model
        pbar.set_description("Training initial model")
        train_model(
            Unc_Model.model,
            initial_dataset,
            converge=opt.converge_train,
            only_final=opt.train_only_final,
        )

        # Obtain accuracy of the initial model on the test dataset
        accs = test_model(
            Unc_Model.model,
            test_data,
            opt.test_b_size,
            class_normalized=opt.normalized_accuracy,
        )
        print("Test accuracy: ", accs)

        # Save initial model weights
        save_model(Unc_Model.model, run_i_loc, "init_model" + str(sim_i) + ".pt")

        # For each initial model run simulations
        for trial_i in range(
            opt.init_sim_for_model, opt.init_sim_for_model + opt.n_sim_per_initial_model
        ):
            # Set random seeds for reproducibility
            random.seed(trial_i * simcoef_int + simsum_int)
            torch.manual_seed(trial_i * simcoef_int + simsum_int)
            np.random.seed(trial_i * simcoef_int + simsum_int)

            # Create a directory to save results for each simulation
            trial_loc = create_run_dir(run_i_loc, "trial")

            # Creating a list of Sim object for each benchmark method
            Sim_List = [
                Sim(params, benchmark, device, copy.deepcopy(Unc_Model.model))
                for benchmark in benchmark_methods
            ]

            # Create observed data distributions for each simulation
            x_dist, N_x = Sim_List[0].create_xdist(
                trial_i * simcoef_int + simsum_int, y_train
            )

            # Create observed data indices based on the observed data distributions
            obs_inds = Sim_List[0].create_obs_ind(
                N_x, y_train, trial_i * simcoef_int + simsum_int
            )

            # Run simulations for each benchmark method
            for i, Sim_Model in enumerate(Sim_List):
                pbar.set_description("Running " + benchmark_methods[i])
                Sim_Model.sim_round(
                    0,
                    trial_i * simcoef_int + simsum_int,
                    X_train,
                    y_train,
                    test_data,
                    base_inds,
                    obs_inds,
                    clip_embs,
                )

            pbar.set_description("Saving results")
            # Save results for each benchmark method
            for i, Sim_Model in enumerate(Sim_List):
                Sim_Model.save_infos(trial_loc, benchmark_methods[i])

            # Plot results for each benchmark method

            plot_accs(
                [Sim_Model.accs for Sim_Model in Sim_List],
                benchmark_methods,
                trial_loc + "/Accs.jpg",
            )
            pbar.update(1)

        # Combine results for each intial model
        pbar.set_description("Combining results")
        run_ids = [i for i in range(opt.n_trial)]
        combine_sims(
            run_ids,
            run_i_loc,
            run_i_loc,
            benchmark_methods,
            name="trial",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-loc",
        type=str,
        default="/store/datasets/AdversarialWeather",
        help="Location of dataset labels",
    )
    parser.add_argument(
        "--img-loc",
        type=str,
        default="/store/datasets/AdversarialWeather",
        help="Location of dataset images",
    )
    parser.add_argument(
        "--clip-emb-loc",
        type=str,
        default="/store/datasets/AdversarialWeather",
        help="Location of dataset clip embeddings",
    )
    parser.add_argument(
        "--emb-loc",
        type=str,
        default="/store/datasets/AdversarialWeather/features/resnet50",
        help="Location of outputs of backbone network",
    )
    parser.add_argument("--gpu-no", type=int, default=4, help="GPU number to use")
    parser.add_argument(
        "--n-environment",
        type=int,
        default=5,
        help="Number of different environments to simulate",
    )
    parser.add_argument(
        "--n-device-per-environment",
        type=int,
        default=5,
        help="Number of different devices per environment to simulate",
    )
    parser.add_argument(
        "--n-init-model",
        type=int,
        default=1,
        help="Number of different initial models to simulate",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=5,
        help="Number of data collection rounds for each simulation",
    )
    parser.add_argument(
        "--n-epoch", type=int, default=300, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--b-size", type=int, default=10000, help="Batch size for training"
    )
    parser.add_argument(
        "--init-model-ind", type=int, default=0, help="Simulation index to start from"
    )
    parser.add_argument(
        "--test-b-size", type=int, default=40000, help="Batch size for testing"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--initial-train-dataset-size",
        type=int,
        default=20,
        help="Number of initial training samples",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=1000,
        help="Number of observations which each device observes",
    )
    parser.add_argument(
        "--n-cache", type=int, default=1, help="Number of cached samples per device"
    )
    parser.add_argument(
        "--run-loc",
        type=str,
        default="./runs/AdversarialWeather",
        help="Location to save simulation results",
    )
    parser.add_argument(
        "--n-sim-per-initial-model",
        type=int,
        default=5,
        help="Number of simulations per initial model",
    )
    parser.add_argument(
        "--init-sim-for-model",
        type=int,
        default=0,
        help="Initial simulation index for each initial model",
    )
    parser.add_argument(
        "--unc-type", type=str, default="clip", help="Uncertainty type to use"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="AdversarialWeather",
        help="Dataset that is being used",
    )
    parser.add_argument(
        "--use-embeddings",
        type=int,
        default=1,
        help="Whether use pretrained model backbone",
    )
    parser.add_argument(
        "--converge-train",
        type=int,
        default=1,
        help="Whether to check for convergence during training",
    )
    parser.add_argument(
        "--cache-all",
        type=int,
        default=0,
        help="Whether to cache all samples druing training",
    )
    parser.add_argument(
        "--dirichlet",
        type=int,
        default=1,
        help="Whether to use dirichlet distribution for observation distribution",
    )
    parser.add_argument(
        "--dirichlet-base",
        type=int,
        default=1,
        help="Whether to use dirichlet distribution for initial dataset distribution",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=1,
        help="Alpha parameter for observation distribution",
    )
    parser.add_argument(
        "--dirichlet-base-alpha",
        type=float,
        default=5,
        help="Alpha parameter for initial dataset distribution",
    )
    parser.add_argument(
        "--cache-in-first",
        type=int,
        default=1,
        help="Whether to cache samples in first round",
    )
    parser.add_argument(
        "--train-only-final",
        type=int,
        default=0,
        help="Whether to train only the final layers of the network",
    )
    parser.add_argument(
        "--normalized-accuracy",
        type=int,
        default=0,
        help="Whether to use normalized accuracy",
    )
    parser.add_argument(
        "--cache-selection",
        type=str,
        default="facility",
        help="Submodular objective function for cache selection",
    )
    parser.add_argument(
        "--backbone-network",
        type=str,
        default="resnet50",
        help="Backbone network to use",
    )

    opt = parser.parse_args()

    device = torch.device(
        "cuda:" + str(opt.gpu_no) if (torch.cuda.is_available()) else "cpu"
    )

    print(
        "Using torch %s %s"
        % (
            torch.__version__,
            torch.cuda.get_device_properties(opt.gpu_no)
            if torch.cuda.is_available()
            else "CPU",
        )
    )

    run_sim(opt, device)
