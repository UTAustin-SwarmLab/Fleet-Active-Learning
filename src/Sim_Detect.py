import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch


from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *


def run_sim(opt, device):
    # Benchmark Methods to compare our Interactive policy against
    benchmark_methods = ["Distributed", "Centralized", "Interactive"]

    # Load dataset and crate train and test splits
    X_train, X_test, y_train, y_test = load_datasets(
        opt["label_loc"],
        opt["dataset_type"],
        img_loc=opt["img_loc"],
        emb_loc=opt["emb_loc"],
        use_embs=opt["use_embeddings"],
        backbone=opt["backbone_network"],
    )

    # If using clip embeddings, load them
    if opt["unc_type"] == "clip":
        clip_embs = get_clip_embeddings(
            X_train, opt["clip_emb_loc"], opt["dataset_type"], opt["label_loc"], device
        )
    else:
        clip_embs = None

    # If X_train is a tuple, only use the first element
    if isinstance(X_train, tuple):
        X_train = X_train[0]
        X_test = X_test[0]

    # Number of classes in the dataset
    n_class = len(np.unique(y_train))

    # Number of features in the dataset
    n_features = X_train.shape[1] if opt["use_embeddings"] else 1000

    # Create dictionary of parameters
    params = opt.copy()

    # Add additional parameters
    params["n_total_device"] = opt["n_environment"] * opt["n_device_per_environment"]
    params["n_class"] = n_class
    params["n_features"] = n_features

    # Create a progress bar
    pbar = tqdm(total=opt["n_init_model"] * opt["n_sim_per_initial_model"])

    # Create a directory to save results for each initial model
    run_loc = create_run_dir(opt["run_loc"])

    # Run simulations
    for sim_i in range(
        opt["init_sim_for_model"], opt["init_sim_for_model"] + opt["n_init_model"]
    ):
        # Set random seeds for reproducibility
        random.seed(sim_i)
        torch.manual_seed(sim_i)
        np.random.seed(sim_i)

        # Creating a random similarity coefficient and similarity sum to create a random integer for each simulation
        simcoef_int = np.random.randint(low=1, high=100)
        simsum_int = np.random.randint(low=1, high=100)

        # Obtain initial model for the simulation
        model = get_model(
            opt["dataset_type"],
            opt["dataset_type"],
            device,
            opt["b_size"],
            opt["n_epoch"],
            opt["lr"],
            n_class,
            use_embs=opt["use_embeddings"],
            n_features=n_features,
        )

        # Create a Sim object for the initial model
        Unc_Model = Sim_Detect(params, "Base", device, model, 0)

        # Create initial training data indices
        Unc_Model.create_base_inds(y_train, sim_i, sim_i)
        base_inds = Unc_Model.dataset_ind[sim_i]

        # Create initial training dataset
        initial_dataset = create_detection_dataset(
            X_train[tuple(Unc_Model.dataset_ind[sim_i])], X_test
        )

        # Train initial model
        pbar.set_description("Training initial model")
        Unc_Model.model.train(
            data=initial_dataset,
            epochs=Unc_Model.params["n_epoch"],
            save=False,
            device=Unc_Model.device,
            val=False,
            pretrained=True,
            batch=Unc_Model.params["b_size"],
            verbose=False,
            plots=False,
            cache=Unc_Model.params["cache_all"],
            workers=Unc_Model.params["n_workers"],
        )

        # Obtain accuracy of the metrics
        metrics = Unc_Model.model.val(
            batch=Unc_Model.params["test_b_size"], device=Unc_Model.device
        )
        results = metrics.results_dict

        print("Initial mAP50: ", results["metrics/mAP50(B)"])

        # For each initial model run simulations
        for trial_i in range(
            opt["init_sim_for_model"],
            opt["init_sim_for_model"] + opt["n_sim_per_initial_model"],
        ):
            # Set random seeds for reproducibility
            random.seed(trial_i * simcoef_int + simsum_int)
            torch.manual_seed(trial_i * simcoef_int + simsum_int)
            np.random.seed(trial_i * simcoef_int + simsum_int)

            # Create a directory to save results for each simulation
            trial_loc = create_run_dir(run_loc, "trial")

            # Creating a list of Sim object for each benchmark method
            Sim_List = [
                Sim_Detect(params, benchmark, device, Unc_Model.model, results)
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
                    X_test,
                    base_inds,
                    obs_inds,
                    clip_embs,
                )

            pbar.set_description("Saving results")
            # Save results for each benchmark method
            for i, Sim_Model in enumerate(Sim_List):
                Sim_Model.save_infos(trial_loc, benchmark_methods[i])

            # Plot Results for each metric
            precisions = [Sim_Model.precision for Sim_Model in Sim_List]
            recalls = [Sim_Model.recall for Sim_Model in Sim_List]
            map50s = [Sim_Model.map50 for Sim_Model in Sim_List]
            map50_95s = [Sim_Model.map50_95 for Sim_Model in Sim_List]
            fitnesses = [Sim_Model.fitness for Sim_Model in Sim_List]

            # Metric Names
            metrics = ["precision", "recall", "map50", "map50-95", "fitness"]

            # Save locations for each metric
            save_locs = [trial_loc + "/" + metric + ".jpg" for metric in metrics]

            # Plot values for each metric
            plot_values(precisions, benchmark_methods, save_locs[0], "Precision")
            plot_values(recalls, benchmark_methods, save_locs[1], "Recall")
            plot_values(map50s, benchmark_methods, save_locs[2], "mAP50")
            plot_values(map50_95s, benchmark_methods, save_locs[3], "mAP50-95")
            plot_values(fitnesses, benchmark_methods, save_locs[4], "Fitness")
            pbar.update(1)

    # Combine results for each initial model
    pbar.set_description("Combining results")
    run_ids = [i for i in range(opt["n_init_model"] * opt["n_sim_per_initial_model"])]
    combine_det_sims(
        run_ids,
        run_loc,
        run_loc,
        benchmark_methods,
        name="trial",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, default="./configs/deepdrive_detection_test.json"
    )

    # Parse arguments
    opt = parser.parse_args()

    # Load config file
    opt = load_config(opt.config_path)

    device = torch.device(
        "cuda:" + str(opt["gpu_no"]) if (torch.cuda.is_available()) else "cpu"
    )

    print(
        "Using torch %s %s"
        % (
            torch.__version__,
            torch.cuda.get_device_properties(opt["gpu_no"])
            if torch.cuda.is_available()
            else "CPU",
        )
    )

    run_sim(opt, device)
