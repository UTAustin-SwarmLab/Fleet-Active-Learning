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
from typing import List, Tuple, Dict
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
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from utils.plotting_utils import *
from utils.dataset_utils import *
from utils.submodular_maximization import *


# Combines multiple trials
def combine_sims(run_ids, run_loc, target_run_loc, sim_types, name="run"):
    with open(
        run_loc + "/" + name + str(run_ids[0]) + "/" + sim_types[0] + "_params.json"
    ) as f:
        params = json.load(f)

    Acc = dict()

    for sim_type in sim_types:
        params = dict()
        with open(
            run_loc + "/" + name + str(run_ids[0]) + "/" + sim_type + "_params.json"
        ) as f:
            params = json.load(f)
        seeds = list()
        obs_ind = dict()
        dataset_ind = dict()
        Acc[sim_type] = np.zeros((len(run_ids), params["n_rounds"] + 1))
        tot_sim = 0

        for i, run_i in enumerate(run_ids):
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_params.json"
            ) as f:
                new_params = json.load(f)

            if new_params != params:
                print("Error in run" + str(run_i) + ". Params don't match.")
                continue

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_seeds.json"
            ) as f:
                new_seed = json.load(f)

            if any(s in seeds for s in new_seed):
                print("Error in run" + str(run_i) + ". The sim seed is already added.")
                continue
            else:
                seeds += new_seed
            tot_sim += 1
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_obs_ind.json"
            ) as f:
                new_obs_ind = json.load(f)
            obs_ind.update(new_obs_ind)

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_dataset_ind.json"
            ) as f:
                new_dataset_ind = json.load(f)
            dataset_ind.update(new_dataset_ind)

            Acc[sim_type][i, :] = np.load(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_acc.npy"
            )

        params["n_sim"] = tot_sim

        with open(target_run_loc + "/" + sim_type + "_params.json", "w") as outfile:
            json.dump(params, outfile)

        with open(target_run_loc + "/" + sim_type + "_obs_ind.json", "w") as outfile:
            json.dump(obs_ind, outfile)

        with open(
            target_run_loc + "/" + sim_type + "_dataset_ind.json", "w"
        ) as outfile:
            json.dump(dataset_ind, outfile)

        with open(target_run_loc + "/" + sim_type + "_seeds.json", "w") as outfile:
            json.dump(seeds, outfile)

        with open(target_run_loc + "/" + sim_type + "_acc.npy", "wb") as outfile:
            np.save(outfile, Acc[sim_type])

    Accs = []
    for sim_type in sim_types:
        Accs.append(Acc[sim_type])

    plot_accs(Accs, sim_types, target_run_loc + "/Accs.jpg")


# Combines multiple trials
def combine_FL_sims(run_ids, run_loc, target_run_loc, sim_types, name="run"):
    with open(
        run_loc + "/" + name + str(run_ids[0]) + "/" + sim_types[0] + "_params.json"
    ) as f:
        params = json.load(f)

    Acc = dict()

    for sim_type in sim_types:
        params = dict()
        with open(
            run_loc + "/" + name + str(run_ids[0]) + "/" + sim_type + "_params.json"
        ) as f:
            params = json.load(f)
        seeds = list()
        obs_ind = dict()
        dataset_ind = dict()
        Acc[sim_type] = np.zeros((len(run_ids), params["n_rounds"] + 1))
        tot_sim = 0

        for i, run_i in enumerate(run_ids):
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_params.json"
            ) as f:
                new_params = json.load(f)

            if new_params != params:
                print("Error in run" + str(run_i) + ". Params don't match.")
                continue

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_seeds.json"
            ) as f:
                new_seed = json.load(f)

            if any(s in seeds for s in new_seed):
                print("Error in run" + str(run_i) + ". The sim seed is already added.")
                continue
            else:
                seeds += new_seed
            tot_sim += 1
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_obs_ind.json"
            ) as f:
                new_obs_ind = json.load(f)
            obs_ind.update(new_obs_ind)

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_dataset_ind.json"
            ) as f:
                new_dataset_ind = json.load(f)
            dataset_ind.update(new_dataset_ind)

            Acc[sim_type][i, :] = np.load(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_acc.npy"
            )

        params["n_sim"] = tot_sim

        with open(target_run_loc + "/" + sim_type + "_params.json", "w") as outfile:
            json.dump(params, outfile)

        with open(target_run_loc + "/" + sim_type + "_obs_ind.json", "w") as outfile:
            json.dump(obs_ind, outfile)

        with open(
            target_run_loc + "/" + sim_type + "_dataset_ind.json", "w"
        ) as outfile:
            json.dump(dataset_ind, outfile)

        with open(target_run_loc + "/" + sim_type + "_seeds.json", "w") as outfile:
            json.dump(seeds, outfile)

        with open(target_run_loc + "/" + sim_type + "_acc.npy", "wb") as outfile:
            np.save(outfile, Acc[sim_type])

    Accs = []
    for sim_type in sim_types:
        Accs.append(Acc[sim_type])

    plot_FL_accs(Accs, sim_types, target_run_loc + "/Accs.jpg")


# Combines multiple trials
def combine_det_sims(
    run_ids=[0, 1],
    run_loc="./runs/DeepDrive-Detection/run9",
    target_run_loc="./runs/DeepDrive-Detection/run9",
    sim_types=["Distributed", "Centralized", "Interactive"],
    name="trial",
):
    with open(
        run_loc + "/" + name + str(run_ids[0]) + "/" + sim_types[0] + "_params.json"
    ) as f:
        params = json.load(f)

    metrics = ["precision", "recall", "map50", "map50_95", "fitness"]

    Metrics = dict()

    for sim_type in sim_types:
        params = dict()
        with open(
            run_loc + "/" + name + str(run_ids[0]) + "/" + sim_type + "_params.json"
        ) as f:
            params = json.load(f)
        seeds = list()
        obs_ind = dict()
        dataset_ind = dict()
        Metrics[sim_type] = [np.zeros((len(run_ids), 2)) for i in range(len(metrics))]
        tot_sim = 0

        for i, run_i in enumerate(run_ids):
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_params.json"
            ) as f:
                new_params = json.load(f)

            if new_params != params:
                print("Error in run" + str(run_i) + ". Params don't match.")
                continue

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_seeds.json"
            ) as f:
                new_seed = json.load(f)

            if any(s in seeds for s in new_seed):
                print("Error in run" + str(run_i) + ". The sim seed is already added.")
                continue
            else:
                seeds += new_seed
            tot_sim += 1
            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_obs_ind.json"
            ) as f:
                new_obs_ind = json.load(f)
            obs_ind.update(new_obs_ind)

            with open(
                run_loc + "/" + name + str(run_i) + "/" + sim_type + "_dataset_ind.json"
            ) as f:
                new_dataset_ind = json.load(f)
            dataset_ind.update(new_dataset_ind)

            for j in range(len(metrics)):
                Metrics[sim_type][j][i, :] = np.load(
                    run_loc
                    + "/"
                    + name
                    + str(run_i)
                    + "/"
                    + sim_type
                    + "_"
                    + metrics[j]
                    + ".npy"
                )

        params["n_sim"] = tot_sim

        with open(target_run_loc + "/" + sim_type + "_params.json", "w") as outfile:
            json.dump(params, outfile)

        with open(target_run_loc + "/" + sim_type + "_obs_ind.json", "w") as outfile:
            json.dump(obs_ind, outfile)

        with open(
            target_run_loc + "/" + sim_type + "_dataset_ind.json", "w"
        ) as outfile:
            json.dump(dataset_ind, outfile)

        with open(target_run_loc + "/" + sim_type + "_seeds.json", "w") as outfile:
            json.dump(seeds, outfile)

        for j in range(len(metrics)):
            with open(
                target_run_loc + "/" + sim_type + "_" + metrics[j] + ".npy", "wb"
            ) as outfile:
                np.save(outfile, Metrics[sim_type][j])

    Ms = [[] for j in range(len(metrics))]
    for j in range(len(metrics)):
        Ms[j].append(Metrics[sim_types[0]][j][:, 0])

    for j in range(len(metrics)):
        for sim_type in sim_types:
            Ms[j].append(Metrics[sim_type][j][:, 1])

    plot_sim_types = sim_types.copy()
    plot_sim_types.insert(0, "Initial")

    for i in range(len(metrics)):
        for sim_type in plot_sim_types:
            print(
                sim_type
                + " "
                + metrics[i]
                + " mean: "
                + str(np.mean(Ms[i][plot_sim_types.index(sim_type)]))
            )
            print(
                sim_type
                + " "
                + metrics[i]
                + " std: "
                + str(np.std(Ms[i][plot_sim_types.index(sim_type)]))
            )

    plot_boxplot_values(
        Ms[0], plot_sim_types, target_run_loc + "/Precision_box.jpg", "Precision"
    )
    plot_boxplot_values(Ms[1], plot_sim_types, target_run_loc + "/Recall_box.jpg", "Recall")
    plot_boxplot_values(Ms[2], plot_sim_types, target_run_loc + "/mAP50_box.jpg", "mAP50")
    plot_boxplot_values(
        Ms[3], plot_sim_types, target_run_loc + "/mAP50-95_box.jpg", "mAP50-95"
    )
    plot_boxplot_values(
        Ms[4], plot_sim_types, target_run_loc + "/Fitness_box.jpg", "Fitness"
    )


def clip_obtain_embeddings(X_train, inds, train_embs, dataset_type):
    emb_size = train_embs[list(train_embs.keys())[0]].shape[1]
    embeddings = np.zeros((len(inds), emb_size))

    if dataset_type == "CIFAR10" or dataset_type == "MNIST":
        for i in range(len(embeddings)):
            embeddings[i] = train_embs[inds[i]]
    elif dataset_type == "AdverseWeather":
        for i in range(len(embeddings)):
            embeddings[i] = train_embs["/".join(X_train[inds[i]].split("/")[-4:])]

    elif dataset_type == "DeepDrive":
        for i in range(len(embeddings)):
            embeddings[i] = train_embs[X_train[inds[i]].split("/")[-1]]

    return embeddings


# Creates new directories for simulations
def create_run_dir(run_loc, name="run"):
    run_i = 0
    if os.path.exists(run_loc):
        exps = os.listdir(run_loc)
    else:
        os.makedirs(run_loc)
        exps = []

    for i in range(len(exps)):
        if name + str(run_i) in exps:
            run_i += 1
        else:
            break

    os.makedirs(run_loc + "/" + name + str(run_i))
    return run_loc + "/" + name + str(run_i)


def convert_BDD_2_YOLO(
    annotations_loc="./bdd100k/labels/det_20/det_train.json",
    imgs_path="./bdd100k/images/100k",
    train: bool = True,
):
    """
    Convert BDD100K annotations to YOLO format
    :param annotations_loc: Location of the annotations file
    :param imgs_path: Location of the images
    :param train: Whether the images are train or val
    """

    label_map = {
        "motorcycle": 0,
        "rider": 1,
        "car": 2,
        "bicycle": 3,
        "bus": 4,
        "pedestrian": 5,
        "traffic sign": 6,
        "truck": 7,
        "traffic light": 8,
        "train": 9,
    }

    # Load the label map
    with open(annotations_loc) as f:
        annotations = json.load(f)

    # Loop through the annotations
    for annot in annotations:
        # Get the image location
        if train:
            img_loc = imgs_path + "/train/" + annot["name"]
        else:
            img_loc = imgs_path + "/val/" + annot["name"]

        # Get the location of the text file to save the labels
        txt_loc = img_loc.replace(".jpg", ".txt")
        txt_loc = txt_loc.replace("/images/", "/labels/")

        # Loop through the labels and convert them to YOLO format
        labels = []
        width = 1280
        height = 720

        # If there are no labels, skip the image
        if "labels" not in annot.keys():
            continue

        # Loop through the labels and convert them to YOLO format
        for label in annot["labels"]:
            if label["category"] in label_map.keys():
                class_id = label_map[label["category"]]
                x_mid = (
                    label["box2d"]["x1"]
                    + (label["box2d"]["x2"] - label["box2d"]["x1"]) / 2
                )
                y_mid = (
                    label["box2d"]["y1"]
                    + (label["box2d"]["y2"] - label["box2d"]["y1"]) / 2
                )
                w = label["box2d"]["x2"] - label["box2d"]["x1"]
                h = label["box2d"]["y2"] - label["box2d"]["y1"]
                x_mid = x_mid / width
                y_mid = y_mid / height
                w = w / width
                h = h / height
                labels.append(" ".join(list(map(str, [class_id, x_mid, y_mid, w, h]))))

        # Join the labels
        labels_txt = "\n".join(labels)

        # Create the directory if it doesn't exist
        loc = os.path.dirname(txt_loc)
        if not os.path.exists(loc):
            os.makedirs(loc)

        # Save the labels
        with open(txt_loc, "w") as f:
            f.write(labels_txt)


def calculate_objective_value(
    dataset_loc, dataset_type, img_loc, clip_emb_loc, run_loc
):
    sim_types = ["Distributed", "Centralized", "Interactive"]
    X_train, X_test, y_train, y_test = load_datasets(
        dataset_loc, dataset_type, img_loc=img_loc
    )

    if dataset_type == "AdverseWeather":
        embs = np.load(clip_emb_loc + "/clip_embs.npy", allow_pickle=True).item()
        train_embs = {
            "/".join(X_train[i].split("/")[-4:]): embs[
                "/".join(X_train[i].split("/")[-4:])
            ]
            for i in range(len(X_train))
        }
    else:
        train_embs = np.load(clip_emb_loc + "/train_embs.npy", allow_pickle=True).item()

    dataset_inds = []

    for sim_type in sim_types:
        with open(run_loc + "/" + sim_type + "_dataset_ind.json") as f:
            dataset_ind = json.load(f)

        dataset_inds.append(dataset_ind)

    sim_keys = list(dataset_inds[0].keys())
    Values = [
        np.zeros((len(dataset_inds[0]), len(dataset_inds[0][sim_keys[0]])))
        for i in range(len(sim_types))
    ]

    all_inds = [i for i in range(len(X_train))]
    all_embs = clip_obtain_embeddings(X_train, all_inds, train_embs, dataset_type)
    pbar = tqdm(
        total=len(sim_types) * len(sim_keys) * len(dataset_inds[0][sim_keys[0]])
    )

    for i in range(len(sim_types)):
        for j in range(len(sim_keys)):
            for k in range(len(dataset_inds[i][sim_keys[j]])):
                dataset_ind = dataset_inds[i][sim_keys[j]][k].copy()

                dataset_embs = clip_obtain_embeddings(
                    X_train, dataset_ind, train_embs, dataset_type
                )

                dist_w_all_training = pairwise_distances(
                    all_embs, dataset_embs, metric="euclidean"
                )

                M = 1 / (1 + 0.01 * dist_w_all_training)

                Values[i][j][k] = (np.max(M, axis=1)).sum()

                if k != 0:
                    Values[i][j][k] -= Values[i][j][0]

                pbar.update(1)
            Values[i][j][0] = 0

    for sim_type in sim_types:
        np.save(
            run_loc + "/" + sim_type + "_submodular_values.npy",
            Values[sim_types.index(sim_type)],
        )


def generate_objective_value_plots(
    dataset_loc, dataset_type, img_loc, clip_emb_loc, run_loc
):
    sim_types = ["Distributed", "Centralized", "Interactive"]
    if os.path.exists(run_loc + "/" + sim_types[0] + "_submodular_values.npy"):
        Values = [
            np.load(run_loc + "/" + sim_type + "_submodular_values.npy")
            for sim_type in sim_types
        ]
    else:
        calculate_objective_value(
            dataset_loc, dataset_type, img_loc, clip_emb_loc, run_loc
        )
        Values = [
            np.load(run_loc + "/" + sim_type + "_submodular_values.npy")
            for sim_type in sim_types
        ]

    plot_values(
        Values,
        sim_types,
        save_loc=run_loc + "/submodular_values.jpg",
        y_label="$f(\mathcal{D}_c^r)$ Value of Submodular Objective",
    )


# Creates labels for Adverse Dataset from foldernames
def create_labels_Adverse_Weather(dataset_loc, out_loc, label_map, frame_per_image):
    get_key = lambda x: "/".join(x.split("/")[-4:])

    daytime_label = dict()
    weather_label = dict()
    category_num = []
    image_locs = []

    img_extensions = ["png", "bmp", "jpeg"]

    for dirpath, dirnames, filenames in os.walk(dataset_loc):
        # Check if the file name is an image
        filenames = list(
            filter(lambda x: x.split(".")[-1] in img_extensions, filenames)
        )
        # Check if the name starts with "Pic"
        filenames = list(filter(lambda x: x.split(".")[0][:3] == "Pic", filenames))
        # Filter the images containing special characters
        filenames = list(filter(lambda x: x.split(".")[0][3:].isnumeric(), filenames))
        filenames.sort(key=lambda x: int(x.split(".")[0][3:]))
        i = 0
        for filename in filenames:
            if i % frame_per_image == 0:
                image_locs.append(os.path.join(dirpath, filename))
                i = 1
            else:
                i += 1

    for i in image_locs:
        dirs = i.split("/")
        category_num.append(int(dirs[-4]))
        daytime_label[get_key(i)] = label_map[int(dirs[-4])]["Daytime"]
        weather_label[get_key(i)] = label_map[int(dirs[-4])]["Weather"]

    with open(out_loc + "/daytime_labels.json", "w") as outfile:
        json.dump(daytime_label, outfile)
    with open(out_loc + "/weather_labels.json", "w") as outfile:
        json.dump(weather_label, outfile)

    print("Total size:", len(daytime_label))


# Creates Adverse Weather dataset labels
def label_Adverse(
    data_loc="/store/datasets/AdverseWeather/Recordings",
    label_map_loc="/store/datasets/AdverseWeather/label_maps.yml",
    frame_per_image=10,
    out_loc="/store/datasets/AdverseWeather",
):
    with open(label_map_loc, "r") as f:
        label_map = yaml.safe_load(f)

    create_labels_Adverse_Weather(data_loc, out_loc, label_map, frame_per_image)


def create_tsne_plots(
    dataset_loc="/store/datasets/CIFAR10",
    dataset_type="CIFAR10",
    img_loc="/store/datasets/CIFAR10",
    clip_emb_loc="/store/datasets/CIFAR10",
    run_loc="./runs/CIFAR10/run10",
):
    sim_types = ["Distributed", "Centralized", "Interactive"]

    X_train, X_test, y_train, y_test = load_datasets(
        dataset_loc, dataset_type, img_loc=img_loc
    )

    if dataset_type == "AdverseWeather":
        embs = np.load(clip_emb_loc + "/clip_embs.npy", allow_pickle=True).item()
        train_embs = {
            "/".join(X_train[i].split("/")[-4:]): embs[
                "/".join(X_train[i].split("/")[-4:])
            ]
            for i in range(len(X_train))
        }
    else:
        train_embs = np.load(clip_emb_loc + "/train_embs.npy", allow_pickle=True).item()

    dataset_inds = []

    for sim_type in sim_types:
        with open(run_loc + "/" + sim_type + "_dataset_ind.json") as f:
            dataset_ind = json.load(f)

        dataset_inds.append(dataset_ind)

    sim_keys = list(dataset_inds[0].keys())

    all_inds = [i for i in range(len(X_train))]
    all_embs = clip_obtain_embeddings(X_train, all_inds, train_embs, dataset_type)
    all_tsne = TSNE(
        n_components=2, n_jobs=-1, init="pca", learning_rate="auto", verbose=1
    ).fit_transform(all_embs)
    # Save the TSNE Embeddings

    np.save(run_loc + "/embs_tsne.npy", all_tsne)

    dataset_tsnes = []

    for i in range(len(sim_types)):
        dataset_ind = dataset_inds[i][sim_keys[0]][-1].copy()
        dataset_tsne = all_tsne[dataset_ind]
        dataset_tsnes.append(dataset_tsne)

    plot_tsne(all_tsne, dataset_tsnes, run_loc + "/tsne.jpg")


def generate_random_points_within_rotated_ellipse(
    a, b, center, num_points, rotation_angle=0
):
    points = []
    while len(points) < num_points:
        # Generate random point within a unit circle
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Scale the random point to fit within the ellipse
        x *= a
        y *= b

        # Rotate the point
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        point = np.dot(rotation_matrix, [x, y])
        points.append(point + center)
    return points


def create_example_plot(save_loc="./"):
    n_iter = 3
    n_cache = 4
    n_samples = 1000

    centers = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
    a_s = [1.5, 1.5, 1.5]
    b_s = [0.5, 0.5, 0.5]
    n_repeat = 3
    n_device = len(centers)
    n_device_show = len(centers)
    rotation_angles = [np.pi / 6, np.pi / 2, -np.pi / 6]

    seed = 12

    np.random.seed(seed)

    X = np.zeros((0, 2))
    y = np.zeros((0))
    for j in range(n_repeat):
        for i in range(len(centers)):
            X = np.concatenate(
                (
                    X,
                    generate_random_points_within_rotated_ellipse(
                        a_s[i], b_s[i], centers[i], n_samples, rotation_angles[i]
                    ),
                ),
                axis=0,
            )
            y = np.concatenate((y, np.ones((n_samples)) * i + j * len(centers)), axis=0)

    X_b = X.copy()
    y_b = y.copy()

    base_embeddings = np.zeros((1, 2))
    embeddings = np.zeros((n_device * n_repeat, n_samples, 2))
    obs_inds = []

    for i in range(n_device * n_repeat):
        embeddings[i] = X[y == i]
        obs_inds.append(np.where(y == i)[0].tolist())

    distributed_policy = FacilityLocation(
        embeddings, base_embeddings, obs_inds, n_iter, n_cache
    )
    dist_inds = distributed_policy.sample_caches("Distributed")
    interactive_policy = FacilityLocation(
        embeddings, base_embeddings, obs_inds, n_iter, n_cache
    )
    centr_inds = interactive_policy.sample_caches("Interactive")

    X_dist = np.concatenate((X[dist_inds], base_embeddings), axis=0)
    X_centr = np.concatenate((X[centr_inds], base_embeddings), axis=0)

    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

    legends = ["Robot %d" % (i + 1) for i in range(n_device_show)]

    obs_inds = np.stack(obs_inds).reshape(-1)

    colors = plt.cm.get_cmap("Greens", 5)
    colors = [
        plt.cm.get_cmap("Blues", 3),
        plt.cm.get_cmap("Oranges", 4),
        plt.cm.get_cmap("Greens", 4),
    ]

    for i in range(n_device_show):
        plt.scatter(
            X[obs_inds[i * n_samples : (i + 1) * n_samples], 0],
            X[obs_inds[i * n_samples : (i + 1) * n_samples], 1],
            color=colors[i](1),
            label=legends[i],
        )

    plt.scatter(
        X_dist[:, 0],
        X_dist[:, 1],
        marker="x",
        color="r",
        s=200,
        label="Distributed",
        linewidths=5,
    )
    plt.scatter(
        X_centr[:, 0],
        X_centr[:, 1],
        marker="o",
        color="b",
        s=200,
        facecolors="none",
        label="Oracle",
        linewidths=5,
    )

    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["legend.labelspacing"] = 0.4
    handles = [
        plt.scatter(
            [], [], marker="o", color=colors[0](1), label="Robot 1 Observations", s=150
        ),
        plt.scatter(
            [], [], marker="o", color=colors[1](1), label="Robot 2  Observations", s=150
        ),
        plt.scatter(
            [], [], marker="o", color=colors[2](1), label="Robot 3  Observations", s=150
        ),
        plt.scatter(
            [],
            [],
            marker="x",
            color="r",
            s=200,
            label="Distributed Action",
            linewidths=5,
        ),
        plt.scatter(
            [],
            [],
            marker="o",
            color="b",
            s=200,
            facecolors="none",
            label="Interactive Action",
            linewidths=5,
        ),
    ]
    legend = plt.legend(
        handles=handles, frameon=True, loc="upper left", bbox_to_anchor=(-0.1, 1.1)
    )

    for text in legend.get_texts():
        text.set_weight("bold")

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(os.path.join(save_loc, f"toy_example.png"), dpi=600)


def create_accuracy_objective_plots(
    run_loc="./runs/AdverseWeather/run22",
    sim_types=["Distributed", "Oracle", "Interactive"],
):
    Values = [
        np.load(run_loc + "/" + sim_type + "_submodular_values.npy")
        for sim_type in sim_types
    ]

    Accs = [np.load(run_loc + "/" + sim_type + "_acc.npy") for sim_type in sim_types]

    plot_accs(Accs, sim_types, run_loc + "/Accs.jpg")
    plot_values(
        Values,
        sim_types,
        run_loc + "/submodular_values.jpg",
        y_label=" Submodular Objective $f(\mathcal{D}_c^r)$",
    )


def load_config(config_loc: str) -> Dict:
    with open(config_loc) as f:
        config = json.load(f)
    return config
