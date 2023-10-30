import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.mnist_model import MNISTClassifier
import os
from utils.cifar10_model import *
from utils.adverseweather import *
import torchvision.models as vsmodels
from ultralytics_yolo.ultralytics import YOLO
from PIL import Image
import torchvision.datasets as datasets
import numpy as np
import json
import clip


# Function to initialize weights
def init_weights(m):
    """
    Initializes the weights of the model
    :param m: Model
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)


# Function to get model accuracy
def test_model(
    model,
    test_dataset,
    b_size: int = 100,
    class_normalized: bool = True,
    detection: bool = False,
) -> float:
    """
    Tests the model on the test dataset
    :param model: Model to be tested
    :param test_dataset: Test dataset
    :param test_b_size: Batch size for the test dataset
    :return: Accuracy of the model
    """

    if detection:
        metrics = model.val(batch=b_size)
        return metrics.results_dict

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size)
    correct = [0 for i in range(model.n_class)]
    total = [0 for i in range(model.n_class)]

    with torch.no_grad():
        for x, y in test_loader:
            out, _ = model(x.to(model.device))
            _, pred = torch.max(out.data, 1)
            for i in range(model.n_class):
                pred_i = pred == i
                y_i = y == i
                total[i] += y_i.sum().item()
                correct[i] += (pred_i.cpu() & y_i).sum().item()

    if class_normalized:
        ratios = [correct[i] / total[i] for i in range(model.n_class)]
        return sum(ratios) / len(ratios)
    else:
        return sum(correct) / sum(total)


# Function to train model
def train_model(
    model,
    train_dataset,
    silent: bool = True,
    converge: bool = False,
    only_final: bool = False,
    detection: bool = False,
    params=None,
    device=None,
):
    """
    Trains the model on the train dataset
    :param model: Model to be trained
    :param train_dataset: Train dataset
    :param silent: If True, no progress bar is shown
    :param converge: If True, the model is trained till convergence
    """

    if detection:
        model.train(
            data=train_dataset,
            epochs=params["n_epoch"],
            save=False,
            device=device,
            val=False,
            pretrained=True,
            batch=params["b_size"],
            verbose=False,
            plots=False,
        )

        return

    model.train()

    if only_final:
        for param in model.parameters():
            param.requires_grad = False

        # Set the requires_grad attribute of the final layer parameters to True
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=model.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=model.lr)

    lr_sch = lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model.b_size, shuffle=True, worker_init_fn=0
    )

    if not silent:
        pbar = tqdm([i for i in range(model.n_epoch)], total=model.n_epoch)
    else:
        pbar = [i for i in range(model.n_epoch)]

    if converge:
        acc_final = 0
        best_acc = 0
        attempts = 0
        epoch = 1

    if not converge:
        for epoch in pbar:
            for x, y in dataloader:
                model.zero_grad()

                out, _ = model(x.to(model.device))
                loss = model.loss_fn(out, y.to(model.device))

                loss.backward()
                optimizer.step()

            lr_sch.step()

            if not silent:
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )
                s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
                pbar.set_description(s)
    else:
        while epoch < model.n_epoch:
            acc_final = 0.0

            for x, y in dataloader:
                model.zero_grad()

                out, _ = model(x.to(model.device))
                loss = model.loss_fn(out, y.to(model.device))

                acc_final += torch.sum(
                    (torch.max(out, 1)[1] == y.to(model.device)).float()
                ).data.item()

                loss.backward()
                optimizer.step()

            acc_final /= len(train_dataset)
            lr_sch.step()

            if epoch >= model.n_epoch // 2 and acc_final < 0.2:
                if only_final:
                    for param in model.parameters():
                        param.requires_grad = False

                    # Set the requires_grad attribute of the final layer parameters to True
                    for param in model.fc.parameters():
                        param.requires_grad = True
                    optimizer = optim.Adam(model.fc.parameters(), lr=model.lr)
                    model.fc.apply(init_weights)
                else:
                    model.apply(init_weights)
                    optimizer = optim.Adam(model.parameters(), lr=model.lr)
                lr_sch = lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.99, last_epoch=-1
                )
                epoch = 0
            else:
                epoch += 1


# Function to get model
def get_model(
    model_name,
    dataset_name: str,
    device,
    b_size: int = 100,
    n_epoch: int = 100,
    lr: float = 0.001,
    n_class: int = 5,
    use_embs: bool = False,
    n_features: int = 2048,
):
    """
    Returns the model
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :param device: Device to be used
    :param b_size: Batch size
    :param n_epoch: Number of epochs
    :return: Model
    """

    if model_name == "MNIST":
        model = MNISTClassifier()
    elif model_name == "CIFAR10":
        if use_embs:
            model = FinalLayerCIFAR10(n_features, n_class)
        else:
            model = CifarResNet(BasicBlock, [2] * 4)
    elif model_name == "AdverseWeather" or model_name == "DeepDrive":
        if use_embs and model_name == "DeepDrive":
            model = FinalLayer(n_features, n_class)
        elif use_embs and model_name == "AdverseWeather":
            model = FinalLayerAdverse(n_features, n_class)
        else:
            model = vsmodels.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_class)
            model.emb_size = num_features
    elif model_name == "DeepDrive-Detection":
        model = YOLO("yolov8s.yaml")
        return model
    else:
        raise ValueError("Model not found")
    model.device = device
    model.to(device)
    if model_name != "DeepDrive-Detection":
        model.apply(init_weights)
    model.loss_fn = nn.CrossEntropyLoss()
    model.dataset_name = dataset_name
    model.model_name = model_name
    model.b_size = b_size
    model.n_epoch = n_epoch
    model.lr = lr
    model.n_class = n_class

    return model


# Function to get backbone model
def get_backbone_model(model_name: str, device) -> tuple:
    """
    Returns the backbone model
    :param model_name: Name of the model
    :param device: Device to be used
    :return: Model, Preprocess, Number of features
    """

    if model_name == "resnet50":
        weights = vsmodels.ResNet50_Weights.DEFAULT
        model = vsmodels.resnet50(weights=weights)
        preprocess = weights.transforms()
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "resnet101":
        weights = vsmodels.ResNet101_Weights.DEFAULT
        model = vsmodels.resnet101(weights=weights)
        preprocess = weights.transforms()
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "resnet152":
        weights = vsmodels.ResNet152_Weights.DEFAULT
        model = vsmodels.resnet152(weights=weights)
        preprocess = weights.transforms()
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "vith14":
        weights = vsmodels.ViT_H_14_Weights.DEFAULT
        model = vsmodels.vit_h_14(weights=weights)
        preprocess = weights.transforms()
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
    else:
        raise Exception("Model name not found")

    return model, preprocess, num_features


# Function to save model
def save_model(model, save_loc: str, name: str):
    """
    Saves the model weights
    :param model: Model to be saved
    :param save_loc: Location to save the model
    :param name: Name of the model
    """

    isExist = os.path.exists(save_loc)

    if not isExist:
        os.makedirs(save_loc)

    torch.save(model.state_dict(), save_loc + "/" + name)


# Function to load model
def load_model(model, loc: str):
    """
    Loads the model weights
    :param model: Model to be loaded
    :param loc: Location of the model weights
    """
    model.load_state_dict(torch.load(loc))

    return model


def create_embeddings(model_name, device, dataset_type, dataset_loc, save_loc):
    """
    Function to create outputs for the dataset for specific backbone model
    :param model_name: Name of the backbone model
    :param device: Device to be used
    :param dataset_type: Type of the dataset
    :param dataset_loc: Location of the dataset
    :param save_loc: Location to save the embeddings
    """

    if model_name == "clip":
        if dataset_type in ["CIFAR10", "MNIST", "AdverseWeather"]:
            model, preprocess = clip.load("RN50", device=device, jit=False)
        elif dataset_type == "DeepDrive":
            model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)
    else:
        model, preprocess, num_features = get_backbone_model(model_name, device)

    model.eval()
    model.to(device)

    if dataset_type == "CIFAR10":
        trainset = datasets.CIFAR10(root=dataset_loc, train=True, download=True)
        testset = datasets.CIFAR10(root=dataset_loc, train=False, download=True)

        X_train = trainset.data
        X_test = testset.data

        train_embs = dict()
        test_embs = dict()

        # get embeddings for train images

        x_train = tqdm(X_train)
        k = 0

        with torch.no_grad():
            for x in x_train:
                img = Image.fromarray(x)
                img = preprocess(img).unsqueeze(0).to(device)
                features = model(img)
                train_embs[k] = features[0].cpu().numpy()
                k += 1

        x_test = tqdm(X_test)

        k = 0

        with torch.no_grad():
            for x in x_test:
                img = Image.fromarray(x)
                img = preprocess(img).unsqueeze(0).to(device)
                features = model(img)
                test_embs[k] = features[0].cpu().numpy()
                k += 1

        # get embeddings for test images
        # save embeddings

        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        np.save(save_loc + "/train_embs.npy", train_embs)
        np.save(save_loc + "/test_embs.npy", test_embs)
    elif dataset_type == "AdverseWeather":
        get_key = lambda x: "/".join(x.split("/")[-4:])

        with open(dataset_loc + "/weather_labels.json", "r") as file:
            weather_labels = json.load(file)

        img_locs = list(weather_labels.keys())

        img_locs = list(map(lambda x: dataset_loc + "/" + x, img_locs))

        embs = dict()

        # get embeddings for train images

        with torch.no_grad():
            for i in tqdm(range(len(img_locs))):
                img = Image.open(img_locs[i])
                img = preprocess(img).unsqueeze(0).to(device)
                features = model(img)
                embs[get_key(img_locs[i])] = features[0].cpu().numpy()

        # get embeddings for test images
        # save embeddings
        os.makedirs(save_loc)

        np.save(save_loc + "/embs.npy", embs)
    elif dataset_type == "DeepDrive":
        with open(dataset_loc + "/det_train.json", "r") as file:
            train_labels = json.load(file)

        with open(dataset_loc + "/det_val.json", "r") as file:
            val_labels = json.load(file)

        train_locs = list(
            map(lambda x: dataset_loc + "/train/" + x["name"], train_labels)
        )
        test_locs = list(map(lambda x: dataset_loc + "/val/" + x["name"], val_labels))

        train_embs = dict()
        test_embs = dict()

        # get embeddings for train images
        with torch.no_grad():
            for i in tqdm(range(len(train_locs))):
                img = Image.open(train_locs[i])
                img = preprocess(img).unsqueeze(0).to(device)
                features = model(img)
                train_embs[train_locs[i].split("/")[-1]] = features[0].cpu().numpy()

        # get embeddings for test images
        with torch.no_grad():
            for i in tqdm(range(len(test_locs))):
                img = Image.open(test_locs[i])
                img = preprocess(img).unsqueeze(0).to(device)
                features = model(img)
                test_embs[test_locs[i].split("/")[-1]] = features[0].cpu().numpy()

        # save embeddings
        os.makedirs(save_loc)

        np.save(save_loc + "/train_embs.npy", train_embs)
        np.save(save_loc + "/test_embs.npy", test_embs)


def get_clip_embeddings(X, clip_emb_loc, dataset_type, dataset_loc, device):
    """
    Get the embeddings for the dataset using CLIP model
    :param X: Dataset
    :param clip_emb_loc: Location of the embeddings
    :param dataset_type: Type of the dataset
    :param dataset_loc: Location of the dataset
    :param device: Device to be used
    """

    if dataset_type == "AdverseWeather":
        if not os.path.exists(clip_emb_loc + "/clip_embs.npy"):
            create_embeddings("clip", device, dataset_type, dataset_loc, clip_emb_loc)
    elif (
        dataset_type == "CIFAR10"
        or dataset_type == "DeepDrive"
        or dataset_type == "DeepDrive-Detection"
    ):
        if not os.path.exists(clip_emb_loc + "/train_embs.npy"):
            create_embeddings("clip", device, dataset_type, dataset_loc, clip_emb_loc)

    if dataset_type == "AdverseWeather":
        embs = np.load(clip_emb_loc + "/clip_embs.npy", allow_pickle=True).item()
        clip_embs = {
            "/".join(X[i].split("/")[-4:]): embs["/".join(X[i].split("/")[-4:])]
            for i in range(len(X))
        }
    elif dataset_type == "CIFAR10" or dataset_type == "DeepDrive-Detection":
        clip_embs = np.load(clip_emb_loc + "/train_embs.npy", allow_pickle=True).item()
    else:
        clip_embs = np.load(clip_emb_loc + "/train_embs.npy", allow_pickle=True).item()
        embs = dict()
        for i in range(len(X[1])):
            embs[i] = clip_embs[X[1][i].split("/")[-1]]
        clip_embs = embs

    return clip_embs
