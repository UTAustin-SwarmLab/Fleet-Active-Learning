import torchvision.datasets as datasets
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as trfm
from sklearn.model_selection import train_test_split
import json
import numpy as np
import os
from utils.model_utils import *


# Function to Split Dataset
def split_dataset(dataset: Dataset, split_ratio: float, random_state: int = 1) -> tuple:
    """
    :param dataset: Dataset to be split
    :param split_ratio: Ratio of split
    :param random_state: Random state
    :return: trainset and testset
    """

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state),
    )
    return trainset, testset


# Function to filter the dataset containing only the given indices


def filter_indices(X_train, indices):
    if np.isnumpy(X_train):
        X_train = X_train[indices]


# Function to load datasets
def create_datasets(X, y, type, cache_all=False, cache_in_first=False, use_embs=False):
    if type == "MNIST":
        dataset = create_MNIST_datasets(X, y)
    elif type == "CIFAR10":
        if use_embs:
            dataset = create_Embedding_dataset(X, y)
        else:
            dataset = create_CIFAR10_datasets(X, y)
    elif type == "AdverseWeather" or type == "DeepDrive":
        if use_embs:
            dataset = create_Embedding_dataset(X, y)
        else:
            dataset = create_AdverseWeather_dataset(X, y, cache_all)
            dataset.set_use_cache(cache_in_first)
    else:
        print("Dataset not found")
        exit()

    return dataset


# Function to Create MNIST Dataset
def create_MNIST_datasets(X, y):
    """
    :param X: data
    :param y: labels
    :return: MNIST Dataset
    """
    transform = trfm.Compose([trfm.ToTensor(), trfm.Normalize((0.1307), (0.3081))])
    dataset = MNISTDataset(X, y, transform)
    return dataset


# Function to Create CIFAR10 Dataset
def create_CIFAR10_datasets(X, y):
    """
    :param X: data
    :param y: labels
    :return: CIFAR10 Dataset
    """
    transform = trfm.Compose(
        [
            trfm.ToTensor(),
            trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = CIFAR10Dataset(X, y, transform)
    return dataset


# Creates AdverseWeather Dataset
def create_AdverseWeather_dataset(X, y, cache_all=False):
    transform = trfm.Compose(
        [
            trfm.Resize(256),
            trfm.CenterCrop(224),
            trfm.ToTensor(),
            trfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = AdverseWeatherDataset(X, y, transform, cache_all)
    return dataset


# Create Embedding Dataset
def create_Embedding_dataset(X, y):
    dataset = EmbeddingDataset(X, y)
    return dataset


def create_detection_dataset(X_train, X_test):
    data = {}
    data["train"] = (X_train, 0)
    data["val"] = (X_test[:], 0)
    data["nc"] = 10
    data["names"] = {
        0: "motorcycle",
        1: "rider",
        2: "car",
        3: "bicycle",
        4: "bus",
        5: "pedestrian",
        6: "traffic sign",
        7: "truck",
        8: "traffic light",
        9: "train",
    }

    return data


# Function to Load Dataset
def load_datasets(
    save_dir: str,
    type: str,
    cache_all=False,
    test_ratio=0.125,
    img_loc="/store/datasets/bdd100k/images_resized/100k",
    emb_loc="/store/datasets/bdd100k/features/resnet50",
    use_embs=False,
    device="cpu",
    backbone="resnet50",
) -> tuple:
    """
    :param save_dir: directory to save the dataset and labels
    :param type: type of dataset to load
    :param cache_all: whether to cache all the images
    :param test_ratio: ratio of test set
    :param img_loc: location of images
    :param emb_loc: location of embeddings
    :param use_embs: whether to use embeddings
    :param device: device to load to create embeddings
    :param backbone: backbone to use for creating embeddings
    :return train_data, train_labels, test_data, test_labels
    """

    if type == "MNIST":
        trainset = datasets.MNIST(root=save_dir, train=True, download=True)
        testset = datasets.MNIST(root=save_dir, train=False, download=True)
    elif type == "FashionMNIST":
        trainset = datasets.FashionMNIST(root=save_dir, train=True, download=True)
        testset = datasets.FashionMNIST(root=save_dir, train=False, download=True)
    elif type == "KMNIST":
        trainset = datasets.KMNIST(root=save_dir, train=True, download=True)
        testset = datasets.KMNIST(root=save_dir, train=False, download=True)
    elif type == "CIFAR10":
        trainset = datasets.CIFAR10(root=save_dir, train=True, download=True)
        testset = datasets.CIFAR10(root=save_dir, train=False, download=True)
        if use_embs:
            if not os.path.exists(emb_loc + "/train_embs.npy"):
                create_embeddings(backbone, device, type, save_dir, emb_loc)

            train_embs = np.load(emb_loc + "/train_embs.npy", allow_pickle=True).item()
            test_embs = np.load(emb_loc + "/test_embs.npy", allow_pickle=True).item()

            input_size = train_embs[0].shape[0]

            train_data = np.zeros((len(train_embs), input_size), dtype=np.float32)
            test_data = np.zeros((len(test_embs), input_size), dtype=np.float32)

            for i in range(len(train_embs)):
                train_data[i] = train_embs[i]

            for i in range(len(test_embs)):
                test_data[i] = test_embs[i]

            X_train = train_data
            X_test = test_data

            y_train = torch.tensor(trainset.targets, dtype=int)
            y_test = torch.tensor(testset.targets, dtype=int)

            return X_train, X_test, y_train, y_test
    elif type == "AdverseWeather":
        with open(save_dir + "/weather_labels.json", "r") as file:
            weather_labels = json.load(file)
        with open(save_dir + "/daytime_labels.json", "r") as file:
            daytime_labels = json.load(file)
        get_key = lambda x: "/".join(x.split("/")[-4:])
        img_locs = list(daytime_labels.keys())

        img_locs = list(map(lambda x: img_loc + "/" + x, img_locs))
        X = img_locs

        y = [
            "".join(weather_labels[get_key(x)]) + daytime_labels[get_key(x)]
            for x in img_locs
        ]

        classes = list(set(y))

        classes.sort()

        y = torch.tensor(list(map(lambda x: classes.index(x), y)), dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=1
        )

        train_locs = X_train
        test_locs = X_test

        if use_embs:
            if not os.path.exists(emb_loc + "/embs.npy"):
                create_embeddings(backbone, device, type, save_dir, emb_loc)

            embs = np.load(emb_loc + "/embs.npy", allow_pickle=True).item()
            input_size = embs[get_key(X_train[0])].shape[0]
            train_data = np.zeros((len(X_train), input_size), dtype=np.float32)
            test_data = np.zeros((len(X_test), input_size), dtype=np.float32)

            for i, x_train in enumerate(X_train):
                train_data[i] = embs[get_key(x_train)]

            for i, x_test in enumerate(X_test):
                test_data[i] = embs[get_key(x_test)]

            X_train = train_data
            X_test = test_data

        return (
            (np.array(X_train), train_locs),
            (np.array(X_test), test_locs),
            y_train,
            y_test,
        )
    elif type == "DeepDrive":
        with open(save_dir + "/det_train.json", "r") as file:
            train_labels = json.load(file)

        with open(save_dir + "/det_val.json", "r") as file:
            val_labels = json.load(file)

        label_map = {
            "rainy": 0,
            "snowy": 1,
            "clear": 2,
            "overcast": 3,
            "partly cloudy": 4,
            "foggy": 5,
            "undefined": 6,
        }

        w_val_labels = list(map(lambda x: x["attributes"]["weather"], val_labels))
        w_train_labels = list(map(lambda x: x["attributes"]["weather"], train_labels))

        train_locs = list(map(lambda x: img_loc + "/train/" + x["name"], train_labels))
        test_locs = list(map(lambda x: img_loc + "/val/" + x["name"], val_labels))
        y_train = torch.tensor(
            list(map(lambda x: label_map[x], w_train_labels)), dtype=int
        )
        y_test = torch.tensor(
            list(map(lambda x: label_map[x], w_val_labels)), dtype=int
        )

        filt_test1 = (y_test != 6).tolist()
        filt_train1 = (y_train != 6).tolist()

        filt_test2 = (y_test != 5).tolist()
        filt_train2 = (y_train != 5).tolist()

        filt_train = list(map(lambda x, y: x and y, filt_train1, filt_train2))
        filt_test = list(map(lambda x, y: x and y, filt_test1, filt_test2))

        train_locs = [train_locs[i] for i in range(len(train_locs)) if filt_train[i]]
        test_locs = [test_locs[i] for i in range(len(test_locs)) if filt_test[i]]

        X_train = train_locs
        X_test = test_locs

        if use_embs:
            if not os.path.exists(emb_loc + "/train_embs.npy"):
                create_embeddings(backbone, device, type, save_dir, emb_loc)

            train_embs = np.load(emb_loc + "/train_embs.npy", allow_pickle=True).item()
            test_embs = np.load(emb_loc + "/test_embs.npy", allow_pickle=True).item()

            input_size = train_embs[X_train[0].split("/")[-1]].shape[0]

            train_data = np.zeros((len(X_train), input_size), dtype=np.float32)
            test_data = np.zeros((len(X_test), input_size), dtype=np.float32)

            for i, x_train in enumerate(X_train):
                train_data[i] = train_embs[x_train.split("/")[-1]]

            for i, x_test in enumerate(X_test):
                test_data[i] = test_embs[x_test.split("/")[-1]]

            X_train = train_data
            X_test = test_data

        y_train = y_train[filt_train]
        y_test = y_test[filt_test]

        return (
            (np.array(X_train), train_locs),
            (np.array(X_test), test_locs),
            y_train,
            y_test,
        )
    elif type == "DeepDrive-Detection":
        with open(save_dir + "/det_train.json", "r") as file:
            train_labels = json.load(file)

        with open(save_dir + "/det_val.json", "r") as file:
            val_labels = json.load(file)

        label_map = {
            "rainy": 0,
            "snowy": 1,
            "clear": 2,
            "overcast": 3,
            "partly cloudy": 4,
            "foggy": 5,
            "undefined": 6,
        }

        w_val_labels = list(map(lambda x: x["attributes"]["weather"], val_labels))
        w_train_labels = list(map(lambda x: x["attributes"]["weather"], train_labels))

        train_locs = list(map(lambda x: img_loc + "/train/" + x["name"], train_labels))
        test_locs = list(map(lambda x: img_loc + "/val/" + x["name"], val_labels))
        y_train = torch.tensor(
            list(map(lambda x: label_map[x], w_train_labels)), dtype=int
        )
        y_test = torch.tensor(
            list(map(lambda x: label_map[x], w_val_labels)), dtype=int
        )

        X_train = train_locs
        X_test = test_locs

        return np.array(X_train), np.array(X_test), y_train, y_test
    else:
        print("Dataset not found")
        exit()

    if not torch.is_tensor(trainset.data):
        X_train = torch.tensor(trainset.data)
        X_test = torch.tensor(testset.data)
    else:
        X_train = trainset.data.clone().detach()
        X_test = testset.data.clone().detach()

    if not torch.is_tensor(trainset.targets):
        y_train = torch.tensor(trainset.targets, dtype=int)
        y_test = torch.tensor(testset.targets, dtype=int)
    else:
        y_train = trainset.targets.clone().detach()
        y_test = testset.targets.clone().detach()

    return X_train, X_test, y_train, y_test


# Dataset Class for the MNIST Dataset
class MNISTDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        :param X: data
        :param y: labels
        :param transform: transformation to apply to the data
        """
        self.data = X.numpy()
        self.label = y.numpy()
        self.transform = transform

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.label)

    def __getitem__(self, index):
        """
        :param index: index of the data to return
        :return: data and label at the specified index
        """
        x = Image.fromarray(self.data[index], mode="L")

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x, y)

    def pin_memory(self):
        """
        :return: pinned memory version of the dataset
        """

        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()

        return self


# Dataset Class for the CIFAR10 Dataset
class CIFAR10Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        :param X: data
        :param y: labels
        :param transform: transformation to apply to the data
        """
        self.data = X.numpy()
        self.label = y.numpy()
        self.transform = transform

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.label)

    def __getitem__(self, index):
        """
        :param index: index of the data to return
        :return: data and label at the specified index
        """
        x = Image.fromarray(self.data[index])

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x, y)

    def pin_memory(self):
        """
        :return: pinned memory version of the dataset
        """

        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()

        return self


# Dataset Class for the AdverseWeather and BDD Datasets
class AdverseWeatherDataset(Dataset):
    def __init__(self, X, y, transform=None, cache_all=False):
        if cache_all:
            self.img_locs = X.numpy()
        else:
            self.img_locs = X
        self.label = y
        self.transform = transform
        self.cache_all = cache_all
        self.cached_flag = False

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.cache_all:
            x = Image.fromarray(self.img_locs[index])
        elif self.cached_flag:
            x = self.imgs[index]
        else:
            x = Image.open(self.img_locs[index])

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x, y)

    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()

        return self

    def set_use_cache(self, use_cache):
        if use_cache:
            self.imgs = [None for _ in range(len(self.img_locs))]
            w = 455
            h = 256
            for img_ind, img_loc in enumerate(self.img_locs):
                self.imgs[img_ind] = Image.open(img_loc)
            self.cached_flag = True
        else:
            self.imgs = None


# Creates Embeddings Dataset for the AdverseWeather and BDD Datasets
class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
