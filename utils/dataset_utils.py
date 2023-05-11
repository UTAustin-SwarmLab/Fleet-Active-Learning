import torchvision.datasets as datasets
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as trfm
import yaml
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import json
import numpy as np

# Function to Split Dataset
def split_dataset(dataset:Dataset,split_ratio:float,random_state:int = 1) -> tuple:
    '''
    :param dataset: Dataset to be split
    :param split_ratio: Ratio of split
    :param random_state: Random state
    :return: trainset and testset
    '''

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
    return trainset, testset

# Function to filter the dataset containing only the given indices

def filter_indices(X_train,indices):

    if np.isnumpy(X_train):
        X_train = X_train[indices]

# Function to load datasets
def create_datasets(X,y,type,cache_all=False):

    if type == "MNIST":
        dataset = create_MNIST_datasets(X,y)
    elif type == "CIFAR10":
        dataset = create_CIFAR10_datasets(X,y)
    elif type == "AdversarialWeather" or type == "DeepDrive":
        dataset = create_AdversarialWeather_dataset(X,y,cache_all)
    else:
        print("Dataset not found")
        exit()
    
    return dataset

# Function to Create MNIST Dataset
def create_MNIST_datasets(X,y):
    """
    :param X: data
    :param y: labels
    :return: MNIST Dataset
    """
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.1307), (0.3081))])
    dataset = MNISTDataset(X,y,transform)
    return dataset

# Function to Create CIFAR10 Dataset
def create_CIFAR10_datasets(X,y):
    """
    :param X: data
    :param y: labels
    :return: CIFAR10 Dataset
    """
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = CIFAR10Dataset(X,y,transform)
    return dataset

# Creates AdversarialWeather Dataset
def create_AdversarialWeather_dataset(X,y,cache_all=False):
    transform  = trfm.Compose([
        trfm.Resize(256),
        trfm.RandomCrop(224),
        trfm.ToTensor(),
        trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset = AdversarialWeatherDataset(X,y,transform,cache_all)
    return dataset

# Function to Load Dataset
def load_datasets(save_dir:str,type:str,cache_all=False,test_ratio=0.2,img_loc="/store/datasets/bdd100k/images_resized/100k") -> tuple:
    """
    :param save_dir: directory to save the dataset
    :param type: type of dataset to load
    :return: trainset and testset
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
    elif type == "AdversarialWeather":
        with open(save_dir+"/weather_labels.yml","r") as file:
            weather_labels = yaml.safe_load(file)
        with open(save_dir+"/daytime_labels.yml","r") as file:
            daytime_labels = yaml.safe_load(file)
        
        img_locs = list(daytime_labels.keys())

        if cache_all:
            w = 455
            h = 256    
            X = torch.zeros((len(img_locs),256,455,3),dtype=torch.uint8)
            sim_bar = tqdm(img_locs,total=len(img_locs))
            for ind,i in enumerate(sim_bar):
                Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
                X[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
        else:
            X = img_locs

        y = ["".join(weather_labels[x])+daytime_labels[x] for x in img_locs]

        classes = list(set(y))

        classes.sort()

        y = torch.tensor(list(map(lambda x:classes.index(x),y)),dtype=int)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_ratio,stratify=y,random_state=1)

        if cache_all:
            return X_train, X_test, y_train, y_test, classes
        else:
            return np.array(X_train), np.array(X_test), y_train, y_test
    elif type == "DeepDrive":
         
        with open(save_dir+"/det_train.json","r") as file:
            train_labels = json.load(file)
        
        with open(save_dir+"/det_val.json","r") as file:
            val_labels = json.load(file)

        label_map = {
        "rainy":0,
        "snowy":1,
        "clear": 2,
        "overcast":3,
        "partly cloudy": 4,
        "foggy": 5,
        "undefined":6
        }

        w_val_labels = list(map(lambda x:x["attributes"]["weather"],val_labels))
        w_train_labels = list(map(lambda x:x["attributes"]["weather"],train_labels)) 

        train_locs = list(map(lambda x:img_loc+"/train/"+x["name"],train_labels))
        test_locs = list(map(lambda x:img_loc+"/val/"+x["name"],val_labels))
        y_train = torch.tensor(list(map(lambda x:label_map[x],w_train_labels)),dtype=int)
        y_test = torch.tensor(list(map(lambda x:label_map[x],w_val_labels)),dtype=int)

        filt_test1 = (y_test!=6).tolist()
        filt_train1 = (y_train!=6).tolist()
    
        filt_test2 = (y_test!=5).tolist()
        filt_train2 = (y_train!=5).tolist()
    
        filt_train = list(map(lambda x,y: x and y , filt_train1, filt_train2))
        filt_test = list(map(lambda x,y: x and y , filt_test1, filt_test2))

        train_locs = [train_locs[i] for i in range(len(train_locs)) if filt_train[i]]
        test_locs = [test_locs[i] for i in range(len(test_locs)) if filt_test[i]]

        if cache_all:

            sim_bar = tqdm(train_locs,total=len(train_locs))
            X_train = torch.zeros((len(train_locs),256,455,3),dtype=torch.uint8)
            w = 455
            h = 256   
            for ind,i in enumerate(sim_bar):              
                Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
                X_train[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
            
            sim_bar = tqdm(test_locs,total=len(test_locs))
            X_test = torch.zeros((len(test_locs),256,455,3),dtype=torch.uint8)
            for ind,i in enumerate(sim_bar):
                Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
                X_test[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
            
        else:
            X_train = train_locs
            X_test = test_locs

        y_train = y_train[filt_train]    
        y_test = y_test[filt_test]

        if cache_all:
            return X_train, X_test, y_train, y_test, classes
        else:
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
        y_train = torch.tensor(trainset.targets,dtype=int)
        y_test = torch.tensor(testset.targets,dtype=int)
    else:
        y_train = trainset.targets.clone().detach()
        y_test = testset.targets.clone().detach()
    
    return X_train, X_test, y_train, y_test

# Dataset Class for the MNIST Dataset
class MNISTDataset(Dataset):
    def __init__(self,X,y,transform=None):
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

    def __getitem__(self,index):
        """
        :param index: index of the data to return
        :return: data and label at the specified index
        """
        x = Image.fromarray(self.data[index], mode='L')

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)
    
    def pin_memory(self):
        """
        :return: pinned memory version of the dataset
        """
        
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        
        return self

# Dataset Class for the CIFAR10 Dataset
class CIFAR10Dataset(Dataset):
    def __init__(self,X,y,transform=None):
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

    def __getitem__(self,index):
        """
        :param index: index of the data to return
        :return: data and label at the specified index
        """
        x = Image.fromarray(self.data[index])

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)
    
    def pin_memory(self):
        """
        :return: pinned memory version of the dataset
        """
        
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        
        return self

# Dataset Class for the AdversarialWeather and BDD Datasets
class AdversarialWeatherDataset(Dataset):
    def __init__(self,X,y,transform=None,cache_all=False):
        if cache_all:
            self.img_locs = X.numpy()
        else:
            self.img_locs = X
        self.label = y
        self.transform = transform
        self.cache_all = cache_all

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):

        if self.cache_all:
            x = Image.fromarray(self.img_locs[index])
        else:
            x = Image.open(self.img_locs[index])
        
        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)
    
    def pin_memory(self):
        
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        
        return self
    