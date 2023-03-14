import torchvision.datasets as datasets
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as trfm

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

# Loads MNIST dataset from torchvision.datasets with specified type
def load_MNIST_datasets(save_dir:str,type:str) -> tuple:

    '''
    :param save_dir: directory to save the dataset
    :param type: type of dataset to load
    :return: trainset and testset
    '''

    if type == "MNIST":
        trainset = datasets.MNIST(root=save_dir, train=True, download=True)
        testset = datasets.MNIST(root=save_dir, train=False, download=True)
    elif type == "FashionMNIST":
        trainset = datasets.FashionMNIST(root=save_dir, train=True, download=True)
        testset = datasets.FashionMNIST(root=save_dir, train=False, download=True)
    elif type == "KMNIST":
        trainset = datasets.KMNIST(root=save_dir, train=True, download=True)
        testset = datasets.KMNIST(root=save_dir, train=False, download=True)
    else:
        print("Dataset not found")
        exit()
    
    return trainset, testset

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

