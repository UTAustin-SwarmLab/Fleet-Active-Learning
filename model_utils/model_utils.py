import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from model_utils.mnist import SmallMNISTClassifier,LargeMNISTClassifier
import os

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
def test_model(model,test_dataset,b_size:int=100):
    """
    Tests the model on the test dataset
    :param model: Model to be tested
    :param test_dataset: Test dataset
    :param test_b_size: Batch size for the test dataset
    :return: Accuracy of the model
    """
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in test_loader:
            out = model(x.to(model.device))
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y.to(model.device)).sum().item()
    return correct/total

# Function to train model
def train_model(model,train_dataset,silent:bool=True):
    """
    Trains the model on the train dataset
    :param model: Model to be trained
    :param train_dataset: Train dataset
    :param silent: If True, no progress bar is shown
    """

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=model.b_size, shuffle=True, worker_init_fn=0)

    if not silent:
        pbar = tqdm([i for i in range(model.n_epoch)], total=model.n_epoch)
    else:
        pbar = [i for i in range(model.n_epoch)]

    for epoch in pbar:
        for x,y in dataloader:

            model.zero_grad()

            out = model(x.to(model.device))
            loss = model.loss_fn(out, y.to(model.device))

            loss.backward()
            optimizer.step()

        lr_sch.step()

        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)

# Function to get model
def get_model(model_name,dataset_name:str,device,b_size:int=100,n_epoch:int=100,lr:float=0.001):
    """
    Returns the model
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :param device: Device to be used
    :param b_size: Batch size
    :param n_epoch: Number of epochs
    :return: Model
    """

    if model_name == "SmallMNISTClassifier":
        model = SmallMNISTClassifier()
    elif model_name == "LargeMNISTClassifier":
        model = LargeMNISTClassifier()
    else:
        raise ValueError("Model not found")
    
    model.device = device
    model.to(device)
    model.apply(init_weights)
    model.loss_fn = nn.CrossEntropyLoss()
    model.dataset_name = dataset_name
    model.model_name = model_name
    model.b_size = b_size
    model.n_epoch = n_epoch
    model.lr = lr

    return model

# Function to save model
def save_model(model,save_loc:str,name:str):
    """
    Saves the model weights
    :param model: Model to be saved
    :param save_loc: Location to save the model
    :param name: Name of the model
    """

    isExist = os.path.exists(save_loc)
    
    if not isExist:
        os.makedirs(save_loc)
    
    torch.save(model.state_dict(), save_loc+"/"+name)

# Function to load model
def load_model(model,loc:str):
    """
    Loads the model weights
    :param model: Model to be loaded
    :param loc: Location of the model weights
    """

    model.load_state_dict(torch.load(loc))

    return model

