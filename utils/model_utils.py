import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.mnist import MNISTClassifier
import os
from utils.cifar10 import *
from utils.adversarialweather import *
import torchvision.models as vsmodels

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
def test_model(model,test_dataset,b_size:int=100,class_normalized:bool=True) -> float:
    """
    Tests the model on the test dataset
    :param model: Model to be tested
    :param test_dataset: Test dataset
    :param test_b_size: Batch size for the test dataset
    :return: Accuracy of the model
    """
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size)
    correct = [0 for i in range(model.n_class)]
    total = [0 for i in range(model.n_class)]

    with torch.no_grad():
        for x,y in test_loader:
            out, _ = model(x.to(model.device))
            _, pred = torch.max(out.data, 1)
            for i in range(model.n_class):
                pred_i = pred == i
                y_i = y == i
                total[i] += y_i.sum().item()
                correct[i] += (pred_i.cpu() & y_i).sum().item()

    if class_normalized:
        ratios = [correct[i]/total[i] for i in range(model.n_class)]
        return sum(ratios)/len(ratios)
    else:
        return sum(correct)/sum(total)

# Function to train model
def train_model(model,train_dataset,silent:bool=True,converge:bool=False,only_final:bool=False):
    """
    Trains the model on the train dataset
    :param model: Model to be trained
    :param train_dataset: Train dataset
    :param silent: If True, no progress bar is shown
    :param converge: If True, the model is trained till convergence
    """

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

    
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=model.b_size, shuffle=True, worker_init_fn=0)

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
            for x,y in dataloader:

                model.zero_grad()

                out,_ = model(x.to(model.device))
                loss = model.loss_fn(out, y.to(model.device))

                loss.backward()
                optimizer.step()

            lr_sch.step()

            if not silent:
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
                pbar.set_description(s)
    else:
        while epoch < model.n_epoch:

            acc_final = 0.

            for x,y in dataloader:

                model.zero_grad()

                out,_ = model(x.to(model.device))
                loss = model.loss_fn(out, y.to(model.device))

                acc_final += torch.sum((torch.max(out,1)[1] == y.to(model.device)).float()).data.item()

                loss.backward()
                optimizer.step()
            
            acc_final /= len(train_dataset)
            lr_sch.step()

            if epoch >= model.n_epoch//2 and acc_final < 0.2:
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
                lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
                epoch = 0
            else:
                epoch += 1
    
# Function to get model
def get_model(model_name,dataset_name:str,device,b_size:int=100,n_epoch:int=100,lr:float=0.001,n_class:int=5,
              use_embs:bool=False,n_features:int=2048):
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
        model = CifarResNet(BasicBlock,[2]*4)
    elif model_name == "AdversarialWeather" or model_name == "DeepDrive":
        if use_embs:
            model = FinalLayer(n_features, n_class)
        else:
            model = vsmodels.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features,n_class)
            model.emb_size = num_features
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
    model.n_class = n_class

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