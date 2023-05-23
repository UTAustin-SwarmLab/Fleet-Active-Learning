import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
import torch
import copy
import os
import torchvision.models as vsmodels
from torch.utils.data import Dataset
from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *
from PIL import Image


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_model(model_name,device):

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

obtain_embs = 1

# Location of the dataset labels
label_loc = "/store/datasets/AdversarialWeather"
# Location of the images that will be used
dataset_loc = "/store/datasets/AdversarialWeather/Recordings"
# gpu no for running model
device_no = 4
# Device no
device = torch.device("cuda:"+str(device_no) if (torch.cuda.is_available()) else "cpu")
model_name = "vith14"
save_loc = "/store/datasets/AdversarialWeather/features/"+model_name

if obtain_embs:

    model, preprocess, num_features = get_model(model_name,device)
    model.eval()

    model.to(device)

    with open(label_loc+"/weather_labels.json","r") as file:
        weather_labels = json.load(file)

    img_locs = list(weather_labels.keys())

    img_locs = list(map(lambda x:dataset_loc+"/"+x,img_locs))

    embs = dict()

    get_key = lambda x: "/".join(x.split("/")[-4:])

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

    np.save(save_loc+"/embs.npy",embs)

embs = np.load(save_loc+"/embs.npy",allow_pickle=True).item()

X_train,X_test,y_train,y_test = load_datasets(label_loc,"AdversarialWeather",img_loc=dataset_loc)

n_class = len(np.unique(y_train))
input_size = embs[get_key(X_train)].shape[0]

train_data = np.zeros((len(X_train),input_size),dtype=np.float32)
test_data = np.zeros((len(X_test),input_size),dtype=np.float32)

for i,x_train in enumerate(X_train):
    train_data[i] = embs[get_key(x_train)]

for i,x_test in enumerate(X_test):
    test_data[i] = embs[get_key(x_test)]

train_features = torch.tensor(train_data).float()
test_features = torch.tensor(test_data).float()

final_model = FinalLayer(input_size,n_class)

final_model.apply(init_weights)
final_model.to(device)
final_model.lr = 0.001
final_model.b_size = 70000
final_model.n_epoch = 500
final_model.device = device
final_model.loss_fn = nn.CrossEntropyLoss()
final_model.n_class = n_class

# Create the dataset and dataloader
train_dataset = CustomDataset(train_features, y_train)
test_dataset = CustomDataset(test_features, y_test)

train_model(final_model,train_dataset,silent=False)

accs = test_model(final_model,test_dataset,b_size=70000,class_normalized=True)

print("Normalized accuracy:", accs)

accs = test_model(final_model,test_dataset,b_size=70000,class_normalized=False)

print("Regular accuracy:", accs)