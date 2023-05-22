import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy
import os
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

obtain_embs = 1

# Location of the dataset labels
label_loc = "/store/datasets/bdd100k/labels/det_20"
# Location of the images that will be used
dataset_loc = "/store/datasets/bdd100k/images/100k"
# gpu no for running model
device_no = 4
# Device no
device = torch.device("cuda:"+str(device_no) if (torch.cuda.is_available()) else "cpu")
model_name = "resnet101"
save_loc = "/store/datasets/bdd100k/features/"+model_name

if obtain_embs:

    model = vsmodels.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Identity()
    model.eval()

    transform = trfm.Compose([
            trfm.Resize(256),
            trfm.CenterCrop(224),
            trfm.ToTensor(),
            trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    with open(label_loc+"/det_train.json","r") as file:
        train_labels = json.load(file)

    with open(label_loc+"/det_val.json","r") as file:
        val_labels = json.load(file)

    train_locs = list(map(lambda x:dataset_loc+"/train/"+x["name"],train_labels))
    test_locs = list(map(lambda x:dataset_loc+"/val/"+x["name"],val_labels))


    train_embs = dict()
    test_embs = dict()

    # get embeddings for train images

    with torch.no_grad():
        for i in tqdm(range(len(train_locs))):
            img = Image.open(train_locs[i])
            img = transform(img).to(device).reshape(1,3,224,224)
            features = model(img)
            train_embs[train_locs[i].split("/")[-1]] = features[0].cpu().numpy()

    # get embeddings for test images

    with torch.no_grad():
        for i in tqdm(range(len(test_locs))):
            img = Image.open(test_locs[i])
            img = transform(img).to(device).reshape(1,3,224,224)
            features = model(img)
            test_embs[test_locs[i].split("/")[-1]] = features[0].cpu().numpy()

    # save embeddings
    os.makedirs(save_loc)

    np.save(save_loc+"/train_embs.npy",train_embs)
    np.save(save_loc+"/test_embs.npy",test_embs)

train_embs = np.load(save_loc+"/train_embs.npy",allow_pickle=True).item()
test_embs = np.load(save_loc+"/test_embs.npy",allow_pickle=True).item()

X_train,X_test,y_train,y_test = load_datasets(label_loc,"DeepDrive",img_loc=dataset_loc)

n_class = len(np.unique(y_train))
input_size = train_embs[X_train[0].split("/")[-1]].shape[0]

train_data = np.zeros((len(X_train),input_size))
test_data = np.zeros((len(X_test),input_size))

for i,x_train in enumerate(X_train):
    train_data[i] = train_embs[x_train.split("/")[-1]]

for i,x_test in enumerate(X_test):
    test_data[i] = test_embs[x_test.split("/")[-1]]

train_features = torch.tensor(train_data).float()
test_features = torch.tensor(test_data).float()

final_model = FinalLayer(input_size,n_class)

final_model.apply(init_weights)
final_model.to(device)
final_model.lr = 0.001
final_model.b_size = 70000
final_model.n_epoch = 300
final_model.device = device
final_model.loss_fn = nn.CrossEntropyLoss()
final_model.n_class = n_class

# Create the dataset and dataloader
train_dataset = CustomDataset(train_features, y_train)
test_dataset = CustomDataset(test_features, y_test)

train_model(final_model,train_dataset,silent=False)

accs = test_model(final_model,test_dataset,b_size=70000,class_normalized=False)

print(accs)