import numpy as np
from tqdm import tqdm
import torch
import argparse
import torch
import copy

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *
import clip
from PIL import Image

device = "cuda:1" if (torch.cuda.is_available()) else "cpu"

# load model and image preprocessing
model, preprocess = clip.load("RN50", device=device, jit=False)

# Process images from DeepDrive dataset 

save_dir = "/store/datasets/CIFAR10"

trainset = datasets.CIFAR10(root=save_dir, train=True, download=True)
testset = datasets.CIFAR10(root=save_dir, train=False, download=True)

X_train = trainset.data
X_test = testset.data

train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

train_embs = dict()
test_embs = dict()

k=0

x_train = tqdm(X_train)

for x in x_train:
    img = Image.fromarray(x)
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    train_embs[k] = image_features.cpu().numpy()
    k+=1

k=0
x_test = tqdm(X_test)
for x in x_test:
    img = Image.fromarray(x)
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    test_embs[k] = image_features.cpu().numpy()
    k+=1

save_loc = "/store/datasets/CIFAR10"

np.save(save_loc+"/train_embs.npy",train_embs)
np.save(save_loc+"/test_embs.npy",test_embs)