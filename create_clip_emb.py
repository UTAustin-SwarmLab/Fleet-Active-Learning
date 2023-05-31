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

device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

# load model and image preprocessing
model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)

# Process images from DeepDrive dataset 
dataset_loc = "/store/datasets/bdd100k/images/100k"
label_loc = "/store/datasets/bdd100k/labels/det_20"

with open(label_loc+"/det_train.json","r") as file:
    train_labels = json.load(file)

with open(label_loc+"/det_val.json","r") as file:
    val_labels = json.load(file)

train_locs = list(map(lambda x:dataset_loc+"/train/"+x["name"],train_labels))
test_locs = list(map(lambda x:dataset_loc+"/val/"+x["name"],val_labels))

# load image
image = Image.open(train_locs[0])

# pre-process image
image = preprocess(image).unsqueeze(0).to(device)
print("\n\nTensor shape:")
print(image.shape)

train_embs = dict()
test_embs = dict()

# get embeddings for train images

for i in tqdm(range(len(train_locs))):
    image = Image.open(train_locs[i])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    train_embs[train_locs[i].split("/")[-1]] = image_features.cpu().numpy()

# get embeddings for test images

for i in tqdm(range(len(test_locs))):
    image = Image.open(test_locs[i])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    test_embs[test_locs[i].split("/")[-1]] = image_features.cpu().numpy()

# save embeddings

save_loc = "/store/datasets/bdd100k/clip_embeddings"

np.save(save_loc+"/train_embs.npy",train_embs)
np.save(save_loc+"/test_embs.npy",test_embs)

