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

save_dir = "/store/datasets/AdversarialWeather"

with open(save_dir+"/weather_labels.yml","r") as file:
    weather_labels = yaml.safe_load(file)
with open(save_dir+"/daytime_labels.yml","r") as file:
    daytime_labels = yaml.safe_load(file)

img_locs = list(daytime_labels.keys())

# load image
image = Image.open(img_locs[0])

# pre-process image
image = preprocess(image).unsqueeze(0).to(device)
print("\n\nTensor shape:")
print(image.shape)

train_embs = dict()

# get embeddings for train images

for i in tqdm(range(len(img_locs))):
    image = Image.open(img_locs[i])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    train_embs[img_locs[i].split("/")[-1]] = image_features.cpu().numpy()

# save embeddings

np.save(save_dir+"/clip_embs.npy",train_embs)

