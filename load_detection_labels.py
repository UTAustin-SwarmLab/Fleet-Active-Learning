import json
from utils.dataset_utils import *
from ultralytics_yolo.ultralytics import YOLO
import copy

save_dir = "./bdd100k/labels/det_20"
clip_emb_loc = "./bdd100k/clip_embeddings"
with open(save_dir+"/det_train.json","r") as file:
    train_labels = json.load(file)

with open(save_dir+"/det_val.json","r") as file:
    val_labels = json.load(file)

X_train, X_test, y_train, y_test = load_datasets(save_dir,"DeepDrive-Detection",img_loc="./bdd100k/images/100k")
train_embs = np.load(clip_emb_loc+"/train_embs.npy",allow_pickle=True).item()

data = {}
data["train"] = (X_train[:10],0)
data["val"] = (X_train[:10],0)
data["nc"] = 10
data["names"] = {0: "motorcycle",1: "rider",2: "car",3: "bicycle",4: "bus",
                    5: "pedestrian",6: "traffic sign",7: "truck",8: "traffic light",9: "train"}

model = YOLO("yolov8s.yaml")

gpu_no = 0 

device = torch.device("cuda:"+str(gpu_no) if (torch.cuda.is_available()) else "cpu")

batch_size = 64

model = copy.deepcopy(model)

model.train(data=data,epochs=200,save=False,device= device, val=False,pretrained=True,
batch=batch_size,verbose=False,plots=False)
metrics = model.val()
print(2)

['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness']
metrics.results_dic.keys()