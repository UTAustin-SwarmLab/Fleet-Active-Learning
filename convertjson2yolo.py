import os
import json


annotations_loc = "./bdd100k/labels/det_20/det_train.json"
imgs_path = "./bdd100k/images/100k"
label_map = {"motorcycle": 0,
            "rider":1,
            "car":2,
            "bicycle":3,
            "bus":4,
            "pedestrian":5,
            "traffic sign":6,
            "truck":7,
            "traffic light":8,
            "train":9}

def convert_BDD_2_YOLO(annotations_loc,imgs_path,train:bool=True):

    with open(annotations_loc) as f:
        annotations = json.load(f)
    
    for annot in annotations:
        if train:
            img_loc = imgs_path+"/train/"+ annot["name"]
        else:
            img_loc = imgs_path+"/val/"+ annot["name"]
        txt_loc = img_loc.replace(".jpg",".txt")
        txt_loc = txt_loc.replace("/images/","/labels/")

        labels = []
        width = 1280
        height = 720

        if "labels" not in annot.keys():
            continue

        for label in annot["labels"]:
            if label["category"] in label_map.keys():
                class_id = label_map[label["category"]]
                x_mid = label["box2d"]["x1"] + (label["box2d"]["x2"] - label["box2d"]["x1"])/2
                y_mid = label["box2d"]["y1"] + (label["box2d"]["y2"] - label["box2d"]["y1"])/2
                w = label["box2d"]["x2"] - label["box2d"]["x1"]
                h = label["box2d"]["y2"] - label["box2d"]["y1"]
                x_mid = x_mid/width
                y_mid = y_mid/height
                w = w/width
                h = h/height
                labels.append(" ".join(list(map(str,[class_id,x_mid,y_mid,w,h]))))

        
        labels_txt = "\n".join(labels)

        loc = os.path.dirname(txt_loc)

        if not os.path.exists(loc):
            os.makedirs(loc)

        with open(txt_loc,"w") as f:
            f.write(labels_txt)

convert_BDD_2_YOLO(annotations_loc,imgs_path,train=True)

annotations_loc = "./bdd100k/labels/det_20/det_val.json"
convert_BDD_2_YOLO(annotations_loc,imgs_path,train=False)

