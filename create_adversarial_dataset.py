import json 
import os
import yaml

get_key = lambda x: "/".join(x.split("/")[-4:])

# Creates labels for Adversarial Dataset from foldernames
def create_labels_Adverse_Weather(dataset_loc,out_loc,label_map,frame_per_image):
    
    daytime_label = dict()
    weather_label = dict()
    category_num = []
    image_locs = []

    img_extensions = ["png","bmp","jpeg"]

    for (dirpath,dirnames,filenames) in os.walk(dataset_loc):
        # Check if the file name is an image
        filenames = list(filter(lambda x: x.split(".")[-1] in img_extensions,filenames))
        # Check if the name starts with "Pic"
        filenames = list(filter(lambda x: x.split(".")[0][:3] == "Pic",filenames))
        # Filter the images containing special characters
        filenames = list(filter(lambda x: x.split(".")[0][3:].isnumeric(),filenames))
        filenames.sort(key=lambda x: int(x.split(".")[0][3:]))
        i = 0
        for filename in filenames:
            if i%frame_per_image == 0:
                image_locs.append(os.path.join(dirpath,filename))
                i = 1
            else:
                i +=1

    for i in image_locs:
        dirs =  i.split("/")
        category_num.append(int(dirs[-4]))
        daytime_label[get_key(i)] = label_map[int(dirs[-4])]["Daytime"]
        weather_label[get_key(i)] = label_map[int(dirs[-4])]["Weather"]
    
    with open(out_loc+'/daytime_labels.json', 'w') as outfile:
        json.dump(daytime_label, outfile)
    with open(out_loc+'/weather_labels.json', 'w') as outfile:
        json.dump(weather_label, outfile)
    
    print("Total size:", len(daytime_label))

# Creates Adversarial Weather dataset labels
def label_Adverse(data_loc,label_map_loc,frame_per_image,out_loc):

    with open(label_map_loc,"r") as f:
        label_map = yaml.safe_load(f) 

    create_labels_Adverse_Weather(data_loc,out_loc,label_map,frame_per_image)


dataset_loc = "/store/datasets/AdversarialWeather/Recordings"
out_loc = "/store/datasets/AdversarialWeather"

frame_per_image = 10
label_map_loc = "/store/datasets/AdversarialWeather/label_maps.yml"

label_Adverse(dataset_loc,label_map_loc,frame_per_image,out_loc)