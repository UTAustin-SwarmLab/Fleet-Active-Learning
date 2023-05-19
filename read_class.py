import yaml
import json
import os

# Function that converts all yaml files in a directory to json files
# Function that finds all yaml files with .yml extension in a directory and 
# subdirectories and converts them to json files with the same name in same
# directory

weather_label_loc = "/store/datasets/AdversarialWeather/weather_labels.json"
daytime_label_loc = "/store/datasets/AdversarialWeather/daytime_labels.json"

# Read weather labels
with open(weather_label_loc,"r") as file:
    weather_label =json.load(file)

# Read daytime labels
with open(daytime_label_loc,"r") as file:
    daytime_label =json.load(file)

daytime_label_new = {}
weather_label_new = {}

for key in daytime_label.keys():
    new_key = "/".join(key.split("/")[-4:])
    daytime_label_new[new_key] = daytime_label[key]
    weather_label_new[new_key] = weather_label[key]

with open(weather_label_loc,"w") as file:
    json.dump(weather_label_new,file)

with open(daytime_label_loc,"w") as file:
    json.dump(daytime_label_new,file)

