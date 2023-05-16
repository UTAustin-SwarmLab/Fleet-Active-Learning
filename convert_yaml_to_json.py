import yaml
import json
import os

# Function that converts all yaml files in a directory to json files
# Function that finds all yaml files with .yml extension in a directory and 
# subdirectories and converts them to json files with the same name in same
# directory

def convert_yaml_to_json(dir_path):
    # get all yaml files in directory and subdirectories
    yaml_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".yml"):
                yaml_files.append(os.path.join(root, file))

    # convert all yaml files to json files
    for yaml_file in yaml_files:
        with open(yaml_file,"r") as file:
            yaml_dict = yaml.load(file,Loader=yaml.FullLoader)
        with open(yaml_file.replace(".yml",".json"),"w") as file:
            json.dump(yaml_dict,file)

    return

convert_yaml_to_json("./runs/MNIST/run99")