import argparse

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, default="./configs/create_YOLO_labels_deepdrive.json"
    )

    # Parse arguments
    opt = parser.parse_args()

    # Load config file
    opt = load_config(opt.config_path)

    # Convert BDD labels to YOLO format
    convert_BDD_2_YOLO(opt["train_annotations_loc"], opt["imgs_path"], train=True)
    convert_BDD_2_YOLO(opt["val_annotations_loc"], opt["imgs_path"], train=False)
