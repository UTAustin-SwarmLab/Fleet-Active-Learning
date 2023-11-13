import argparse

from utils.general_utils import *
from utils.dataset_utils import *
from utils.model_utils import *
from utils.sim_utils import *
from utils.plotting_utils import *

label_map = {
    1: {"Daytime": "Afternoon", "Weather": ["Sunny"]},
    2: {"Daytime": "Afternoon", "Weather": ["Sunny"]},
    3: {"Daytime": "Afternoon", "Weather": ["Sunny"]},
    4: {"Daytime": "Sunset", "Weather": ["Overcast", "Snow"]},
    5: {"Daytime": "Dusk", "Weather": []},
    6: {"Daytime": "Sunset", "Weather": ["Snow"]},
    7: {"Daytime": "Dusk", "Weather": ["Snow"]},
    8: {"Daytime": "Afternoon", "Weather": ["Rain", "Fog"]},
    9: {"Daytime": "Afternoon", "Weather": ["Overcast"]},
    10: {"Daytime": "Afternoon", "Weather": ["Overcast", "Sleet"]},
    11: {"Daytime": "Afternoon", "Weather": ["Overcast", "Sleet"]},
    12: {"Daytime": "Afternoon", "Weather": ["Overcast", "Sleet"]},
    13: {"Daytime": "Afternoon", "Weather": ["Overcast", "Sleet"]},
    14: {"Daytime": "Dusk", "Weather": ["Snow"]},
    15: {"Daytime": "Sunset", "Weather": ["Sunny", "Overcast"]},
    16: {"Daytime": "Sunset", "Weather": ["Sunny", "Overcast"]},
    17: {"Daytime": "Afternoon", "Weather": ["Overcast", "Snow"]},
    18: {"Daytime": "Afternoon", "Weather": ["Overcast", "Snow"]},
    19: {"Daytime": "Afternoon", "Weather": ["Overcast"]},
    20: {"Daytime": "Dusk", "Weather": []},
    21: {"Daytime": "Dusk", "Weather": []},
    22: {"Daytime": "Dusk", "Weather": []},
    23: {"Daytime": "Dusk", "Weather": []},
    24: {"Daytime": "Afternoon", "Weather": ["Overcast", "Rain"]},
    25: {"Daytime": "Afternoon", "Weather": ["Overcast", "Rain"]},
    26: {"Daytime": "Afternoon", "Weather": ["Overcast", "Rain"]},
    27: {"Daytime": "Afternoon", "Weather": ["Overcast", "Rain"]},
    28: {"Daytime": "Dusk", "Weather": []},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, default="./configs/create_labels_adverseweather.json"
    )

    # Parse arguments
    opt = parser.parse_args()

    # Load config file
    opt = load_config(opt.config_path)

    create_labels_Adverse_Weather(
        opt["dataset_loc"], opt["out_loc"], label_map, opt["frame_per_image"]
    )
