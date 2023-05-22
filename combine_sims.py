from utils.general_utils import *

run_ids = [i for i in range(25)]
run_i_loc = "./runs/CIFAR10/run99"

combine_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")