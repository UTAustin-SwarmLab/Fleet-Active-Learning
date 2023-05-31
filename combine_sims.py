from utils.general_utils import *

run_ids = [i for i in range(5)]
run_i_loc = "./runs/MNIST/run6"

combine_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")