from utils.general_utils import *

run_ids = [i for i in range(2)]
run_i_loc = "./runs/DeepDrive-Detection/run9"

#combine_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")
combine_det_sims(run_ids,run_i_loc,run_i_loc,["Distributed","Oracle","Interactive"],name="trial")
