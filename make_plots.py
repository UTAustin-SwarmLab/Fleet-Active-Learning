from utils.plotting_utils import *

run_loc = "./runs/AdversarialWeather/run22"

sim_types = ["Distributed","Oracle","Interactive"]

Values = [np.load(run_loc+"/"+sim_type+"_submodular_values.npy") for sim_type in sim_types]

Accs = [np.load(run_loc+"/"+sim_type+"_acc.npy") for sim_type in sim_types]

plot_accs(Accs,sim_types,run_loc+"/Accs.jpg")
plot_values(Values,sim_types,run_loc+"/submodular_values.jpg",y_label=" Submodular Objective $f(\mathcal{D}_c^r)$")

