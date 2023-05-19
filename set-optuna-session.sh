#!/bin/bash

# default values for hyperparameters, can be overwritten by command line arguments
script_name="Sim_optuna.py"
n_process=1
n_opt_trial="1"
coop="0"
dataset_loc="/store/datasets/AdversarialWeather"
img_loc="/store/datasets/AdversarialWeather/Recordings_resized"
clip_emb_loc="/store/datasets/AdversarialWeather"
gpu_no="1"
cache_all="0"
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
n_unique_device="20"
n_same_device="2"
n_sim="3"
n_rounds="5"
n_epoch="200"
b_size="128"
init_sim="0"
n_iter="3"
n_class="10"
test_b_size="256"
lr="0.1"
n_size="30"
n_obs="2000"
n_cache="2"
run_loc="./runs/AdversarialWeather"
n_trial="3" 
init_trial="0"
unc_type="clip"
dataset_type="AdversarialWeather"
converge_train="1"
dirichlet="1"
dirichlet_base="1"
dirichlet_alpha="1"
dirichlet_base_alpha="5"

# parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --script-name)
      script_name="$2"; shift ;;
    --n-process)
      n_process=$2; shift ;;
    --n-opt-trial)
      n_opt_trial="$2"; shift ;;
    --coop)
      coop="$2"; shift ;;
    --dataset-loc)
      dataset_loc="$2"; shift ;;
    --img-loc)
      img_loc="$2"; shift ;;
    --clip-emb-loc)
      clip_emb_loc="$2"; shift ;;
    --gpu-no)
      gpu_no="$2"; shift ;;
    --cache-all)
      cache_all="$2"; shift ;;
    --n-unique-device)
      n_unique_device="$2"; shift ;;
    --n-same-device)
      n_same_device="$2"; shift ;;
    --n-sim)
      n_sim="$2"; shift ;;
    --n-rounds)
      n_rounds="$2"; shift ;;
    --n-epoch)
      n_epoch="$2"; shift ;;
    --b-size)
      b_size="$2"; shift ;;
    --init-sim)
      init_sim="$2"; shift ;;
    --n-iter)
      n_iter="$2"; shift ;;
    --n-class)
      n_class="$2"; shift ;;
    --test-b-size)
      test_b_size="$2"; shift ;;
    --lr)
      lr="$2"; shift  ;;
    --n-size)
      n_size="$2"; shift ;;
    --n-obs)
      n_obs="$2"; shift ;;
    --n-cache)
      n_cache="$2"; shift ;;
    --run-loc)
      run_loc="$2"; shift  ;;
    --n-trial)
      n_trial="$2"; shift ;;
    --init-trial)
      init_trial="$2"; shift ;;
    --unc-type)
      unc_type="$2"; shift ;;
    --dataset-type)
      dataset_type="$2"; shift ;;
    --converge-train)
      converge_train="$2"; shift ;;
    --dirichlet)
      dirichlet="$2"; shift ;;
    --dirichlet-base)
      dirichlet_base="$2"; shift ;;
    --dirichlet-alpha)
      dirichlet_alpha="$2"; shift ;;
    --dirichlet-base-alpha)
      dirichlet_base_alpha="$2"; shift ;;
    *)
      echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# start optuna session
tmux kill-session -t optuna
tmux new-session -d -s optuna
for i in $(seq 0 $((n_process - 1))); do
  tmux send-keys -t optuna:$i 'conda activate adv_cv' Enter
  tmux send-keys -t optuna:$i 'sleep '$((i + 1))'; python3 '$script_name' --process-id '$i' --n-opt-trial '$n_opt_trial' --coop '$coop' --dataset-loc '$dataset_loc' --img-loc '$img_loc' --clip-emb-loc '$clip_emb_loc' --gpu-no '$((i % gpu_count))' --n-unique-device '$n_unique_device' --n-same-device '$n_same_device' --n-sim '$n_sim' --cache-all '$cache_all' --n-rounds '$n_rounds' --n-epoch '$n_epoch' --b-size '$b_size' --init-sim '$init_sim' --n-iter '$n_iter' --n-class '$n_class' --test-b-size '$test_b_size' --lr '$lr' --n-size '$n_size' --n-obs '$n_obs' --n-cache '$n_cache' --run-loc '$run_loc' --n-trial '$n_trial' --init-trial '$init_trial' --unc-type '$unc_type' --dataset-type '$dataset_type' --converge-train '$converge_train' --dirichlet '$dirichlet' --dirichlet-base '$dirichlet_base' --dirichlet-alpha '$dirichlet_alpha' --dirichlet-base-alpha '$dirichlet_base_alpha Enter
  tmux new-window -d
done
tmux attach -t optuna
