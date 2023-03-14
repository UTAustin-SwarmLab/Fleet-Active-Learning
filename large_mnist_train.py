import torch
from dataset_utils.dataset_utils import *
from model_utils.mnist import *
from model_utils.model_utils import *

# This is the code to train the LargeMNISTClassifier 
# on different MNIST datasets

datasets = ["MNIST","FashionMNIST","KMNIST"]

for dataset in datasets:
    train_dataset, test_dataset = load_MNIST_datasets("/store/datasets",dataset)
    train_dataset = create_MNIST_datasets(train_dataset.data,train_dataset.targets)
    test_dataset = create_MNIST_datasets(test_dataset.data,test_dataset.targets)
    Large_Model = get_model("LargeMNISTClassifier",dataset,device=torch.device("cuda:4"),b_size=10000)

    # Train the model
    train_model(Large_Model, train_dataset, silent=False)

    # Test the model
    acc = test_model(Large_Model, test_dataset, b_size=64)
    print("Accuracy on "+ dataset+ " the test set: " + str(acc))
    # Save the model
    save_model(Large_Model, "models", dataset+"_Large.pt")