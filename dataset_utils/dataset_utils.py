import torchvision.datasets as datasets

def load_MNIST_datasets(save_dir,type):

    if type == "MNIST":
        trainset = datasets.MNIST(root=save_dir, train=True, download=True)
        testset = datasets.MNIST(root=save_dir, train=False, download=True)
    elif type == "FashionMNIST":
        trainset = datasets.FashionMNIST(root=save_dir, train=True, download=True)
        testset = datasets.FashionMNIST(root=save_dir, train=False, download=True)
    elif type == "KMNIST":
        trainset = datasets.KMNIST(root=save_dir, train=True, download=True)
        testset = datasets.KMNIST(root=save_dir, train=False, download=True)
    elif type == "EMNIST":
        trainset = datasets.EMNIST(root=save_dir, train=True, download=True)
        testset = datasets.EMNIST(root=save_dir, train=False, download=True)
    else:
        print("Dataset not found")
        exit()
    
    return trainset, testset


