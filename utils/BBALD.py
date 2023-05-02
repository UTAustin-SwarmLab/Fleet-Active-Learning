import torch.nn as nn
import torch
import math

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=(1,1)),nn.ReLU(True))
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d(0.3)
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=(1,1)), nn.ReLU(True), nn.MaxPool2d(2))
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d(0.3)
        self.conv3 = nn.Sequential(nn.Conv2d(32,32,3,padding=(1,1)), nn.ReLU(True))
        self.conv3_drop = consistent_mc_dropout.ConsistentMCDropout2d(0.3)
        self.conv4 = nn.Sequential(nn.Conv2d(32,32,3,padding=(1,1)), nn.ReLU(True), nn.MaxPool2d(2))
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout(0.3)
        self.fc1 = nn.Sequential(nn.Linear(7*7*32, 128), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(128,num_classes))


    def mc_forward_impl(self, input: torch.Tensor):
        input = self.conv1_drop(self.conv1(input))
        input = self.conv2_drop(self.conv2(input))
        input = self.conv3_drop(self.conv3(input))
        input = self.conv4(input)
        input = input.view(-1, 7*7*32)
        emb = self.fc1(self.fc1_drop(input))
        out = F.log_softmax(self.fc2(emb))
        return out

def run_BBALD_sim(opt,device,model_weights):
    train_dataset, test_dataset = repeated_mnist.create_repeated_MNIST_dataset(num_repetitions=1, add_noise=False)

    num_initial_samples = opt.n_size
    num_classes = 10

    initial_samples = active_learning.get_balanced_sample_indices(
    repeated_mnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes)

    max_training_samples = opt.n_size + opt.n_cache*opt.n_rounds
    acquisition_batch_size = opt.n_cache
    num_inference_samples = 100
    num_test_inference_samples = 5
    num_samples = 100000

    test_batch_size = 512
    batch_size = 64
    scoring_batch_size = 128
    training_iterations = opt.n_epoch

    use_cuda = torch.cuda.is_available()

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    active_learning_data = active_learning.ActiveLearningData(train_dataset)

    # Split off the initial samples first.
    active_learning_data.acquire(initial_samples)

    # THIS REMOVES MOST OF THE POOL DATA. UNCOMMENT THIS TO TAKE ALL UNLABELLED DATA INTO ACCOUNT!
    active_learning_data.extract_dataset_from_pool(40000)

    train_loader = torch.utils.data.DataLoader(
    active_learning_data.training_dataset,
    sampler=active_learning.RandomFixedLengthSampler(active_learning_data.training_dataset, training_iterations),
    batch_size=batch_size,
    **kwargs,)

    pool_loader = torch.utils.data.DataLoader(
    active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs)

    # Run experiment
    test_accs = []
    test_loss = []
    added_indices = []

    pbar = tqdm(initial=len(active_learning_data.training_dataset), total=max_training_samples, desc="Training Set Size")

    while True:
        model = BayesianCNN(num_classes).to(device=device)
        optimizer = torch.optim.Adam(model.parameters())

        model.train()

        # Train
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(device=device)
            target = target.to(device=device)

            optimizer.zero_grad()

            prediction = model(data, 1).squeeze(1)
            loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        # Test
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", leave=False):
                data = data.to(device=device)
                target = target.to(device=device)

                prediction = torch.logsumexp(model(data, num_test_inference_samples), dim=1) - math.log(
                num_test_inference_samples
            )
                loss += F.nll_loss(prediction, target, reduction="sum")

                prediction = prediction.max(1)[1]
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        loss /= len(test_loader.dataset)
        test_loss.append(loss)

        percentage_correct = correct / len(test_loader.dataset)
        test_accs.append(percentage_correct)

        if len(active_learning_data.training_dataset) >= max_training_samples:
            break

        # Acquire pool predictions
        N = len(active_learning_data.pool_dataset)
        logits_N_K_C = torch.empty((N, num_inference_samples, num_classes), dtype=torch.double, pin_memory=use_cuda)

        with torch.no_grad():
            model.eval()

            for i, (data, _) in enumerate(tqdm(pool_loader, desc="Evaluating Acquisition Set", leave=False)):
                data = data.to(device=device)

                lower = i * pool_loader.batch_size
                upper = min(lower + pool_loader.batch_size, N)
                logits_N_K_C[lower:upper].copy_(model(data, num_inference_samples).double(), non_blocking=True)

        with torch.no_grad():
            candidate_batch = batchbald.get_batchbald_batch(
                logits_N_K_C, acquisition_batch_size, num_samples, dtype=torch.double, device=device
            )

        dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)

        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))
    
    return np.array(test_accs)