import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core

global_seed = 101
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)


badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    #model=net,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

badnets.model = core.models.ResNet(18)
# train_register_forwardhook_for_resnet(badnets.model,'resnet18',0)
# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '2',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_CIFAR10_BadNets'
    # 'experiment_name': 'train_poisoned_MNIST_BadNets'
}

badnets.train(schedule)
infected_model = badnets.get_model()
torch.save(infected_model.state_dict(),"Myinfectedmodel3.pth.tar")


# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])
# trainset = dataset('data', train=True, transform=transform_train, download=True)
#
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('data', train=False, transform=transform_test, download=True)
#
#
# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(18),
#     # model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     seed=global_seed,
#     deterministic=deterministic
# )
#
# poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
#
#
#
# # Train Benign Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '3',
#     'GPU_num': 1,
#
#     'benign_training': True, # Train Benign Model
#     'batch_size': 128,
#     'num_workers': 4,
#
#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],
#
#     'epochs': 200,
#
#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'train_benign_CIFAR10_BadNets'
#     # 'experiment_name': 'train_benign_MNIST_BadNets'
# }
#
# badnets.train(schedule)
# benign_model = badnets.get_model()
# torch.save(benign_model.state_dict(),"Mybenigndmodel1_101.pth.tar")
#
# # Test Benign Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '3',
#     'GPU_num': 1,
#
#     'batch_size': 128,
#     'num_workers': 4,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'test_benign_CIFAR10_BadNets'
#     # 'experiment_name': 'test_benign_MNIST_BadNets'
# }
# badnets.test(test_schedule)