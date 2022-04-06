import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import core

#import pretrainedmodels

from RegisterHook import register_hook_for_resnet

global_seed = 666
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


# net = pretrainedmodels.__dict__['resnet18'](num_classes=10, pretrained='cifar10')


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

myfile = open('test.txt','a')


for i in range(0,5):
    badnets.model = core.models.ResNet(18)
    gamma=pow(10,-i)
    print(gamma)
    myfile.write(str(gamma))
    myfile.write('\n')
    register_hook_for_resnet(badnets.model, arch='resnet18', gamma=gamma)
    print(id(badnets.model))

# Train Infected Model
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
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

    # Test Infected Model
    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,

        'save_dir': 'experiments',
        'experiment_name': 'test_poisoned_CIFAR10_BadNets'
        # 'experiment_name': 'test_poisoned_MNIST_BadNets'
    }
    badnets.test(test_schedule)