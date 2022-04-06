import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core
import cv2


global_seed = 101
deterministic = True
torch.manual_seed(global_seed)

# # Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR10
# #dataset = torchvision.datasets.MNIST
#
#
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


transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/yangsheng/data/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root='/data/yangsheng/data/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)





pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    #model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()


# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '2',
    'GPU_num': 1,

    'benign_training': False,
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
    'experiment_name': 'train_poisoned_CIFAR10_Blended'
}

blended.train(schedule)
infected_model = blended.get_model()
torch.save(infected_model.state_dict(),"Myinfectedmodel1_blended_101_GTSRB.pth.tar")