import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import core


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

global_seed = 101
deterministic = True
torch.manual_seed(global_seed)


# ============== Label Consistent attack ResNet-18 on CIFAR-10 ==============
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=False)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=False)

adv_model = core.models.ResNet(18)
adv_model.load_state_dict(torch.load("/data/yangsheng/Mybenigndmodel1_101.pth.tar"))

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 128,
    'num_workers': 8,

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
    'experiment_name': 'train_poisioned_CIFAR10_LabelConsistent'
}


eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.05

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

label_consistent.train()
infectedmodel=label_consistent.get_model()
torch.save(infectedmodel.state_dict(),"Myinfectedmodel1_labelconsistent_101.pth.tar")