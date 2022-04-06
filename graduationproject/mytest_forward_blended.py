import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import core
import numpy as np

from RegisterHook import register_forwardhook_for_resnet

global_seed = 101
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
#dataset = torchvision.datasets.MNIST


transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

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


for j in range(1,9):
    loc='layer'+str(int((j+1)/2)).replace('.0','')+'.'+str((j+1)%2)+'.shortcut'
    print(loc)
    for i in range(1,11):
        gamma=i*0.1
        print(gamma)
        mymodel=core.models.ResNet(18)
        mymodel.load_state_dict(torch.load("Myinfectedmodel1_blended_101.pth.tar"))

        register_forwardhook_for_resnet(mymodel,'resnet18',gamma,loc)

        blended.model=mymodel

        # Test Infected Model
        test_schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '3',
            'GPU_num': 1,

            'batch_size': 128,
            'num_workers': 4,

            'save_dir': 'experiments',
            'experiment_name': 'test_poisoned_CIFAR10_blended'
        }
        blended.test(test_schedule)

import csv
with open("data_global_seed101_blended_detail.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(blended.testdata)