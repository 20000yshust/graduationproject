import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import core
import numpy as np

from RegisterHook import register_forwardhook_for_resnet

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


global_seed = 101
deterministic = True
torch.manual_seed(global_seed)


transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=False)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=False)

identity_grid,noise_grid=gen_grid(32,4)
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()


for j in range(1,9):
    loc='layer'+str(int((j+1)/2)).replace('.0','')+'.'+str((j+1)%2)+'.shortcut'
    print(loc)
    for i in range(1,11):
        gamma=i*0.1
        print(gamma)
        mymodel=core.models.ResNet(18)
        mymodel.load_state_dict(torch.load("Myinfectedmodel_wanet_101.pth.tar"))

        register_forwardhook_for_resnet(mymodel,'resnet18',gamma,loc)

        wanet.model=mymodel

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
        wanet.test(test_schedule)

import csv
with open("data_global_seed101_wanet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(wanet.testdata)