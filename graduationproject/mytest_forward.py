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

global_seed =666
deterministic = True
torch.manual_seed(global_seed)


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


for j in range(1,9):
    loc='layer'+str(int((j+1)/2)).replace('.0','')+'.'+str((j+1)%2)+'.shortcut'
    print(loc)
    for i in range(1,11):
        gamma=i*0.1
        print(gamma)
        mymodel=core.models.ResNet(18)
        mymodel.load_state_dict(torch.load("Myinfectedmodel3.pth.tar"))

        register_forwardhook_for_resnet(mymodel,'resnet18',gamma,loc)

        badnets.model=mymodel

        # Test benign Model
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

# import csv
# with open("data_global_seed666_benign.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(badnets.testdata)