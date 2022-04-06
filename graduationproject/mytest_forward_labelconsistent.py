import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import core
import numpy as np

from RegisterHook import register_forwardhook_for_resnet


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

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
adv_model.load_state_dict(torch.load("/data/yangsheng/Mybenigndmodel1_666.pth.tar"))
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
    seed=global_seed,
    deterministic=True
)

poisoned_train_dataset, poisoned_test_dataset = label_consistent.get_poisoned_dataset()


for j in range(1,9):
    loc='layer'+str(int((j+1)/2)).replace('.0','')+'.'+str((j+1)%2)+'.shortcut'
    print(loc)
    for i in range(1,11):
        gamma=i*0.1
        print(gamma)
        mymodel=core.models.ResNet(18)
        mymodel.load_state_dict(torch.load("Myinfectedmodel1_labelconsistent_666.pth.tar"))

        register_forwardhook_for_resnet(mymodel,'resnet18',gamma,loc)

        label_consistent.model=mymodel

        # Test Infected Model
        test_schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '3',
            'GPU_num': 1,

            'batch_size': 128,
            'num_workers': 4,

            'save_dir': 'experiments',
            'experiment_name': 'test_poisoned_CIFAR10_label_consistent'
        }
        label_consistent.test(test_schedule)

import csv
with open("data_global_seed666_labelconsistent_detail.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(label_consistent.testdata)