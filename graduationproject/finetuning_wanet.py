import copy
import os
import sys
import time

import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import core as core
from torch.utils.data import random_split
import random

from core.utils import Log
from RegisterHook import register_forwardhook_for_resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

global_seed = 101
deterministic = True
torch.manual_seed(global_seed)



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


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def adjust_learning_rate(lr, optimizer, epoch):
    if epoch in [20]:
        lr*=0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(model, schedule,train_dataset):
    print("--------fine tuning-------")
    current_schedule=schedule

    if 'pretrain' in current_schedule:
        model.load_state_dict(torch.load(current_schedule['pretrain']), strict=False)

    # Use GPU
    if 'device' in current_schedule and current_schedule['device'] == 'GPU':
        if 'CUDA_VISIBLE_DEVICES' in current_schedule:
            os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

        assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
        assert current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
        print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")

        if current_schedule['GPU_num'] == 1:
            device = torch.device("cuda:0")
        else:
            gpus = list(range(current_schedule['GPU_num']))
            model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
            # TODO: DDP training
            pass
    # Use CPU
    else:
        device = torch.device("cpu")

    if current_schedule['benign_training'] is True:
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_workers=current_schedule['num_workers'],
            drop_last=True,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

    iteration = 0
    last_time = time.time()

    for i in range(current_schedule['epochs']):
        adjust_learning_rate(current_schedule['lr'],optimizer, i)
        for batch_id, batch in enumerate(train_loader):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            predict_digits = model(batch_img)
            loss = torch.nn.functional.cross_entropy(predict_digits, batch_label)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % current_schedule['log_iteration_interval'] == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                    time.localtime()) + f"Epoch:{i + 1}/{current_schedule['epochs']}, iteration:{batch_id + 1}/{len(poisoned_train_dataset) // current_schedule['batch_size']}, lr: {current_schedule['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
                last_time = time.time()
                print(msg)




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

fttrainset,fttestset=random_split(testset,[1000,9000])



for i in range(0,36):
    mymodel=core.models.ResNet(18)
    mymodel.load_state_dict(torch.load("Myinfectedmodel_wanet_101.pth.tar"))
    print(i*0.01)
    register_forwardhook_for_resnet(mymodel,"resnet18",1,"layer2.1.shortcut")
    register_forwardhook_for_resnet(mymodel,"resnet18",1,"layer3.1.shortcut")

    for name, child in mymodel.named_children():
        if not "layer2" in name and not"layer3" in name:
            for param in child.parameters():
                param.requires_grad = False

    identity_grid, noise_grid = gen_grid(32, 4)
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


    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '3',
        'GPU_num': 1,

        'benign_training': True, # Train Benign Model
        'batch_size': 128,
        'num_workers': 4,

        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [],

        'epochs': 10,
        'log_iteration_interval': 100,
    }

    # train(mymodel,schedule,fttrainset)

    identity_grid, noise_grid = gen_grid(32, 4)
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
    with open("tuning_data_global_seed101_wanet.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(wanet.testdata)



