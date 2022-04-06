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
import argparse



parser = argparse.ArgumentParser(description='Pruning ResNet-18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument("--prune-layer", type=str, default="layer2")
parser.add_argument("--prune-rate", type=str, default="np.linspace(0, 1, 20, endpoint=False)")
parser.add_argument("--checkpoint-path", type=str, default="./Myinfectedmodel2.pth.tar")
parser.add_argument("--outfile", type=str, default="CIFAR10_ResNet-18_BadNets_pruned.txt")
#Device options
parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Define the model
Model = core.models.ResNet(18)
# Model = core.models.vgg19_bn(43)

Model.load_state_dict(torch.load(args.checkpoint_path))
Model.requires_grad_(False)
Model.cuda()
Model.eval()


# Define settings involved in attacks
global_seed = 789
deterministic = True
torch.manual_seed(global_seed)


# Define evaluation
def test_eval(model, data_loader):
    print(" Eval:")
    acc = 0.0
    total_sample = 0
    total_correct = 0

    # Evaluating benign test accuracy
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        total_sample += inputs.shape[0]

        preds = model(inputs)
        correct_num = torch.sum(torch.argmax(preds, 1) == targets)
        total_correct += correct_num
        acc = total_correct * 100.0 / total_sample

        print(batch_idx, len(data_loader), "ACC: {:.4f}".format(acc))

    return acc


# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask

def get_pruned_model(model, tr_loader, prune_rate, layer_to_prune):
    # prune silent activation
    print("======== pruning... ========")
    with torch.no_grad():
        container = []
        def forward_hook(module, input, output):
            container.append(output)
        hook = getattr(model, layer_to_prune).register_forward_hook(forward_hook)
        print("Forwarding all training set")

        model.eval()
        for data, _ in tr_loader:
            model(data.cuda())
        hook.remove()

    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    # print('activation shape: ',activation.shape)
    # print(activation[seq_sort].detach().cpu().numpy())
    num_channels = len(activation)
    prunned_channels = int(num_channels*prune_rate)
    mask = torch.ones(num_channels).cuda()
    mask[seq_sort[:prunned_channels]]=0
    if len(container.shape)==4:
        mask = mask.reshape(1,-1,1,1)
    setattr(model, layer_to_prune, MaskedLayer(getattr(model, layer_to_prune),mask))
    print("======== pruning complete ========")


# ## Settings for BadNets
# transform_train = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
#     RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     ToTensor()
# ])
#
# trainset = DatasetFolder(
#     root='./datasets/GTSRB/train', # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
#
# transform_test = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
#     ToTensor()
# ])

# testset = DatasetFolder(
#     root='./datasets/GTSRB/testset', # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
#
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-2, -2] = 255
# pattern[-2, -4] = 255
# pattern[-4, -2] = 255
# pattern[-3, -3] = 255
#
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-4:-1, -4:-1] = 1.0

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
    model=Model,
    #model=net,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_train_index=4,
    poisoned_transform_test_index=2,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
benign_train_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4,
                                drop_last=True, pin_memory=True)

benign_test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)



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


badnets.test(schedule=schedule, model=Model, test_dataset=testset, poisoned_test_dataset=poisoned_test_dataset)


with open(args.outfile, "w+") as outs:
    for pruning_rate in eval(args.prune_rate):
        pruned_model = copy.deepcopy(Model)
        get_pruned_model(pruned_model, benign_train_loader, pruning_rate, args.prune_layer)
        BA = test_eval(pruned_model, benign_test_loader)
        ASR = test_eval(pruned_model, poisoned_test_loader)
        outs.write("%0.3f %0.4f %0.4f\n" % (pruning_rate, BA, ASR))



if __name__ == "__main__":
     main()
