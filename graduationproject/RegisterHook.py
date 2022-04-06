import numpy as np
import torch
import torch.nn as nn


def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.Sequential):
            print(grad_in)
            return (gamma * grad_in[0],)
    return _backward_hook

def forward_hook(gamma):
    # implement SGM through grad through ReLU
    def _forward_hook(module, input, output):
        if isinstance(module, nn.Sequential):
            return gamma * output
    return _forward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        # print(name)
        # print(module)
        if 'shortcut' in name and not '0.shortcut' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        # if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
        #     print(name)
        #     module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)

def register_forwardhook_for_resnet(model, arch, gamma,loc):
    # if arch in ['resnet50', 'resnet101', 'resnet152']:
    #     gamma = np.power(gamma, 0.5)
    forward_hook_backdoor = forward_hook(gamma)

    for name, module in model.named_modules():
        # print(name)
        # print(module)
        if loc in name:
            hook=module.register_forward_hook(forward_hook_backdoor)
    return hook


def train_register_forwardhook_for_resnet(model, arch, gamma):
    # if arch in ['resnet50', 'resnet101', 'resnet152']:
    #     gamma = np.power(gamma, 0.5)
    forward_hook_backdoor = forward_hook(gamma)

    for name, module in model.named_modules():
        # print(name)
        # print(module)
        if 'shortcut' in name:
            module.register_forward_hook(forward_hook_backdoor)