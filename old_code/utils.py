# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params

from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2


def create_model(model_type=None, n_classes=120, input_size=224, checkpoint=None, pretrained=False, width_mult=1.):
    if model_type == 'mobilenet_v1':
        model = MobileNet(n_class=n_classes, profile='normal')
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(n_class=n_classes, input_size=input_size, width_mult=width_mult)
    elif model_type == 'mobilenet_v2_torchhub':
        model = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=pretrained)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
        feature_size = model.classifier[1].weight.data.size()[1]
        replace_classifier = torch.nn.Linear(feature_size, n_classes)
        model.classifier[1] = replace_classifier
    elif model_type is None:
        model = None
    else:
        raise RuntimeError('Unknown model_type.')

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        
    return model


class TrainDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]
        
        transform_set = [transforms.Lambda(lambda x: x),
                         transforms.RandomRotation(30),
                         # transforms.RandomPerspective(),
                         transforms.ColorJitter(),
                         transforms.RandomHorizontalFlip(p=1)]
        self.transform = transforms.RandomChoice(transform_set)
        
        # self.transform = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        
    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)     # (C, H, W)
        x = torch.from_numpy(x).type(torch.float)#.type(torch.uint8)        # convert to Tensor to use torchvision.transforms
        x = self.transform(x)
        return x, instance['label']


class EvalDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)
        x = torch.from_numpy(x).type(torch.float) #.type(torch.uint8)
        return x, instance['label']


def count_flops(model, log=None):
    dummy_input = torch.rand([1, 3, 256, 256])
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")
    if log is not None:
        log.write(f"FLOPs: {flops}, params: {params}\n")
    return flops, params
