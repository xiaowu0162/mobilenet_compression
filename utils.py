# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.data import Dataset
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params

from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2


def create_model(model_type=None, n_classes=120, input_size=224, checkpoint=None, pretrained=False):
    if model_type == 'mobilenet_v1':
        model = MobileNet(n_class=n_classes, profile='normal')
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(n_class=n_classes, input_size=input_size, width_mult=1.)
    elif model_type == 'mobilenet_v2_torchhub':
        model = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=pretrained)
        feature_size = model.classifier[1].weight.data.size()[1]
        replace_classifier = torch.nn.Linear(feature_size, n_classes)
        model.classifier[1] = replace_classifier
    elif model_type is None:
        model = None
    else:
        raise RuntimeError('Unknown model_type.')

    return model


class TrainDataset(Dataset):
    pass


class EvalDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index]).item()
        return instance['input'], instance['label']


def count_flops(model):
    dummy_input = torch.rand([1, 3, 256, 256])
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")
