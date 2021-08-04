import torch
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params

from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2


def create_model(model_type=None, checkpoint=None, pretrained=False):
    if model_type == 'mobilenet_v1':
        model = MobileNet(n_class=1000, profile='normal')
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(n_class=1000, input_size=224, width_mult=1.)
    elif model_type == 'mobilenet_v2_torchhub':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
    elif model_type is None:
        model = None
    else:
        raise RuntimeError('Unknown model_type.')

    return model


def count_flops(model):
    dummy_input = torch.rand([1, 3, 256, 256])
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")