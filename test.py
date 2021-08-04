import torch
import numpy as np

from nni.compression.pytorch.utils.counter import count_flops_params

from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2


def main(mobilenet_version='v1'):
    assert mobilenet_version in ['v1', 'v2']

    if mobilenet_version == 'v1':
        model = MobileNet(n_class=1000, profile='normal')
    else:
        model = MobileNetV2(n_class=1000, input_size=224, width_mult=1.)

    print(model)
    
    dummy_input = torch.rand([1, 3, 224, 224])
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")


if __name__ == '__main__':
    main('v2')
