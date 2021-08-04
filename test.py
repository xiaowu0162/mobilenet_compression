# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

from utils import *


def main(model='mobilenet_v1', checkpoint=None, pretrained=False, n_classes=120, input_size=224):
    model = create_model(model_type=model, pretrained=pretrained, n_classes=n_classes, input_size=input_size,
                         checkpoint=checkpoint)
    print(model)
    count_flops(model)



if __name__ == '__main__':
    main('mobilenet_v2_torchhub', pretrained=True, n_classes=120, input_size=256)
