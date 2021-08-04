# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

from utils import *


def main(model='mobilenet_v1', checkpoint=None, pretrained=False):
    model = create_model(model_type=model, checkpoint=checkpoint, pretrained=pretrained)
    print(model)
    count_flops(model)



if __name__ == '__main__':
    main('mobilenet_v2', pretrained=False)
