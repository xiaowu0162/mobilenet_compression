# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = True                      # load imagenet weight (only for 'mobilenet_v2_torchhub')
checkpoint_dir = './experiments/pretrained_mobilenet_v2_best/'
checkpoint = checkpoint_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120
batch_size = 8

def run_test():
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)
    for name, weight in model.named_parameters():
        print(name, weight.max().item(), weight.min().item())


if __name__ == '__main__':
    run_test()
    
