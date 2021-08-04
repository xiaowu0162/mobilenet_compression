# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import *


def main(model='mobilenet_v1', checkpoint=None, pretrained=False, n_classes=120, input_size=224):
    model = create_model(model_type=model, pretrained=pretrained, n_classes=n_classes, input_size=input_size,
                         checkpoint=checkpoint)
    print(model)
    count_flops(model)

    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for i, (inputs, labels) in enumerate(test_dataloader):
        preds = model(inputs)
        print(inputs.size(), labels.size(), preds.size())

if __name__ == '__main__':
    main('mobilenet_v2_torchhub', pretrained=True, n_classes=120, input_size=256)
