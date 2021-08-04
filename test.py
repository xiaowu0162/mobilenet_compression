# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *


def main(model='mobilenet_v1', checkpoint=None, pretrained=False, n_classes=120, input_size=224):
    model = create_model(model_type=model, pretrained=pretrained, n_classes=n_classes, input_size=input_size,
                         checkpoint=checkpoint)
    print(model)
    count_flops(model)

    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
            inputs = inputs.float()
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()
    print('Final loss: {}\nFinal accuracy: {}'.format(final_loss, final_acc))

if __name__ == '__main__':
    main('mobilenet_v2_torchhub', pretrained=True, n_classes=120, input_size=256)
