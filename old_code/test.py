# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = True                      # load imagenet weight (only for 'mobilenet_v2_torchhub')
checkpoint_dir = './experiments/pretrained_mobilenet_v2_best/'
# checkpoint_dir = 'pretrained_baseline_mobilenet_v2_torchhub_202108230905_adam_noreg_kd/'
checkpoint = checkpoint_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120
batch_size = 32

def run_test():
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)
    # count_flops(model)
    
    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
        end_time_raw = perf_counter()

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()
    print('Final loss: {}\nFinal accuracy: {}'.format(final_loss, final_acc))
    print("Inference elapsed raw time: {}ms".format(end_time_raw - start_time_raw))


if __name__ == '__main__':
    torch.set_num_threads(16)
    run_test()
    
