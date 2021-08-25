# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from time import gmtime, strftime
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *

from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, RandomFilterPruner


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = True                     # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = 'pretrained_random_pruning_baseline_{}_{}_adam_noreg_kd'.format(model_type, strftime("%Y%m%d%H%M", gmtime()))
os.mkdir(experiment_dir)
checkpoint = None
teacher_checkpoint = './experiments/pretrained_mobilenet_v2_best/checkpoint_best.pt'
input_size = 224
n_classes = 120

# optimization parameters
batch_size = 32
n_epochs = 160
learning_rate = 1e-3         # 1e-4 for finetuning, 1e-3 (?) for training from scratch
weight_decay = 0 #1e-3          # l2 regularization


def run_validation(model, valid_dataloader):
    model.eval()
    
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(valid_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)

            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    valid_loss = np.array(loss_list).mean()
    valid_acc = np.array(acc_list).mean()
    
    return valid_loss, valid_acc


def run_pretrain():
    log = open(experiment_dir + '/pretrain.log', 'w')

    # train with knowledge distillation
    teacher_model = create_model(model_type='mobilenet_v2_torchhub', pretrained=True, n_classes=n_classes,
                                 input_size=input_size, checkpoint=teacher_checkpoint)
    teacher_model = teacher_model.to(device)
    print(teacher_model)
    
    model = copy.deepcopy(teacher_model)
    model = model.to(device)

    # first prune the model to 0.5 sparsity
    config_list = [{
        'op_types': ['Conv2d'],
        'op_names': ['features.{}.conv.1.0'.format(x) for x in range(2, 18)],
        'sparsity': 0.5
    },{
        'op_types': ['Conv2d'],
        'op_names': ['features.{}.conv.2'.format(x) for x in range(2, 18)],
        'sparsity': 0.5                   
    }]

    pruner = RandomFilterPruner(model, config_list)
    pruner.compress()
    pruner.export_model('./base_model_temp.pth', './base_mask_temp.pth')
    dummy_input = torch.rand(1,3,224,224).cuda()
    pruner._unwrap_model()
    ms = ModelSpeedup(model, dummy_input, './base_mask_temp.pth')
    ms.speedup_model()
    
    print(model)

    # count_flops(model)

    train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True) 

    # for distillation
    alpha, temperature = 0.99, 8
    # alpha, temperature = 0, 1
    
    best_valid_acc = 0.0
    for epoch in range(n_epochs):
        print('Start training epoch {}'.format(epoch))
        loss_list = []

        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            
            preds = model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds/temperature, dim=1),
                                       F.softmax(teacher_preds/temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()            
            
        # validation
        valid_loss, valid_acc = run_validation(model, valid_dataloader)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
                  (epoch, train_loss, valid_loss, valid_acc))
        
        # save
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), experiment_dir + '/checkpoint_best.pt')

    log.close()


if __name__ == '__main__':
    torch.set_num_threads(16)    
    run_pretrain()
    
