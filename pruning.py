# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from time import gmtime, strftime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    TaylorFOWeightFilterPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = False                     # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = './experiments/pretrained_mobilenet_v2_best/'
checkpoint = experiment_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120

# optimization parameters    (for finetuning)
batch_size = 32
n_epochs = 10
learning_rate = 1e-4         # 1e-4 for finetuning, 1e-3 (?) for training from scratch

# pruning parameters
pruner_type = 'l1'
sparsity = 0.5


pruner_type_to_class = {'level': LevelPruner,
                        'l1': L1FilterPruner,
                        'l2': L2FilterPruner,
                        'slim': SlimPruner,
                        'fpgm': FPGMPruner,
                        'taylor': TaylorFOWeightFilterPruner,
                        'agp': AGPPruner,
                        'activationmeanrank': ActivationMeanRankFilterPruner,
                        'apoz': ActivationAPoZRankFilterPruner}


def run_test(model):
    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()

    return final_loss, final_acc


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


def run_finetune(model):
    log = open(experiment_dir + '/finetune_{}.log'.format(strftime("%Y%m%d%H%M", gmtime())), 'w')
    
    train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_acc = 0.0
    for epoch in range(n_epochs):
        print('Start training epoch {}'.format(epoch))
        loss_list = []

        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
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

    log.close()


def main():
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)

    # evaluation before pruning 
    count_flops(model)
    initial_loss, initial_acc = run_test(model)
    print('Before Pruning:\nLoss: {}\nAccuracy: {}'.format(initial_loss, initial_acc))
    
    # pruning
    config_list = [{'op_names': ['features.{}.conv.2'.format(x) for x in range(2, 18)],
                        'sparsity': sparsity                    
                    }]
    if pruner_type in ['l1', 'l2', 'level', 'fpgm']:
        pruner = pruner_type_to_class[pruner_type](model, config_list)
        pruner.compress()
        
    
    # finetuning
    run_finetune(model)
    
    # model speedup

    # final evaluation
    count_flops(model)
    final_loss, final_acc = run_test(model)
    print('After Pruning:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))


if __name__ == '__main__':
    main()
    
