# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
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
    ActivationAPoZRankFilterPruner,
    AMCPruner
)

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = False                     # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = './experiments/pretrained_mobilenet_v2_best/'
log_name_additions = ''
checkpoint = experiment_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120

# reduce CPU usage
train_dataset, train_dataloader = None, None
# train_dataset_for_pruner, train_dataloader_for_pruner = None, None
valid_dataset, valid_dataloader = None, None
test_dataset, test_dataloader = None, None 

# optimization parameters    (for finetuning)
batch_size = 32
n_epochs = 30
learning_rate = 1e-4         # 1e-4 for finetuning, 1e-3 (?) for training from scratch

pruner_type_to_class = {'level': LevelPruner,
                        'l1': L1FilterPruner,
                        'l2': L2FilterPruner,
                        'slim': SlimPruner,
                        'fpgm': FPGMPruner,
                        'taylor': TaylorFOWeightFilterPruner,
                        'agp': AGPPruner,
                        'activationmeanrank': ActivationMeanRankFilterPruner,
                        'apoz': ActivationAPoZRankFilterPruner,
                        'amc': AMCPruner}


def run_test(model):
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


def run_validation(model, dataloader):
    model.eval()
    
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
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


def validator(dataloader, model):
    _, acc = run_validation(model, dataloader)
    return acc


def run_finetune(model, log):    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 

    best_valid_acc = 0.0
    best_model = None
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

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model).to(device)

    log.write("Best validation accuracy: {}".format(best_valid_acc))

    model = best_model


def trainer_helper(model, criterion, optimizer):
    print("Running trainer in tuner")
    for epoch in range(1):
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader_for_pruner)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
    

def main(pruner_type):
    log = open(experiment_dir + '/prune_{}_{}{}.log'.format(pruner_type, strftime("%Y%m%d%H%M", gmtime()), log_name_additions), 'w')
    
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)

    # evaluation before pruning 
    count_flops(model, log)
    
    initial_loss, initial_acc = run_test(model)
    print('Before Pruning:\nLoss: {}\nAccuracy: {}'.format(initial_loss, initial_acc))
    log.write('Before Pruning:\nLoss: {}\nAccuracy: {}\n'.format(initial_loss, initial_acc))
    
    # pruning
    config_list = [{
        'op_types': ['Conv2d'],
    }]
    
    kwargs = {
        'evaluator': validator,
        'val_loader': valid_dataloader,
        'flops_ratio': 0.5,
        'lbound': 0.3,
        'rbound': 0.8,
        'model_type': 'mobilenetv2',
        'train_episode': 800
    }

    pruner = pruner_type_to_class[pruner_type](model, config_list, **kwargs)
    pruner.compress()
    pruner.export_model('./model_temp.pth', './mask_temp.pth')
    
    # model speedup
    dummy_input = torch.rand(1,3,224,224).cuda()
    pruner._unwrap_model()
    ms = ModelSpeedup(model, dummy_input, './mask_temp.pth')
    ms.speedup_model()
    print(model)
    count_flops(model, log)
    intermediate_loss, intermediate_acc = run_test(model)
    print('Before Finetuning:\nLoss: {}\nAccuracy: {}'.format(intermediate_loss, intermediate_acc))
    log.write('Before Finetuning:\nLoss: {}\nAccuracy: {}\n'.format(intermediate_loss, intermediate_acc))

    # finetuning
    run_finetune(model, log)
    
    # final evaluation
    final_loss, final_acc = run_test(model)
    print('After Pruning:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))
    log.write('After Pruning:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))

    # clean up
    filePaths = ['./model_tmp.pth', './mask_tmp.pth']
    for f in filePaths:
        if os.path.exists(f):
            os.remove(f)
            
    log.close()


if __name__ == '__main__':
    # create here and reuse
    train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_dataset_for_pruner = EvalDataset('./data/stanford-dogs/Processed/train')
    # train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.set_num_threads(16)

    main('amc')
