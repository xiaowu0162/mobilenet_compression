# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.nn.functional as F
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


os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = False                     # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = './experiments/pretrained_mobilenet_v2_best/'
log_name_additions = '_kd'
checkpoint = experiment_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120

# reduce CPU usage
train_dataset, train_dataloader = None, None
train_dataset_for_pruner, train_dataloader_for_pruner = None, None
valid_dataset, valid_dataloader = None, None
test_dataset, test_dataloader = None, None 

# optimization parameters    (for finetuning)
batch_size = 32
n_epochs = 30
learning_rate = 1e-4         # 1e-4 for finetuning, 1e-3 (?) for training from scratch

# for distillation
use_distillation = True
# alpha = 0.1
# temperature = 3


# pruning parameters
# pruner_type_list = ['slim']
# sparsity_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sparsity_list = [0.4, 0.5, 0.6, 0.7]


pruner_type_to_class = {'level': LevelPruner,
                        'l1': L1FilterPruner,
                        'l2': L2FilterPruner,
                        'slim': SlimPruner,
                        'fpgm': FPGMPruner,
                        'taylorfo': TaylorFOWeightFilterPruner,
                        'mean_activation': ActivationMeanRankFilterPruner,
                        'apoz': ActivationAPoZRankFilterPruner}


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


def run_finetune_distillation(student_model, teacher_model, alpha, temperature, log):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=0.9)            

    best_valid_acc = 0.0
    best_model = None
    for epoch in range(n_epochs):
        print('Start training epoch {}'.format(epoch))
        loss_list = []

        # train
        student_model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            
            preds = student_model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds/temperature, dim=1),
                                       F.softmax(teacher_preds/temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        valid_loss, valid_acc = run_validation(student_model, valid_dataloader)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
                  (epoch, train_loss, valid_loss, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(student_model).to(device)

    log.write("Best validation accuracy: {}".format(best_valid_acc))

    student_model = best_model
    return student_model


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
    return model


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

            
def trainer_helper_with_distillation(model, teacher_model, alpha, temperature, optimizer):
    print("Running trainer in tuner")
    for epoch in range(1):
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader_for_pruner)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            preds = model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds/temperature, dim=1),
                                       F.softmax(teacher_preds/temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss.backward()
            optimizer.step()
    

def main(sparsity, pruner_type, num_iterations, epochs_per_iteration, alpha=0, temperature=1):
    log = open(experiment_dir + '/prune_AGP{}_{}_{}_{}iter_{}epoch_alpha{}_temp{}{}.log'.format(pruner_type, sparsity, strftime("%Y%m%d%H%M", gmtime()), num_iterations, epochs_per_iteration, alpha, temperature, log_name_additions), 'w')
    
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)
    
    teacher_model = None
    if use_distillation:
        teacher_model = copy.deepcopy(model)

    # evaluation before pruning 
    count_flops(model, log)
    
    initial_loss, initial_acc = run_test(model)
    print('Before Pruning:\nLoss: {}\nAccuracy: {}'.format(initial_loss, initial_acc))
    log.write('Before Pruning:\nLoss: {}\nAccuracy: {}\n'.format(initial_loss, initial_acc))
    
    # pruning
    if pruner_type == 'slim':
        config_list = [{
            'op_names': ['features.{}.conv.1.1'.format(x) for x in range(2, 18)],
            'sparsity': sparsity                    
            },{
            'op_types': ['BatchNorm2d'],
            'op_names': ['features.{}.conv.3'.format(x) for x in range(2, 18)],
            'sparsity': sparsity                    
        }]
    else:
        config_list = [{
            'op_names': ['features.{}.conv.1.0'.format(x) for x in range(2, 18)],
            'sparsity': sparsity                    
            },{
            'op_names': ['features.{}.conv.2'.format(x) for x in range(2, 18)],
            'sparsity': sparsity                    
        }]

    kwargs = {}
    kwargs['pruning_algorithm'] = pruner_type
    kwargs['num_iterations'] = num_iterations                # int(sparsity/0.1)
    kwargs['epochs_per_iteration'] = epochs_per_iteration    # 1

    # if pruner_type in ['slim', 'taylor', 'activationmeanrank', 'apoz']:
    def trainer(model, optimizer, criterion, epoch):
        if not use_distillation:
            return trainer_helper(model, criterion, optimizer)
        else:
            return trainer_helper_with_distillation(model, teacher_model, alpha, temperature, optimizer)
    kwargs.update({
        'trainer': trainer,
        'optimizer': torch.optim.Adam(model.parameters()),
        'criterion': nn.CrossEntropyLoss()
    })
    if pruner_type == 'slim':
        kwargs['sparsifying_training_epochs'] = 10

    pruner = AGPPruner(model, config_list, **kwargs)
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
    if use_distillation:
        model = run_finetune_distillation(model, teacher_model, alpha, temperature, log)
    else:
        model = run_finetune(model, log)
        
    
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
    train_dataset_for_pruner = EvalDataset('./data/stanford-dogs/Processed/train')
    train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.set_num_threads(16)
    
    for pruner_type in ['l1', 'taylor', 'fpgm']:
        for sparsity in sparsity_list:
            # for n_iter, n_epoch in [(int(sparsity/0.1), 1), (int(sparsity/0.1), 2), (int(sparsity/0.1), 3), (int(sparsity/0.1), 4)]:
            for n_epoch in [1]:
                for n_iter in [2, 4, 8]:
                    for alpha in [0.99, 0.95]:
                        for temperature in [8]:
                            main(sparsity, pruner_type, n_iter, n_epoch, alpha=alpha, temperature=temperature)

    # main(0.4, 'l1', 2, 1)
