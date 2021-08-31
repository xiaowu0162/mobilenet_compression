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
    ActivationAPoZRankFilterPruner
)

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = False                     # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = './experiments/pretrained_mobilenet_v2_best/'
checkpoint = experiment_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120

pruner_type_to_class = {'level': LevelPruner,
                        'l1': L1FilterPruner,
                        'l2': L2FilterPruner,
                        'slim': SlimPruner,
                        'fpgm': FPGMPruner,
                        'taylor': TaylorFOWeightFilterPruner,
                        'agp': AGPPruner,
                        'activationmeanrank': ActivationMeanRankFilterPruner,
                        'apoz': ActivationAPoZRankFilterPruner}

def main():
    log = open('model_size.csv', 'a')
    log.write('description,flops,params\n')
    for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pruner_type = 'l1'
            
        model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                             input_size=input_size, checkpoint=checkpoint)
        model = model.to(device)
        print(model)
            
        # pruning
        config_list = [{
            'op_names': ['features.{}.conv.1.0'.format(x) for x in range(2, 18)],
            'sparsity': sparsity                    
        }]
            
        pruner = pruner_type_to_class[pruner_type](model, config_list)
        pruner.compress()
        pruner.export_model('./model_tmp.pth', './mask_tmp.pth')
            
        # model speedup
        dummy_input = torch.rand(1,3,224,224).cuda()
        pruner._unwrap_model()
        ms = ModelSpeedup(model, dummy_input, './mask_tmp.pth')
        ms.speedup_model()

        flops, params = count_flops(model)
        log.write('conv.1.0 pruned with sparsity {},{},{}\n'.format(sparsity,flops, params))
        log.flush()
        
        # clean up
        filePaths = ['./model_tmp.pth', './mask_tmp.pth']
        for f in filePaths:
            if os.path.exists(f):
                os.remove(f)
                
    log.close()


if __name__ == '__main__':
    main()
