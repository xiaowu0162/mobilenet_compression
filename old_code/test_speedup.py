import torch
from torchvision.models import mobilenet_v2
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner

from utils import *

model = mobilenet_v2(pretrained=True)
dummy_input  = torch.rand(8, 3, 416, 416)
count_flops(model)

#cfg_list = [{'op_names':['features.2.conv.0.0', 'features.2.conv.1.0', 'features.2.conv.2'], 'sparsity':0.5}]
#cfg_list = [{'op_names':['features.2.conv.0.0', 'features.2.conv.1.0'], 'sparsity':0.5}]
#cfg_list = [{'op_names':['features.{}.conv.0.0'.format(x) for x in range(2, 18)], 'sparsity':0.5}]
#cfg_list = [{'op_names':['features.{}.conv.1.0'.format(x) for x in range(2, 18)], 'sparsity':0.5}]
cfg_list = [{'op_names':['features.{}.conv.2'.format(x) for x in range(2, 18)], 'sparsity':0.5}]
# cfg_list = [{'op_types':['Conv2d'], 'sparsity':0.5}]

pruner = L1FilterPruner(model, cfg_list)
pruner.compress()
pruner.export_model('./model', './mask')
# need call _unwrap_model if you want run the speedup on the same model
pruner._unwrap_model()

# Speedup the nanodet
ms = ModelSpeedup(model, dummy_input, './mask')
ms.speedup_model()

count_flops(model)

model(dummy_input)

