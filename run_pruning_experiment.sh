#!/bin/bash


for n_iter in 32 64 128; do
    python pruning_experiments.py \
	   --experiment_dir pretrained_mobilenet_v2_torchhub/ \
	   --checkpoint_name 'checkpoint_best.pt' \
	   --pruner_name agp \
	   --agp_n_iters ${n_iter} \
	   --agp_pruning_alg slim \
	   --speed_up \
	   --kd
done
