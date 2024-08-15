#!/bin/sh
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --use_kl True --schedule_sampler loss-second-moment"
NUM_GPUS=4
# shellcheck disable=SC2086
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir /data/Dynamics/cifar_data/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
