#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 10 --timestep_respacing ddim1000 --use_ddim True"

#export TRAIN_DATE="v1_2025_05_08"
export TRAIN_DATE="v1_2025_11_01"

python scripts/cfg_image_train.py --data_dir data/CheXpert-v1.0 --result_dir results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE} --dataset chexpert --lr_anneal_steps 50000 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
python scripts/cfg_image_train.py --data_dir data/CheXpert-v1.0_deblur --result_dir results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE} --dataset chexpert --lr_anneal_steps 50000 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS