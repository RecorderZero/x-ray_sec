#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond -1 --clf_free False"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 11 --timestep_respacing ddim1000 --use_ddim True --clf_free False"
export CLASSIFIER_FLAGS="--image_size 256 --in_channels 1 --out_channels 2 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

export TRAIN_DATE="2025_12_27"

python scripts/image_train.py --data_dir data/CheXpert-v1.0 --dataset chexpert --result_dir results/ddim_base_${TRAIN_DATE} --lr_anneal_steps 50000 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
python scripts/image_train.py --data_dir data/CheXpert-v1.0_deblur --dataset chexpert --result_dir results/ddim_deblur_${TRAIN_DATE} --lr_anneal_steps 50000 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS