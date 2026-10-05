#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 10 --timestep_respacing ddim1000 --use_ddim True"

export GUIDANCE_SCALE=4.0
export TRAIN_DATE="2025_05_03"

for i in 500; do
    python scripts/cfg_image_sample.py --data_dir data/chexpert_new_sub/testing --result_dir "results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/inference_logs_gs_${GUIDANCE_SCALE}_noise_${i}" --model_path results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level $i --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
done

for i in 500; do
    python scripts/cfg_image_sample.py --data_dir data/chexpert_new_deblur_sub/testing --result_dir "results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/inference_logs_gs_${GUIDANCE_SCALE}_noise_${i}" --model_path results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level $i --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
done