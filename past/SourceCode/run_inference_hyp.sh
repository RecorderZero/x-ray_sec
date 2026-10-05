#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 11 --timestep_respacing ddim1000 --use_ddim True"

export TRAIN_DATE="v1_2025_05_08"

#data/chexpert_old_sub/testing
#data/chexpert_old_deblur_sub/testing

initial=4.0
end=8.0
inc=2.0

for noise_level in 250 500 750; do
    for guidance_scale in $(seq $initial $inc $end); do
        python scripts/cfg_image_sample.py --data_dir data/tif_pleural/testing --result_dir "results/cfg_tif_pleural_all_hyp_${TRAIN_DATE}/inference_logs_noise_${noise_level}_gs_${guidance_scale}" --model_path results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level $noise_level --guidance_scale $guidance_scale $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
        python scripts/cfg_image_sample.py --data_dir data/tif_pleural_deblur/testing --result_dir "results/deblur_cfg_tif_pleural_all_hyp_${TRAIN_DATE}/inference_logs_noise_${noise_level}_gs_${guidance_scale}" --model_path results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level $noise_level --guidance_scale $guidance_scale $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
    done
done

# python scripts/cfg_image_sample.py --data_dir data/tif_pleural/testing --result_dir "results/tif_pleural_inference_gs_${GUIDANCE_SCALE}_noise_500" --model_path results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# python scripts/cfg_image_sample.py --data_dir data/tif_pleural_deblur/testing --result_dir "results/tif_pleural_deblur_inference_gs_${GUIDANCE_SCALE}_noise_500" --model_path results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS