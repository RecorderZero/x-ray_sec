#!/bin/bash
# Unconditional DDIM Inference Script
# 與 CFG 版本比較用：移除 Classifier-Free Guidance 引導

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 11 --timestep_respacing ddim1000 --use_ddim True"

export TRAIN_DATE="v1_2025_05_08"

# ============================================================================
# Unconditional DDIM (No CFG Guidance)
# guidance_scale=0: CFG 公式變成 eps = eps_cond，移除引導增強
# ============================================================================

# 原始影像 - Unconditional DDIM
python scripts/cfg_image_sample.py \
    --data_dir data/tif_pleural/testing \
    --result_dir "results/tif_pleural_uncond_ddim" \
    --model_path results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt \
    --dataset chexpert \
    --noise_level 500 \
    --guidance_scale 0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# Deblur 影像 - Unconditional DDIM
python scripts/cfg_image_sample.py \
    --data_dir data/tif_pleural_deblur/testing \
    --result_dir "results/Model/tif_pleural_deblur_uncond_ddim" \
    --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt \
    --dataset chexpert \
    --noise_level 500 \
    --guidance_scale 0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

echo "Unconditional DDIM inference complete!"
echo "Results saved to:"
echo "  - results/tif_pleural_uncond_ddim"
echo "  - results/tif_pleural_deblur_uncond_ddim"
