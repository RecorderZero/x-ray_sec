#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 3 --timestep_respacing ddim1000 --use_ddim True --data_filter False"
export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"
export DECRYPTO_FLAGS="--anonymization_mode deanonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"

export TRAIN_DATE="v1_2025_05_08"
export DECRYPTO_DATE="2026_01_24"

# python scripts/cfg_image_sample.py --data_dir /workspace/CheXpert-v1.0 --result_dir "results/cfg_chexpert_all_test_${TRAIN_DATE}/inference_logs_gs_${GUIDANCE_SCALE}_noise_500" --model_path results/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
# python scripts/cfg_image_sample.py --data_dir /workspace/CheXpert-v1.0_deblur --result_dir "results/deblur_cfg_chexpert_all_test_${TRAIN_DATE}/inference_logs_gs_${GUIDANCE_SCALE}_noise_500" --model_path results/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale $GUIDANCE_SCALE $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

#################################CFG_DDIM
# No_Deblur
# python scripts/cfg_image_sample.py --data_dir data/tif_pleural/testing --result_dir "results/tif_pleural_inference" --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir data/tif_pleural/testing --result_dir results/tif_pleural_inference_crypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_cfg_pleural_inference_decrypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

# Deblur
# python scripts/cfg_image_sample_anonymization.py --data_dir data/tif_pleural_deblur/testing --result_dir results/tif_pleural_deblur_inference_crypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_deblur_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_cfg_pleural_deblur_inference_decrypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

#################################Uncond_DDIM
# No_Deblur
python scripts/cfg_image_sample_anonymization.py --data_dir data/tif_pleural/testing --result_dir results/tif_uncond_pleural_inference_crypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_uncond_pleural_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_uncond_pleural_inference_decrypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

# Deblur
python scripts/cfg_image_sample_anonymization.py --data_dir data/tif_pleural_deblur/testing --result_dir results/tif_uncond_pleural_deblur_inference_crypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_uncond_pleural_deblur_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_uncond_pleural_deblur_inference_decrypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

###############################################################
# cfg_sample 90_Crypto
# python scripts/cfg_image_sample_anonymization.py --data_dir data/CheXpert-v1.0 --result_dir results/tif_pleural_inference_crypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir data/CheXpert-v1.0_deblur --result_dir results/tif_pleural_deblur_inference_crypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS

# cfg_sample 90_DeCrypto
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_cfg_pleural_inference_decrypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_deblur_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_cfg_pleural_deblur_inference_decrypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 4.0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

# uncond_sample_90_Crypto
# python scripts/cfg_image_sample_anonymization.py --data_dir data/CheXpert-v1.0 --result_dir results/tif_pleural_uncond_inference_crypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir data/CheXpert-v1.0_deblur --result_dir results/tif_pleural_uncond_deblur_inference_crypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS

# uncond_sample_90_DeCrypto
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_uncond_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_uncond_pleural_inference_decrypto --model_path results/Model/cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS
# python scripts/cfg_image_sample_anonymization.py --data_dir results/tif_pleural_uncond_deblur_inference_crypto_${DECRYPTO_DATE} --result_dir results/tif_uncond_pleural_deblur_inference_decrypto --model_path results/Model/deblur_cfg_chexpert_p_uncond_0.1_${TRAIN_DATE}/modelchexpert050000.pt --dataset chexpert --noise_level 500 --guidance_scale 0 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS
