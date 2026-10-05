#!/bin/bash

export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond -1 --clf_free False"
export SAMPLE_FLAGS="--batch_size 1 --num_samples 3 --timestep_respacing ddim1000 --use_ddim True --clf_free False --data_filter False"
export CLASSIFIER_FLAGS="--image_size 256 --in_channels 1 --out_channels 2 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_password Ki@13579 --use_signed_permutation True"
export DECRYPTO_FLAGS="--anonymization_mode deanonymize --anonymization_key_password Ki@13579 --use_signed_permutation True"


export TRAIN_DATE="2025_12_27"
export DECRYPTO_DATE="2026_01_24"

######################
# Without Deblur
python scripts/classifier_sample_known_anonymization.py --data_dir data/tif_pleural/testing --model_path results/Model/ddim_base_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_base_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_inference_crypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
python scripts/classifier_sample_known_anonymization.py --data_dir results/clf_tif_pleural_inference_crypto_${DECRYPTO_DATE} --model_path results/Model/ddim_base_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_base_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_inference_decrypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

# With Deblur
# python scripts/classifier_sample_known_anonymization.py --data_dir data/tif_pleural_deblur/testing --model_path results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_deblur_inference_crypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/classifier_sample_known_anonymization.py --data_dir results/clf_tif_pleural_deblur_inference_crypto_${DECRYPTO_DATE} --model_path results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_deblur_inference_decrypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS

######################
# sample 90_Crypto
# python scripts/classifier_sample_known_anonymization.py --data_dir data/CheXpert-v1.0 --model_path results/Model/ddim_base_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_base_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_inference_crypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# python scripts/classifier_sample_known_anonymization.py --data_dir data/CheXpert-v1.0_deblur --model_path results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_deblur_inference_crypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $CRYPTO_FLAGS
# sample 90_decrypto
# python scripts/classifier_sample_known_anonymization.py --data_dir results/clf_tif_pleural_inference_crypto_${DECRYPTO_DATE} --model_path results/Model/ddim_base_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_base_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_inference_decrypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS
# python scripts/classifier_sample_known_anonymization.py --data_dir results/clf_tif_pleural_deblur_inference_crypto_${DECRYPTO_DATE} --model_path results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpert050000.pt --classifier_path  results/Model/ddim_deblur_${TRAIN_DATE}/modelchexpertclass020000.pt --result_dir results/clf_tif_pleural_deblur_inference_decrypto --dataset chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $DECRYPTO_FLAGS


