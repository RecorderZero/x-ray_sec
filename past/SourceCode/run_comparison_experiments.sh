#!/bin/bash
# =============================================================================
# Anonymization Algorithm Comparison Experiments
# =============================================================================
# This script runs the same anonymization/de-anonymization experiments using
# three different sampling methods to compare the effectiveness of the
# encryption/decryption algorithm.
#
# Methods compared:
# 1. CFG (Classifier-Free Guidance)
# 2. CLF (Classifier Guidance)
# 3. Unconditional DDIM (No Guidance) - Baseline
#
# The encryption/decryption operates at x_T level, so results should be
# consistent across all methods, proving the algorithm's robustness.
# =============================================================================

# Configuration
DATA_DIR="data/tif_pleural/testing"
MODEL_PATH="results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_11_27/modelchexpert050000.pt"
# CLASSIFIER_PATH="checkpoints/classifier.pt"  # Only for CLF method
RESULT_BASE="results/ddim_comparison_experiments"
DATASET="chexpert"
NUM_SAMPLES=2
NOISE_LEVEL=500
BATCH_SIZE=1

# Encryption settings
KEY_PASSWORD="Ki@13579"
USE_SIGNED_PERMUTATION="--use_signed_permutation True"  # Comment out to use basic Rademacher

# Model architecture parameters (from your training config)
export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --timestep_respacing ddim1000 --use_ddim True"

# Timestamp for this experiment run
TIMESTAMP=$(date +"%Y%m%d")
EXPERIMENT_DIR="${RESULT_BASE}"

echo "=============================================="
echo "Anonymization Algorithm Comparison Experiments"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Results directory: ${EXPERIMENT_DIR}"
echo "Key password: ${KEY_PASSWORD}"
echo "Using signed permutation: ${USE_SIGNED_PERMUTATION}"
echo ""

# Create experiment directory
mkdir -p ${EXPERIMENT_DIR}

# =============================================================================
# Experiment 1: Unconditional DDIM (Baseline)
# =============================================================================
echo ""
echo "=========================================="
echo "Experiment 1: Unconditional DDIM (Baseline)"
echo "=========================================="

# 1.1 Anonymization
echo "[1.1] Unconditional DDIM - Anonymization"
python scripts/unconditional_ddim_anonymization.py \
    --data_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --result_dir ${EXPERIMENT_DIR}/unconditional \
    --dataset ${DATASET} \
    --noise_level ${NOISE_LEVEL} \
    --anonymization_mode anonymize \
    --anonymization_key_password "${KEY_PASSWORD}" \
    ${USE_SIGNED_PERMUTATION} \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# 1.2 De-anonymization
echo "[1.2] Unconditional DDIM - De-anonymization"
python scripts/unconditional_ddim_anonymization.py \
    --data_dir results/ddim_comparison_experiments/unconditional/unconditional_anonymize_2026_01_04/anonymized \
    --model_path ${MODEL_PATH} \
    --result_dir ${EXPERIMENT_DIR}/unconditional_deano \
    --dataset ${DATASET} \
    --noise_level ${NOISE_LEVEL} \
    --anonymization_mode deanonymize \
    --anonymization_key_password "${KEY_PASSWORD}" \
    ${USE_SIGNED_PERMUTATION} \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# # =============================================================================
# # Experiment 2: CFG (Classifier-Free Guidance)
# # =============================================================================
# echo ""
# echo "=========================================="
# echo "Experiment 2: CFG (Classifier-Free Guidance)"
# echo "=========================================="

# # 2.1 Anonymization
# echo "[2.1] CFG - Anonymization"
# python scripts/cfg_image_sample_anonymization.py \
#     --data_dir ${DATA_DIR} \
#     --model_path ${MODEL_PATH} \
#     --result_dir ${EXPERIMENT_DIR}/cfg \
#     --dataset ${DATASET} \
#     --noise_level ${NOISE_LEVEL} \
#     --guidance_scale 4.0 \
#     --anonymization_mode anonymize \
#     --anonymization_key_password "${KEY_PASSWORD}" \
#     ${USE_SIGNED_PERMUTATION} \
#     $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# # 2.2 De-anonymization + Anomaly Detection
# echo "[2.2] CFG - De-anonymization + Anomaly Detection"
# python scripts/cfg_image_sample_anonymization.py \
#     --data_dir ${EXPERIMENT_DIR}/cfg/*/anonymized \
#     --model_path ${MODEL_PATH} \
#     --result_dir ${EXPERIMENT_DIR}/cfg_deano \
#     --dataset ${DATASET} \
#     --noise_level ${NOISE_LEVEL} \
#     --guidance_scale 4.0 \
#     --anonymization_mode deanonymize \
#     --anonymization_key_password "${KEY_PASSWORD}" \
#     ${USE_SIGNED_PERMUTATION} \
#     $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# # =============================================================================
# # Experiment 3: Classifier Guidance
# # =============================================================================
# echo ""
# echo "=========================================="
# echo "Experiment 3: Classifier Guidance"
# echo "=========================================="

# # 3.1 Anonymization
# echo "[3.1] Classifier Guidance - Anonymization"
# python scripts/classifier_sample_known_anonymization.py \
#     --data_dir ${DATA_DIR} \
#     --model_path ${MODEL_PATH} \
#     --classifier_path ${CLASSIFIER_PATH} \
#     --result_dir ${EXPERIMENT_DIR}/classifier \
#     --dataset ${DATASET} \
#     --noise_level ${NOISE_LEVEL} \
#     --classifier_scale 100 \
#     --anonymization_mode anonymize \
#     --anonymization_key_password "${KEY_PASSWORD}" \
#     ${USE_SIGNED_PERMUTATION} \
#     $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# # 3.2 De-anonymization + Anomaly Detection
# echo "[3.2] Classifier Guidance - De-anonymization + Anomaly Detection"
# python scripts/classifier_sample_known_anonymization.py \
#     --data_dir ${EXPERIMENT_DIR}/classifier/*/anonymized \
#     --model_path ${MODEL_PATH} \
#     --classifier_path ${CLASSIFIER_PATH} \
#     --result_dir ${EXPERIMENT_DIR}/classifier_deano \
#     --dataset ${DATASET} \
#     --noise_level ${NOISE_LEVEL} \
#     --classifier_scale 100 \
#     --anonymization_mode deanonymize \
#     --anonymization_key_password "${KEY_PASSWORD}" \
#     ${USE_SIGNED_PERMUTATION} \
#     $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# # =============================================================================
# # Summary
# # =============================================================================
# echo ""
# echo "=============================================="
# echo "Experiment Complete!"
# echo "=============================================="
# echo ""
# echo "Results saved to: ${EXPERIMENT_DIR}"
# echo ""
# echo "Directory structure:"
# echo "  ${EXPERIMENT_DIR}/"
# echo "  ├── unconditional/     # Baseline (no guidance)"
# echo "  │   ├── *_anonymize/   # Anonymization results"
# echo "  │   └── *_deanonymize/ # De-anonymization results"
# echo "  ├── cfg/               # CFG guidance"
# echo "  │   ├── *_anonymize/   # Anonymization results"
# echo "  │   └── *_deanonymize/ # De-anonymization + anomaly detection"
# echo "  └── classifier/        # Classifier guidance"
# echo "      ├── *_anonymize/   # Anonymization results"
# echo "      └── *_deanonymize/ # De-anonymization + anomaly detection"
# echo ""
# echo "Comparison metrics to analyze:"
# echo "  1. Anonymization quality (visual inspection)"
# echo "  2. De-anonymization accuracy (MSE, SSIM, PSNR)"
# echo "  3. Key sensitivity (wrong key test)"
# echo "  4. Distribution preservation (histogram analysis)"
# echo ""
