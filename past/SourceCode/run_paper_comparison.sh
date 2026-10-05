#!/bin/bash
# ============================================================================
# Paper Comparison Experiments: Encryption Algorithm Evaluation
# ============================================================================
# This script runs all three diffusion model variants to compare the
# encryption/decryption algorithm (Rademacher vs Signed Permutation)
#
# Experiments:
# 1. CFG DDIM (Classifier-Free Guidance)
# 2. Classifier Guidance DDIM
# 3. Unconditional DDIM (baseline, no guidance)
#
# For each variant, we test:
# - Rademacher key (basic)
# - Signed Permutation key (enhanced)
# ============================================================================

# === Configuration ===
DATA_DIR="data/chexpert/testing"
MODEL_PATH="checkpoints/model.pt"
CLASSIFIER_PATH="checkpoints/classifier.pt"
RESULT_BASE="results/paper_comparison"
DATASET="chexpert"
NOISE_LEVEL=500
NUM_SAMPLES=50
BATCH_SIZE=1
PASSWORD="paper_experiment_key_2024"

# Model architecture parameters (from your training config)
export MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --timestep_respacing ddim1000 --use_ddim True"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${RESULT_BASE}/${TIMESTAMP}"
mkdir -p $RESULT_DIR

echo "=============================================="
echo "Paper Comparison Experiments"
echo "Timestamp: $TIMESTAMP"
echo "Results Directory: $RESULT_DIR"
echo "=============================================="

# ============================================================================
# Experiment 1: Unconditional DDIM (Baseline - No Guidance)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 1: Unconditional DDIM"
echo "=============================================="

# 1.1 Unconditional DDIM + Rademacher Key (Roundtrip)
echo "[1.1] Unconditional DDIM + Rademacher Key (roundtrip)"
python scripts/unconditional_ddim_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/uncond_ddim_rademacher \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode roundtrip \
    --anonymization_key_password "${PASSWORD}_rademacher" \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/uncond_ddim_rademacher.log

# 1.2 Unconditional DDIM + Signed Permutation Key (Roundtrip)
echo "[1.2] Unconditional DDIM + Signed Permutation Key (roundtrip)"
python scripts/unconditional_ddim_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/uncond_ddim_signed_perm \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode roundtrip \
    --anonymization_key_password "${PASSWORD}_signed_perm" \
    --use_signed_permutation \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/uncond_ddim_signed_perm.log

# ============================================================================
# Experiment 2: CFG DDIM (Classifier-Free Guidance)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 2: CFG DDIM"
echo "=============================================="

# 2.1 CFG DDIM + Rademacher Key (Anonymize -> Deanonymize)
echo "[2.1] CFG DDIM + Rademacher Key (anonymize)"
python scripts/cfg_image_sample_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/cfg_ddim_rademacher_ano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode anonymize \
    --anonymization_key_password "${PASSWORD}_rademacher" \
    --guidance_scale 4.0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/cfg_ddim_rademacher_ano.log

echo "[2.1b] CFG DDIM + Rademacher Key (deanonymize)"
python scripts/cfg_image_sample_anonymization.py \
    --data_dir ${RESULT_DIR}/cfg_ddim_rademacher_ano/*/anonymized \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/cfg_ddim_rademacher_deano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode deanonymize \
    --anonymization_key_password "${PASSWORD}_rademacher" \
    --guidance_scale 4.0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/cfg_ddim_rademacher_deano.log

# 2.2 CFG DDIM + Signed Permutation Key
echo "[2.2] CFG DDIM + Signed Permutation Key (anonymize)"
python scripts/cfg_image_sample_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/cfg_ddim_signed_perm_ano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode anonymize \
    --anonymization_key_password "${PASSWORD}_signed_perm" \
    --use_signed_permutation \
    --guidance_scale 4.0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/cfg_ddim_signed_perm_ano.log

echo "[2.2b] CFG DDIM + Signed Permutation Key (deanonymize)"
python scripts/cfg_image_sample_anonymization.py \
    --data_dir ${RESULT_DIR}/cfg_ddim_signed_perm_ano/*/anonymized \
    --model_path $MODEL_PATH \
    --result_dir ${RESULT_DIR}/cfg_ddim_signed_perm_deano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode deanonymize \
    --anonymization_key_password "${PASSWORD}_signed_perm" \
    --use_signed_permutation \
    --guidance_scale 4.0 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/cfg_ddim_signed_perm_deano.log

# ============================================================================
# Experiment 3: Classifier Guidance DDIM
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 3: Classifier Guidance DDIM"
echo "=============================================="

# 3.1 Classifier Guidance + Rademacher Key
echo "[3.1] Classifier Guidance + Rademacher Key (anonymize)"
python scripts/classifier_sample_known_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --classifier_path $CLASSIFIER_PATH \
    --result_dir ${RESULT_DIR}/clf_ddim_rademacher_ano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode anonymize \
    --anonymization_key_password "${PASSWORD}_rademacher" \
    --classifier_scale 100 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/clf_ddim_rademacher_ano.log

echo "[3.1b] Classifier Guidance + Rademacher Key (deanonymize)"
python scripts/classifier_sample_known_anonymization.py \
    --data_dir ${RESULT_DIR}/clf_ddim_rademacher_ano/*/anonymized \
    --model_path $MODEL_PATH \
    --classifier_path $CLASSIFIER_PATH \
    --result_dir ${RESULT_DIR}/clf_ddim_rademacher_deano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode deanonymize \
    --anonymization_key_password "${PASSWORD}_rademacher" \
    --classifier_scale 100 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/clf_ddim_rademacher_deano.log

# 3.2 Classifier Guidance + Signed Permutation Key
echo "[3.2] Classifier Guidance + Signed Permutation Key (anonymize)"
python scripts/classifier_sample_known_anonymization.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --classifier_path $CLASSIFIER_PATH \
    --result_dir ${RESULT_DIR}/clf_ddim_signed_perm_ano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode anonymize \
    --anonymization_key_password "${PASSWORD}_signed_perm" \
    --use_signed_permutation \
    --classifier_scale 100 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/clf_ddim_signed_perm_ano.log

echo "[3.2b] Classifier Guidance + Signed Permutation Key (deanonymize)"
python scripts/classifier_sample_known_anonymization.py \
    --data_dir ${RESULT_DIR}/clf_ddim_signed_perm_ano/*/anonymized \
    --model_path $MODEL_PATH \
    --classifier_path $CLASSIFIER_PATH \
    --result_dir ${RESULT_DIR}/clf_ddim_signed_perm_deano \
    --dataset $DATASET \
    --noise_level $NOISE_LEVEL \
    --anonymization_mode deanonymize \
    --anonymization_key_password "${PASSWORD}_signed_perm" \
    --use_signed_permutation \
    --classifier_scale 100 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS \
    2>&1 | tee ${RESULT_DIR}/clf_ddim_signed_perm_deano.log

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "Experiments Complete!"
echo "=============================================="
echo "Results saved to: $RESULT_DIR"
echo ""
echo "Experiment Summary:"
echo "  1. Unconditional DDIM (baseline)"
echo "     - Rademacher: ${RESULT_DIR}/uncond_ddim_rademacher"
echo "     - Signed Perm: ${RESULT_DIR}/uncond_ddim_signed_perm"
echo ""
echo "  2. CFG DDIM"
echo "     - Rademacher: ${RESULT_DIR}/cfg_ddim_rademacher_*"
echo "     - Signed Perm: ${RESULT_DIR}/cfg_ddim_signed_perm_*"
echo ""
echo "  3. Classifier Guidance DDIM"
echo "     - Rademacher: ${RESULT_DIR}/clf_ddim_rademacher_*"
echo "     - Signed Perm: ${RESULT_DIR}/clf_ddim_signed_perm_*"
echo ""
echo "Check metrics.csv in each directory for quantitative results."
