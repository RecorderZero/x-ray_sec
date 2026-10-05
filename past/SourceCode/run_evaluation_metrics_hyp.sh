#!/bin/bash
# export SAMPLE_LABELS="DDIM DDIM+Deblur CFG CFG+Deblur FPGAN FPGAN+Deblur"
export SAMPLE_LABELS="noise_250_gs_4.0 noise_500_gs_4.0 noise_750_gs_4.0 noise_500_gs_6.0 noise_500_gs_8.0 d_noise_250_gs_4.0 d_noise_500_gs_4.0 d_noise_750_gs_4.0 d_noise_500_gs_6.0 d_noise_500_gs_8.0"
export LABEL_SET_1="L=250 L=500 L=750"
export LABEL_SET_2="w=4.0 w=6.0 w=8.0"

export N_SAMPLES=11
export INFERENCE_DATE="2025_07_10"

# python scripts/split_evaluation_metrics.py --npz_paths results/all_test_run/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                      --x_labels ${SAMPLE_LABELS} \
#                                      --result_dir results/latest_final_all_test \
#                                      --pred_mask_threshold 0.5 \
#                                      --include_miou True \
#                                      --split_size 10

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/latest_tif_pleural_11/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/latest_tif_pleural_11/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/latest_tif_pleural_11/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/latest_tif_pleural_11/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/latest_tif_pleural_11/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                      --x_labels ${SAMPLE_LABELS} \
#                                      --result_dir results/latest_tif_pleural_11_june \
#                                      --pred_mask_threshold 0.5 \
#                                      --include_miou True \
#                                      --gt_mask_path /workspace/tif_pleural_256_bbox_masks

# python scripts/evaluation_metrics.py --npz_paths results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_250_gs_4.0_${INFERENCE_DATE}/noise_250_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_750_gs_4.0_${INFERENCE_DATE}/noise_750_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_6.0_${INFERENCE_DATE}/noise_500_gs_6.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_8.0_${INFERENCE_DATE}/noise_500_gs_8.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_250_gs_4.0_${INFERENCE_DATE}/noise_250_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_750_gs_4.0_${INFERENCE_DATE}/noise_750_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_6.0_${INFERENCE_DATE}/noise_500_gs_6.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_8.0_${INFERENCE_DATE}/noise_500_gs_8.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                      --x_labels ${SAMPLE_LABELS} \
#                                      --result_dir results/latest_tif_pleural_hyp \
#                                      --pred_mask_threshold 0.5 \
#                                      --include_miou True \
#                                      --gt_mask_path /workspace/tif_pleural_256_bbox_masks

# CFG by noise level
python scripts/evaluation_metrics.py --npz_paths results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_250_gs_4.0_${INFERENCE_DATE}/noise_250_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_750_gs_4.0_${INFERENCE_DATE}/noise_750_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${LABEL_SET_1} \
                                     --result_dir results/latest_tif_pleural_hyp \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
                                     --img_filename cfg_noise_level_image \
                                     --csv_filename cfg_noise_level_metrics \
                                     --save_image False

# CFG by guidance scale
python scripts/evaluation_metrics.py --npz_paths results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_6.0_${INFERENCE_DATE}/noise_500_gs_6.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_8.0_${INFERENCE_DATE}/noise_500_gs_8.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${LABEL_SET_2} \
                                     --result_dir results/latest_tif_pleural_hyp \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
                                     --img_filename cfg_guidance_scale_image \
                                     --csv_filename cfg_guidance_scale_metrics \
                                     --save_image False

# Deblur CFG by noise level
python scripts/evaluation_metrics.py --npz_paths results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_250_gs_4.0_${INFERENCE_DATE}/noise_250_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_750_gs_4.0_${INFERENCE_DATE}/noise_750_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${LABEL_SET_1} \
                                     --result_dir results/latest_tif_pleural_hyp \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
                                     --img_filename deblur_cfg_noise_level_image \
                                     --csv_filename deblur_cfg_noise_level_metrics \
                                     --save_image False

# Deblur CFG by guidance scale
python scripts/evaluation_metrics.py --npz_paths results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_4.0_${INFERENCE_DATE}/noise_500_gs_4.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_6.0_${INFERENCE_DATE}/noise_500_gs_6.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/deblur_cfg_tif_pleural_all_hyp_v1_2025_05_08/inference_logs_noise_500_gs_8.0_${INFERENCE_DATE}/noise_500_gs_8.0_modelchexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${LABEL_SET_2} \
                                     --result_dir results/latest_tif_pleural_hyp \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
                                     --img_filename deblur_cfg_guidance_scale_image \
                                     --csv_filename deblur_cfg_guidance_scale_metrics \
                                     --save_image False