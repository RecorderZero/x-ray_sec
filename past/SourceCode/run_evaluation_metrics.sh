#!/bin/bash
export N_SAMPLES=90
export SAMPLE_LABELS="FPGAN FPGAN+Deblur CLF-DDIM CLF-DDIM+Deblur CFG-DDIM CFG-DDIM+Deblur"
export DEBLUR_LABELS="FPGAN+Deblur CLF-DDIM+Deblur CFG-DDIM+Deblur"
export BASE_LABELS="FPGAN CLF-DDIM CFG-DDIM"
# export SAMPLE_LABELS="noise_250_gs_4.0 noise_500_gs_4.0 noise_750_gs_4.0 noise_500_gs_6.0 noise_500_gs_8.0 d_noise_250_gs_4.0 d_noise_500_gs_4.0 d_noise_750_gs_4.0 d_noise_500_gs_6.0 d_noise_500_gs_8.0"


# python scripts/evaluation_metrics.py --npz_paths results/all_test_run/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                  results/all_test_run/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                      --model_labels ${SAMPLE_LABELS} \
#                                      --result_dir results/latest_final_all_test \
#                                      --pred_mask_threshold 0.5 \
#                                      --include_miou True \
#                                      --save_image True \
#                                      --img_filename latest_final_all_test_image \
#                                      --csv_filename latest_final_all_test_metrics \
#                                      --split_size 10

# Pixel-wise AUROC
python scripts/evaluation_metrics.py --npz_paths results/all_test_run/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/all_test_run/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/all_test_run/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${BASE_LABELS} \
                                     --result_dir results/latest_final_all_test \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --save_image False \
                                     --img_filename latest_final_base_all_test_image \
                                     --csv_filename latest_final_base_all_test_metrics

# Pixel-wise AUROC Deblur
python scripts/evaluation_metrics.py --npz_paths results/all_test_run/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/all_test_run/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                                 results/all_test_run/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
                                     --model_labels ${DEBLUR_LABELS} \
                                     --result_dir results/latest_final_all_test \
                                     --pred_mask_threshold 0.5 \
                                     --include_miou True \
                                     --save_image False \
                                     --img_filename latest_final_deblur_all_test_image \
                                     --csv_filename latest_final_deblur_all_test_metrics

# for i in {0..10}; do
#     python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                     results/latest_tif_pleural_11/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                     results/latest_tif_pleural_11/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                         --model_labels ${BASE_LABELS} \
#                                         --result_dir results/latest_tif_pleural_11_specific_cases \
#                                         --pred_mask_threshold 0.5 \
#                                         --include_miou True \
#                                         --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                         --img_filename external_test_set_comparison \
#                                         --csv_filename external_test_set_metrics \
#                                         --data_idx $i
# done

# for i in {0..10}; do
#     python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                     results/latest_tif_pleural_11/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                                     results/latest_tif_pleural_11/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                         --model_labels ${DEBLUR_LABELS} \
#                                         --result_dir results/latest_tif_pleural_11_specific_cases \
#                                         --pred_mask_threshold 0.5 \
#                                         --include_miou True \
#                                         --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                         --img_filename external_test_set_deblur_comparison \
#                                         --csv_filename external_test_set_deblur_metrics \
#                                         --data_idx $i
# done

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_fpgan_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "FPGAN" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename fpgan_base_image \
#                                     --csv_filename fpgan_base_metrics

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "CLF-DDIM" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename ddim_base_image \
#                                     --csv_filename ddim_base_metrics

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "CFG-DDIM" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename cfg_base_image \
#                                     --csv_filename cfg_base_metrics

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_fpgan_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "FPGAN+Deblur" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename fpgan_deblur_image \
#                                     --csv_filename fpgan_deblur_metrics

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_ddim_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "CLF-DDIM+Deblur" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename ddim_deblur_image \
#                                     --csv_filename ddim_deblur_metrics

# python scripts/evaluation_metrics.py --npz_paths results/latest_tif_pleural_11/samples_cfg_deblur_chexpert050000_${N_SAMPLES}x1x256x256.npz \
#                                     --model_labels "CFG-DDIM+Deblur" \
#                                     --result_dir results/latest_tif_pleural_11_june_best \
#                                     --pred_mask_threshold 0.5 \
#                                     --include_miou True \
#                                     --gt_mask_path /workspace/tif_pleural_256_bbox_masks \
#                                     --img_filename cfg_deblur_image \
#                                     --csv_filename cfg_deblur_metrics