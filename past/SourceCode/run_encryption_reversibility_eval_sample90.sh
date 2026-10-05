#!/bin/bash
#===============================================================================
# 加解密可逆性評估腳本
# 
# 目的：評估三個模型（CFG-DDIM、CLF-DDIM、Uncond-DDIM）
#       在兩種加密方案（Rademacher、Signed Permutation）下的可逆性
#       
# 比較方式：Base 模型輸出 vs 加解密後輸出
#
# 評估指標：
#   - PSNR: 峰值信噪比（越高越好，理想為 ∞）
#   - SSIM: 結構相似性（越接近 1 越好）
#   - FID:  分布距離（越低越好，理想為 0）
#   - MAE:  平均絕對誤差（越低越好，理想為 0）
#   - MSE:  均方誤差（越低越好，理想為 0）
#   - Cosine Similarity: 餘弦相似度（越接近 1 越好）
#
# 數學公式：
# ┌─────────────────────────────────────────────────────────────────┐
# │ 流程: x₀ → x_T → Encrypt → Decrypt → x_T' → x_rec             │
# │                                                                 │
# │ 可逆性驗證: x_T ≈ x_T'                                         │
# │             Base_Output ≈ Decrypted_Output                     │
# │                                                                 │
# │ Rademacher:                                                     │
# │   D(E(x_T)) = k ⊙ (k ⊙ x_T) = x_T  ✓                          │
# │                                                                 │
# │ Signed Permutation:                                             │
# │   D(E(x_T)) = r ⊙ π⁻¹(π(r ⊙ x_T)) = r ⊙ r ⊙ x_T = x_T  ✓     │
# └─────────────────────────────────────────────────────────────────┘
#===============================================================================

set -e  # 遇到錯誤時停止

#-------------------------------------------------------------------------------
# 配置參數 (請根據實際情況修改)
#-------------------------------------------------------------------------------
export N_SAMPLES=90
export RESULT_DIR="results/encryption_reversibility_evaluation"

#-------------------------------------------------------------------------------
# NPZ 檔案路徑配置
# *** 請根據實際生成的 .npz 檔案路徑修改以下變數 ***
#-------------------------------------------------------------------------------

# Base 模型輸出目錄（無加密）
NPZ_BASE_DIR="results/Sample90/all_test_run"

# Rademacher 加解密後輸出目錄
NPZ_RAD_DIR="results/Sample90/rademacher_test_run"

# Signed Permutation 加解密後輸出目錄
NPZ_SP_DIR="results/Sample90/signed_perm_test_run"

#-------------------------------------------------------------------------------
# NPZ 檔案路徑定義_NO Deblur
#-------------------------------------------------------------------------------
# Base 模型 NPZ
NPZ_CFG_BASE="${NPZ_BASE_DIR}/samples_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_CLF_BASE="${NPZ_BASE_DIR}/samples_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_UNCOND_BASE="${NPZ_BASE_DIR}/samples_uncond_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"

# Rademacher 解密後 NPZ
NPZ_CFG_RAD="${NPZ_RAD_DIR}/samples_cfg_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_CLF_RAD="${NPZ_RAD_DIR}/samples_ddim_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_UNCOND_RAD="${NPZ_RAD_DIR}/samples_uncond_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"

# Signed Permutation 解密後 NPZ
NPZ_CFG_SP="${NPZ_SP_DIR}/samples_cfg_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_CLF_SP="${NPZ_SP_DIR}/samples_ddim_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"
NPZ_UNCOND_SP="${NPZ_SP_DIR}/samples_uncond_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"

#-------------------------------------------------------------------------------
# NPZ 檔案路徑定義 Deblur
#-------------------------------------------------------------------------------
# Base 模型 NPZ
# NPZ_CFG_BASE="${NPZ_BASE_DIR}/samples_deblur_cfg_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_CLF_BASE="${NPZ_BASE_DIR}/samples_deblur_ddim_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_UNCOND_BASE="${NPZ_BASE_DIR}/samples_deblur_uncond_base_chexpert050000_${N_SAMPLES}x1x256x256.npz"

# # Rademacher 解密後 NPZ
# NPZ_CFG_RAD="${NPZ_RAD_DIR}/samples_deblur_cfg_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_CLF_RAD="${NPZ_RAD_DIR}/samples_deblur_ddim_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_UNCOND_RAD="${NPZ_RAD_DIR}/samples_deblur_uncond_rademacher_chexpert050000_${N_SAMPLES}x1x256x256.npz"

# # Signed Permutation 解密後 NPZ
# NPZ_CFG_SP="${NPZ_SP_DIR}/samples_deblur_cfg_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_CLF_SP="${NPZ_SP_DIR}/samples_deblur_ddim_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"
# NPZ_UNCOND_SP="${NPZ_SP_DIR}/samples_deblur_uncond_signed_perm_chexpert050000_${N_SAMPLES}x1x256x256.npz"

#===============================================================================
# 開始評估
#===============================================================================
echo "============================================================"
echo "  加解密可逆性評估"
echo "============================================================"
echo ""
echo "Base 模型 NPZ:"
echo "  CFG-DDIM:   ${NPZ_CFG_BASE}"
echo "  CLF-DDIM:   ${NPZ_CLF_BASE}"
echo "  Uncond-DDIM: ${NPZ_UNCOND_BASE}"
echo ""
echo "Rademacher 解密後 NPZ:"
echo "  CFG-DDIM:   ${NPZ_CFG_RAD}"
echo "  CLF-DDIM:   ${NPZ_CLF_RAD}"
echo "  Uncond-DDIM: ${NPZ_UNCOND_RAD}"
echo ""
echo "Signed Permutation 解密後 NPZ:"
echo "  CFG-DDIM:   ${NPZ_CFG_SP}"
echo "  CLF-DDIM:   ${NPZ_CLF_SP}"
echo "  Uncond-DDIM: ${NPZ_UNCOND_SP}"
echo ""

mkdir -p ${RESULT_DIR}

#-------------------------------------------------------------------------------
# 執行可逆性評估（同時評估兩種加密方案）
#-------------------------------------------------------------------------------
echo "執行完整可逆性評估..."
echo ""

python scripts/evaluation_metrics.py \
    --eval_mode reversibility \
    --result_dir ${RESULT_DIR} \
    --base_npz_paths ${NPZ_CFG_BASE} ${NPZ_CLF_BASE} ${NPZ_UNCOND_BASE} \
    --rademacher_npz_paths ${NPZ_CFG_RAD} ${NPZ_CLF_RAD} ${NPZ_UNCOND_RAD} \
    --signed_perm_npz_paths ${NPZ_CFG_SP} ${NPZ_CLF_SP} ${NPZ_UNCOND_SP} \
    --include_roc True \
#     --include_anomaly_roc True \
#     --gt_mask_path /workspace/CheXlocalize/256_segmentations_pleural_effusion

#===============================================================================
# 額外評估：僅 Rademacher（如果只想評估單一方案）
#===============================================================================
# echo ""
# echo "[Optional] 僅評估 Rademacher..."
# 
# python scripts/evaluation_metrics.py \
#     --eval_mode reversibility \
#     --result_dir ${RESULT_DIR}/rademacher_only \
#     --base_npz_paths ${NPZ_CFG_BASE} ${NPZ_CLF_BASE} ${NPZ_UNCOND_BASE} \
#     --rademacher_npz_paths ${NPZ_CFG_RAD} ${NPZ_CLF_RAD} ${NPZ_UNCOND_RAD}

#===============================================================================
# 額外評估：僅 Signed Permutation
#===============================================================================
# echo ""
# echo "[Optional] 僅評估 Signed Permutation..."
# 
# python scripts/evaluation_metrics.py \
#     --eval_mode reversibility \
#     --result_dir ${RESULT_DIR}/signed_perm_only \
#     --base_npz_paths ${NPZ_CFG_BASE} ${NPZ_CLF_BASE} ${NPZ_UNCOND_BASE} \
#     --signed_perm_npz_paths ${NPZ_CFG_SP} ${NPZ_CLF_SP} ${NPZ_UNCOND_SP}

#===============================================================================
# 生成總結報告
#===============================================================================
echo ""
echo "============================================================"
echo "  生成總結報告"
echo "============================================================"

SUMMARY_FILE="${RESULT_DIR}/reversibility_evaluation_report.md"

cat > ${SUMMARY_FILE} << 'EOF'
# 加解密可逆性評估報告

## 1. 評估目的

驗證加解密操作的可逆性，確保：
$$\mathcal{D}(\mathcal{E}(x_T)) = x_T$$

## 2. 評估架構

```
原始影像 x₀
    │
    ▼ DDIM Forward
潛在空間 x_T ─────────────────────────────────────┐
    │                                             │
    ├── [Base] ──────────────────────────────┐   │
    │       直接 DDIM Backward               │   │
    │              │                         │   │
    │              ▼                         │   │
    │       Base Output                      │   │
    │                                        │   │
    ├── [Rademacher]                         │   │
    │       加密: k ⊙ x_T                    │   │
    │              │                         │   │
    │       解密: k ⊙ (k ⊙ x_T) = x_T       │   │
    │              │                         │   │
    │       DDIM Backward                    │   │
    │              │                         │   │
    │              ▼                         │   │
    │       Rademacher Output                │   │
    │                                        │   │
    └── [Signed Permutation]                 │   │
            加密: π(r ⊙ x_T)                 │   │
                   │                         │   │
            解密: r ⊙ π⁻¹(...) = x_T        │   │
                   │                         │   │
            DDIM Backward                    │   │
                   │                         │   │
                   ▼                         │   │
            SignedPerm Output                │   │
                                             │   │
                                             ▼   │
                            比較: Base ≈ Encrypted_Decrypted
```

## 3. 評估模型

| 模型 | 說明 | 特點 |
|------|------|------|
| CFG-DDIM | Classifier-Free Guidance | 無分類器引導，訓練時隨機丟棄條件 |
| CLF-DDIM | Classifier Guidance | 使用獨立分類器的梯度進行引導 |
| Uncond-DDIM | Unconditional | 無任何引導，純粹的擴散過程 |

## 4. 加密方案

### 4.1 Rademacher 加密

**密鑰生成：**
$$k_i = 2 \cdot \text{Bernoulli}(0.5) - 1 \in \{-1, +1\}$$

**加密/解密（自逆）：**
$$\mathcal{E}_k(x) = \mathcal{D}_k(x) = k \odot x$$

**可逆性證明：**
$$\mathcal{D}_k(\mathcal{E}_k(x)) = k \odot (k \odot x) = (k \odot k) \odot x = 1 \odot x = x$$

### 4.2 Signed Permutation 加密

**密鑰結構：**
$$\mathcal{K} = (r, \pi), \quad r \in \{-1,+1\}^d, \pi \in S_d$$

**加密：**
$$\mathcal{E}_{(r,\pi)}(x) = \pi(r \odot x)$$

**解密：**
$$\mathcal{D}_{(r,\pi)}(x) = r \odot \pi^{-1}(x)$$

**可逆性證明：**
$$\mathcal{D}(\mathcal{E}(x)) = r \odot \pi^{-1}(\pi(r \odot x)) = r \odot (r \odot x) = x$$

## 5. 評估指標

| 指標 | 公式 | 理想值 | 說明 |
|------|------|--------|------|
| PSNR | $10 \log_{10}\frac{MAX^2}{MSE}$ | ∞ | 峰值信噪比 |
| SSIM | 結構相似性 | 1.0 | 結構一致性 |
| FID | Fréchet距離 | 0 | 分布差異 |
| MAE | $\frac{1}{n}\sum\|x - \hat{x}\|$ | 0 | 平均絕對誤差 |
| MSE | $\frac{1}{n}\sum(x - \hat{x})^2$ | 0 | 均方誤差 |
| Cosine Sim | $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\|\mathbf{a}\|\| \times \|\|\mathbf{b}\|\|}$ | 1.0 | 餘弦相似度 |

## 6. 預期結果

### 理想情況（完美可逆）：
- PSNR → ∞（或非常高，如 >40 dB）
- SSIM → 1.0
- FID → 0
- MAE → 0
- MSE → 0
- Cosine Similarity → 1.0

### 實際情況：
由於浮點數精度限制，可能存在微小誤差，但應該：
- PSNR > 30 dB
- SSIM > 0.99
- Cosine Similarity > 0.9999
- MAE < 1e-6

## 7. 結果文件

```
results/encryption_reversibility_evaluation/
├── reversibility_metrics.csv                    # 量化指標（含 Cosine Similarity）
├── reversibility_summary.png                    # 指標條形圖總結（6 個子圖）
├── reversibility_roc_curves.png                 # 可逆性 ROC 曲線
├── anomaly_detection_roc_comparison.png         # 異常檢測 ROC 比較
├── anomaly_detection_auroc.csv                  # 異常檢測 AUROC 數據
├── roc_curves_summary.png                       # 綜合 ROC 曲線總結
├── roc_auc_summary.csv                          # 所有 ROC AUC 數據
│
├── # 個別模型比較圖（每個模型 + 加密方案一張）
├── reversibility_CFG-DDIM_rademacher_comparison.png
├── reversibility_CFG-DDIM_signed_perm_comparison.png
├── reversibility_CLF-DDIM_rademacher_comparison.png
├── reversibility_CLF-DDIM_signed_perm_comparison.png
├── reversibility_Uncond-DDIM_rademacher_comparison.png
├── reversibility_Uncond-DDIM_signed_perm_comparison.png
│
├── # 跨模型比較圖（每種加密方案，比較三個模型）
├── reversibility_rademacher_cross_model_comparison.png      # Rademacher 跨模型交錯排列
├── reversibility_rademacher_cross_model_with_diff.png       # Rademacher 跨模型含差異圖
├── reversibility_rademacher_base_first_comparison.png       # Rademacher Base優先排列
├── reversibility_signed_perm_cross_model_comparison.png     # Signed Perm 跨模型交錯排列
├── reversibility_signed_perm_cross_model_with_diff.png      # Signed Perm 跨模型含差異圖
├── reversibility_signed_perm_base_first_comparison.png      # Signed Perm Base優先排列
│
└── reversibility_evaluation_report.md           # 本報告
```

## 8. 結論判讀

- 若所有指標接近理想值 → **加解密具有良好可逆性**
- 若 PSNR/SSIM/Cosine Similarity 下降明顯 → **存在可逆性問題，需檢查加解密實現**
- 若不同模型表現差異大 → **模型特性影響可逆性**

EOF

echo "報告已生成: ${SUMMARY_FILE}"

#===============================================================================
# 完成
#===============================================================================
echo ""
echo "============================================================"
echo "  評估完成！"
echo "============================================================"
echo ""
echo "結果目錄: ${RESULT_DIR}"
echo ""
echo "生成的文件:"
ls -la ${RESULT_DIR}/ 2>/dev/null || echo "  (尚未生成文件，請確認 NPZ 路徑正確)"
echo ""
echo "可逆性評估說明："
echo "  - PSNR 越高越好（理想為無窮大）"
echo "  - SSIM 越接近 1 越好"
echo "  - Cosine Similarity 越接近 1 越好"
echo "  - MAE/MSE 越接近 0 越好"
echo "  - 若 Base ≈ Decrypted，表示加解密具有良好可逆性"
echo ""
