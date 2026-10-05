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
| MAE | $\frac{1}{n}\sum|x - \hat{x}|$ | 0 | 平均絕對誤差 |
| MSE | $\frac{1}{n}\sum(x - \hat{x})^2$ | 0 | 均方誤差 |

## 6. 預期結果

### 理想情況（完美可逆）：
- PSNR → ∞（或非常高，如 >40 dB）
- SSIM → 1.0
- FID → 0
- MAE → 0
- MSE → 0

### 實際情況：
由於浮點數精度限制，可能存在微小誤差，但應該：
- PSNR > 30 dB
- SSIM > 0.99
- MAE < 1e-6

## 7. 結果文件

```
results/encryption_reversibility_evaluation/
├── reversibility_metrics.csv                    # 量化指標（含 AUROC）
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
├── # 跨模型比較圖（每種加密方案一張，比較三個模型）
├── reversibility_rademacher_cross_model_comparison.png      # Rademacher 跨模型簡潔版
├── reversibility_rademacher_cross_model_with_diff.png       # Rademacher 跨模型含差異圖
├── reversibility_signed_perm_cross_model_comparison.png     # Signed Perm 跨模型簡潔版
├── reversibility_signed_perm_cross_model_with_diff.png      # Signed Perm 跨模型含差異圖
│
└── reversibility_evaluation_report.md           # 本報告
```

## 8. 結論判讀

- 若所有指標接近理想值 → **加解密具有良好可逆性**
- 若 PSNR/SSIM 下降明顯 → **存在可逆性問題，需檢查加解密實現**
- 若不同模型表現差異大 → **模型特性影響可逆性**

