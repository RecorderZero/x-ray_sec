# Secure and Reversible Anonymization Integration Report

## 基於論文 "Secure and Reversible Face Anonymization with Diffusion Models" 的匿名化技術整合報告

### 作者: Pol Labarbarie, Vincent Itier, William Puech
### 整合目標: 胸部X光影像擴散模型異常檢測系統

---

## 1. 概述

本報告詳細說明如何將論文中提出的安全可逆匿名化技術整合到現有的胸部X光影像擴散模型異常檢測系統中。

### 1.1 論文核心概念

論文提出了基於 Rademacher key 的匿名化方法：

1. **Rademacher Key**: `k ∈ {-1, +1}^d`，一個二元符號向量
2. **匿名化公式**: `x_ano_T = M ⊙ (k ⊙ x_T) + (1 - M) ⊙ x_T`
3. **去匿名化公式**: `x_rec_T = M ⊙ (k ⊙ x_ano_T) + (1 - M) ⊙ x_ano_T`
4. **可逆性保證**: 由於 `k ⊙ k = 1`，應用兩次密鑰可完美恢復原始資料

### 1.2 整合方案（像素空間適配版）

由於本專案的 UNet 在**像素空間**操作（不同於論文中使用的 Stable Diffusion 潛在空間 UNet），匿名化流程完全在像素空間進行：

- 實現 Rademacher key 生成與管理
- 實現 DDIM 前向/後向匿名化流程（像素空間）
- 密鑰直接應用於像素空間的噪聲表示 x_T
- 保持原有異常檢測功能不變
- **不需要 VAE**

---

## 2. 新增與修改的檔案

### 2.1 新增檔案

| 檔案路徑 | 說明 |
|---------|------|
| `guided_diffusion/anonymization.py` | Rademacher key 匿名化模組 |
| `scripts/cfg_image_sample_anonymization.py` | 支援匿名化的 CFG 採樣腳本 |
| `scripts/classifier_sample_known_anonymization.py` | 支援匿名化的 Classifier Guidance 採樣腳本 |
| `run_clf_inference_anonymization.sh` | Classifier Guidance 匿名化推論腳本 |

### 2.2 修改檔案

| 檔案路徑 | 修改內容 |
|---------|---------|
| `guided_diffusion/script_util.py` | 新增匿名化相關的配置函數 |
| `guided_diffusion/gaussian_diffusion.py` | 新增匿名化相關的 DDIM 採樣方法 |

---

## 3. 詳細程式碼說明

### 3.1 匿名化模組 (`guided_diffusion/anonymization.py`)

實現論文中的 Rademacher key 匿名化方案。

```python
class RademacherKey:
    """
    Rademacher key 用於安全匿名化。
    產生 {-1, +1} 的值，機率相等。
    
    密鑰可以：
    - 隨機生成
    - 從密碼/種子派生
    - 從檔案載入
    """
    
class DiffusionAnonymizer(nn.Module):
    """
    擴散模型的安全可逆匿名化模組。
    
    在像素空間進行匿名化操作：
    - 密鑰直接應用於 x_T（噪聲表示）
    - 支援選擇性遮罩（全圖、中心、邊緣）
    """
```

**主要功能：**
- `generate_key()`: 生成 Rademacher 密鑰
- `anonymize_latent()`: 匿名化（應用密鑰）
- `deanonymize_latent()`: 去匿名化（再次應用密鑰）
- `AnonymizationMask`: 用於選擇性匿名化的遮罩生成

### 3.2 DDIM 匿名化流程 (`guided_diffusion/gaussian_diffusion.py`)

新增的方法：

```python
def ddim_anonymization_forward(self, model, x_0, noise_level, ...):
    """DDIM 前向過程：x_0 -> x_T"""
    
def ddim_anonymization_backward(self, model, x_T, noise_level, ...):
    """DDIM 後向過程：x_T -> x_0"""
    
def ddim_sample_loop_anonymization(self, model, shape, img, anonymizer, key, mode, ...):
    """
    完整的匿名化/去匿名化流程（像素空間）：
    1. DDIM Forward: x_0 -> x_T
    2. Apply Key: x_T -> x_ano_T (或 x_ano_T -> x_rec_T)
    3. DDIM Backward: x_ano_T -> x_ano (或 x_rec_T -> x_rec)
    """
    
def ddim_sample_loop_anomaly_detection_with_deanonymization(self, ...):
    """去匿名化後進行異常檢測"""
```

---

## 4. 使用方式

### 4.1 原始訓練（無匿名化）

使用 `cfg_image_train.py` 進行標準訓練：

```bash
python scripts/cfg_image_train.py \
    --data_dir data/CheXpert-v1.0 \
    --result_dir results/cfg_chexpert_p_uncond_0.1_v1 \
    --dataset chexpert \
    --lr_anneal_steps 50000 \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --lr 1e-4 \
    --batch_size 4 \
    --p_uncond 0.1 \
    --dropout 0.1
```

### 4.2 原始採樣/推論（無匿名化）

使用 `cfg_image_sample.py` 進行標準異常檢測：

```bash
python scripts/cfg_image_sample.py \
    --data_dir data/tif_pleural/testing \
    --result_dir results/tif_pleural_inference \
    --model_path results/cfg_chexpert_p_uncond_0.1_v1/modelchexpert050000.pt \
    --dataset chexpert \
    --noise_level 500 \
    --guidance_scale 4.0 \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --batch_size 1 \
    --num_samples 11 \
    --timestep_respacing ddim1000 \
    --use_ddim True
```

### 4.3 匿名化採樣

使用 `cfg_image_sample_anonymization.py` 將原始影像匿名化：

```bash
python scripts/cfg_image_sample_anonymization.py \
    --data_dir data/tif_pleural/testing \
    --result_dir results/anonymization_output \
    --model_path results/cfg_chexpert_p_uncond_0.1_v1/modelchexpert050000.pt \
    --dataset chexpert \
    --noise_level 500 \
    --guidance_scale 4.0 \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --batch_size 1 \
    --num_samples 11 \
    --timestep_respacing ddim1000 \
    --use_ddim True \
    --anonymization_mode anonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/my_key.pt
```

### 4.4 去匿名化 + 異常檢測

將匿名化影像去匿名化並進行異常檢測：

```bash
python scripts/cfg_image_sample_anonymization.py \
    --data_dir data/anonymized_images \
    --result_dir results/deanonymization_output \
    --model_path results/cfg_chexpert_p_uncond_0.1_v1/modelchexpert050000.pt \
    --dataset chexpert \
    --noise_level 500 \
    --guidance_scale 4.0 \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --batch_size 1 \
    --num_samples 11 \
    --timestep_respacing ddim1000 \
    --use_ddim True \
    --anonymization_mode deanonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/my_key.pt
```

### 4.5 參數說明

#### 模型參數 (MODEL_FLAGS)

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--unet_version` | UNet 版本 | `v1` |
| `--image_size` | 影像大小 | `256` |
| `--in_channels` | 輸入通道數 | `1` |
| `--num_channels` | 模型通道數 | `128` |
| `--num_classes` | 類別數 | `2` |
| `--class_cond` | 類別條件 | `True` |
| `--num_res_blocks` | 殘差塊數量 | `2` |
| `--num_heads` | 注意力頭數 | `1` |
| `--learn_sigma` | 學習 sigma | `True` |
| `--use_scale_shift_norm` | 使用 scale shift norm | `False` |
| `--attention_resolutions` | 注意力解析度 | `16` |

#### 擴散參數 (DIFFUSION_FLAGS)

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--diffusion_steps` | 擴散步數 | `1000` |
| `--noise_schedule` | 噪聲排程 | `linear` |
| `--rescale_learned_sigmas` | 重縮放學習的 sigmas | `False` |
| `--rescale_timesteps` | 重縮放時間步 | `False` |

#### 訓練參數 (TRAIN_FLAGS)

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--lr` | 學習率 | `1e-4` |
| `--batch_size` | 批次大小 | `4` |
| `--p_uncond` | 無條件機率 | `0.1` |
| `--dropout` | Dropout 率 | `0.1` |
| `--lr_anneal_steps` | 學習率退火步數 | `50000` |

#### 採樣參數 (SAMPLE_FLAGS)

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--batch_size` | 批次大小 | `1` |
| `--num_samples` | 採樣數量 | `10` |
| `--timestep_respacing` | 時間步重分配 | `ddim1000` |
| `--use_ddim` | 使用 DDIM | `True` |
| `--noise_level` | 噪聲水平 | `500` |
| `--guidance_scale` | 引導尺度 | `4.0` |

#### 匿名化參數 (ANONYMIZATION_FLAGS)

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--anonymization_mode` | 模式：`anonymize`、`deanonymize`、或 `None` | `None` |
| `--anonymization_key_seed` | 密鑰種子（用於可重現的密鑰生成） | `None` |
| `--anonymization_key_path` | 密鑰檔案路徑（用於保存/載入密鑰） | `None` |
| `--anonymization_key_password` | 從密碼派生密鑰 | `None` |
| `--anonymization_mask_type` | 遮罩類型：`full`、`center`、`margin` | `full` |
| `--anonymization_margin` | 遮罩邊緣大小（像素） | `0` |

---

## 5. 流程圖

### 5.1 架構說明

**重要**：本專案的 UNet 在**像素空間**操作（輸入 1×256×256），與論文中使用的 Stable Diffusion 的潛在空間 UNet 不同。

因此，匿名化流程**完全在像素空間進行**：
- DDIM Forward/Backward：在像素空間進行（使用 UNet）
- 密鑰操作：直接在像素空間的 x_T 上進行
- **不需要 VAE**

### 5.2 匿名化流程（像素空間）

```
原始影像 (x_0)
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  DDIM Forward Process (像素空間)    │
│  x_0 → x_1 → ... → x_T              │
│  使用 UNet 進行確定性前向擴散        │
│  (將乾淨影像轉換為噪聲表示)          │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  Apply Rademacher Key (像素空間)    │
│  x_ano_T = M ⊙ (k ⊙ x_T) + (1-M) ⊙ x_T  │
│  直接在噪聲表示上應用密鑰翻轉符號     │
│  k ∈ {-1, +1}^(1×256×256)           │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  DDIM Backward Process (像素空間)   │
│  x_ano_T → ... → x_ano_1 → x_ano    │
│  使用 UNet 進行確定性後向擴散        │
│  (將匿名化的噪聲轉換為乾淨影像)      │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
匿名化影像 (x_ano)
※ 看起來像原始 X 光，但身份特徵已改變
```

### 5.3 去匿名化 + 異常檢測流程

```
匿名化影像 (x_ano)
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  DDIM Forward Process (像素空間)    │
│  x_ano → x_ano_T                    │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  Apply Same Key (Inverse)           │
│  x_rec_T = M ⊙ (k ⊙ x_ano_T) + (1-M) ⊙ x_ano_T │
│  由於 k⊙k=1，恢復原始噪聲表示        │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  DDIM Backward Process (像素空間)   │
│  x_rec_T → x_rec (恢復的原始影像)    │
└─────────────────────────────────────┘
    │ shape: (1, 256, 256)
    ▼
┌─────────────────────────────────────┐
│  Standard Anomaly Detection         │
│  (CFG toward healthy reconstruction)│
│  x_rec → x_healthy                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Compute Anomaly Map                │
│  diff = |x_rec - x_healthy|         │
└─────────────────────────────────────┘
    │
    ▼
異常圖 (diff)
```

### 5.4 完整系統架構圖

```
┌──────────────────────────────────────────────────────────────────┐
│                      推論/採樣階段                                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 匿名化模式:                                                  │ │
│  │                                                              │ │
│  │ x_0 ──[DDIM_Fwd]──→ x_T ──[Apply_Key]──→ x_ano_T ──[DDIM_Bwd]──→ x_ano │
│  │                                                              │ │
│  │ 所有操作都在像素空間 (1×256×256) 進行                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 去匿名化 + 異常檢測模式:                                     │ │
│  │                                                              │ │
│  │ x_ano ──[DDIM_Fwd]──→ x_ano_T ──[Apply_Key]──→ x_rec_T       │ │
│  │                                       │                      │ │
│  │                                       ▼                      │ │
│  │                              [DDIM_Bwd] → x_rec              │ │
│  │                                       │                      │ │
│  │                                       ▼                      │ │
│  │                            [CFG 異常檢測] → x_healthy        │ │
│  │                                       │                      │ │
│  │                                       ▼                      │ │
│  │                         |x_rec - x_healthy| → 異常圖         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 5.5 與論文原始設計的差異

| 項目 | 論文 (Stable Diffusion) | 本專案（適配版）|
|------|------------------------|-----------------|
| UNet 輸入 | 潛在空間 (4×32×32) | 像素空間 (1×256×256) |
| DDIM 操作空間 | 潛在空間 | 像素空間 |
| 密鑰操作空間 | 潛在空間 z_T | 像素空間 x_T |
| 密鑰維度 | (4×32×32) = 4,096 | (1×256×256) = 65,536 |
| 需要 VAE | 是 | 否 |

### 5.6 為什麼匿名化後的影像看起來像原圖？

這是因為：
1. **DDIM 是確定性的**：相同的輸入總是產生相同的輸出
2. **密鑰操作在噪聲空間**：在 x_T（高度噪聲）上翻轉符號，對最終影像的視覺影響較小
3. **保持整體結構**：密鑰操作改變的是細節特徵，而不是整體結構

這確保了：
- 匿名化影像仍然是有效的 X 光影像
- 可以用於醫學分析
- 但無法通過密鑰反推原始影像（除非有正確的密鑰）

---

## 6. 密鑰管理

### 6.1 密鑰生成選項

```python
from guided_diffusion.anonymization import RademacherKey

# 方法 1: 隨機生成（不可重現）
key = RademacherKey(shape=(1, 256, 256))

# 方法 2: 使用種子生成（可重現）
key = RademacherKey(shape=(1, 256, 256), seed=42)

# 方法 3: 從密碼派生（可重現，安全性更高）
key = RademacherKey(shape=(1, 256, 256), password="secure_password_123")
```

### 6.2 密鑰保存與載入

```python
# 保存密鑰
key.save("keys/my_secret_key.pt")

# 載入密鑰
loaded_key = RademacherKey.load("keys/my_secret_key.pt")
```

### 6.3 密鑰安全性

- 密鑰是匿名化/去匿名化的核心
- 需要妥善保管密鑰檔案
- 建議使用密碼派生密鑰，並安全保管密碼
- 不同的密鑰會產生不同的匿名化結果

---

## 7. 遮罩類型

### 7.1 可用的遮罩類型

```
┌─────────────────────────────────────────────────────────────────┐
│ full (全圖匿名化)           center (中心匿名化)    margin (邊緣保留) │
│ ┌───────────────┐         ┌───────────────┐    ┌───────────────┐ │
│ │███████████████│         │   ┌───────┐   │    │███████████████│ │
│ │███████████████│         │   │███████│   │    │█             █│ │
│ │███████████████│         │   │███████│   │    │█             █│ │
│ │███████████████│         │   │███████│   │    │█             █│ │
│ │███████████████│         │   └───────┘   │    │███████████████│ │
│ └───────────────┘         └───────────────┘    └───────────────┘ │
│ M = 全部為 1               M = 中心為 1         M = 邊緣為 0       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 遮罩公式

```
x_ano_T = M ⊙ (k ⊙ x_T) + (1 - M) ⊙ x_T

其中：
- M = 1 的區域：應用密鑰翻轉（匿名化）
- M = 0 的區域：保持原樣（不匿名化）
```

### 7.3 使用範例

```bash
# 全圖匿名化（預設）
--anonymization_mask_type full

# 只匿名化中心區域
--anonymization_mask_type center --anonymization_margin 32

# 保留邊緣，匿名化內部
--anonymization_mask_type margin --anonymization_margin 16
```

---

## 8. 輸出檔案結構

### 8.1 匿名化模式輸出

```
{result_dir}/
├── anonymized/                    # 匿名化後的影像
│   ├── image001_anonymized.png
│   ├── image002_anonymized.png
│   └── ...
├── original/                      # 原始影像（供參考）
│   ├── image001_original.png
│   └── ...
├── latest_run_anonymize_xxx.png   # 所有結果合併圖
├── samples_anonymize_xxx.npz      # NPZ 格式數據
└── key_info.txt                   # 密鑰資訊
```

### 8.2 去匿名化模式輸出

```
{result_dir}/
├── recovered/                     # 恢復的原始影像
│   ├── image001_recovered.png
│   └── ...
├── healthy/                       # 健康重建影像
│   ├── image001_healthy.png
│   └── ...
├── heatmaps/                      # 異常熱圖
│   ├── image001_heatmap.png
│   └── ...
├── latest_run_deanonymize_xxx.png
├── samples_deanonymize_xxx.npz
└── key_info.txt
```

---

## 9. 重要注意事項

### 9.1 密鑰一致性

- 匿名化和去匿名化**必須使用相同的密鑰**
- 建議使用 `--anonymization_key_seed` 或 `--anonymization_key_path` 確保一致性
- 密鑰丟失將無法恢復原始影像

### 9.2 原有功能相容性

- 不使用 `--anonymization_mode` 參數時，系統行為與原有完全相同
- 原有的異常檢測功能不受影響
- 已訓練的模型可以直接使用，無需重新訓練

### 9.3 效能考量

- 匿名化需要額外的 DDIM forward 和 backward 過程
- 處理時間約為標準採樣的 2 倍
- 建議使用 GPU 加速

---

## 10. 錯誤排除

### 10.1 常見問題

**Q: 匿名化後的影像是黑白噪聲**
A: 確保使用正確的 `--noise_level` 參數，建議範圍 300-700

**Q: 去匿名化後影像與原始不一致**
A: 確認使用了相同的密鑰（seed、password 或 key_path）

**Q: 記憶體不足**
A: 減小 `--batch_size` 或降低 `--image_size`

### 10.2 驗證匿名化正確性

```python
# 測試可逆性
x_ano = anonymize(x_0, key)
x_rec = deanonymize(x_ano, key)
assert torch.allclose(x_0, x_rec, atol=1e-5)  # 應該幾乎相等
```

---

## 11. Classifier Guidance 匿名化整合

### 11.1 概述

除了 CFG (Classifier-Free Guidance) 之外，本專案也支援使用 Classifier Guidance 進行異常檢測。Classifier Guidance 使用獨立訓練的分類器來引導擴散過程。

### 11.2 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│              Classifier Guidance 匿名化工作流程                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  訓練階段:                                                       │
│  1. 訓練 UNet 擴散模型: image_train.py                          │
│  2. 訓練分類器: classifier_train.py                              │
│                                                                 │
│  推論階段:                                                       │
│  3. 標準推論: classifier_sample_known.py                        │
│  4. 匿名化推論: classifier_sample_known_anonymization.py        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 訓練指令

#### 訓練 UNet 模型

```bash
python scripts/image_train.py \
    --data_dir data/CheXpert-v1.0 \
    --dataset chexpert \
    --result_dir results/ddim_base \
    --lr_anneal_steps 50000 \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --lr 1e-4 \
    --batch_size 4 \
    --p_uncond -1 \
    --clf_free False
```

#### 訓練分類器

```bash
python scripts/classifier_train.py \
    --data_dir data/CheXpert-v1.0 \
    --dataset chexpert \
    --result_dir results/ddim_base \
    --iterations 20000 \
    --image_size 256 \
    --in_channels 1 \
    --out_channels 2 \
    --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 4 \
    --classifier_width 32 \
    --classifier_pool attention \
    --classifier_resblock_updown True \
    --classifier_use_scale_shift_norm True \
    --lr 1e-4 \
    --batch_size 4
```

### 11.4 匿名化推論指令

#### 匿名化影像

```bash
python scripts/classifier_sample_known_anonymization.py \
    --data_dir data/tif_pleural/testing \
    --model_path results/ddim_base/modelchexpert050000.pt \
    --classifier_path results/ddim_base/modelchexpertclass020000.pt \
    --result_dir results/anonymized_output \
    --dataset chexpert \
    --classifier_scale 100 \
    --noise_level 500 \
    --anonymization_mode anonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/patient_key.pt \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --batch_size 1 \
    --num_samples 11 \
    --timestep_respacing ddim1000 \
    --use_ddim True
```

#### 去匿名化 + Classifier Guidance 異常檢測

```bash
python scripts/classifier_sample_known_anonymization.py \
    --data_dir results/anonymized_output/anonymized \
    --model_path results/ddim_base/modelchexpert050000.pt \
    --classifier_path results/ddim_base/modelchexpertclass020000.pt \
    --result_dir results/deanonymized_output \
    --dataset chexpert \
    --classifier_scale 100 \
    --noise_level 500 \
    --anonymization_mode deanonymize \
    --anonymization_key_seed 42 \
    --anonymization_key_path keys/patient_key.pt \
    --unet_version v1 \
    --image_size 256 \
    --in_channels 1 \
    --num_channels 128 \
    --num_classes 2 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --batch_size 1 \
    --num_samples 11 \
    --timestep_respacing ddim1000 \
    --use_ddim True
```

### 11.5 使用 Shell 腳本

也可以直接使用提供的 shell 腳本：

```bash
# 執行匿名化 + 去匿名化推論
chmod +x run_clf_inference_anonymization.sh
./run_clf_inference_anonymization.sh
```

### 11.6 CFG vs Classifier Guidance 比較

| 特性 | CFG (Classifier-Free) | Classifier Guidance |
|------|----------------------|---------------------|
| 額外模型 | 不需要 | 需要訓練分類器 |
| 腳本 | `cfg_image_sample_anonymization.py` | `classifier_sample_known_anonymization.py` |
| 引導方式 | 無條件/有條件輸出加權 | 分類器梯度引導 |
| 參數 | `--guidance_scale` | `--classifier_scale` |
| 訓練複雜度 | 較低 | 較高（需訓練分類器）|

---

## 12. 結論

本整合方案成功將論文中的安全可逆匿名化技術適配到像素空間的胸部X光擴散模型異常檢測系統中：

1. **像素空間適配**：完全在像素空間進行操作，無需 VAE
2. **功能完整**：支援匿名化、去匿名化和異常檢測
3. **向後相容**：不影響原有功能，所有原始參數保持不變
4. **安全性**：使用 Rademacher key 確保可逆性和安全性
5. **靈活性**：支援多種遮罩類型和密鑰管理方式
6. **雙重引導支援**：同時支援 CFG 和 Classifier Guidance 兩種引導方式
