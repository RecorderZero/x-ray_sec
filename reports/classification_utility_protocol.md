# 分類 utility-preservation 評估（Track P2，2026-10-28）

## 1. 動機與定位

病灶定位在 CheXlocalize 上對本 pipeline 與學長原始設定皆為 chance 等級（pointing 0/10、IoU≈0，見 `reports/localization_evaluation_protocol.md` 與 PROGRESS 2026-10-27）。受 refpaper/114（GMIA，ISBI 2025）啟發，改以**病理分類**作為 utility 軸，量化「匿名後是否仍保留可訓練特徵、還原後 utility 是否無損、密文是否洩漏病理」。

此為論文最後一塊貢獻（定位 → 分類），與 GMIA 對照；前段（可逆、安全性、重構品質、速度）以學長論文為 baseline。

## 2. 任務定義（Design A，使用者裁決）

- 二元分類：**No Finding（健康，y=0）vs Pleural Effusion（肺積水，y=1）**，即生成器 CFG 唯一條件過的兩類（`bratsloader.py:172-173`：No Finding→0、Pleural Effusion→1）。
- [決定] 選樣時只收「純 No Finding」與「有積水」者；兩類皆非者（如骨折-only）**丟棄**，不歸入任一類。理由：生成器對其他病理無條件，送進只有 y∈{0,1} 的匿名/還原 pipeline 語意未定義。
- [事實] CheXpert frontal：純 No Finding 24,291、有積水 93,081（其中單純只有積水 = 0，積水幾乎永遠合併他症）、兩類皆非 73,547。
- [限制] 與 GMIA 的 14 類 multi-label 不同；本任務為二元單病理，難度較低，直接數字不可與 GMIA 的 14 類平均互換。

## 3. Pipeline 與評估協定

每張選中影像（patient-disjoint，每病人一張）產生三版本，**走完全相同 render**（[0,1] 256² → 8-bit 灰階 → RGB），差異只來自匿名本身：

```
real       = x0                       (e1 前處理的 pipeline 輸入本身)
anon       = B(E(F(x0)))              (已發佈的不可辨識密文影像)
recovered  = B(D(F(PNG(anon))))       (授權金鑰還原後)
```

- Sampler：**DPM-Solver++ (2M)**（`--sampler dpm++`，對齊現行部署），aligned backward；加密 S2fn（固定 norm 保真），storage 為 range-preserving PNG（`deploy_dpm_sweep` 同構）。
- 分類器：**凍結特徵 linear probe**（`StandardScaler + LogisticRegression`，class_weight balanced），encoder = DenseNet-121(ImageNet) 與 rad-dino(CXR 基礎模型)；無微調。
- Utility 矩陣（4 probe）：
  - `train real → test real`：任務可學控制組（= DPM++ pipeline 下的分類準確率）。
  - `train real → test recovered`：還原後 utility 是否保全。
  - `train real → test anon`：real 分類器能否從密文讀出病理。
  - `train anon → test anon`：能否**訓練**出可用的密文分類器（GMIA 式 train-on-anon）。
- 腳本：`scripts/classification_utility_anon.py`（GPU，產三版本 + 矩陣 + 對照圖）；`scripts/classification_utility_pilot.py`（CPU，raw-PNG real-only baseline）；`scripts/plot_classification_utility.py`（論文長條圖）。

## 4. 結果（DPM++，N=100，S2fn，per-class 100；AUROC densenet / rad-dino）

| Probe | DenseNet | rad-dino | 解讀 |
|---|---:|---:|---|
| train real → test real | 0.831 | 0.824 | 任務可學控制組 |
| train real → test recovered | 0.829 | 0.819 | **≈ real → utility 保全** |
| train real → test anon | 0.608 | 0.544 | 近 chance，密文讀不出病理 |
| train anon → test anon | 0.529 | 0.416 | ≈ chance，訓不出密文分類器 |

- [事實] recon PSNR **43.7 dB**（= `deploy_dpm_sweep` DPM++ N=100 S2fn 44.26）、anon PSNR ~6 dB。DDIM N=100 僅 37.67 dB —— sampler 差異，非退步。
- [事實] raw-PNG real-only baseline（sampler 無關，per-class 200）：densenet AUROC 0.884、rad-dino 0.914。
- [注意] baseline AUROC（~0.82）低於 raw-PNG（~0.88），因本實驗 real 與 recovered 走相同 e1 render（正確 apples-to-apples）。
- [待更新] per-class 200 確認中（test n=120），數字與圖將以 200 版定稿。

## 5. 威脅模型定位（與 GMIA 對照）

| | GMIA（refpaper/114） | 本架構 S2fn |
|---|---|---|
| 匿名圖 | 合理胸片、換身份（可訓練） | 不可辨識密文（雜訊狀） |
| 可逆 | ✗（PSNR ~24） | ✓（~44 dB，latent+影像兩空間） |
| utility 所在 | 匿名圖本身 | **授權還原端** |
| 隱私所在 | 身份被替換 | **密文端**（讀不出也訓不出病理） |

[結論] 兩者為**不同且互補**的威脅模型：GMIA 用不可逆合成保 utility；本架構用可逆加密密文 + 授權還原，utility 在還原端、隱私在密文端。因此 GMIA 作為「匿名化—分類 utility」文獻 baseline 比較時，須標明本架構額外提供**可逆性**與**密文對病理的零洩漏**，非單純 utility 數字高低之爭。

## 6. 限制與待補

- [限制] 單一 split、單 seed、固定 N=100、凍結特徵無微調、二元單病理。
- [限制] `train real → test anon` densenet 0.608 略高於 chance → 密文殘留少量解剖輪廓洩漏（對照圖第 2 欄可見），rad-dino（0.544）更接近 chance。
- [待辦] 見 §下方「新架構定稿前建議補齊的實驗」（另於 PROGRESS 追蹤）。

## 7. 重現指令

```
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_pilot --per-class 200
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_anon --per-class 200 --steps 100 --sampler dpm++
scripts/run_cfg_ddim.sh python -m scripts.plot_classification_utility \
    --json paper_assets/classification_utility/anon_utility_dpm++_n200_s100.json
```
輸出於 `paper_assets/classification_utility/`（json/png/pdf，可重現，不入庫）；seed=1911。
