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

## 4. 結果（DPM++，N=100，S2fn，per-class 200；AUROC，50 patient-splits，mean±95%CI）

**4a. 凍結特徵 linear probe（item 2，`anon_utility_dpm++_n200_s100_k50.json`）**

| Probe | DenseNet | rad-dino | 解讀 |
|---|---:|---:|---|
| train real → test real | 0.916 ±0.005 | 0.937 ±0.004 | 任務可學控制組 |
| train real → test recovered | **0.904 ±0.006** | **0.937 ±0.004** | **≈ real → utility 保全** |
| train real → test anon | 0.502 ±0.012 | 0.499 ±0.015 | = chance，密文讀不出病理 |
| train anon → test anon | 0.510 ±0.013 | 0.525 ±0.011 | = chance，訓不出密文分類器 |

**4b. 微調分類器（item 3，GMIA 式訓練 DenseNet-121，5 splits，`finetune_utility_dpm++_n200_s100_k5.json`）**

| Probe | 微調 DenseNet AUROC |
|---|---:|
| train real → test real | 0.886 ±0.026 |
| train real → test recovered | **0.888 ±0.027**（≈ real） |
| train real → test anon | 0.442 ±0.037（chance） |
| train anon → test anon | 0.494 ±0.040（chance） |

凍結探針與真實訓練分類器**結論一致**：還原保全 utility、密文 = chance。

- [事實] recon PSNR **43.6 dB**（400 張；= `deploy_dpm_sweep` DPM++ N=100 S2fn 44.26 級距）、anon PSNR ~5.9 dB。DDIM N=100 僅 37.67 dB —— 為 sampler 差異，非退步。
- [事實] raw-PNG real-only baseline（sampler 無關，per-class 200）：densenet AUROC 0.884、rad-dino 0.914。
- [注意] 本實驗 real 與 recovered 走相同 e1 render（正確 apples-to-apples）。

**4c. Privacy–Utility 聯合表（item 1，同 DPM++/S2 pipeline、同凍結 encoder；`privacy_utility_table.md`）**

| Encoder | 病理 real→recovered | 身份 P 連結 AUC | 身份 **S2** 連結 AUC |
|---|---:|---:|---:|
| DenseNet-121 | 0.904 | 0.821 | **0.381** |
| rad-dino | 0.937 | 0.975 | **0.486** |

外部攻擊者（Packhauser Siamese，latent 密文）：plaintext top-1 0.15 → **S2 top-1 0.05 = random**（AUC 0.445）。
[結論] 同一條 pipeline、同一批 encoder：**讀病理(utility) ~0.9、讀身份(privacy) S2 ~random** —— 隱私與 utility 在同一操作點同時成立。

**4d. Utility vs 重構步數 N（item 6，per-class 100、20 splits、`anon_utility_dpm++_n100_s{50,100,200}_k20.json`、`fig_utility_vs_n_k20.png`）**

| N | recon PSNR | DenseNet real→rec | DenseNet real→real | rad-dino real→rec | rad-dino real→real |
|---:|---:|---:|---:|---:|---:|
| 50 | 38.9 dB | 0.882 ±0.019 | 0.903 | 0.903 ±0.016 | 0.895 |
| 100 | 43.7 dB | 0.894 ±0.019 | 0.903 | 0.897 ±0.017 | 0.895 |
| 200 | 46.1 dB | 0.901 ±0.020 | 0.903 | 0.895 ±0.016 | 0.895 |

[結論] recovered utility **隨重構品質單調逼近 real**：DenseNet 在 N=50(38.9 dB)仍有小落差(0.882 vs 0.903),N=200(46.1 dB)已實質無損(0.901 vs 0.903);rad-dino(CXR 基礎模型)對 N 不敏感,N=50 即飽和。此把分類 utility 直接綁到 DPM++／步數貢獻:**更好的重構 → 更無損的 utility**,且即使最快設定(N=50)utility 仍 ~0.9。

## 5. 威脅模型定位（與 GMIA 量化對照，item 4；僅表格比較，未復現 GMIA）

| 維度 | GMIA（refpaper/114，14 類 multi-label） | 本架構 S2fn（二元 No Finding/Effusion） |
|---|---|---|
| 匿名化型態 | 不可逆合成、換身份 | **可逆加密密文**（雜訊狀、不可辨識） |
| 重構/還原 PSNR | 24.3 dB（不可逆） | **43.6 dB（可逆，latent+影像兩空間）** |
| real 分類 AUROC | 0.774 | 0.92 / 0.94（densenet / rad-dino） |
| utility 保全 AUROC | 0.737（**在匿名圖**） | 0.904 / 0.937（**在還原圖**） |
| 匿名/密文圖可訓練性 | 可（匿名圖 utility 0.737） | **= chance（密文,設計使然）** |
| inner 連結風險（匿名↔匿名） | **高 0.899（Acc）** | **~random（AUC 0.38–0.49）** |
| outer 連結風險（匿名↔真實） | ~0（去識別） | ~random（S2 外部 top-1 0.05） |
| MIA（評估階段） | 0.0 | （未量，另列） |

[結論] GMIA 把 utility 留在**匿名圖**，代價是**高 inner 連結風險**且**不可逆**；本架構把匿名圖做成**不可連結密文**（inner 與 outer 風險皆 ~random）、utility 以**授權還原無損取回**。兩者威脅模型不同；數值 AUROC 因資料集/任務（14 類 vs 二元）不同不可直接比大小，應比**結構性質**：我們多了可逆性、消滅了 inner-risk。

## 6. 限制與待補

- [限制] 固定 N=100、二元單病理（生成器僅條件過兩類，無法如 GMIA 做 14 類）；CheXpert 單一資料集。
- [事實] 大樣本（50 splits、per-class 200）下 `train real → test anon` 已收斂到 0.50（n100 時的 0.61 殘留為小樣本雜訊）；密文無顯著病理洩漏。
- [待辦] 外部資料集 / 外部分類器泛化；本架構自身的 MIA 量測（對齊 GMIA 的 MIA 欄）。

## 7. 重現指令

```
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_pilot --per-class 200          # raw-PNG baseline
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_anon --per-class 200 --steps 100 --sampler dpm++ --splits 50
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_finetune --per-class 200 --steps 100 --splits 5
scripts/run_cfg_ddim.sh python -m scripts.build_privacy_utility_table \
    --utility paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json
scripts/run_cfg_ddim.sh python -m scripts.plot_classification_utility \
    --json paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json --out paper_assets/classification_utility/fig_utility_matrix_k50.png
for N in 50 100 200; do scripts/run_cfg_ddim.sh python -m scripts.classification_utility_anon \
    --per-class 100 --steps $N --sampler dpm++ --splits 20 --save-examples 0; done     # item 6 sweep
scripts/run_cfg_ddim.sh python -m scripts.plot_utility_vs_n \
    --jsons paper_assets/classification_utility/anon_utility_dpm++_n100_s*_k20.json \
    --out paper_assets/classification_utility/fig_utility_vs_n_k20.png                 # item 6 figure
```
輸出於 `paper_assets/classification_utility/`（json/png/pdf，可重現，不入庫）；seed=1911。
