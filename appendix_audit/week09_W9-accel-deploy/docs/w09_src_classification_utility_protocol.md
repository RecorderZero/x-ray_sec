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

## 4. 結果（DPM++，N=100，S2fn，**null backward**，per-class 200；AUROC，50 patient-splits，mean±95%CI）

> [事實] 本節全數於最終部署條件（反演/生成皆 null、forward/backward 對齊，2026-10-29 重量）量測，supersede 先前 y=0 數字。對照：null 比 y=0 recon +~3 dB（N=100 43.7→46.7 dB），二元 utility 結論不變（架構/保真不敏感），且密文病理洩漏更乾淨（real→anon 由 y=0 的 0.60 殘留降到 null 的 0.50 chance）。

**4a. 凍結特徵 linear probe（item 2，`anon_utility_dpm++_n200_s100_k50.json`；recon 46.7 dB）**

| Probe | DenseNet | rad-dino | 解讀 |
|---|---:|---:|---|
| train real → test real | 0.916 ±0.005 | 0.937 ±0.004 | 任務可學控制組 |
| train real → test recovered | **0.909 ±0.006** | **0.937 ±0.004** | **≈ real → utility 保全** |
| train real → test anon | 0.501 ±0.011 | 0.497 ±0.016 | = chance，密文讀不出病理 |
| train anon → test anon | 0.507 ±0.013 | 0.525 ±0.012 | = chance，訓不出密文分類器 |

**4b. 微調分類器（item 3，GMIA 式訓練 DenseNet-121，5 splits，`finetune_utility_dpm++_n200_s100_k5.json`）**

| Probe | 微調 DenseNet AUROC |
|---|---:|
| train real → test real | 0.883 ±0.029 |
| train real → test recovered | **0.884 ±0.028**（≈ real） |
| train real → test anon | 0.452 ±0.025（chance） |
| train anon → test anon | 0.500 ±0.035（chance） |

凍結探針與真實訓練分類器**結論一致**：還原保全 utility、密文 = chance。

- [事實] recon PSNR **46.7 dB**（400 張；= `deploy_dpm_sweep` DPM++ N=100 S2fn null 47.06 級距）、anon PSNR ~5.9 dB。DDIM N=100 僅 ~38 dB —— 為 sampler 差異，非退步。
- [事實] raw-PNG real-only baseline（sampler 無關，per-class 200）：densenet AUROC 0.884、rad-dino 0.914。
- [注意] 本實驗 real 與 recovered 走相同 e1 render（正確 apples-to-apples）。

**4c. Privacy–Utility 聯合表（item 1，同 DPM++/S2 pipeline、同凍結 encoder；`privacy_utility_table.md`）**

| Encoder | 病理 real→recovered | 身份 P 連結 AUC | 身份 **S2** 連結 AUC |
|---|---:|---:|---:|
| DenseNet-121 | 0.909 | 0.811 | **0.481** |
| rad-dino | 0.937 | 0.977 | **0.513** |

外部攻擊者（Packhauser Siamese，latent 密文）：plaintext top-1 0.15（AUC 0.764）→ **S2 top-1 0.05 = random**（AUC 0.445）。
- [事實/AF-047] 各欄 n 與方案不同，明確標註：病理 utility = per-class 200（test 120/split × 50 patient-splits，S2fn）；身份連結 = 20 patient pairs（gallery+query，S2）；外部 re-id = Packhauser Siamese（latent 密文）。身份用 S2、utility 用 S2fn（同 per-image-nonce 正交族，S2fn 為其固定 norm 保真變體，匿名圖皆雜訊狀）。單一 null run 組裝，無跨來源拼接（`privacy_utility_table.{md,csv,json}`）。
[結論] 同一條 null pipeline、同一批 encoder：**讀病理(utility) ~0.9、讀身份(privacy) S2 ~random（0.48–0.51）** —— 隱私與 utility 在同一操作點同時成立。

**4d. Utility vs 重構步數 N（item 6，per-class 100、20 splits、`anon_utility_dpm++_n100_s{50,100,200}_k20.json`、`fig_utility_vs_n_k20.png`）**

| N | recon PSNR | DenseNet real→rec | DenseNet real→real | rad-dino real→rec | rad-dino real→real |
|---:|---:|---:|---:|---:|---:|
| 50 | 39.8 dB | 0.889 ±0.018 | 0.903 | 0.902 ±0.016 | 0.895 |
| 100 | 46.8 dB | 0.900 ±0.019 | 0.903 | 0.898 ±0.016 | 0.895 |
| 200 | 52.5 dB | 0.904 ±0.019 | 0.903 | 0.896 ±0.016 | 0.895 |

[結論] recovered utility **隨重構品質單調逼近 real**：DenseNet 在 N=50(39.8 dB)仍有小落差(0.889 vs 0.903),N=200(52.5 dB)已實質無損(0.904 vs 0.903);rad-dino(CXR 基礎模型)對 N 不敏感,N=50 即飽和。此把分類 utility 直接綁到 DPM++／步數貢獻:**更好的重構 → 更無損的 utility**,且即使最快設定(N=50)utility 仍 ~0.9。

**4e. 學長架構 baseline(同任務、同 render、同凍結 encoder;`classification_utility_legacy.py`、`legacy_utility_S1*_n100_t500_k20.json`)**

把學長**原始匿名化架構**(legacy dense DDIM T=500、generation y=0 條件 `guidance_scale=-1` 無 CFG 混合、加密 S1=SignedPermutationKey)接上**完全相同**的二元任務/選樣/render/凍結探針,只換架構。偽健康圖 CFG(gs=4,`|x0−healthy|`)是定位工具、對二元分類無貢獻,故省略 —— 架構化簡為 invert→加密→generate(anon)、invert→解密→generate(recover),正是使用者描述的流程。per-class 100、20 splits,AUROC mean±95%CI。

| 架構 | sampler | 加密 | storage | recon PSNR | 速度 | DenseNet real→rec | rad-dino real→rec |
|---|---|---|---|---:|---:|---:|---:|
| **本架構** | DPM++ N=100 (null) | S2fn | 保值域 | 46.7 dB | **6.3 s/img** | 0.900 | 0.898 |
| 學長(+我們儲存) | legacy DDIM T=500 | S1 | 保值域 | 38.9 dB | 30.8 s/img | 0.887 ±0.021 | 0.900 ±0.017 |
| 學長原生 | legacy DDIM T=500 | S1 | 原生(值域丟棄) | **31.8 dB** | 30.8 s/img | 0.899 ±0.017 | 0.893 ±0.019 |

(三者 real→real 參考一致:DenseNet 0.903、rad-dino 0.895;real→anon 0.535/0.553、anon→anon 0.520/0.492 在學長兩版完全相同,因 anon 影像與 storage 無關。)

- [事實] 學長原生儲存 recon **31.8 dB**,與學長論文自報的 ~30 dB 吻合 → 忠實重現;換成我們的保值域儲存升到 38.9 dB(**儲存貢獻 ~+7 dB**),再換 DPM++ null 升到 46.7 dB(**sampler+對齊貢獻 再 ~+7.8 dB**)。
- [結論] **二元分類 utility 對架構不敏感**:recon 橫跨 31.8→38.9→46.7 dB,recovered AUROC 都 ≈ real(~0.90)。即二元肺積水分類不需要 PSNR 所量的高頻細節,**utility 保留是「可逆匿名化家族」的共通性質,非本架構改進獨有**。
- [結論] 因此本架構的差異化**不在「分類能否存活」,而在**:(i) **速度 ~4.9×**(6.3 vs 30.8 s/img,DPM++ N=100 取代 dense T=500);(ii) **重構保真** 31.8→46.7 dB(保值域 + DPM++ + null 對齊;同算力 N=499 達 54.2 dB);(iii) **安全性**(S2 固定 norm + per-image nonce vs S1 固定全域 key;量化見 §4c 與 t2bb 連結實驗)。此為誠實定位,不過度宣稱 utility 為本架構獨佔。

**4f. 外部資料集泛化 + 文獻定位(`classification_utility_external.py`、`external_utility_nih_n150.json`)**

把 CheXpert 上 train 好的 **train-real 凍結探針直接搬到外部資料集 NIH ChestX-ray14**(不同醫院/機器/人群,**完全不重訓**)測試。Design A 標籤跨集乾淨對映(NIH "No Finding"=y0、"Effusion"=y1);兩集走相同 e1 preprocess 與 render,唯一變數是資料集。per-class 150、patient-disjoint(已下載 images_001 archive,NIH 本機 `dataset/nih-cxr14/`)。

| encoder | in-domain CheXpert→CheXpert | **external CheXpert→NIH** | 落差 |
|---|---:|---:|---:|
| DenseNet-121 | 0.900 | **0.823** (acc 0.730) | −0.077 |
| rad-dino | 0.902 | **0.836** (acc 0.767) | −0.066 |

- [結論] 跨資料集 AUROC 僅掉 ~0.07(0.90→0.82–0.84),對零重訓的 cross-dataset CXR 泛化屬**良好**(文獻 domain-shift 落差常 0.05–0.15)。→ 分類器學到的是**真實病理特徵、非 CheXpert 專屬 artifact**,佐證 utility 的外部效度。
- [限制/範圍] 本測試在**真實 NIH 影像**上評估「utility 分類器是否泛化」(使用者所問),**非**把我們的匿名/還原 pipeline 跑在 NIH 上(生成器未在 NIH 訓練,超出範圍)。
- [文獻定位] CheXpert 肺積水分類公開水準:原始 baseline **0.928**、ResNet-152 **0.937**、DenseNet-169 **0.934**(2020 架構比較,全資料集/14 類多標籤/完整測試集)。我們:real→real 0.903–0.916 / rad-dino **0.937**;recovered 0.887–0.909 / 0.937 —— **落在公開正常區間(0.90–0.94)**,rad-dino 貼 SOTA 級,DenseNet 略低屬合理(小子集 per-class 100–200 的凍結線性探針,非全資料微調)。→ 我們的 utility 精準度為**臨床典型水準**,非離群。

**4g. 高頻保真 proxy（HF-FIDELITY，回應 AF-046；`hf_fidelity_check.py`、`hf_fidelity.json`，200 張 patient-disjoint、兩類各 100）**

為佐證「還原不系統性抹除細節（小病灶 ≈ 局部高頻）」，在**與部署完全一致的操作點**（DPM++ 2M N=100、S2fn、range-preserving PNG、**null backward**、seed 1911、e1 render）量還原圖對原圖的影像端高頻保真。**非醫療偵測、不下診斷結論。**

| 指標（recovered vs x0） | 值 | C1 x0-vs-x0（上界） | C3 anon-vs-x0（下界） |
|---|---:|---:|---:|
| RAPSD 高頻帶 retention（**headline**） | **0.964** | 1.000 | 6.756† |
| RAPSD 中 / 低頻帶 retention | 0.998 / 0.999 | 1.000 | — |
| 高通 SSIM（σ=1.0） | **0.988** | 1.000 | 0.068 |
| Sobel 梯度 r | 0.997 | 1.000 | 0.006 |
| Laplacian 變異比 | 0.896 | 1.000 | — |
| 32×32 分塊 HF retention mean / p05 / **min** | 0.946 / 0.799 / **0.650** | 1.000 | — |

†C3 高頻 retention 爆高，因 anon 為**高頻噪聲狀密文**（功率本就高）；其結構性崩塌正確反映在 HP-SSIM 0.068、Sobel-r 0.006（≈0），故 C3 以結構指標觸底判定，非功率 retention。模糊校準尺（C2，x0 vs GaussianBlur）：σ=0.5 高頻 retention 0.678 / HP-SSIM 0.971；σ=1.0 0.407 / 0.800；σ=2.0 0.374 / 0.677。

- [事實] 工程門檻全數通過：數值有限、C1 到頂（1.000）、C3 結構觸底（HP-SSIM 0.068）、操作點與部署一致（`engineering_gate.pass=true`）。好壞照報、未設未登記的 retention 通過門檻。
- [結論] 還原圖高頻 retention **0.964**、HP-SSIM **0.988** —— **比最輕的 σ=0.5 模糊還銳**（0.678 / 0.971），即還原**未系統性平滑高頻**；分塊最差 patch（min 0.650）仍保留逾六成局部高頻功率。此為影像端間接佐證，進一步由 §4h 的獨立第三方分類器直接驗證（見 §6 limitation 的升級）。

**4h. 獨立第三方裁判多病理偵測保全（AF-046 升級，使用者 2026-10-29 提議；`referee_pathology_check.py`、`referee_pathology_n400.json`，400 張 patient-disjoint frontal、full multi-label GT）**

用**我們從未接觸、零重訓**的第三方預訓練 CXR 分類器（torchxrayvision DenseNet-121，公開權重）當裁判，直接量「還原圖是否保留多種病理的可偵測性」——把 §4g 的影像端 proxy 升級為**真正的偵測指標**。同一批 400 張影像、同一 xrv 前處理，real / recovered / anon 皆過同一裁判，故 Δ = AUC(rec) − AUC(real) 只隔離 round-trip 效應（非資料集/前處理差異）。操作點與部署完全一致（DPM++ N=100、S2fn、null backward、range-preserving、seed 1911）。兩組權重：`densenet121-res224-all`（18 病理，含 Lung Lesion/Pneumothorax 等焦點病灶）與 `densenet121-res224-chex`（CheXpert 子集）。11 個 CheXpert∩xrv 重疊病理可評分。

| 病理（n_pos） | all: real / **rec** / anon | chex: real / **rec** / anon |
|---|---|---|
| Enlarged Cardiomediastinum (272) | 0.874 / **0.876** / 0.531 | 0.860 / **0.861** / 0.453 |
| Cardiomegaly (231) | 0.858 / **0.855** / 0.532 | 0.872 / **0.873** / 0.475 |
| Lung Opacity (282) | 0.911 / **0.914** / 0.565 | 0.892 / **0.894** / 0.488 |
| Lung Lesion (54) | 0.664 / **0.661** / 0.469 | 0.612 / **0.611** / 0.456 |
| Edema (175) | 0.828 / **0.822** / 0.564 | 0.827 / **0.827** / 0.487 |
| Consolidation (209) | 0.865 / **0.866** / 0.508 | 0.873 / **0.874** / 0.488 |
| Pneumonia (86) | 0.675 / **0.656** / 0.481 | 0.629 / **0.629** / 0.413 |
| Atelectasis (221) | 0.871 / **0.871** / 0.548 | 0.837 / **0.838** / 0.508 |
| Pneumothorax (44) | 0.748 / **0.737** / 0.475 | 0.759 / **0.757** / 0.514 |
| Effusion (184) | 0.889 / **0.890** / 0.560 | 0.870 / **0.869** / 0.523 |
| Fracture (110) | 0.503 / **0.502** / 0.513 | 0.591 / **0.589** / 0.504 |
| **MACRO** | 0.790 / **0.786** / 0.522 | 0.784 / **0.784** / 0.483 |

- [事實] **還原 ≈ 真圖（逐病理）**：11 病理最大 |Δ| = 0.019（Pneumonia, all），macro Δ = **−0.003**（all）／**−0.000**（chex）。獨立裁判在還原圖上的每病理 AUC 與真圖幾乎不可區分。
- [事實] **焦點/小病灶亦保全**（直接回應 §6 limitation #2）：Lung Lesion（rec 0.661 ≈ real 0.664）、Pneumothorax（0.737 ≈ 0.748）、Pneumonia（0.656 ≈ 0.675）—— 偵測能力未因 round-trip 流失。
- [事實] **密文非診斷性**：anon macro 0.522（all）／0.483（chex），逐病理皆 ≈ chance，對真圖訓練的裁判而言密文不可讀（符合設計）。
- [限制] **Fracture 在真圖上即 ≈ chance**（0.503 / 0.591）——裁判本身在 256² CXR 讀不出骨折，故 rec ≈ real ≈ chance；誠實反映「真圖上不存在的訊號，round-trip 既不無中生有也不抹除」。Nodule / Mass 非 CheXpert 欄位未評分，以同族 Lung Lesion 代表焦點病灶。此為 ML-裁判 proxy、非放射科醫師判讀；CheXpert 標籤本身帶不確定性；好壞照報。

## 5. 威脅模型定位（與 GMIA 量化對照，item 4；僅表格比較，未復現 GMIA）

| 維度 | GMIA（refpaper/114，14 類 multi-label） | 本架構 S2fn（二元 No Finding/Effusion） |
|---|---|---|
| 匿名化型態 | 不可逆合成、換身份 | **可逆加密密文**（雜訊狀、不可辨識） |
| 重構/還原 PSNR | 24.3 dB（不可逆） | **46.7 dB（可逆，latent+影像兩空間；同算力 N=499 達 54.2 dB）** |
| real 分類 AUROC | 0.774 | 0.92 / 0.94（densenet / rad-dino） |
| utility 保全 AUROC | 0.737（**在匿名圖**） | 0.909 / 0.937（**在還原圖**） |
| 匿名/密文圖可訓練性 | 可（匿名圖 utility 0.737） | **= chance（密文,設計使然）** |
| inner 連結風險（匿名↔匿名） | **高 0.899（Acc）** | **~random（S2 AUC 0.48–0.51）** |
| outer 連結風險（匿名↔真實） | ~0（去識別） | ~random（S2 外部 top-1 0.05） |
| MIA（membership inference） | 0.0（學習式匿名器需量測） | **結構性 N/A（見下）** |

[MIA 定位] GMIA 的匿名器是**在病患群體上訓練的生成模型**,故「某病患是否在匿名器訓練集」可被攻擊,須量 MIA(其報 0.0)。本架構的匿名化是 **per-image 線性自逆加密 + 公開預訓練 diffusion**,加密端**沒有訓練集**,membership 概念不成立;唯一可談的是 diffusion 預訓練資料的 membership,但它是公開固定模型、與匿名/還原操作無關。更實際的是:攻擊者無金鑰時只看到**雜訊狀密文**(§4c 已證連結 inner/outer 皆 ~random、外部 re-id top-1 0.05=random),還原需授權金鑰 —— 故本架構對「從發佈影像推斷 membership」的風險由**不可連結密文**結構性緩解,非靠學習式匿名器的隱私-utility 取捨。因此此欄與 GMIA 不對等,不強行給可比數字。

[結論] GMIA 把 utility 留在**匿名圖**，代價是**高 inner 連結風險**且**不可逆**；本架構把匿名圖做成**不可連結密文**（inner 與 outer 風險皆 ~random）、utility 以**授權還原無損取回**。兩者威脅模型不同；數值 AUROC 因資料集/任務（14 類 vs 二元）不同不可直接比大小，應比**結構性質**：我們多了可逆性、消滅了 inner-risk。

## 6. 限制與待補

**AF-046 三點 utility limitation（必載，裁決 AUD-20261028-02；HF-FIDELITY 見 §4g）**

1. [限制] utility 保全僅在**肺積水（粗大、低頻 finding）**上以分類驗證；該軸恰為生成器 CFG 唯一條件過的病理軸。
2. [限制→大幅收斂] **多病理偵測保全現有直接證據**：除 §4g 影像端高頻 proxy（high-band retention 0.964、HP-SSIM 0.988、分塊 min 0.650）外，§4h 以**獨立第三方零重訓裁判**（torchxrayvision，兩組公開權重）在 **11 病理**上直接量得 **recovered ≈ real（macro Δ ≤ 0.003，最大逐病理 |Δ|=0.019）**，含焦點病灶 Lung Lesion / Pneumothorax / Pneumonia。機制上唯一可能抹除高頻者為**擴散重構的平滑**；原裁決所列的另一機制「健康條件還原」**已不適用** —— 本最終版還原改為**無條件（null，與反演對齊；supersede AF-018/021 的 y=0）**，不再把還原預偏向健康類。**殘留界線**：Nodule / Mass 非 CheXpert 欄位未直接評分（以同族 Lung Lesion 代表）；裁判為 ML-proxy 非放射科判讀，且 256² 解析度下裁判對 Fracture 本即 chance；故仍不宣稱「全部小病灶之臨床偵測保全」，但「小/高頻病理未做偵測驗證」的原限制已由 §4h 實質回應。
3. [限制] **不宣稱一般診斷 utility 保全**；NIH 泛化（§4f）只支持「偵測任務」跨資料集，不支持「還原保全」跨資料集（NIH 跑的是真實影像，非走過本匿名/還原 pipeline 的 NIH）。

- [限制] 固定 N=100、二元單病理（生成器僅條件過兩類，無法如 GMIA 做 14 類）；匿名/還原 pipeline 僅在 CheXpert（生成器訓練集）上評估。
- [事實] 大樣本（50 splits、per-class 200）下 `train real → test anon` = 0.501/0.497（null）；y=0 舊版曾有 0.60 殘留，null 對齊後降到 chance，密文無顯著病理洩漏。
- [事實] 外部資料集泛化已補（§4f）：train-real 探針零重訓搬到 NIH ChestX-ray14,AUROC 0.823/0.836（落差僅 ~0.07），分類器學到真實病理特徵。
- [決定] MIA 不量測、改結構性說明（§5）：本架構加密端無訓練集,membership 概念不成立,風險由不可連結密文結構性緩解,與 GMIA 不對等。

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
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_legacy \
    --per-class 100 --noise-level 500 --splits 20 --scheme S1 --storage range-preserving  # §4e 學長+我們儲存
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_legacy \
    --per-class 100 --noise-level 500 --splits 20 --scheme S1 --storage legacy            # §4e 學長原生儲存
scripts/run_cfg_ddim.sh python -m scripts.classification_utility_external \
    --per-class 150 --nih-root dataset/nih-cxr14                                          # §4f 外部泛化 (NIH)
scripts/run_cfg_ddim.sh python -m scripts.p11_experiment_matrix                          # 重生實驗矩陣 (41 cells)
scripts/run_cfg_ddim.sh python -m scripts.plot_architecture_comparison                   # 圖3 三方架構對照
scripts/run_cfg_ddim.sh python -m scripts.plot_external_generalization                   # 圖4 外部泛化
```
輸出於 `paper_assets/classification_utility/`（json/png/pdf，可重現，不入庫）；seed=1911。
NIH 取得:`dataset/nih-cxr14/`(Data_Entry_2017_v2020.csv + images_001 archive,HF/NIH Box 直連,不入庫)。

## 8. 論文圖組（最終）

| 圖 | 檔案 | 訊息 |
|---|---|---|
| 圖1 utility 矩陣 | `fig_utility_matrix_k50.png` | 4-probe:real→real/recovered 高且齊,anon/anon→anon = chance |
| 圖2 utility-vs-N | `fig_utility_vs_n_k20.png` | recovered AUROC 隨 recon PSNR 單調逼近 real;N=50 已 ~0.9 |
| 圖3 三方架構對照 | `fig_architecture_comparison.png` | recovered utility 對架構不敏感(平 ~0.90);差異在速度/保真(標註) |
| 圖4 外部泛化 | `fig_external_generalization.png` | CheXpert→NIH 零重訓,AUROC 落差僅 ~0.07 |
| 定性對照圖 | `anon_utility_examples_*`、`legacy_utility_examples_*` | real / anon / recovered 三欄並排 |
