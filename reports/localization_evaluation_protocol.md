# 最新平行分支定位評估（LOC-VAL，2026-10-27）

## 範圍與階段

使用者新增授權：以最新 DPM-Solver++(2M)／aligned backward／解密後 latent 分岔流程，評估胸腔積液定位，並研究 CFG++。不是回復前作「還原圖再 inversion」流程。舊凍結論文結果不覆寫。

1. 已完成資料 preflight、數值 smoke 工具、CFG++ paper-variant 單測。
2. 先做 CheXlocalize validation 陽性／陰性小實驗；不得當 formal 結果。
3. 接著 validation 校準 guidance 與 threshold，凍結設定後才開 test 結果。
4. 完整 P/S0/S1/S2 authenticated 部署比較與 storage-controlled 歸因；目前 S2 smoke 僅數值 fixture。
5. BioViL-T、AFLoc 公開模型用固定積液 prompt 重跑；完整報告輸入另列，不與 class-only CFG 混比。
6. ChestX-Det 外部驗證。外部模型尚未下載或執行。

工程成功門檻：正確幾何、有限值、固定 split/樣本、輸出與 hash 完整、gs0 分支算式一致。定位分數無優勝門檻，失敗結果照報。

## 新流程與條件語意

`x -> inversion -> z -> transform -> anonymous decode -> PNG -> inversion -> inverse transform -> z_rec`

從 `z_rec` 分別執行 `recovery=backward(z_rec)` 與 `healthy=guided_backward(z_rec)`；兩分支沒有資料相依，不代表必須在 GPU 同時排程。不 reinvert recovery。

- 現有 inversion：`null=True` 無條件。
- 現有 recovery／anonymous backward：`y=0, p_uncond=-1, clf_free=True`、未設 null，**健康類別條件**，非文件部分段落所稱「無條件」。
- CFG：`omega=1+gs`；`gs=0` 是 conditional recovery，非 unconditional。
- 先分別取得 clipped branch x0（既有模型 clip 區間 [-1,1]），再混合 `x0_c+gs*(x0_c-x0_u)`。這保留 layerA 的 clip-before-mix，**不等於** raw-epsilon guidance 後再 clip mixed x0。
- 原始影像前處理沿用 e1：grayscale/equalizeHist/INTER_AREA256/JPEG100/minmax[0,1]，無 crop；不借用前作 Q11 的另一套 bicubic loader。
- 「4-pass」是產生 recovery 的主路徑，不含 healthy 額外成本。CFG gs!=0 與 CFG++ 每步各需兩次 model call；gs0 使用一次。總時間另報，不當成只需四次單分支 decode。
- 小實驗 N_steps=100；不可與正式樣本數 N_images=200 混淆。

## gs0 失敗與修復

初次 `LOC-SMOKE_20261027T061435155266Z_2084a51b` 因 gs0 Linf=0.0024699867 > 1e-4 而停止。原因不是不同 class，而是 layerA helper 使用不同捨入係數做多餘的 float32 `x0 -> eps -> x0`。

只在新 localization module 改為直接混合 x0，未改 legacy/frozen 程式、未放寬 gate。診斷 `LOC-SMOKE_20261027T065416809243Z_1ac6beb8`（單張 dev）42秒完成：

| 指標 | aligned DDIM | DPM++ |
|---|---:|---:|
| 舊 gs0 vs recovery Linf | 0.00246999 | 0.00018325 |
| 新 gs0 vs recovery Linf | 0 | 0 |
| 新舊 gs2 heatmap Pearson | 0.99999876 | 0.99999731 |

這只驗證計算一致性，不證明醫療定位正確。

## 資料與隔離

使用者提供 `dataset/chexlocalize/`。`.cache/localization/{val,test}_manifest.json` 記錄完整影像／標註 hash、原始幾何、排除名單；原圖與 mask 不上 Git。

| 項目 | Validation | Test |
|---|---:|---:|
| 已標註正面影像 | 171 | 413 |
| 保留病患 | 170 | 403 |
| 積液陽性 label/mask | 64/64 | 105/105 |
| 陰性 | 107 | 308 |
| 無標註正面影像（排除） | 31 | 105 |
| 側面影像（排除） | 32 | 150 |

完整 CSV 的 val200/test500 病患集合互斥，標註 RLE 解碼／大小／binary／image checksum 均通過。缺標註不等於陰性。test 目前只做資料完整性檢查，未做模型輸出與選參。

正式 N_images=200 政策不變；test 積液陽性只有105張，不能假裝有200個獨立陽性樣本。正式抽樣／陰陽比例／patient-cluster CI 在 pilot 之後另凍結；不可重複抽樣補足陽性。

## 評估協定

差分為 `abs(recovery-healthy)` 原始 float。256² 熱圖以 bilinear/align_corners=False 放回原始 H×W；GT 保持原始解析度。

- 官方風格：逐熱圖 min-max 後 threshold；常數圖明確為零。首輪固定 threshold=0.5，**未校準 exploratory**，不能拿來直接對照論文排行榜。
- 指標：IoU、Dice、pixel average precision（ties一起處理）、pointing hit（row-major first argmax）、陰性預測面積比例；另報 raw mean/max、raw>0.01 比例以辨識小殘差被正規化放大的情形。
- GT陽性全部納入，空預測 IoU/Dice=0；GT陰性 overlap/AP/pointing為null，誤報另報。常數圖AP=陽性像素比例是無資訊排名基準。
- 不使用 official `true_positive_only` 來排除「有病灶但沒預測到」的案例。
- 臨床品質與影像 PSNR、兩熱圖相似度是不同問題。

## CFG++ 實驗分支

[CFG++ ICLR2025論文](https://arxiv.org/html/2406.08070v2#A1)將 guided x0 與 unconditional renoising 結合。lambda∈[0,1] 不等於本專案 gs；不能僅把 `_cfg_eps` 的係數改小。

本實作採 paper v2 Eq26 的 DPM++2M VP轉換，二階差分為 unconditional-current−unconditional-previous；[官方程式](https://github.com/CFGpp-diffusion/CFGpp/blob/main/latent_diffusion.py)中的另一 variant 使用 guided-current−previous-unconditional，兩者不可混稱。本分支保留本地 clip-before-mix 慣例，只改 healthy，不改 encryption/inversion/recovery。

lambda0 是 unconditional 負控制；因 recovery 為 conditional，**其差分不必為零**。首輪比較 CFG gs0/2 與 CFG++ lambda0/0.5/1，只為工程與探索，不依兩張分數決定最優。

## 執行與依賴

`pycocotools==2.0.11 --no-deps` 僅裝於已 gitignore 的 `.cache/localization_deps`；未升級 CFG_DDIM 或修改全域套件。每次用 `PYTHONPATH=/data2/paper/.cache/localization_deps scripts/run_cfg_ddim.sh ...`。

```bash
PYTHONPATH=/data2/paper/.cache/localization_deps scripts/run_cfg_ddim.sh python -m scripts.managed_run --task-id LOC-VAL --gpu-id 0 --validate-artifacts -- python -m scripts.chexlocalize_pilot --include-s2
```

輸出在 `artifacts/runs/LOC-VAL/<run_id>/`：status/log/manifest、per_sample.csv、localization_summary.json、PNG比較圖與NPZ。只將小型研究摘要寫到 reports；run、圖像陣列與資料集不提交。

正式長任務開始前，依 smoke 耗時預估；超過30分鐘 GPU 須另請使用者裁決。prototype CFG++ 有逐步 finite 同步檢查，不能把本輪時間直接當演算法速度排名。

## 第一輪標註 smoke 結果

`LOC-VAL_20261027T065700039777Z_791d3192`：2026-10-27，224秒 wall time（核心計時217.7秒），2張影像、40條件列，managed artifact validation valid=true，exit0。PID579850已結束。所有本輪程序均已結束。

僅有1張陽性，固定未校準 threshold0.5；以下不是平均泛化表現，不作方法選擇：

| 陽性案例方案 | IoU | Dice | pixel AP | pointing |
|---|---:|---:|---:|---|
| aligned-DDIM + original + CFG gs2 | 0.0000 | 0.0000 | 0.1060 | miss |
| DPM++ + original + CFG gs2 | 0.0218 | 0.0428 | 0.1196 | miss |
| DPM++ + S2數值fixture + CFG gs2 | 0.0490 | 0.0934 | 0.1277 | hit |
| DPM++ + S2數值fixture + CFG++ lambda0.5 | 0.0100 | 0.0198 | 0.0767 | miss |
| DPM++ + S2數值fixture + CFG++ lambda1 | 0.0000 | 0.0000 | 0.0528 | miss |

陰性案例 DPM++/S2：CFGgs2 的 normalized預測面積0.160%，CFG++lambda0.5為0.405%、lambda1為1.747%。對應 raw差分>0.01面積為14.60%、78.23%、88.10%，提示 CFG++ 在本地適配下改變較廣的影像內容，不能單憑 normalized熱圖看起來集中就認定有較好病灶特異性。

已人工檢視輸出比較圖；原圖/還原/偽健康/差分/原始GT五欄正常。需更多 validation 樣本與 guidance/threshold 校準，尚無證據支持 CFG++ 優於 CFG，也不以此否定 CFG++ 論文。

下一階段建議20張validation pilot，現有觀測外推約35–45分鐘GPU佔用（包含CPU評分時保留GPU模型），**尚未啟動，等待使用者裁決**。低成本替代為4張約7–10分鐘；CPU-only驗證無法代替模型效果實驗。

## 主要來源連結

- [CheXlocalize](https://aimi.stanford.edu/datasets/chexlocalize)／[官方評估](https://github.com/rajpurkarlab/cheXlocalize)
- [BioViL-T公開模型](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)
- [AFLoc官方實作](https://github.com/YH0517/AFLoc)
- [ChestX-Det](https://github.com/Deepwise-AILab/ChestX-Det-Dataset)
