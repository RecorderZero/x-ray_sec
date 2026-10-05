# Shell 腳本用途與使用建議

本文件整理 `past/SourceCode/` 內的 22 個 `.sh` 檔案。判斷基準是目前預計延續學長的 **CheXpert、CFG-DDIM、影像匿名化／解匿名化與異常偵測** 工作。

## 先看結論

目前所有腳本都通過 `bash -n` 語法檢查，但除了 `start_visdom_cfg_ddim.sh` 外，多數腳本都不能直接原封不動執行。常見原因包括：

- 資料路徑仍指向舊環境的 `/workspace/...` 或目前不存在的 `data/tif_pleural/...`。
- `past/SourceCode/data/CheXpert-v1.0/` 目前只有 CSV，`train/`、`valid/` 沒有實際影像。
- 目前確認存在的 checkpoint 是 `results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt`；不少腳本少了 `Model/`，或引用其他不存在的日期。
- 多個腳本將日期、輸出資料夾或解匿名化輸入寫死，未必能接到本次執行產生的目錄。
- 部分腳本一次連續執行多組訓練／推論；不要在尚未核對參數與路徑時直接執行。
- `Ki@13579`、`paper_experiment_key_2024` 是明文寫在腳本裡的實驗密碼，只能視為可重現實驗參數，不應作為真正的安全密鑰。

建議分級：

- **核心**：延續目前 CFG-DDIM／匿名化研究時很可能會使用。
- **條件式**：只有加入特定比較實驗或重新訓練時才需要。
- **舊版／參考**：主要用來追溯學長實驗，不建議直接執行。
- **工具**：環境或視覺化輔助。

## 快速索引

| 檔案 | 類型 | 在做什麼 | 之後會不會用到 | 現在能否直接跑 |
|---|---|---|---|---|
| `start_visdom_cfg_ddim.sh` | 工具 | 啟動 CFG_DDIM 環境的 Visdom | **會，常用** | 可以；port 8850 未被占用時 |
| `run_cfg_inference.sh` | 核心 | CFG-DDIM 一般異常偵測推論 | **很可能會** | 不行，需修資料與模型路徑 |
| `run_cfg_inference_Crypto.sh` | 核心 | CFG 模型的匿名化與解匿名化流程 | **很可能會** | 不建議，活動命令其實是 `guidance_scale=0` 且日期寫死 |
| `run_uncond_ddim_inference.sh` | 核心／基線 | 無 guidance 的 DDIM 推論基線 | **做比較時會** | 不行，路徑不一致 |
| `run_uncond_inference_Crypto.sh` | 條件式 | 90 張無 guidance 的匿名化／解匿名化 | **做隱私基線時會** | 不行，資料與中間目錄尚未備妥 |
| `run_inference_hyp.sh` | 核心 | 掃描 noise level 與 CFG guidance scale | **調參時會** | 不行，資料與模型路徑需更新 |
| `run_evaluation_metrics_hyp.sh` | 核心 | 評估上述超參數掃描結果 | **調參後會** | 不行，NPZ 與 mask 路徑不存在 |
| `run_evaluation_metrics.sh` | 核心 | 比較 FPGAN、CLF-DDIM、CFG-DDIM | **最終比較時可能會** | 不行，所需 NPZ 尚不存在 |
| `run_figure_maker.sh` | 核心／製圖 | 從多模型 NPZ 產生論文比較圖 | **寫報告時可能會** | 不行，NPZ 與 `/workspace` mask 路徑需更新 |
| `run_encryption_reversibility_eval_sample3.sh` | 核心／評估 | 用 3 張影像快速檢查加解密可逆性 | **很可能會** | 不行，需先產生指定 NPZ；mask 路徑也要改 |
| `run_encryption_reversibility_eval_sample90.sh` | 核心／評估 | 用 90 張影像做正式可逆性評估 | **正式實驗時會** | 不行，需先產生 Sample90 NPZ |
| `run_paper_comparison.sh` | 條件式／整合 | 一次比較 Uncond、CFG、CLF 與兩種密鑰 | **論文完整比較時可能會** | 不行，仍是 placeholder 路徑且計算量大 |
| `run_comparison_experiments.sh` | 舊版／部分實驗 | 匿名化方法比較的早期版本 | **通常不必** | 不行，只有 Unconditional 區塊啟用且日期寫死 |
| `run_cfg_training.sh` | 條件式／訓練 | 重新訓練原圖與 deblur CFG 模型 | **需要新模型時才用** | 不行，訓練影像與 deblur 資料不存在 |
| `run_clf_model_training.sh` | 條件式／訓練 | 訓練 classifier-guidance 使用的 diffusion model | **要做 CLF 比較才用** | 不行，資料未備妥 |
| `run_clf_classifier_training.sh` | 條件式／訓練 | 訓練 classifier-guidance 的獨立分類器 | **要做 CLF 比較才用** | 不行，資料未備妥 |
| `run_clf_inference_Crypto.sh` | 條件式 | CLF-DDIM 的匿名化／解匿名化推論 | **要做 CLF 比較才用** | 不行，CLF model、classifier 與資料皆未備妥 |
| `run_training.sh` | 舊版／參考 | 2025-04-26 的舊 CFG 訓練設定 | **不建議使用** | 不行，舊架構與 `/workspace` 路徑 |
| `run_training_v2.sh` | 舊版／參考 | 2025-05-07 的第二版 CFG 訓練 | **通常不必** | 不行，已被較新的設定取代且路徑失效 |
| `run_inference.sh` | 舊版／參考 | 2025-04-26 模型、guidance 8 的推論 | **不建議使用** | 不行，資料不存在且輸出名稱使用未定義 `$i` |
| `run_inference_v2.sh` | 舊版／參考 | 2025-05-03 模型、guidance 4 的推論 | **通常不必** | 不行，舊 checkpoint 與資料不存在 |
| `run_experiment_orig.sh` | 舊版／參考 | 最早期 CFG 單組推論 | **不建議使用** | 不行，路徑過時且輸出名稱使用未定義 `$i` |

## 逐檔說明

### 1. `start_visdom_cfg_ddim.sh`

- 使用 `conda` 環境 `CFG_DDIM` 啟動 Visdom。
- 固定使用 port `8850`，只綁定 `127.0.0.1`。
- Visdom session 儲存在 `/data2/paper/.visdom`，可用 `VISDOM_DATA_DIR` 覆寫。
- **建議：保留並使用。** 跑推論或訓練前，在另一個 terminal 啟動即可。

### 2. `run_cfg_inference.sh`

- 呼叫 `scripts/cfg_image_sample.py`。
- 對一般影像和 deblur 影像各跑一次 CFG-DDIM。
- 固定 `noise_level=500`、`guidance_scale=4.0`、每組 11 張。
- 使用 v1 U-Net：1 channel、128 base channels、attention resolution 16。
- **用途：** 現有 CFG 模型的標準異常偵測基線，之後很可能需要。
- **執行前：** 把 `data/tif_pleural/...` 換成目前資料位置，並把 checkpoint 改到現有的 `results/Model/...`。

### 3. `run_cfg_inference_Crypto.sh`

- 呼叫 `scripts/cfg_image_sample_anonymization.py`。
- 定義 Signed Permutation、seed 42、密碼，以及 anonymize／deanonymize 參數。
- 檔名雖是 CFG，但目前未註解的四個命令全部使用 `guidance_scale=0`，實際上是在跑無 guidance 版本。
- 依序嘗試原圖匿名化、原圖解匿名化、deblur 匿名化、deblur 解匿名化，每組 3 張。
- **用途：** 目前匿名化研究的核心入口之一。
- **執行前：** 決定要比較 CFG (`4.0`) 還是 Uncond (`0`)，修正 `DECRYPTO_DATE`，確認解匿名化輸入只指向 `anonymized/`，並更新資料／checkpoint 路徑。

### 4. `run_uncond_ddim_inference.sh`

- 仍使用 CFG sample 程式與同一個 CFG checkpoint，但將 `guidance_scale=0`，作為沒有 CFG 增強的 DDIM 基線。
- 對原圖與 deblur 影像各跑 11 張。
- **用途：** 證明改善是否真的來自 CFG guidance 時會用到。
- **注意：** 第一個 checkpoint 路徑缺少 `results/Model/`；第二個輸出寫到 `results/Model/`，但最後顯示的路徑又不同，執行前應統一。

### 5. `run_uncond_inference_Crypto.sh`

- 以 `guidance_scale=0` 對原圖與 deblur 資料各做 90 張 Signed Permutation 匿名化，再接解匿名化。
- 日期使用執行當天，計算量明顯高於 sample 3 腳本。
- **用途：** Unconditional DDIM 的隱私／可逆性正式比較。
- **執行前：** 確認 CheXpert 影像已放好、中間輸出目錄與 `anonymized/` 層級正確，避免解匿名化 loader 把 original、heatmap 等其他 PNG 一起讀入。

### 6. `run_inference_hyp.sh`

- 對 `noise_level = 250, 500, 750` 與 `guidance_scale = 4, 6, 8` 做巢狀掃描。
- 原圖與 deblur 各跑一遍，共 18 次 inference，每次 11 張。
- **用途：** 選擇 CFG 的 noise level 與 guidance scale，之後若要重做超參數實驗會使用。
- **成本：** 以目前單張、500-step 約 23 秒估算，完整執行需相當長時間；正式執行前應先用 1–3 張 smoke test。

### 7. `run_evaluation_metrics_hyp.sh`

- 讀取 `run_inference_hyp.sh` 產生的 NPZ。
- 分別比較三種 noise level 和三種 guidance scale，並拆成原圖／deblur 四組結果。
- 呼叫 `scripts/evaluation_metrics.py` 計算與輸出 CSV；目前 `save_image=False`。
- **用途：** 超參數掃描完成後，用來決定最佳 `noise_level` 與 `guidance_scale`。
- **執行前：** 更新 `INFERENCE_DATE`、NPZ 路徑及 `/workspace/tif_pleural_256_bbox_masks`。

### 8. `run_evaluation_metrics.sh`

- 比較 FPGAN、CLF-DDIM、CFG-DDIM 三種方法，另有三種 deblur 版本。
- 目前真正執行的是 base 三模型與 deblur 三模型各一組 pixel-wise AUROC／mIoU 評估。
- 後半部保留許多已註解的單模型與逐案例評估命令。
- **用途：** 最終模型比較表與定量結果可能會用到。
- **執行前：** 必須先有 `results/all_test_run/` 下的六個 90-sample NPZ，並確認 ground-truth mask 的預設或顯式路徑。

### 9. `run_figure_maker.sh`

- 呼叫 `scripts/figure_maker_select.py`。
- 將 FPGAN、CLF-DDIM、CFG-DDIM 的原圖組與 deblur 組分別排成比較圖。
- 目前讀取 11-sample NPZ，預計輸出最佳案例 grid 與 CSV。
- **用途：** 論文、proposal 或簡報需要並排視覺比較時會用到。
- **執行前：** 更新 NPZ 和 ground-truth mask 路徑；`FILTER_INDICES` 目前沒有真正傳入。

### 10. `run_encryption_reversibility_eval_sample3.sh`

- 讀取 CFG、CLF、Uncond 三種 base 輸出，以及 Rademacher、Signed Permutation 解密後輸出，共九個 3-sample NPZ。
- 呼叫 `evaluation_metrics.py --eval_mode reversibility` 計算 PSNR、SSIM、FID、MAE、MSE、ROC 與 anomaly ROC。
- 執行後另外用 heredoc 產生 `reversibility_evaluation_report.md`。
- **用途：** 正式大量實驗前的快速可逆性驗證，建議保留。
- **執行前：** 先產生 `results/Sample3/...` 的九個 NPZ，並更新 `/workspace/CheXlocalize/...` mask 路徑。

### 11. `run_encryption_reversibility_eval_sample90.sh`

- 與 sample 3 版相同，但改讀取 90-sample NPZ，並在報告中增加 Cosine Similarity。
- 目前 anomaly ROC 與 ground-truth mask 參數被註解，只保留一般 ROC。
- **用途：** 方法確認無誤後的正式可逆性實驗。
- **注意：** sample 3 與 sample 90 都寫入同一個 `results/encryption_reversibility_evaluation`，可能覆蓋同名結果；正式使用時應拆開輸出目錄。

### 12. `run_paper_comparison.sh`

- 企圖一次完成三種採樣法：Unconditional DDIM、CFG-DDIM、Classifier Guidance DDIM。
- 每種方法比較 Rademacher 與 Signed Permutation；CFG／CLF 另分 anonymize、deanonymize，共 10 個主要推論命令，每組 50 張。
- 使用 timestamp 建立結果目錄並以 `tee` 保存 log。
- **用途：** 適合整理成最終論文比較的總控腳本。
- **目前狀態：** `data/chexpert/testing`、`checkpoints/model.pt`、`checkpoints/classifier.pt` 都是 placeholder；計算量大，部分輸出 glob 也需要依程式實際日期命名重新核對。不要直接執行。

### 13. `run_comparison_experiments.sh`

- 原設計也是比較 Unconditional、CFG、Classifier Guidance 的匿名化效果。
- 目前只有 Unconditional + Signed Permutation 的 anonymize／deanonymize 兩步啟用；CFG 與 CLF 全部被註解。
- 解匿名化路徑寫死為 `unconditional_anonymize_2026_01_04`。
- **用途：** 早期比較實驗原型，功能已大致被 `run_paper_comparison.sh` 涵蓋，通常只需參考。

### 14. `run_cfg_training.sh`

- 使用 `scripts/cfg_image_train.py` 訓練一般與 deblur 兩個 CFG diffusion model。
- 設定為 `p_uncond=0.1`、batch size 4、50,000 steps、v1 U-Net。
- **用途：** 若要改 loss、scheduler、loop engineering 或重新訓練新提案模型，這是較接近現有 checkpoint 架構的訓練入口。
- **執行前：** 目前一般 CheXpert 目錄沒有影像，deblur 目錄不存在；也應更換新的實驗名稱，避免混淆舊結果。

### 15. `run_clf_model_training.sh`

- 呼叫 `scripts/image_train.py`，訓練 classifier-guidance 所需的 unconditional／class-conditional diffusion backbone。
- 一般與 deblur 各訓練 50,000 steps，`clf_free=False`。
- **用途：** 只有在實驗需要 CLF-DDIM baseline 時使用。
- **注意：** 它只訓練 diffusion model，不包含獨立 classifier。

### 16. `run_clf_classifier_training.sh`

- 呼叫 `scripts/classifier_train.py`，訓練 2-class 的獨立 noisy-image classifier。
- 一般與 deblur 各 20,000 iterations。
- **用途：** 與 `run_clf_model_training.sh` 配套；要跑 classifier guidance 時兩者 checkpoint 都需要。

### 17. `run_clf_inference_Crypto.sh`

- 使用 diffusion model 加獨立 classifier，執行 CLF-DDIM Signed Permutation 匿名化與解匿名化。
- 目前只啟用 no-deblur 的 3 張流程，`classifier_scale=100`；deblur 和 90 張版本留在註解中。
- **用途：** 若論文要比較 CFG 與 classifier guidance，才需要使用。
- **目前狀態：** `results/Model/ddim_base_2025_12_27/` 的 model 與 classifier checkpoint 都不存在。

### 18. `run_training.sh`

- 2025-04-26 的早期 CFG 訓練設定：2 heads、scale-shift norm、attention 32/16/8、batch size 8。
- 會先將參數寫入結果資料夾，再訓練原圖與 deblur 模型。
- **用途：** 僅用於追溯舊 checkpoint 架構。
- **問題：** 使用失效的 `/workspace` 路徑，而且結果資料夾若不存在，最前面的 shell redirect 就會先失敗。

### 19. `run_training_v2.sh`

- 2025-05-07 的第二版 CFG 訓練設定，架構已接近目前 v1 checkpoint。
- 同樣訓練原圖與 deblur 模型並保存參數。
- **用途：** 歷史參考；實際重新訓練優先整理 `run_cfg_training.sh`。
- **問題：** 仍使用 `/workspace` 路徑，且沒有先建立結果資料夾。

### 20. `run_inference.sh`

- 對 2025-04-26 舊架構模型跑原圖與 deblur inference，`guidance_scale=8.0`、10 張。
- **用途：** 對應早期 checkpoint 的歷史推論。
- **問題：** 資料和 checkpoint 目前不存在；結果目錄名稱使用未定義的 `$i`，可能產生錯誤或不完整命名。

### 21. `run_inference_v2.sh`

- 對 2025-05-03 模型跑原圖與 deblur inference。
- `guidance_scale=4.0`、`noise_level=500`、10 張；兩個只有單一值的 `for` loop 是舊調參寫法殘留。
- **用途：** 歷史參考；目前應改用較新的 `run_cfg_inference.sh`。

### 22. `run_experiment_orig.sh`

- 最早期的單組 CFG inference，使用 guidance 2.0、noise 500、10 張。
- **用途：** 只適合追溯最初實驗設定。
- **問題：** 沒有 `unet_version`，使用舊模型架構、舊資料／checkpoint 路徑，輸出名稱又引用未定義 `$i`，不建議執行。

## 建議保留的實際工作流

若目前主線是「CFG-DDIM + 可逆匿名化 + 異常偵測」，建議日後優先整理並使用以下順序：

1. `start_visdom_cfg_ddim.sh`：啟動視覺化。
2. `run_cfg_inference.sh`：建立未加密的 CFG baseline。
3. `run_cfg_inference_Crypto.sh`：產生匿名化與解匿名化結果。
4. `run_uncond_ddim_inference.sh`：建立無 guidance baseline。
5. `run_encryption_reversibility_eval_sample3.sh`：先用 3 張做 smoke evaluation。
6. `run_inference_hyp.sh` + `run_evaluation_metrics_hyp.sh`：需要重新調參時才跑。
7. `run_encryption_reversibility_eval_sample90.sh`：方法穩定後做正式評估。
8. `run_evaluation_metrics.sh` + `run_figure_maker.sh`：整理最終數據與論文圖。

CLF 相關三個腳本只有在論文確定要納入 classifier-guidance baseline 時才值得投入訓練成本。`run_training*.sh` 與早期 `run_inference*.sh` 應保留作實驗追溯，不應當成目前預設入口。

## 執行前共同檢查

```bash
cd /data2/paper/past/SourceCode
conda activate CFG_DDIM
bash -n <script-name>.sh
```

然後逐項確認：

1. `data_dir` 真的含有影像，而不只有 CSV。
2. `model_path`／`classifier_path` 存在，且架構旗標與 checkpoint 一致。
3. `num_samples` 先降至 1–3 做 smoke test。
4. anonymize 的輸出目錄能正確接到 deanonymize，最好明確指向 `anonymized/`。
5. 所有 `/workspace/...` 路徑都換成目前 `/data2/paper/...` 下的實際位置。
6. 結果目錄使用新的實驗名稱，不覆蓋學長既有結果。
7. 先估算 GPU 時間，再執行 90 張、多模型或超參數 grid 的完整工作。
