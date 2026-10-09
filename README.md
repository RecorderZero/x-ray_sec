# X-ray Security Research File Index

本檔是專案的「檔案地圖」。未來新增、重新命名或淘汰腳本、CSV/JSON、報告、圖片時，必須在同一個 commit 更新這裡；大型逐樣本輸出、dataset、checkpoint、cache 與 secret 不列入 Git，只記錄其用途與預期位置。

## 目前狀態

- E0.1、E1.1–E1.5 已由 AUD-20261008-02 複驗通過；E2.2 已完成 S0/S1 legacy regression，並補上恆等金鑰 P（AF-017）。
- 2026-10-09 起 P/S0/S1 一律走學長**加密流程**（`ddim_sample_loop_anonymization`）；E1 的推論流程結果只作病灶定位流程的歷史基線。AF-017、AF-020、AF-021 已修正待稽核複驗；AF-019 需待 E2.5／A4 完成。
- 目標 checkpoint 可 strict load，missing/unexpected keys 均為 0。
- E2.1 formal split 已建立且通過獨立驗證；formal N 依使用者裁決先維持 200，僅在小實驗跑通後另建新版 split 擴大。輸入必須使用已凍結的 legacy CheXpert preprocessing：grayscale → histogram equalization → OpenCV `INTER_AREA` 256×256 → JPEG quality 100 round-trip → per-image min-max `[0,1]`。
- 固定 dev split：`dev_v1.1`，20 張、20 位不同病人，健康／積水各 10 張；hash 排除本機路徑。
- 正式 split：`security_v1`，200 位病人；重現學長抽樣後排除 22,253 位曾入選訓練病人，健康／積水各 100 位。
- 主線 P/S0/S1/S2a/S2 的生成一律依學長程式寫死的 `guidance_scale=-1`（y=0 健康類條件生成、單次模型呼叫、無 CFG 混合；與 `guidance_scale=0` 逐位元等價，見 AF-021；使用者 2026-10-09 裁決），inversion 為 `null=True` 無條件；論文用語不得寫成「guidance 0」或「無條件生成」。guidance=4 僅可作病灶健康化附表。
- 主要環境：`CFG_DDIM`、Python 3.10.19、PyTorch 2.14.0+cu130、RTX 5090。

## 核心規劃與紀錄

| 檔案 | 用途 |
|---|---|
| `README.md` | 本索引；回答「檔案在哪裡、做什麼、是否仍使用」。 |
| `PROPOSAL.md` | 研究方法、威脅模型、S0/S1/S2 與論文主張邊界。 |
| `WORKFLOW.md` | 八週任務、子任務 ID、pass criteria 與交付項目。 |
| `PROGRESS.md` | 只增不改的實驗日誌；所有新條目使用最新七段式格式。 |
| `loop_engineering_spec.md` | loop、安全閘門、長任務狀態機、重試與 Git 規則。 |
| `.gitignore` | 排除 dataset、checkpoint、cache、run-level/per-sample 大型輸出與 secrets。 |
| `AUDIT.md` | 稽核方的 finding 與複驗紀錄；實作方唯讀，回覆追加於 `PROGRESS.md`。 |
| `audit/README.md` | 稽核程式與輸出索引；由稽核方維護。 |

## E0/E1 可重現程式

| 檔案 | 主要功能 | 何時使用 |
|---|---|---|
| `scripts/build_e0_evidence_reset.py` | 彙整 d=4,096/d=65,536 與 seed sweep，產生 CSV、舊數字對照及去計時 stable hash。 | preflight 程式或設定改變後。 |
| `scripts/run_cfg_ddim.sh` | 固定 `PYTHONNOUSERSITE=1` 後在 CFG_DDIM 執行命令，阻止 `~/.local` 套件混入。 | 所有文件化的 CFG_DDIM 指令。 |
| `scripts/env_guard.py` | 共用 fail-closed 檢查；套件載自 active `sys.prefix` 外或 user-site 啟用時立即中止。 | E1/E2 runner 與 split builder 啟動時。 |
| `scripts/managed_run.py` | 受控執行命令，保存真實 stdout/stderr/exit code、atomic status、PID、heartbeat 與 task/GPU locks；自身設定寫入 `runner_config.json`，以 `EXPERIMENT_RUN_DIR` 把 run 目錄交給子程序；`--validate-artifacts` 時子程序成功後先以 `validate_run` 驗證 manifest/per-sample/summary，未通過即標為 failed（exit 3）。 | E2.3 起所有長任務的外層 runner。 |
| `scripts/e1_environment_inventory.py` | strict load checkpoint，記錄 Python、PyTorch、CUDA、GPU、Pillow、pytest、OpenCV、模型參數及 hash。 | 環境、套件或 checkpoint 改變後。 |
| `scripts/create_dev_split.py` | 從 CheXpert validation CSV 建立 20 位病人互斥 split；stable hash 不含 local path。 | split 版本升級時；不要為改善結果任意重抽。 |
| `scripts/create_security_split.py` | 重現學長每類 16,000 張抽樣、排除其病人，再建立固定隨機 formal split。 | E2.1 重建或擴大 formal N 時。 |
| `scripts/e1_ddim_runner.py` | headless CFG-DDIM benchmark/smoke，走學長**推論流程**（`ddim_sample_loop_known_progressive`，manifest `pipeline` 註明）；smoke 的 re-inversion 分報共用雜訊（樂觀上限）與未知雜訊（seed+1,000,000）兩版（AF-019）；可在 `managed_run.py` 下寫入受控 run 目錄。 | E1 重驗；`preprocess`／`create_runtime` 供 E2 共用。P/S0/S1 比較改用 `e2_anonymization_runner.py`。 |
| `scripts/e2_legacy_wrapper.py` | 直接呼叫學長 anonymization sampler 的 anonymize／deanonymize；`P` 為恆等金鑰（全 +1 Rademacher），與 S0/S1 只差在金鑰（AF-017）；`legacy_invert`／`apply_legacy_key`／`legacy_generate` 讓 cached x_T 可重用，並以 test 證明與完整 legacy sampler bit-exact（生成固定 guidance=-1，與學長原始碼字面一致）；CLI 檢查 wrapper/direct、guidance 0 等價、deanonymize 與 transform round-trip 皆 bit-exact。 | E2.2 重現與後續 S0/S1 pipeline 共用。 |
| `scripts/run_artifacts.py` | 共用 run 目錄寫入：standalone 時自建目錄、寫 log/exit code 並立即驗證；在 `managed_run.py` 下改寫入 `EXPERIMENT_RUN_DIR`，由 managed_run 擁有 log 並在結束後驗證。 | E1/E2 runner 寫出 manifest、per-sample、summary。 |
| `scripts/e2_anonymization_runner.py` | 學長**加密流程**（`ddim_sample_loop_anonymization`）上的 P/S0/S1 實驗：`benchmark` 量 inversion（null=True）與生成（guidance −1／0）每步成本；`p-smoke` 以恆等金鑰 P 跑 4 張 dev 圖的輸出、重算、re-inversion 與 M1（float／學長 PNG 交接）端到端還原（AF-017、AF-019）；`cache` 為 E2.3 latent cache（batch 1，`latents.npy` 存於 run 目錄，seed 1911 抽樣以 batch 1 重算，MaxAbs ≤ 1e-5）。 | E2.3 起 P/S0/S1 共用；長任務經 `managed_run.py --validate-artifacts`。 |
| `scripts/e2_metric_semantics.py` | E2.4（AF-022）：CPU、seed 1911，量測像素空間 cosine 基準（不同病人、加雜訊至指定 PSNR、去均值、cosine 對 PSNR 表）與「cosine=1 不等於相等」反例，並與稽核方數字比對。 | 指標語意報告的數字來源。 |
| `scripts/artifact_schema.py` | 建立／驗證 manifest、per-sample CSV、summary；檢查 ID、finite 與摘要一致性。 | 新實驗 runner 寫出結果後。 |
| `scripts/check_staged_files.sh` | commit 前拒絕禁傳路徑、secret 名稱及超過 90 MiB 的 staged file。 | 每次 commit 前必跑。 |

## E0/E1 結果檔

| 檔案 | 說明／如何解讀 |
|---|---|
| `canonical_preflight.csv` | E0 唯一 canonical synthetic preflight 摘要；舊 31/32、98/100 不得混用。 |
| `reports/evidence_reset.md` | 說明哪些舊 claim 被撤回或限制，以及現版 preflight 能支持什麼。 |
| `reports/preprocessing_audit.md` | 學長實際 CheXpert 前處理的程式證據、與本 runner 的一致性及剩餘限制。 |
| `reports/metric_semantics.md` | E2.4（AF-022）：前作「Cosine Sim 0.9989」在論文（latent）與程式（輸出影像、base 對解密輸出）的差異與行號；像素 cosine 基準；本專案 latent／影像指標定義與「對 x0／對 P 輸出」並列；論文用語規範。 |
| `reports/t2wb_protocol.md` | AF-019：可逆性以影像端端到端為主指標、latent 只作診斷；加密流程 P smoke 事實；T2-WB 攻擊的 P 正控制／負控制定義，以及 `Inconclusive` 判定門檻提案（待使用者裁決）。 |
| `reports/compute_budget.md` | E2–F9 各 scheme/attack 在加密流程上的 forward／generation 半週期矩陣（實測 batch 1/4/8 成本、guidance −1）與 N=200/500/1,000 GPU 時數；記錄 N=200 裁決，以及 batch 組成會改變結果（batch 4 vs 1 MaxAbs 6.9e-4）而正式 pass 固定 batch 1 的決定。 |
| `artifacts/environment_baseline.txt` | 當次環境、套件實際來源、GPU、checkpoint hash、freeze hash 與 strict-load 結果。 |
| `artifacts/environment_freeze.txt` | 由 CFG_DDIM 的 `pip freeze --all` 產生，供 baseline hash 與重建。 |
| `model_inventory.csv` | 一列式 checkpoint/model inventory，適合程式與試算表讀取。 |
| `splits/dev_v1.csv` | 歷史 dev split；hash 含 local path，不再作新 run 預設。 |
| `splits/dev_v1.1.csv` | 現行 20 張固定 dev images；成員同 v1，stable split hash 排除 local path。 |
| `splits/security_v1.csv` | E2 formal 200 人 split，健康／積水各 100；病人與重建的前作訓練抽樣互斥。 |
| `splits/security_v1_manifest.json` | formal split seed、抽樣規則、排除人數、hash 與 label counts。 |
| `results/E1.2_benchmark.json` | 推論流程 batch 1/4/8 的 cycle 時間、估計 noise=500 時間與 peak VRAM（managed run `E1.2_20261009T153114394206Z_8339c2f5`）；加密流程成本改見 `AF021_anonymization_benchmark.json`。 |
| `results/E1.4_ddim_smoke.csv` | **推論流程**（病灶定位用，`ddim_sample_loop_known_progressive`）四張 x0→z→xrec 的分段 runtime、VRAM、finite、image/latent metrics；re-inversion 分共用雜訊（cos 0.906，樂觀上限）與未知雜訊（cos 0.371）兩版（AF-019）；不是 P/S0/S1 比較用的 P 基線（AF-017）。由 managed run `E1.4_20261009T153121667310Z_aa04e79c` 產生。 |
| `results/AF021_anonymization_benchmark.json` | 加密流程 batch 1/4/8 的 inversion／生成（guidance −1 與 0）每步每張成本與半週期估計；`compute_budget.md` 的依據。 |
| `results/AF017_P_anonymization_smoke.csv` | 恆等金鑰 P 在加密流程上的 4 張 smoke：P 輸出、inversion 重算、re-inversion（float／PNG）latent 指標與低／高頻、\|z\| 診斷，以及 M1 端到端影像指標。 |
| `image/AF017_P_anonymization_smoke.png` | 上述 4 張的 original、P 輸出、M1 float、M1 PNG 與固定色階 0–0.1 差異圖。 |
| `results/E2.4_metric_semantics.json` | E2.4 像素 cosine 基準與反例的數值（dev_v1.1 20 張、190 對），含與稽核方的 `cross_check`。 |
| `results/E2.3_DONE.json` | E2.3 正式 latent cache（`security_v1` 200 張，batch 1）的 metadata：run ID、`latents.npy` 路徑與 SHA-256、shape／finite 檢查、seed 1911 抽 20 張以 batch 1 重算的 MaxAbs（全為 0）。cache 本體在 `artifacts/runs/E2.3/<run_id>/latents.npy`，不入 Git。 |
| `results/E2.3_dev_v1.1_cache.json` | 同上，dev_v1.1 20 張（E2.5 的輸入），抽 5 張重算。 |
| `results/E2.2.json` | 單張 noise=500、guidance=-1 的 P（恆等金鑰）/S0/S1 wrapper 對 legacy direct-call regression，P 另比對無金鑰 forward→backward：anonymize、deanonymize、guidance 0 等價對照與 transform round-trip 的 MaxAbs；不含 raw key。 |
| `image/E1.4_ddim_smoke_contact_sheet.png` | 四列視覺檢查圖；每列是 original、reconstruction、absolute difference。 |
| `artifacts/preflight/canonical_direction_candidates_d4096_n100.json` | d=4,096、N=100 canonical preflight 原始輸出。 |
| `artifacts/preflight/canonical_direction_candidates_d65536_n100.json` | d=65,536、N=100 canonical preflight 原始輸出。 |
| `artifacts/preflight/canonical_sot_seed_sweep_d65536_n100.json` | seeds 0–99、R=1/2/4 bit-exact sweep 原始輸出。 |
| `artifacts/preflight/direction_candidates_cpu*.json` | 舊抽樣數的 historical preflight，只供追溯，不是正式引用來源。 |

## 測試

| 檔案 | 驗證內容 |
|---|---|
| `tests/unit/test_artifact_schema.py` | valid synthetic artifact 應通過；duplicate ID、NaN/Inf、空 numeric fields、缺欄或 exit code 不一致應失敗。 |
| `tests/unit/test_env_guard.py` | active-prefix 正控制，以及越界套件／啟用 user-site 的 fail-closed 負控制。 |
| `tests/unit/test_managed_run.py` | 真 stdout 成功控制與 traceback/非零 exit/status=failed 負控制；`--validate-artifacts` 的有效 artifacts 正控制，以及 NaN、缺 manifest 兩個負控制。 |
| `tests/unit/test_chexpert_preprocessing.py` | 直接載入學長原始前處理函式作 oracle，逐值守住 equalize/INTER_AREA/JPEG/min-max。 |
| `tests/unit/test_split_hash.py` | 驗證更換 local path 前綴不改變 split hash。 |
| `tests/unit/test_run_artifacts.py` | 共用 run 目錄寫入：standalone run 通過 validator 且 git commit 取自 run 開始時；managed 模式沿用 `EXPERIMENT_RUN_DIR`。 |
| `tests/unit/test_staged_guard.py` | 在暫存 repo 驗證 checkpoint 副檔名與 private-key 內容會被拒絕。 |
| `tests/integration/test_legacy_ddim_equivalence.py` | GPU 比對 wrapper 與學長 progressive sampler 的 latent/reconstruction bit-exact。 |
| `tests/test_legacy_repro.py` | E2.2：P/S0/S1 transform 精確可逆；GPU 比對 P/S0/S1 anonymize／deanonymize wrapper 與 legacy sampler bit-exact、guidance −1 與 0 逐位元相同（AF-021），以及恆等金鑰 P 等同無金鑰的 legacy forward→backward（AF-017）；cached x_T 路徑（invert → key → generate）與完整 legacy sampler bit-exact（E2.3）。 |

執行：

```bash
scripts/run_cfg_ddim.sh python -m pytest -q tests/unit
```

## 攻擊與設計驗證程式

| 檔案 | 用途 |
|---|---|
| `attack/README_trackA.md` | Track A 攻擊腳本的入口說明與執行順序。 |
| `attack/attack_keyspace.py` | 審計舊方法 seed/key-space 與 brute-force 正向控制。 |
| `attack/attack_poc.py` | 早期攻擊 proof-of-concept；主要供歷史追溯。 |
| `attack/attack_tier1_real.py` | 對真實 latent 執行 Tier-1 linkage/attack。 |
| `attack/attack_trackA_latent.py` | Track A latent-level invariants、KPA/CPA 相關實驗。 |
| `attack/defense_design_check.py` | 歷史候選防禦檢查；docstring 表格非 canonical，須搭配 evidence reset 解讀。 |
| `attack/validate_sot_bitexact_sweep.py` | 重建指定維度、seed 區間與 R 的 float32 bit-exact 計數。 |
| `attack/validate_direction_candidates.py` | E0 CPU synthetic preflight：SOT-WHT、CDF torus pad、KCI toy chain。 |
| `attack/validate_direction_candidates.py.orig` | 該腳本的歷史備份，不應當作目前執行入口。 |
| `attack/verify_cpa.py` | 驗證固定線性 transform 在 chosen-plaintext queries 下的恢復。 |
| `attack/verify_hh_formula.py` | 驗證 Householder 相關公式與數值實作。 |

## 學長程式：先看哪些文件

| 檔案 | 用途 |
|---|---|
| `past/SourceCode/SHELL_SCRIPTS_GUIDE.md` | 22 個 shell scripts 的逐檔用途、可否直接跑及需要修的路徑。 |
| `past/SourceCode/README/CFG_DDIM_README.md` | CFG-DDIM 資料、preprocessing、模型旗標與訓練／推論使用法。 |
| `past/SourceCode/README/ANONYMIZATION_INTEGRATION_REPORT.md` | 匿名化整合方式、pixel-space latent shape 與參數。 |
| `past/SourceCode/README/SIGNED_PERMUTATION_REPORT.md` | Signed Permutation 實作與過往驗證說明。 |
| `past/SourceCode/安裝環境教學.md` | 學長原環境安裝紀錄；目前以 CFG_DDIM inventory 為準。 |
| `past/SourceCode/requirements.txt` | 舊專案依賴列表；不可直接覆蓋目前 conda 環境。 |

## 學長 shell scripts 快速用途

完整風險與路徑修正見 `past/SourceCode/SHELL_SCRIPTS_GUIDE.md`。

| 檔案 | 主要功能 |
|---|---|
| `start_visdom_cfg_ddim.sh` | 啟動本機 Visdom 8850。 |
| `run_cfg_training.sh` | 訓練原圖／deblur CFG diffusion models。 |
| `run_clf_model_training.sh` | 訓練 classifier-guidance diffusion backbone。 |
| `run_clf_classifier_training.sh` | 訓練 classifier-guidance 的獨立 classifier。 |
| `run_cfg_inference.sh` | CFG-DDIM 一般推論。 |
| `run_cfg_inference_Crypto.sh` | CFG/實際活動命令為 uncond 的匿名化與解匿名化。 |
| `run_clf_inference_Crypto.sh` | CLF-DDIM 匿名化／解匿名化。 |
| `run_uncond_ddim_inference.sh` | guidance=0 的 DDIM baseline。 |
| `run_uncond_inference_Crypto.sh` | uncond 90 張匿名化／解匿名化。 |
| `run_inference_hyp.sh` | 掃描 noise level 與 guidance scale。 |
| `run_evaluation_metrics_hyp.sh` | 評估超參數掃描 NPZ。 |
| `run_evaluation_metrics.sh` | 比較 FPGAN、CLF-DDIM、CFG-DDIM metrics。 |
| `run_figure_maker.sh` | 從 NPZ 產生論文比較圖。 |
| `run_encryption_reversibility_eval_sample3.sh` | 三張快速可逆性評估。 |
| `run_encryption_reversibility_eval_sample90.sh` | 九十張正式可逆性評估。 |
| `run_paper_comparison.sh` | 多模型、多 key scheme 的總比較入口草案。 |
| `run_comparison_experiments.sh` | 早期匿名化比較原型。 |
| `run_training.sh`、`run_training_v2.sh` | 舊版訓練設定，只供追溯。 |
| `run_inference.sh`、`run_inference_v2.sh`、`run_experiment_orig.sh` | 舊版推論設定，只供追溯。 |

## 學長 Python 核心檔案

| 檔案 | 用途 |
|---|---|
| `past/SourceCode/scripts/chexpert_preproc.py` | 原始 CheXpert 做 histogram equalization、256 resize、JPEG 輸出與 label/path 重建。 |
| `past/SourceCode/scripts/cfg_image_train.py` | CFG diffusion 訓練入口。 |
| `past/SourceCode/scripts/cfg_image_sample.py` | CFG DDIM inversion/reconstruction 推論。 |
| `past/SourceCode/scripts/cfg_image_sample_anonymization.py` | 舊 Rademacher/Signed Permutation 匿名化與解匿名化。 |
| `past/SourceCode/scripts/cfg_image_sample_enhanced_anonymization.py` | 後期增強匿名化實驗入口。 |
| `past/SourceCode/scripts/unconditional_ddim_anonymization.py` | guidance=0 匿名化基線。 |
| `past/SourceCode/scripts/image_train.py`、`image_sample.py` | classifier-guidance diffusion backbone 的訓練／採樣。 |
| `past/SourceCode/scripts/classifier_train.py`、`classifier_sample_known*.py` | noisy classifier 訓練與 classifier-guided sampling。 |
| `past/SourceCode/scripts/evaluation_metrics*.py` | 影像品質、定位、可逆性及比較 metrics。 |
| `past/SourceCode/scripts/figure_maker_select.py` | 選案例並排版論文圖片。 |
| `past/SourceCode/scripts/test_enhanced_anonymization.py` | 舊增強匿名化的自測。 |
| `past/SourceCode/guided_diffusion/bratsloader.py` | CheXpert/BRATS dataset、label filtering、min-max loader。 |
| `past/SourceCode/guided_diffusion/gaussian_diffusion.py` | DDPM/DDIM forward、reverse、inversion、anonymization 核心。 |
| `past/SourceCode/guided_diffusion/anonymization.py` | Rademacher、Permutation、Signed Permutation key transforms。 |
| `past/SourceCode/guided_diffusion/script_util.py` | model/diffusion defaults 與 factory。 |
| `past/SourceCode/guided_diffusion/unet_v1.py` | 目前 checkpoint 對應的 v1 U-Net。 |
| `past/SourceCode/guided_diffusion/unet.py`、`unet_v2.py` | 其他歷史 U-Net variants。 |
| `past/SourceCode/guided_diffusion/train_util.py` | training loop、checkpoint 與共用 visualize/min-max。 |
| `past/SourceCode/guided_diffusion/respace.py`、`resample.py` | timestep respacing 與 training schedule sampler。 |
| `past/SourceCode/guided_diffusion/dist_util.py`、`logger.py` | device/distributed/checkpoint I/O 與 logging。 |
| `past/SourceCode/guided_diffusion/nn.py`、`losses.py`、`fp16_util.py` | 網路元件、diffusion losses 與 mixed precision utilities。 |

## 舊結果與大型本機資料

| 路徑 | 用途／規則 |
|---|---|
| `past/SourceCode/results/encryption_reversibility_evaluation*/` | 學長過往可逆性 CSV、報告與比較圖；只作歷史證據，不覆寫。 |
| `past/SourceCode/results/visdom_cfg_inference_*/` | 過往 Visdom inference log/progress。 |
| `past/SourceCode/results/Model/` | checkpoint；Git 排除，禁止上傳。 |
| `dataset/` | 原始資料集；Git 排除且實驗時唯讀。 |
| `artifacts/runs/` | 未來長任務的 run directory；含 log/status/per-sample，本機保存、不提交。 |
| `refpaper/` | 參考論文 PDF；通常不提交大型檔。 |
| `past/Documents/` | 學長論文原始 LaTeX、圖與歷史文件。 |

## 找檔案的建議順序

1. 查任務定義：`WORKFLOW.md`。
2. 查實際做過什麼：`PROGRESS.md`。
3. 查方法與安全主張：`PROPOSAL.md`。
4. 查小型正式結果：`results/`、`reports/`、`splits/`、`image/`。
5. 查學長舊流程：先看 `past/SourceCode/SHELL_SCRIPTS_GUIDE.md`，再讀對應 script。
6. 查長任務：依 `task_id/run_id` 到 `artifacts/runs/`，不要只看 stdout。
