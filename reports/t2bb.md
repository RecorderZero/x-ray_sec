# T2-BB Inner-Risk 實驗：檔案說明與操作流程

> 本檔是 T2-BB（匿名圖之間 anon↔anon linkage）實驗的「導覽 / 操作手冊」。
> 研究設計與完整決策紀錄見 `reports/t2bb_inner_linkage_protocol.md`（§15 為定稿結論）；
> 執行日誌見 `PROGRESS.md`（2026-10-20 條目）。所有 Python 一律經 `scripts/run_cfg_ddim.sh`。

---

## 0. 目前狀態（2026-10-20 actual-M1 pilot 完成）

舊 pilot 證明前作 S0/S1 在**加密 latent 的直接統計**上透明，但該實驗把 latent 正規化成 PNG 後直接評分，並不是部署時真正公開的 DDIM-generated M1。現在新增較嚴格的 `T2-BB-M1-Frozen`：先產生真正的 P/S0/S1/S2 公開匿名 PNG，再由完全不接觸 latent、checkpoint、key 或原圖的 frozen DenseNet121 / RAD-DINO 做 anon↔anon linkage。舊結論保留為「結構探針」，新的 M1 結果另列，不互相取代。

---

## 1. 這是什麼實驗

- **威脅模型 T2-BB**：攻擊者只拿得到匿名產物（沒有金鑰、沒有模型、沒有原圖），問「多張匿名圖是否屬於同一病患」（unlinkability / inner-risk）。
- 和 T2-WB / A4.6 的差別：那裡測 anon↔**real**（outer-risk）；這裡測 anon↔**anon**（inner-risk）。
- 定位：PROPOSAL §3 標 **P2 選做、另表報告**，與主表、F9/G5 凍結範圍無相依。

---

## 2. 檔案清單與說明

### 2.1 設計 / 文件
| 檔案 | 說明 |
|---|---|
| `reports/t2bb_inner_linkage_protocol.md` | 完整設計 doc（§1–§14）＋ 定稿結果結論（§15）。 |
| `reports/t2bb.md` | 本檔：導覽與操作流程。 |
| `PROGRESS.md`（2026-10-20 條目） | 執行日誌（做了什麼、得到什麼）。 |

### 2.2 Split 名單（已入庫）
| 檔案 | 內容 | 備註 |
|---|---|---|
| `splits/t2bb_inner_v1.csv` | **Victim 測試集**：20 病患 × 2 = 40 列 | 同病患跨 study 各取 1 張；`pair_id`/`pair_role`(gallery/query) |
| `splits/t2bb_inner_v1_manifest.json` | victim split 的 provenance | 排除訓練/security 病患、SHA256、label 分布 |
| `splits/t2bb_shadow_v1.csv` | **Shadow 訓練集**：1000 病患 × 2 = 2000 列 | 學習式攻擊者「自己的」標註資料 |
| `splits/t2bb_shadow_v1_manifest.json` | shadow split 的 provenance | 固定全域 permutation(seed 1911)、disjoint 檢查 |

> **「200」不是獨立檔**：它是 `t2bb_shadow_v1.csv` 的**前 400 列（前 200 病患）**。split 以 seed 1911 的固定全域 permutation 排序，任何前綴都巢狀且可重現 —— 這讓「先跑 200、之後補到 1000」不需重建名單（prefix-nesting，protocol §10/§11）。

### 2.3 腳本
| 腳本 | 角色 | 輸入 → 輸出 |
|---|---|---|
| `scripts/build_t2bb_inner_split.py` | 建 victim split | CheXpert CSV → `splits/t2bb_inner_v1.csv` |
| `scripts/build_t2bb_shadow_split.py` | 建 shadow split（含 1000 的固定前綴排序） | CheXpert CSV → `splits/t2bb_shadow_v1.csv` |
| `scripts/t2bb_invert_cache.py` | **增量反演快取**：DDIM 反演 latent，以 `file_sha256` 為鍵、可前綴重用 | split → `artifacts/t2bb_latent_cache/`（gitignored） |
| `scripts/t2bb_step1_image_control.py` | **步驟 1**：影像層 positive control（DenseNet121/AlexNet/rawpix） | victim split → `table_t2bb_step1_image_control.*` |
| `scripts/t2bb_inner_linkage.py` | **步驟 2**：反演+加密(S0/S1/S2)+存檔 view+scorer(cos/abs/sorted_abs/norm/densenet121) | victim split(+快取) → `table_t2bb_inner_linkage.*` |
| `scripts/t2bb_learned_attack.py` | **學習式上界攻擊**：SupCon metric net，shadow 訓練、victim 評估 | shadow+victim split(+快取) → `table_t2bb_learned_attack.*` |
| `scripts/t2bb_m1_generate.py` | **目標端匯出器**：cached latent→P/S0/S1/S2→實際 DDIM generation→部署口徑公開 PNG；公開 manifest 使用 opaque ID，真值另存私有 map | victim split(+快取+checkpoint) → public export + sibling private eval map |
| `scripts/t2bb_m1_frozen_linkage.py` | **真正 BB 評估端**：只讀公開 PNG，以 ImageNet DenseNet121 / CXR RAD-DINO frozen embedding 評分；embedding 完成後才 join 私有 pair truth | public export + private eval map → `table_t2bb_m1_frozen.*` |

### 2.4 反演快取（實驗真正吃的資料，**gitignored、未入庫**）
- 路徑：`artifacts/t2bb_latent_cache/`
  - `latents/<sha>.npy` — 每張圖一個 (256,256) float32 latent（目前 440 個 = victim 40 + shadow 前 200×2）。
  - `manifest.json` — checkpoint SHA256、noise_level=500、sampler、image_size、seed。
  - `index.csv` — sample_id / patient_id / pair_role / file_sha256 / split。
- 為何不入庫：大、且可由 split 完全重生；`.gitignore` 的 `/artifacts/t2bb_latent_cache/` 規則排除。
- **config 鎖**：若 manifest 的 checkpoint/noise level 與當前 run 不符會直接拒絕寫入，避免混入不同設定的 latent。

### 2.5 結果表（已入庫）
| 檔案 | 內容 |
|---|---|
| `paper_assets/tables/table_t2bb_step1_image_control.{csv,json}` | 步驟 1：影像層 Top-1/Top-5/AUC/neg-control |
| `paper_assets/tables/table_t2bb_inner_linkage.{csv,json}` | 步驟 2：各 (scheme, storage, scorer) 的 Top-1/Top-5/AUC + 正/負控制 + decrypt max_abs |
| `paper_assets/tables/table_t2bb_learned_attack.{csv,json}` | 學習式：shadow-val / victim-learned / victim-fixed 三組 + efficacy verdict |
| `paper_assets/tables/table_t2bb_m1_frozen.{csv,json}` | 真正公開 M1 上的 frozen-encoder Top-1/Top-5/AUC、conditional-query bootstrap CI、permutation null/p-value（完成後產生） |

### 2.6 T2-BB-M1-Frozen 的硬隔離

- 公開匯出目錄只含 PNG、opaque 隨機 ID、method/storage/hash/公開 nonce；**不含** `patient_id`、`pair_id`、`pair_role` 或可解碼檔名。
- 評估真值在公開目錄之外的 sibling private map。scorer 先由 PNG 算 embedding/similarity，最後才 join truth 計算指標。
- P/S0/S1 採目前 mixed-deployment 的 predecessor legacy PNG；S2 採 private-range 容器的 public q/255 view。這是端到端方法差異，不是 storage-controlled attribution。
- N=20 為 exploratory pilot、單一隨機 S2 key draw；CI 是固定 gallery 下的 conditional-query bootstrap，不能解讀成正式族群推論。

---

## 3. 操作流程（end-to-end）

> 前置：`dataset/...` CheXpert 四個 batch、`past/SourceCode/...` 前作原始碼、模型 checkpoint 已就位（與 security pipeline 相同）。seed 一律 1911。

### 步驟 A — 建 split（CPU，秒級）
```bash
# A1. victim 20 病患
scripts/run_cfg_ddim.sh python -m scripts.build_t2bb_inner_split \
  --csv "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv" \
  --legacy-csv past/SourceCode/data/CheXpert-v1.0/train.csv \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)" \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)" \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)" \
  --patients 20

# A2. shadow 1000 病患（含固定前綴排序；200 pilot 即其前綴）
scripts/run_cfg_ddim.sh python -m scripts.build_t2bb_shadow_split \
  --csv "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv" \
  --legacy-csv past/SourceCode/data/CheXpert-v1.0/train.csv \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)" \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)" \
  --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)" \
  --patients 1000
```

### 步驟 B — 反演快取（GPU；victim 40 ~5min，shadow 200=400張 ~52min，合計 ~57min）
```bash
# B1. victim 40 張
scripts/run_cfg_ddim.sh python -m scripts.t2bb_invert_cache --split splits/t2bb_inner_v1.csv
# B2. shadow 前 200 病患（400 張）；之後要 1000 時改 --max-patients 1000，只補後面 1600
scripts/run_cfg_ddim.sh python -m scripts.t2bb_invert_cache --split splits/t2bb_shadow_v1.csv --max-patients 200
```
> 反演確定性（DDIM eta=0、固定 checkpoint、null conditioning），同一 `file_sha256` 的 latent 恆等、可重用 → 這是 prefix-nesting 能 reuse 的根據。吞吐約 7.8–8.6 s/img。
> **>30 分鐘的 GPU run 依 CLAUDE.md §3 須先取得使用者裁決。**

### 步驟 C — 步驟 1：影像層 positive control（輕量 GPU/CPU，分鐘級）
```bash
scripts/run_cfg_ddim.sh python -m scripts.t2bb_step1_image_control --split splits/t2bb_inner_v1.csv
```

### 步驟 D — 步驟 2：latent/anon 層 linkage（reuse 快取，CPU 秒級）
```bash
scripts/run_cfg_ddim.sh python -m scripts.t2bb_inner_linkage --split splits/t2bb_inner_v1.csv
# 覆寫既有表需加 --force（output_guard 預設拒絕覆寫）
```

### 步驟 E — 學習式上界攻擊（reuse 快取，CPU 分鐘級）
```bash
scripts/run_cfg_ddim.sh python -m scripts.t2bb_learned_attack \
  --shadow-split splits/t2bb_shadow_v1.csv --max-patients 200 \
  --victim-split splits/t2bb_inner_v1.csv
```
> `--max-patients N` 取 shadow 前 N 病患（讀檔後即切片）；channels 預設 `s0_abs`（前作可遷移的 \|z\| 不變量）與 `s2_anon`（per-image nonce 對照，預期隨機）。

### 步驟 F — 實際公開 M1 + frozen encoder（GPU 約 21–24 分鐘）

```bash
# 先 smoke：2 病患 × 2 影像 × 4 方法 = 16 次 generation
scripts/run_cfg_ddim.sh python -m scripts.t2bb_m1_generate \
  --limit-patients 2 --out-dir artifacts/t2bb_m1_public_smoke

# 完整 victim pilot：20 × 2 × 4 = 160 次 generation
scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id T2BB_M1_GEN --gpu-id 0 \
  --heartbeat-seconds 30 -- python -m scripts.t2bb_m1_generate \
  --out-dir artifacts/t2bb_m1_public_v1

# encoder 端只讀 public export；truth map 僅供最後計分
scripts/run_cfg_ddim.sh python -m scripts.t2bb_m1_frozen_linkage \
  --export-dir artifacts/t2bb_m1_public_v1
```

> smoke 實測為 7.85 秒/generation，因此 160 次約 21 分鐘；低於 30 分鐘 GPU gate。RAD-DINO 執行時 `local_files_only=True`，結果記錄 resolved revision 與權重/config SHA-256。

---

## 4. 要「scale 到 1000」怎麼做（reuse 設計）

名單不用重建（1000 的前綴就是 200）。只需：
```bash
# 1) 補反演後面 800 病患（+1600 張，自動跳過已快取的 400；約 ~3.4h GPU，需 gate）
scripts/run_cfg_ddim.sh python -m scripts.t2bb_invert_cache --split splits/t2bb_shadow_v1.csv --max-patients 1000
# 2) 重訓/重評
scripts/run_cfg_ddim.sh python -m scripts.t2bb_learned_attack \
  --shadow-split splits/t2bb_shadow_v1.csv --max-patients 1000 \
  --victim-split splits/t2bb_inner_v1.csv --force
```
> 定稿判定為**不執行**（cos≈abs≈norm≈0.66 上界 + 學習式過擬合皆指向 latent linkage 本就弱，scale 期望值低）。此處僅留操作說明。

---

## 5. 舊結果摘要（direct-latent 結構探針；非實際公開 M1）

**步驟 1（影像層）**：DenseNet121 Top-1=0.550、AUC=0.811 → 同病患 biometric 訊號在影像層存在。

**步驟 2（latent/anon 層）**：decrypt 正控制 S0/S1 max_abs=0.0、S2=9.5e-7。

| scheme | storage | scorer | Top-1 | AUC |
|---|---|---|---|---|
| plaintext_latent | — | cos / abs / norm | 0.30 / 0.20 / 0.15 | 0.659 / 0.657 / 0.680 |
| S0 = S1 | range_preserving | cos | 0.25 | **0.657**（≈ plaintext cos → 透明性） |
| S0 = S1 | minmax | cos / abs | 0.00 | 0.512（per-image minmax 回隨機） |
| **S2** | minmax | cos / abs | 0.05 | 0.499 |
| **S2** | range_preserving | cos | 0.00 | 0.433 |

**學習式（pilot 200）**：train loss→0.0048 但 shadow-val AUC=0.48、victim AUC=0.468 < 固定 0.657，`efficacy=false`（過擬合；合成 smoke 曾達 Top-1=1.0，證非 trainer bug）。

上述數字回答「latent 結構是否透明」，不再單獨用來結案整個 T2-BB。真正部署輸出的結論以 `table_t2bb_m1_frozen.*` 為準。

## 6. Actual-M1 frozen-encoder 結果（N=20 pilot）

| method | storage | DenseNet Top-1 / AUC | RAD-DINO Top-1 / AUC |
|---|---|---:|---:|
| P | legacy | 0.50 / 0.810 | **0.75 / 0.976** |
| S0 | legacy | 0.00 / 0.434 | 0.10 / 0.532 |
| S1 | legacy | 0.05 / 0.479 | 0.05 / 0.498 |
| S2 | private-range public q | 0.10 / 0.558 | 0.00 / 0.510 |

- [事實] P positive control 很強：DenseNet AUC=0.810、RAD-DINO AUC=0.976，兩者 permutation p=0.0002。這證明實際 DDIM-generated P M1 仍保留可連結的病患訊號，故 encrypted methods 的 null 可以被解讀。
- [事實] S0/S1/S2 的 AUC 全落在 0.434–0.558，所有 permutation p≥0.098；Top-1=0–0.10，接近 20 人 gallery 的 random 0.05。兩個 frozen encoder 都沒有顯著 anon↔anon linkage。
- [結論]（AF-041）在本 pilot 的**雙重前提下——(a) 通用 image-encoder 介面、(b) legacy range-discarding（minmax）儲存**——S0/S1 與 S2 的 encoder 可見 cross-study linkage 皆落回隨機。**此與 §5 的 minmax 列一致（S0/S1 minmax cos 0.512），並非推翻透明性**：在 range_preserving 儲存下 S0/S1 的 `cos` linkage 仍為 **0.657**，scheme-aware 的 latent 攻擊者仍可連結（§6 用的正是 legacy range-discarding PNG，故只在此 PNG 情境下不能沿用 direct-latent probe 的「S0/S1 可被顯著連結」）。正控制 P（同 legacy minmax）仍 0.976，是因 P 的 latent-as-image 保留 CXR 空間結構、而 S0/S1 的 sign-flip/permutation 打亂成 noise-like——此只證明 encoder 能讀 image-domain 特徵，**無法排除 range_preserving 的 `cos` 旁道**，也不是 scheme-level 安全證明。
- [限制] 此 null 只支持「兩個指定 frozen encoder 未成功」，不是通用 BB 安全證明。S2 DenseNet AUC=0.558、p=0.098 為接近但未達顯著的探索性結果；N=20 CI 寬，不能宣稱 S2 優於 S0/S1。若要強化 BB 上界，下一步應是 shadow-trained image encoder 或擴大 victim/key repetitions，而不是從本 pilot 做安全證明。

---

## 7. 注意事項

- **seed=1911** 固定所有隨機來源；影像正規化依 security pipeline（`e1_ddim_runner.preprocess`，[0,1]）。
- **cos scorer 的意義**：是「透明性探針」（正交映射保內積，`cos(z_ano)=cos(z)`），非強 biometric；linkage 判定看 verification AUC 的背景分布，不是單一對的 cos 值。
- **confound 揭露**：split 與前作訓練集/security_v1 病患 disjoint（manifest 記錄），控制記憶化/訓練重疊。
- **快取不入庫**：`artifacts/t2bb_latent_cache/` 已 gitignored；要共享需重跑步驟 B 重生。
- **output_guard**：寫結果表預設拒絕覆寫，需 `--force`。
