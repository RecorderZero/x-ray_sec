# WORKFLOW 修訂版：既有擴散架構的密碼分析與低改動安全強化

> 日期：2026-10-05 至 2026-11-30  
> 研究設計：[PROPOSAL.md](PROPOSAL.md)  
> AI 執行規範：[loop_engineering_spec.md](loop_engineering_spec.md)  
> 原始 WORKFLOW.md 保留不修改

---

## 0. 目標與完成定義

11 月 30 日前完成：

1. 重現並量化前作的 32-bit key-space、magnitude／sorted-magnitude、key reuse、KPA／CPA 與 tampering 問題。
2. 實作完整 S2：
   - 256-bit master key 或 password KDF。
   - HKDF-SHA256 domain separation。
   - ChaCha20／AES-CTR CSPRNG。
   - per-image unique nonce。
   - WHT-based structured transform；R=1/2/3/4 與 permutation on/off 消融後才選正式設定。
   - HMAC-SHA256 transform-then-MAC。
3. 以相同攻擊比較 P、S0、S1、S2a、S2。
4. 分開報告 T1 精確 latent 與 T2-WB target-model anonymous-image inversion；T2-BB surrogate attack 僅在 P0 完成後選做。
5. 完成 security、correctness、quality、runtime 的表格與圖片。
6. 11 月 23 日停止新增方法，11 月 30 日凍結所有實驗。

本 workflow 不要求：

- 重新訓練模型。
- 病灶定位或醫師評估。
- 達成 AES／AEAD 等級的 confidentiality。
- structured transform 在 65,536 維達到 100% bit-exact。
- adaptive norm attack 必須失敗。
- 實作 CDF、EDICT、BDIA 或 KCI/O-BELM。

---

## 1. 固定研究範圍

### 1.1 方案代號

| 代號 | 定義 | 地位 |
|---|---|---|
| P | 不加密 | 正控制 |
| S0 | 前作 Rademacher | 被攻擊基線 |
| S1 | 前作 Signed Permutation | 被攻擊基線 |
| S2a | HKDF＋CSPRNG＋nonce＋WHT-based structured transform，不含 MAC | transform／leakage ablation |
| S2 | S2a＋HMAC-SHA256 | authenticated obfuscation 完整方案 |
| AEAD-ref | AES-GCM／ChaCha20-Poly1305 加密原始 bytes | 安全與 byte-exact 參考 |

原 workflow 的 S3 已取消；完整性不是加分項，而是 S2 的必備部分。

### 1.2 威脅模型

| 代號 | 攻擊者能力 |
|---|---|
| T1 | 取得精確 z_ano、nonce、header、tag |
| T2-WB | 取得 anonymous image 與目標 checkpoint／pipeline，可自行 inversion；P0 主情境 |
| T2-BB | 只有 anonymous image／論文，沒有目標 model，只能做 surrogate／embedding attack；P2 選做 |
| T3 | 取得已知明密文對或可做 chosen-plaintext query |
| T4 | 可修改 image、latent payload、nonce、tag、header 或 authenticated metadata |

### 1.3 儲存模式

| 模式 | ciphertext／資料 | 用途 |
|---|---|---|
| M1 image-only | anonymous image＋nonce＋tag＋manifest | 最接近前作，受 inversion error 影響 |
| M2 latent payload | authenticated z_ano＋anonymous preview | 可逆性上限與部署對照 |

### 1.4 優先級

| 優先級 | 必須完成 |
|---|---|
| P0 | protocol、S0/S1 攻擊、S2 key pipeline、structured transform、MAC、T1/T2-WB、adaptive norm、正式統計、freeze |
| P1 | M2 latent payload、wrong/partial key、quality 消融、runtime／storage |
| P2 | T2-BB surrogate／embedding attack、noise-level sweep、第二資料集、進階 re-ID model |

時間不足時依序取消 P2、M2 大樣本、partial-key 圖；不得取消 MAC、adaptive norm attack 或實驗素材凍結。

---

## 2. 重要修正

### 2.1 Structured transform 數值判準

不得再使用：

> d=65,536、R=1/2/4 必須 bit-exact。

改為：

- float64 MaxAbsError ≤ 1e-12。
- RMSE、cosine 與轉回 float32 後的 bit-exact rate 全部記錄。
- d=4,096 與 d=65,536 分開報告。
- 若 bit-exact rate < 100%，不得以重跑 seed 隱藏。

目前既有 preflight 互相不一致：

- d=4,096：測試樣本 bit-exact rate 100%。
- 舊 artifact 曾記錄 d=65,536 為 31/32。
- 現版 defense_design_check.py 的預設 seed 在 d=65,536、R=1/2/4 均為 False。
- 2026-10-05 對 seeds 0–99、d=65,536、R=2 快速稽核為 98/100。


### 2.2 Security 判準

不得把下列結果當成「S2 已安全」：

- KS p-value > 0.05。
- original magnitude Top-1 降至 random。
- wrong-key image 看起來像雜訊。

必須另外測：

- norm-only linkage。
- same plaintext repeated encryption。
- reused nonce。
- KPA／CPA within one transform。
- cross-nonce transfer。
- tamper rejection。

S2 的 norm-only attack 預期可能成功，這是應報告的 limitation，不是自動觸發重做變換。trace／cosine 只作 diagnostic，不得作為通用安全判準。

### 2.3 MAC 提前為核心工作

HMAC 不再排在後期加分項。Week 3 建立 authenticated container，Week 4 整合 structured transform 時即完成 S2。

驗證 tag 以前：

- 不得執行 inverse transform。
- 不得送入 diffusion generation。
- 不得輸出部分 plaintext。

### 2.4 Nonce 規則

- deployment 使用 128-bit random nonce。
- Study/SOP UID 作為 AAD，不直接當唯一 nonce。
- 同 master key 下禁止 nonce reuse。
- 測試中的 deterministic nonce 只用於 test vector，正式實驗仍需記錄每筆 nonce hash／ID，但不得記錄 secret。

---

## 3. 專案與 artifact 規格

### 3.1 建議目錄

    src/
      pipeline/
      crypto/
        legacy.py
        kdf.py
        prng.py
        sot.py
        container.py
        mac.py
        scheme.py
      attacks/
      eval/
    tests/
      unit/
      integration/
      security/
    experiments/
    artifacts/
      registry/experiment_registry.csv
      runs/<run_id>/
    paper_assets/
      tables/
      figures/
      captions/
      method_notes/
      result_notes/
    reports/
      weekly/
      gates/

不得修改 past/SourceCode/；以 wrapper、subclass 或新 CLI flag 呼叫。

### 3.2 每次 run 必備

- config 與 config hash。
- command、task ID、run ID、時間與 exit code。
- dataset split hash、sample IDs、checkpoint SHA-256。
- Python／PyTorch／CUDA／GPU／dtype／seed／steps。
- source/script SHA-256、workspace/git 狀態與固定 seed list。
- scheme、rounds、nonce mode、container version。
- per-sample CSV。
- summary JSON。
- stdout／stderr。
- 失敗與跳過原因。

不得寫入：

- master key。
- password。
- raw subkey／keystream。
- 可回推 secret 的 seed。

### 3.3 共通指標

Latent：

- cosine、MSE、RMSE、MaxAbsError、bit-exact rate、norm。

Image：

- PSNR、SSIM、LPIPS（可用時）、cosine、L-infinity、pixel equality。

Security：

- Top-1、Top-5、mAP／CMC。
- key recovery／cross-image decryption success。
- tamper accepted／rejected。
- nonce collision／reuse rejection。

Efficiency：

- encrypt／verify／decrypt time。
- total diffusion time。
- peak VRAM。
- payload／tag／manifest size。

### 3.4 每個任務 Definition of Done

每個任務使用三層判定，不以科學結果是否符合預期作為工程完成門檻：

1. **工程正確性硬閘門**：預先寫明 test vector、schema、數值誤差、無 NaN/Inf、fail-closed 等 correctness criteria 與驗證指令；測試退出碼為 0。
2. **證據完整性硬閘門**：有 positive／negative control、固定 split、per-sample rows、summary、config、run ID、程式／checkpoint hash 與失敗原因。
3. **科學結果只報告**：Top-k、attack success、PSNR／SSIM／LPIPS、runtime 與 S2 是否優於 S0 不作「必須好看」的通過條件。不得因結果不符假說自動換 seed、sample、split 或 metric。
4. T2-WB 的 inversion positive control 若不足以支持 latent-level 解讀，任務狀態標為 `Inconclusive`，不得宣稱攻擊失敗或方案安全。
5. 產生必要圖表，更新 PROGRESS.md 與 paper_assets 索引。
6. 若工作區使用 Git，完成對應 atomic commit；不得覆蓋使用者既有修改，且 commit 前必須通過 staged-file guard。

任務狀態僅使用：`Achieved`（兩個硬閘門皆通過）、`Failed`（工程或證據閘門未過）、`Inconclusive`（流程正確但資料／inversion 品質不足以回答 RQ）。研究假說失敗或 S2 被攻擊成功，仍可是 `Achieved`。

---

## 4. Gate 總覽

| Gate | 日期 | 必須完成 | 未通過時處置 |
|---|---|---|---|
| G0 | 10/11 | evidence reset、環境、模型、dev split、artifact schema、DDIM/S0/S1 smoke | 先修基線，不開始正式攻擊 |
| G1 | 10/18 | 前作 key-space、magnitude、KPA/CPA、T2-WB、tamper baseline 完成 | 取消 P2，formal N 下修 |
| G2 | 10/25 | HKDF/CSPRNG/nonce/container/MAC 單元測試全過 | 延後 M2，只完成標準元件 |
| G3 | 11/01 | structured transform、R/P 消融與 S2 dev 整合全過 | 固定最小可行 R；不追 bit-exact |
| G4 | 11/08 | S2 T1/T2-WB、norm、geometry、cross-nonce 與 tamper 重跑完成 | formal N 下修但保留所有 P0 attack cells |
| G5 | 11/22 | 正式 security／quality／runtime 主表與 claim matrix 完成 | 停止新增實驗，只補缺格 |
| G6 | 11/30 | 重現、表圖、caption、manifest、checksum、freeze | 未完成不得宣稱實驗結束 |

---

## 5. 執行日曆

下表自 2026-10-05 起生效；後方原任務 ID 保留作為工作分解，但其舊日期全部由本表取代。

| 週次 | 日期 | 主線 | 對應任務群 |
|---|---|---|---|
| Week 1 | 10/05–10/11 | evidence reset、環境、split、S0/S1 與 DDIM smoke | E1、E2 |
| Week 2 | 10/12–10/18 | 前作 key-space、magnitude、KPA/CPA、T2-WB、tamper baseline | A3、A4 |
| Week 3 | 10/19–10/25 | HKDF、CSPRNG、nonce、container、HMAC | D5 |
| Week 4 | 10/26–11/01 | structured transform、R=1/2/3/4、P on/off、S2 integration | D6 |
| Week 5 | 11/02–11/08 | S2 攻擊重跑：norm、geometry、structured probe、cross-nonce、tamper | D7 |
| Week 6 | 11/09–11/15 | 品質、錯誤金鑰、runtime、storage 與 pilot 統計 | Q8 |
| Week 7 | 11/16–11/22 | formal run、統計、誤差歸因、M1/M2、AEAD-ref | F9、Q10 |
| Week 8 | 11/23–11/29 | 關鍵結果重現、表圖、claim audit、freeze | P11 |
| Freeze | 11/30 | 全測試、結果索引與正式凍結 | Z12 |

### 5.1 八週任務目錄

以下任務已依上方 8 週日曆重新整併；E/A/D/Q/F/P 等任務 ID 保留，方便追蹤既有 artifact。

## Week 1｜10/05–10/11：環境、protocol、split 與基線 smoke

大目標：不修改前作原始碼的前提下，讓模型、資料與結果保存流程可運作。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| E0.1 | evidence reset | 舊 artifact 標為 historical；現版腳本可重建每個引用數字或將該 claim 撤回 | reports/evidence_reset.md、canonical_preflight.csv |
| E1.1 | 環境與版本盤點 | 可載入 checkpoint；記錄 Python/PyTorch/CUDA/GPU；無 unexpected keys | artifacts/environment_baseline.txt、model_inventory.csv |
| E1.2 | 速度與 VRAM benchmark | batch 1/4/8 或可行組合完成；據此決定 formal N | results/E1.2_benchmark.json |
| E1.3 | 固定 dev split | 20 張、patient-disjoint、全部可讀、清單有 SHA-256 | splits/dev_v1.csv |
| E1.4 | DDIM smoke | P 路徑 x0→z→xrec 無 NaN/OOM；latent/image 指標完整 | results/E1.4_ddim_smoke.csv、4 張 contact sheet |
| E1.5 | artifact validator | synthetic run 可產生 manifest、per-sample CSV、summary | tests/unit/test_artifact_schema.py |

G0：E0.1 與 E1.1–E1.5 全過。

---

### E2：前作重現與正確評估

大目標：建立可被後續所有攻擊共用的 S0/S1 與 latent cache。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| E2.1 | formal split | patient-disjoint；最低 200、目標 500–1,000；hash 凍結 | splits/security_v1.csv |
| E2.2 | S0/S1 wrapper | 對 Ki@13579／seed 42 與前作輸出一致 | tests/test_legacy_repro.py、results/E2.2.json |
| E2.3 | latent cache | shape 正確、無 NaN/Inf；抽樣重算 MaxAbs ≤ 1e-5 | cache metadata、results/E2.3_DONE.json |
| E2.4 | metric semantics | 證明 cosine=1 不等於 equality；x0 vs xrec 與前作舊定義分開 | reports/metric_semantics.md |
| E2.5 | S0/S1 完整 dev round-trip | T1、T2-WB 的 latent/image 指標齊全 | table_baseline_correctness.csv |

G0：E0.1、E1.1–E1.5、E2.1–E2.5 全過；所有後續攻擊都能從固定 split 與 cache 重跑。

---

## Week 2｜10/12–10/18：前作 key-space、magnitude、KPA/CPA、T2-WB 與 tamper baseline

大目標：證明前作 short seed 與直接不變量，不預先假設真實攻擊成功率。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| A3.1 | effective key-space audit | 從程式路徑證明只使用 32 bits；宣稱與實際比較表完成 | table_keyspace.csv、method note |
| A3.2 | 小空間 brute force／dictionary | 2^20 縮小空間或字典找回 positive control；若 oracle 不足則保留 analytic 結論 | results/A3.2.json |
| A3.3 | S0/S1 不變量 | S0 abs、S1 sorted-abs 的解析式與數值誤差記錄 | results/A3.3.json |
| A3.4 | T1 gallery | P positive、shuffle negative 正確；報 Top-1/5、mAP、CI | table_T1_magnitude.csv、CMC 圖 |
| P3.1 | 攻擊示意圖 | 固定病例的原圖、latent、abs/sorted profile、匿名圖 | Figure attack overview |

不以 Top-1 必須達某數字作為任務通過條件；protocol 正確且有 controls 即可。

---

### A4：KPA、CPA、T2-WB 與 tamper baseline

大目標：完成修補前的完整攻擊章。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| A4.1 | T2-WB anonymous-image inversion | 使用目標 checkpoint；逐筆保存 S0/S1 的 z_ano vs zhat_ano cosine 與 error | table_T2_WB_inversion.csv |
| A4.2 | T2-WB magnitude gallery | 相同 scorer、split、controls；與 T1 並排 | table_T2_WB_magnitude.csv |
| A4.3 | KPA | S0 positive control 一對恢復；S1 按排序／符號驗證 | table_KPA.csv |
| A4.4 | CPA | verify_cpa.py 的 d-query Q recovery 成功；明列 fixed Q 前提 | table_CPA.csv |
| A4.5 | tamper baseline | 修改 image／latent 後 S0/S1 不會拒絕；保存影響指標 | table_tamper_baseline.csv |
| P4.1 | M1 攻擊素材包 | 每個 claim 對應 CSV／圖／run ID | paper_assets attack package |
| P4.2 | T2-BB surrogate／embedding attack（P2） | 不使用目標 checkpoint；與 T2-WB 分表，不能直接宣稱 key recovery | optional black-box report |

G1：W1–W5 的 baseline evidence 完整；若未完成則取消 P2，並下修 formal N。

---

## Week 3｜10/19–10/25：Key pipeline 與 authenticated container

大目標：完成不依賴 structured transform 的標準基礎元件，避免整合週同時除錯。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| D5.1 | master key／password mode | random 256-bit 預設；Argon2id/scrypt 僅 password mode | src/crypto/kdf.py |
| D5.2 | HKDF domain separation | RFC 5869 vectors全過；每輪 sign-pre/perm/sign-post 與 mac label 互異 | tests/test_kdf.py |
| D5.3 | CSPRNG sign／permutation | permutation 合法、決定性 test vector、rejection sampling 無 modulo bias | src/crypto/prng.py、tests |
| D5.4 | nonce registry | 100,000 次測試無 collision；顯式 reuse 被拒絕 | tests/test_nonce.py、result JSON |
| D5.5 | authenticated container | version、nonce、params、ciphertext ref、AAD schema 可 serialize；未知版本 fail closed | container schema、tests |
| D5.6 | HMAC | tag/ciphertext/header/AAD/wrong-key tamper 全部拒絕；未竄改全接受 | src/crypto/mac.py、tamper matrix |
| D5.7 | no-pickle key format | 新流程不使用 torch.load 載入 key material | tests/test_no_pickle.py |

本週禁止自行實作 SHA、ChaCha 或 HMAC primitive，必須使用成熟函式庫。

---

## Week 4｜10/26–11/01：Structured transform 與 S2 dev 整合

大目標：完成可逆數值測試、分布測試與 authenticated end-to-end dev pipeline。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| D6.1 | torch FWHT transform | CPU/CUDA 可 inverse；d=65,536 float64 MaxAbs ≤1e-12；bit-exact rate只記錄 | src/crypto/sot.py、tests |
| D6.2 | orthogonality／round formula | d=1,024 顯式 Q：orthogonality/norm error ≤1e-12；驗證 Q=Q_R…Q_1 | results/D6.2.json |
| D6.3 | Gaussian synthetic | KS、mean/variance/skew/kurtosis完整；p-value 不作安全證明 | QQ/histogram 圖 |
| D6.4 | true latent shift | 比較 z 與 transformed z 的動差及 generation quality；無預設方向 | results/D6.4.csv |
| D6.5 | R／P 消融 | R=1/2/3/4 × permutation on/off；報 attack、runtime、precision | table_rounds_permutation.csv |
| D6.6 | structural diagnostics | trace、same-vector cosine、basis probe、matrix sparsity完整；明列不是安全證明 | table_structure.csv |
| D6.7 | S2 integration | verify tag→inverse→generation 順序正確；S0 regression 不變 | integration tests、dev samples |

R 的選擇原則：在 original magnitude attack 已大幅下降的設定中選最小 R；不能用 R 消除 norm leakage。

G3：S2 核心單元與 dev integration 全過。

---

## Week 5｜11/02–11/08：S2 攻擊重跑

大目標：用相同攻擊比較 S0/S1/S2a/S2。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| D7.1 | S2 T1/T2-WB 資料 | fixed split 全部完成或失敗有原因；z/zhat 指標齊全 | run artifacts |
| D7.2 | original magnitude attacks | 與 A3/A4 完全相同 scorer、gallery、controls | table_attack_original.csv |
| D7.3 | adaptive norm attack | T1/T2-WB norm-only Top-k 與 norm correlation 完整 | table_norm_leakage.csv |
| D7.4 | cross-nonce KPA/CPA | same nonce positive control 成功；跨 nonce 結果完整 | table_cross_nonce.csv |
| D7.5 | pairwise geometry | fixed-Q 內積／距離應保留；per-image Q 的跨影像 linkage 另報 | table_geometry.csv |
| D7.6 | structured probe | basis／sparse／constant vectors；不得只測 Gaussian | table_structured_probe.csv |
| D7.7 | repeat transform | 相同明文 100 次、nonce 全異、protected payload bytes 全異 | table_repeat.csv |
| D7.8 | tamper suite | S2 所有預定 tamper 100% reject；S2a positive tamper 不拒絕 | table_tamper.csv |
| P7.1 | defense 素材包 | 方法圖、attack table、norm limitation、tamper matrix | paper_assets defense package |

S2 通過的含義：

- 實作符合 protocol。
- 指定弱點的結果已量化。
- 不要求 norm attack 失敗。

---

## Week 6｜11/09–11/15：品質、相似度與成本

大目標：回答「安全強化是否明顯改變模型輸出」。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| Q8.1 | reconstruction fidelity | P/S0/S1/S2 的 latent＋image 指標、CI 完整 | table_fidelity.csv |
| Q8.2 | anonymous image difference | PSNR/SSIM/LPIPS/cosine 與 nearest-neighbor rank | table_anonymity.csv |
| Q8.3 | wrong key／nonce | 正確與錯誤條件並列；不只放示意圖 | table_wrong_key.csv |
| Q8.4 | runtime／memory | transform、KDF、MAC 與 diffusion 成本分開；median/p95 | table_runtime.csv |
| Q8.5 | storage overhead | nonce、tag、manifest、latent payload／image 分開 | table_storage.csv |
| P8.1 | fixed visual grid | 預先固定、median、worst 三類，不得手挑最佳案例 | main figures |

若 S2 PSNR 比 S0 低超過 1 dB，先檢查實作；確認無 bug 後保留為 trade-off，不反覆調到好看。

本週完成修補後 quality、wrong-key、runtime 與 storage pilot，供 Week 7 formal run 固定設定。

---

## Week 7｜11/16–11/22：Formal run、統計、誤差歸因與部署對照

大目標：把 pilot 升級成可寫入論文的正式結果。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| F9.1 | formal security run | 最低 200、目標 500–1,000；每筆有 metrics row | frozen security runs |
| F9.2 | bootstrap CI | Top-k、PSNR、SSIM、runtime 等皆有 95% CI | stats JSON |
| F9.3 | adaptive limitation | norm leakage 的 T1/T2-WB 差異與案例分析完成 | result note |
| F9.4 | statistical comparison | S0/S1/S2 paired test、effect size、multiple-comparison 規則固定 | table_statistics.csv |
| F9.5 | claim-evidence draft | proposal 每個 claim 對應表／圖／run | claim matrix v1 |

G5：主表數字凍結；後續只允許補失敗 cell 或重現。

---

### Q10：誤差歸因、M1/M2 對照與緩衝

大目標：處理 0.9989 的正確解讀，不把 bit-exact 當成 structured transform 必須達成。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| Q10.1 | A0–A4 工程消融 | uint8/min-max、float payload、clip、conditioning、dtype 分項結果 | table_error_attribution.csv |
| Q10.2 | transform-only precision | d=4,096/65,536、R、dtype 的 MaxAbs/RMSE/bit-exact rate | table_numeric_precision.csv |
| Q10.3 | M1 vs M2 | image-only 與 latent-payload 的 security/fidelity/storage 對照 | deployment_modes.md |
| Q10.4 | AEAD-ref | 原始 bytes round-trip 100% equality、tamper reject，作安全下限 | table_aead_reference.csv |
| Q10.5 | optional partial-key | 有餘裕才跑 derived-key Hamming distance、ciphertext correlation | optional figure |

注意：

- M2 的正確比較是 decrypted latent／generated image，不預設 pixel bit-exact。
- 若要宣稱原始 bytes bit-exact，只能引用 AEAD-ref。
- 不在本週加入新 sampler 或新資料集。

---

## Week 8｜11/23–11/29：重現、表圖與 Freeze

大目標：停止功能開發，讓 12 月可直接撰寫。

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| P11.1 | experiment matrix audit | 每個 P0 cell 有 run ID；缺值有原因 | final matrix CSV |
| P11.2 | 關鍵結果重跑 | 主表關鍵方法至少重跑一次，落在預定 tolerance／CI | reproduction report |
| P11.3 | 一鍵建表圖 | 由 frozen CSV/JSON 生成全部表圖，不讀手抄數字 | build scripts |
| P11.4 | 圖片與 captions | PNG 300 dpi＋PDF/SVG；每張有 caption、take-away、限制 | paper_assets |
| P11.5 | 方法與結果 notes | threat model、attacks、S2、metrics、statistics 各有可引用草稿 | method/result notes |
| P11.6 | claim audit | 無 perfect security、IND-CPA、lossless 等無證據字樣 | final claim matrix |
| P11.7 | checksum／manifest | 表圖、config、results 均有 SHA-256 | SHA256SUMS、freeze manifest |

11/23 後禁止：

- 新增方法或資料集。
- 因結果不好看更換 seed／sample。
- 覆蓋舊 run。
- 手動修改表格數字。

---

## 11/30：緩衝與正式凍結

| ID | 子任務 | 通過條件 | 交付 |
|---|---|---|---|
| Z12.1 | 全測試 | unit/integration/security 全跑；P0 failure 有 limitation | final_test_report.md |
| Z12.2 | 結果索引 | 每個 RQ 對應表、圖、run、結論、限制 | results_index.md |
| Z12.3 | 12 月交接 | thesis outline 與插圖位置完成 | december_handoff.md |
| Z12.4 | freeze point | 記錄 workspace/git 狀態、環境、split hash | experiment_freeze.md |

---

## 6. 主表與主圖

### 6.1 主表

| 編號 | 內容 |
|---|---|
| T1 | 前作宣稱與實際 key-space／pipeline |
| T2 | S0/S1 invariants、KPA、CPA、determinism |
| T3 | T1/T2-WB magnitude 與 sorted-magnitude attack；T2-BB 選做另表 |
| T4 | S2 original attack 與 adaptive norm attack |
| T5 | nonce、cross-image KPA／CPA、repeat encryption |
| T6 | tamper rejection matrix |
| T7 | latent/image correctness 與品質 |
| T8 | runtime、VRAM、storage |
| T9 | M1/M2／AEAD-ref 部署比較 |

### 6.2 主圖

| 編號 | 內容 |
|---|---|
| F1 | 前作 pipeline、攻擊點與 S2 修補 |
| F2 | magnitude／sorted magnitude 洩漏 |
| F3 | T1/T2-WB CMC；T2-BB 選做另圖 |
| F4 | structured transform round／permutation／runtime／precision |
| F5 | original attack vs norm-adaptive attack |
| F6 | tamper acceptance／rejection |
| F7 | P/S0/S1/S2 品質 grid |
| F8 | security–fidelity–cost trade-off |

---

## 7. 決策與停損規則

### 7.1 Structured transform 數值誤差

若 float64 MaxAbsError > 1e-12：

1. 檢查 inverse permutation、round normalization 與 dtype。
2. 與 NumPy reference 比較。
3. 兩輪仍無法達標則使用最小通過 R，並記錄限制。

不可以把 bit-exact rate 100% 當成修 bug 的唯一終點。

### 7.2 Original magnitude attack 不下降

若 S2 original magnitude Top-1 仍顯著高：

1. 用 synthetic Gaussian positive control 確認 scorer。
2. 檢查是否誤用了相同 nonce、未套 WHT 或只套 permutation。
3. 確認無 bug 後，視為真實 latent 的額外結構洩漏並如實報告。

### 7.3 Norm attack 成功

不修改方案、不掩蓋結果。其原因由正交不變量直接解釋，寫入限制與 future work；CDF nonlinear transform 可列為後續方向。

### 7.4 T2-WB 太慢

優先：

1. formal N 降到 200。
2. 保留所有方法與 attack cells。
3. 取消 P2。
4. 不得只留下對 S2 有利的樣本。

### 7.5 品質下降

若 S2 相較 S0 明顯下降：

- 檢查真實 z 是否偏離 isotropic Gaussian。
- 報告 R、noise level 與品質 trade-off。
- 不以重新訓練解決，因為超出本研究範圍。

### 7.6 MAC 與 nonce 失敗

若任一預定 tamper 被接受，或 nonce reuse 未被阻擋：

- G3 不通過。
- 取消後續 Week 6–7 的 P1/P2。
- 優先修正 protocol，不得以 limitation 帶過。

---

## 8. 每週固定例行事項

每週最後一個工作日：

1. 執行全部相關 tests。
2. 更新 PROGRESS.md 與週報。
3. 確認 per-sample CSV、summary、config、stdout/stderr 齊全。
4. 更新 paper_assets 索引、caption 草稿與一句 take-away。
5. 檢查 secret 未寫入 artifact。
6. 排定下週長任務；超過 15 分鐘者使用背景工作與 status.json。
7. 檢查是否觸發 Gate 或停損規則。

---

## 9. 與原 WORKFLOW.md 的主要差異

| 原規劃 | 修訂 |
|---|---|
| S3／MAC 是第 10 週加分 | MAC 併入 S2，Week 3–4 必做 |
| d=65,536 structured transform 必須 bit-exact | 改為 MaxAbs ≤1e-12，bit-exact rate如實報 |
| 修補後所有 re-ID 應接近 random | original magnitude 與 adaptive norm 分開；norm 可成功 |
| nonce 後語意上接近 IND-CPA | 只宣稱避免 transform 跨影像重用 |
| Tier 1 應達 image bit-exact | 改為數值上限；byte-exact 只由 AEAD-ref 保證 |
| avalanche 約 50% 輸出元素變化 | 改量 derived-key Hamming、ciphertext correlation；不混用連續值與 bit 指標 |
| HMAC／W7 可刪 | integrity 為醫療資料必要 P0 |
| KCI／exact sampler 可能插入 | 本期不實作，避免擴張架構 |

這些修訂保留原 Claude 方案的低改動優勢，同時避免安全性、bit-exact 與實驗通過條件上的過度宣稱。
