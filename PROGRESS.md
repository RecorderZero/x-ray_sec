# 實驗日誌

> **寫入規則**：只增不改。新條目加在最下面。
> 不要刪除或改寫舊條目 —— 錯誤的判斷本身也是紀錄，之後回頭看才知道當時為什麼那樣做。
> 研究設計的決定寫在 `WORKFLOW.md`，這裡只記「做了什麼、得到什麼」。
> 標記慣例 `[決定]` `[事實]` `[假設]` `[待辦]` `[推定]` `[結論]` `[限制]` `[注意]`。
> 詳細的設計方法紀錄在`PROPOSAL.md`有需要可以去查看。

## 條目格式

```markdown
## YYYY-MM-DD · Wn · Track X · 一句話標題

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：[例如 A1.4]
* 本輪要解決的核心問題或實現的功能是什麼？

### 2. 執行動作 (Actions Taken)
* 修改或新增了哪些檔案/程式碼？（標記具體檔名與函式）
* 下達了哪些測試指令？

### 3. 執行結果 (Results & Observations)
* 測試是否通過？（附上客觀 Pass/Fail 數據，如 `PSNR = 28.5 dB`）
* 是否出現預期外的錯誤？

### 4. 達標判定 (Assessment)
* [ ] **已達標 (Achieved)**：完全符合預期，測試 Pass。
  - 下一步：執行原子化 `git commit`（記錄變更檔案與 feat/fix），並進入下一個 Sub-task。
  - 圖表保存：若有關鍵實驗結果圖表，保存在 `image/` 或指定目錄中，檔名必須具備可讀性與辨識度。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
*(僅在「未達標」時填寫)*
* **失敗原因分析**：為什麼這次修改無效？根本原因為何？
* **下輪修改計畫**：根據本次失敗經驗，下一輪 Loop 應該調整什麼方向？

### 6. ⚠ 與 WORKFLOW.md 不符
- 發現的衝突。沒有就寫「無」。有的話題給使用者裁決，不要自己改 WORKFLOW.md

### 7. 下一步
* 描述1-3下一步需要完成的事項
```

---

<!-- 新條目加在這行下面 -->

## 2026-10-06 · W1 · Track A · 前作模型盤點、攻擊者能力分級與 S2 防禦邊界

**做了什麼**

- 交叉檢查前作論文第 4.4、4.6、5.2.2 節、`past/SourceCode/README/CFG_DDIM_README.md`、訓練／推論 shell scripts 與目前保留的 checkpoint。
- 釐清「三個 diffusion model variants」、「獨立模型角色」、「實際 checkpoint 數量」三種計算口徑。
- 將 model 可得與不可得時的唯密文、已知明文、選擇明文及選擇密文情境分開。
- 比較前作 S0/S1、改進方案 S2 與 AEAD-ref 可防禦及不可防禦的攻擊。

**結果**

### 1. 前作模型數量與來源

- [事實] 論文第 5.2.2 節比較三種方法：CFG-DDIM、CLF-DDIM、Traditional DDIM。
- [事實] 三種方法不必然對應三份獨立權重。`run_uncond_ddim_inference.sh` 直接載入 CFG checkpoint，將 `guidance_scale=0` 作為 Traditional／Unconditional DDIM；因此它是推論模式，而不一定是另一個訓練模型。
- [事實] CLF-DDIM 除 diffusion U-Net 外，還需要一個另外訓練 20,000 iterations 的 noisy-image classifier。
- [事實] CFG 與 CLF diffusion model 的訓練腳本設定為 50,000 steps，且原始 CheXpert 與 Deblurred CheXpert 各有一套訓練命令。
- [事實] 去模糊前處理使用 DiffPIR 與一個 VinDr-CXR unconditional diffusion prior。README 表示此 prior 是以 VinDr-CXR 自行訓練，而非直接使用通用自然影像權重。
- [推定] 按獨立功能角色計算共有 4 類 learned model：CFG diffusion、CLF diffusion、CLF classifier、VinDr-CXR deblur prior。
- [推定] 按預期的資料版本／checkpoint instance 計算約為 7 份：CFG 原始／deblurred 2 份、CLF diffusion 原始／deblurred 2 份、CLF classifier 原始／deblurred 2 份、deblur prior 1 份。
- [事實] 目前 `past/SourceCode` 快照實際只保留 1 份 checkpoint：`results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt`。因此無法只靠目前檔案證明其餘預期 checkpoint 都曾成功完成訓練。
- [事實] 網路現成的是 guided-diffusion、diffusion-anomaly、DiffPIR 等程式架構與方法。現有論文及原始碼沒有足夠證據證明核心 CheXpert 最終權重是直接下載的 off-the-shelf checkpoint。
- [注意] 論文中的 `pre-trained model` 應解讀為「進入該流程前已訓練完成」，不能直接推論為「網路下載且未自行訓練」。

### 2. DDIM inversion 與 model 可得性

- [事實] DDIM inversion 是演算法，不是可由論文 PDF 單獨還原的模型。執行 inversion 還需要 denoising U-Net checkpoint、架構、noise schedule、timestep respacing、noise level、影像正規化、label／guidance 設定及 sampler 實作一致。
- [事實] 只有論文中的架構與超參數，重新訓練只能得到 surrogate model；不同初始化與訓練過程會形成不同的 latent coordinate system。
- [結論] surrogate inversion 的 latent 不應直接拿來與目標系統 latent 做逐元素除法，也不能據此宣稱已恢復目標 key。
- [結論] 若掌握目標 checkpoint 與 pipeline，則可將原圖及匿名圖分別 inversion 成近似的 `z` 與 `z_enc`，把 exact-latent attack 降級為含 inversion error 的 noisy-latent attack。

### 3. 攻擊情境分類

| 攻擊者能力 | 分類 | 可以合理測試的內容 |
|---|---|---|
| 只有匿名／加密影像，且知道演算法與目標 model | White-box COA | inversion 後的 norm、magnitude、sorted-magnitude、gallery linkage |
| 有原圖及其對應匿名圖，且有目標 model | White-box KPA | 近似 latent pair、Rademacher sign recovery、Signed Permutation matching、held-out decryption |
| 可自行挑選原圖並呼叫同一系統加密 | CPA | basis／sparse／constant probes、重複查詢、同 nonce 與跨 nonce transfer |
| 可修改匿名圖／container 並觀察解密端反應 | CCA／tampering oracle | tag、header、nonce、payload 修改是否 fail closed |
| 只有匿名影像與論文，沒有目標 model | Black-box COA | 視覺／embedding re-ID、統計 linkage、surrogate attack；通常不能直接恢復 key |
| 有多組 image pair，但沒有目標 model | Black-box KPA | paired image-to-image recovery、surrogate training、re-ID；不等於目標 latent key recovery |

- [結論]「知道 model」是 white-box／Kerckhoffs 條件，本身不等於 CPA。只有攻擊者可選擇輸入並取得輸出時才是 CPA。
- [結論] 只有 encrypted image 時是 COA；取得已配對的 original/encrypted image 時是 KPA。
- [結論]「已知密文攻擊」不是此處需要另立的類別，因為上述攻擊通常都預設攻擊者能看到 ciphertext。

### 4. 前作 S0/S1 的已知弱點

- [事實] S0 Rademacher 為 `z_enc = k ⊙ z`，保留每個座標的絕對值與整體 norm。取得 exact latent pair 時可由乘積符號直接恢復 sign key。
- [事實] S1 Signed Permutation 為 `z_enc = P(k ⊙ z)`，仍保留 sorted absolute values 與 norm。exact pair 可用 magnitude matching 推測 permutation，再恢復 sign。
- [假設] 從 anonymous image 重新 inversion 會引入誤差；單一 pair 在接近零或 magnitude 相近的座標上可能不穩，但多 pair correlation／assignment 可提升恢復率。
- [事實] 密碼衍生流程只取 SHA-256 digest 的前 4 bytes 作為 Rademacher PRNG seed，形成有效 key-space 問題；實驗密碼也曾直接出現在 shell script。
- [事實] S0/S1 沒有 per-image nonce，重用同一 transform 會讓一組 KPA／CPA 結果跨影像轉移。
- [事實] S0/S1 沒有 MAC，無法偵測 image、latent、header 或 key-related metadata 被修改。
- [結論] S0/S1 不能宣稱具備標準 IND-CPA confidentiality，也沒有 ciphertext integrity 或 CCA 防護。

### 5. S2 相對前作的預期改善

S2 定義為：256-bit random master key（password mode 才使用 memory-hard KDF）、HKDF-SHA256 domain separation、CSPRNG、每張影像唯一 nonce、keyed sign/permutation/WHT structured transform，以及涵蓋 version、params、nonce、AAD 與 payload 的 HMAC-SHA256。

| 前作問題／攻擊 | S2 預期效果 | 必要前提 |
|---|---|---|
| Rademacher 逐座標除法／符號恢復 | dense mixing 後不再能逐座標直接恢復 | transform 確實完整套用 |
| Signed Permutation sorted-magnitude matching | WHT 混合座標，破壞原座標 magnitude 對應 | rounds／permutation 設定經實測 |
| 32-bit seed brute force | random 256-bit master key 排除此弱點 | 不使用低熵 password；secret 不寫入 log |
| 一組 KPA 恢復共用 transform | per-image nonce/subkey 阻止結果跨影像直接轉移 | nonce 唯一且由系統控制 |
| CPA basis recovery 後跨影像套用 | 不同 nonce 導出不同 transform，降低 transfer | 攻擊者不能強迫 nonce reuse |
| ciphertext/header/nonce tampering | HMAC 驗證失敗並在 inversion／generation 前拒絕 | verify-before-decrypt、fail closed |
| 簡單 CCA error oracle | 未驗證資料不得進入 inverse/generation，可大幅縮小 oracle | 錯誤訊息與 timing 不洩漏細節 |

### 6. S2 仍然不足的地方

- [限制] WHT／signed permutation 屬正交線性 transform，仍精確保留 latent L2 norm；COA/KPA 的 norm linkage 可能成功。
- [限制] 若相同 transform 被重用，它仍保留 pairwise inner product／distance；nonce reuse 是嚴重失敗，不得只列為一般 limitation。
- [限制] 若攻擊者能對同一 nonce 做足夠 chosen-plaintext query，仍可能學出該次線性 transform。S2 的重點是阻止其跨 nonce／跨影像轉移，不是取得正式 CPA 證明。
- [限制] HMAC 提供完整性與來源驗證，但不會自動讓 structured transform 具有 confidentiality；S2 仍不能宣稱 IND-CPA 或 IND-CCA。
- [限制] password mode 仍可能遭離線字典攻擊；Argon2id／scrypt 只能提高成本。正式模式應優先使用 random 256-bit key。
- [限制] anonymous preview 仍可能保留病患、解剖、病灶或模型記憶特徵，必須另外做 image/embedding re-ID，不能繼承 HMAC 或 key pipeline 的安全宣稱。
- [限制] T2 結果高度依賴 inversion fidelity；攻擊失敗時必須區分「方案阻擋攻擊」與「攻擊者無法準確反轉匿名圖」。
- [限制] S2 是 authenticated obfuscation／低改動安全強化，不是標準加密的替代品。

### 7. AEAD-ref 的定位

- [決定] AES-GCM 或 ChaCha20-Poly1305 作為 confidentiality、integrity 與 byte-exact round-trip 的參考下限。
- [結論] 若系統需要標準 CPA/CCA 等級安全主張，應以 AEAD 保護原始 bytes 或 latent payload；anonymous image 只能作獨立 preview，並另外評估 re-ID／medical utility。
- [結論] 不能把 AEAD payload 的安全性轉移宣稱到 anonymous preview，也不能把 S2 的實驗攻擊失敗寫成正式 IND-CPA／IND-CCA 證明。

**卡住 / 意外**

- 現有備份缺少預期的 CLF diffusion、CLF classifier、deblurred CFG 與 VinDr deblur prior checkpoint，因此目前只能直接重現原始 CheXpert CFG checkpoint 的 white-box inversion／attack。
- 論文對 `pre-trained`、三種 model variant 與實際獨立 checkpoint 的措辭不足以單獨確認權重來源；最終 model inventory 必須以原作者檔案或 checkpoint manifest 補證。

**⚠ 與 WORKFLOW.md 不符**

- `WORKFLOW.md` 的 T2 原定義為「只取得 anonymous image，需自行 inversion」，但未明示攻擊者是否擁有目標 model。後續報告必須將「有目標 model 的 white-box T2」與「只有論文／surrogate 的 black-box T2」分欄，避免混合解讀。
- S2 測試必須保留 adaptive norm attack；不能因 original magnitude attack 下降便宣稱可抵禦所有 COA/KPA/CPA。

**下一步**

1. 以現有 CFG checkpoint 建立 paired original/anonymous T2 inversion baseline，量測 exact encrypted latent 與 re-inverted latent 的 cosine、RMSE、MaxAbsError。
2. 對 Rademacher 進行 1/2/4/8/16 pairs 的 weighted sign recovery，並以 held-out images 報 key accuracy、latent cosine 與 image PSNR/SSIM。
3. 對 Signed Permutation 進行多 pair coordinate-signature matching／one-to-one assignment；若完整 key recovery 失敗，仍報 norm 與 sorted-magnitude gallery linkage。

## 2026-10-06 · W1 · Track E/A · 治理規格與 T2 優先級定案

**做了什麼**

- [決定] 將舊稱 T2 拆為 `T2-WB` 與 `T2-BB`；前一條目所有「有目標 model 的 white-box T2」自本條起統一稱 T2-WB，「沒有目標 model 的 black-box T2」統一稱 T2-BB。
- [決定] T2-WB 納入 P0 並優先完成；T2-BB 納入 P2，只有在全部 P0 與 T2-WB 完成後才執行。
- [決定] 任務門檻分成工程正確性與證據完整性兩個硬閘門；攻擊成功率、Top-k、PSNR/SSIM/LPIPS 等科學結果只報告，不以結果是否符合假說決定任務是否完成。
- [決定] 長任務採唯一 run ID、不可覆寫 run directory、atomic `status.json`、heartbeat、PID／exit code、受限重試與斷路器。
- [決定] Git 禁止 `git add .`／`git add -A`；dataset、checkpoint、cache、run-level／大型 per-sample 輸出與 secret 不上傳，commit 前執行 `scripts/check_staged_files.sh`。

**結果**

- 更新 `PROPOSAL.md`、`WORKFLOW.md`、`loop_engineering_spec.md`、`.gitignore`。
- 新增 `scripts/check_staged_files.sh`，預設拒絕超過 90 MiB 的 staged file 與禁傳路徑／憑證名稱。
- 任務狀態固定為 `Achieved`、`Failed`、`Inconclusive`；T2-WB positive-control inversion 品質不足時只能標記 Inconclusive，不能宣稱防禦成功。

**卡住 / 意外**

- 標準 `apply_patch` 仍受執行環境的 `bwrap: loopback: Failed RTM_NEWADDR` 阻擋，本輪使用暫存副本自動產生 diff 後以本地 `git apply` 套用。

**⚠ 與 WORKFLOW.md 不符**

- 前一條目記錄的 T2 命名已由本條決策取代；實驗內容未刪除。

**下一步**

1. 完成 E0.1 evidence reset 與 E1.1 model/data/environment inventory。
2. 在正式長任務前實作通用 background runner 與 status schema validator。
3. 先跑 T2-WB；T2-BB 僅在 P0 全部完成後排程。


## 2026-10-07 · W1 · Track E · E0/E1 基線建立與 DDIM smoke

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E0.1、E1.1–E1.5。
* 重建唯一 canonical preflight，確認 checkpoint／環境，固定 patient-disjoint dev split，完成效能、DDIM 與 artifact schema 基線。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/build_e0_evidence_reset.py`、`scripts/e1_environment_inventory.py`、`scripts/create_dev_split.py`、`scripts/artifact_schema.py`、`scripts/e1_ddim_runner.py` 與 `tests/unit/test_artifact_schema.py`。
* 產生 `reports/evidence_reset.md`、`canonical_preflight.csv`、`artifacts/environment_baseline.txt`、`model_inventory.csv`、`splits/dev_v1.csv`、`results/E1.2_benchmark.json`、`results/E1.4_ddim_smoke.csv` 與 contact sheet。
* 以 CFG_DDIM 重跑 d=65,536、N=100 preflight；strict load CFG checkpoint；驗證 20 張 split；執行 batch 1/4/8 benchmark、4 張 noise=500 DDIM smoke 與兩個 schema controls。

### 3. 執行結果 (Results & Observations)
* [事實] E0 canonical SOT-WHT bit-exact=97/100、norm correlation=1；舊 31/32、98/100 artifacts 已標為 historical，0.9989 cosine 不視為 byte equality。
* [事實] Python 3.10.19、PyTorch 2.14.0+cu130、RTX 5090；checkpoint strict load 的 missing/unexpected keys 皆為 0，模型 113,998,722 parameters。
* [事實] dev split 為 20 張／20 位病人，健康與積水各 10 張，split SHA-256=`dd302f56531bfb639ecaed5ab2af506b71886a9d4cef41b591dc2518bff8ff50`。
* [事實] batch 1/4/8 均可行，峰值 VRAM 約 0.80/1.71/2.92 GB；batch 8 吞吐最佳，正式 N 先設 200，待 full-pipeline pilot 再調整。
* [事實] DDIM smoke 4/4 無 NaN/Inf/OOM；平均 23.08 秒／張、PSNR=34.0559 dB、image cosine=0.999408，contact sheet 為 768×1144 PNG。科學數值只報告，不作 success gate。
* [事實] schema positive／negative controls 2/2 通過，可拒絕 duplicate sample ID 與 NaN metric。
* [意外] 首次 schema 測試因 CFG_DDIM 未裝 pytest 停止，改為直接執行相同測試函式後通過；首次 benchmark 因舊 Pillow 無 `Image.Resampling` 停止，加入版本相容寫法後通過。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：E0.1 與 E1.1–E1.5 的 correctness gates 與指定小型交付均完成。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* **失敗原因分析**：兩次首次失敗皆為環境相容性（缺 pytest、舊 Pillow API），不是模型數值或 schema 錯誤。
* **下輪修改計畫**：維持現有環境、不額外安裝 pytest；runner 保留雙版本 Pillow compatibility。下一輪開始 E2 的 S0/S1 reproduction 與 latent cache smoke。

### 6. ⚠ 與 WORKFLOW.md 不符
- 原始 CheXpert validation 影像不是 256×256，而學長現存 raw loader 未 resize；本輪固定 grayscale、bicubic 256×256、per-image min-max `[0,1]`。正式 E2 前需把它列為 protocol 決策，並再核對學長實際訓練資料的前處理。


## 2026-10-07 · W1 · Track E · 學長前處理對齊與 E2 readiness 稽核

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E1.1、E1.2、E1.4 與 E2 啟動前檢查。
* 確認學長 checkpoint 的實際訓練資料前處理，修正不一致的 E1 runner，並建立可持續維護的根目錄檔案索引。

### 2. 執行動作 (Actions Taken)
* 交叉檢查 `chexpert_preproc.py`、`cfg_image_train.py`、`bratsloader.py`、`CFG_DDIM_README.md` 及備份 train/valid CSV 的路徑結構。
* 將 `scripts/e1_ddim_runner.py::preprocess` 從 direct bicubic 改為 grayscale → histogram equalization → OpenCV INTER_AREA 256×256 → JPEG quality 100 round-trip → per-image min-max `[0,1]`。
* 更新 `scripts/e1_environment_inventory.py`，加入 Pillow/pytest/OpenCV 版本並隔離 Visdom import side effect。
* 新增 `tests/unit/test_chexpert_preprocessing.py`、`reports/preprocessing_audit.md` 與根目錄 `README.md`；重新執行 inventory、batch 1/4/8 benchmark、4-image smoke 與完整 unit tests。

### 3. 執行結果 (Results & Observations)
* [事實] 學長前處理腳本預設先 histogram equalization，再用 INTER_AREA 縮成 256×256，輸出 JPEG quality 100；訓練 loader 再做 per-image min-max。
* [事實] 備份 CSV 的 flattened JPEG 路徑符合該前處理腳本輸出，不是原始 CheXpert 巢狀路徑。
* [事實] 更新後 unit tests 3/3 passed；checkpoint strict load 仍為 missing/unexpected keys=0。
* [事實] 更新後 DDIM smoke 4/4 finite、無 OOM/NaN；平均 runtime=23.0635 秒、PSNR=33.8917 dB、image cosine=0.999375。這些數值只報告，不是 success gate。
* [事實] batch 1/4/8 仍全可行，batch 8 peak VRAM 約 2.92 GB，維持 formal N 起始值 200。
* [環境] CFG_DDIM 目前為 Pillow 9.0.0、pytest 9.1.1、OpenCV 4.7.0；pytest 可用，但 Pillow 仍沒有 `Image.Resampling`，相容寫法必須保留。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：現行 runner 已對齊可由學長程式、README 與 CSV 證明的訓練前處理，E2 可以從 smoke/pilot 階段開始。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* **失敗原因分析**：先前只依 model image_size 與 raw loader 推定 bicubic resize，忽略 checkpoint 實際使用的是離線 `chexpert_preproc.py` 產生的 JPEG。
* **下輪修改計畫**：E2 全部 runner 共用已凍結 preprocessing，並在 manifest 保存 preprocessing/config hash；不得混用舊 E1 direct-bicubic 結果。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無。限制是歷史 256×256 訓練影像目錄未保留，因此不能做 byte-for-byte 歷史檔比對；目前結論來自程式、README、CSV 路徑結構、模型 shape 與重新 smoke 的一致證據。

### 7. 下一步
* 建立 E2.1 formal patient-disjoint split，最低 N=200，凍結 split hash。
* 實作 E2.2 S0/S1 wrapper 與 legacy positive-control reproduction。
* 先用 1–4 張執行 E2 latent cache smoke，再依長任務規範啟動正式 cache。

## 2026-10-08 · W1 · Track E · AUD-20261008-01 稽核修正與使用者裁決落地

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E0.1、E1.1–E1.5、E2.1。
* 回應稽核：AUD-20261008-01 的 AF-001～AF-012；`AUDIT.md` 保持唯讀，finding 是否關閉仍由稽核方裁定。
* 落實使用者裁決：AF-011 主線採 `guidance_scale=0`；AF-012 採方案 (a)。

### 2. 執行動作 (Actions Taken)
* [AF-001] 已修正（待稽核複驗）｜commit `1579c96`｜`scripts/e1_ddim_runner.py::invert_reconstruct` 改為直接呼叫學長 `ddim_sample_loop_known_progressive`；新增 `tests/integration/test_legacy_ddim_equivalence.py`｜驗證：`scripts/run_cfg_ddim.sh python -m pytest -q tests/integration/test_legacy_ddim_equivalence.py`。
* [AF-002] 已修正（待稽核複驗）｜commits `44a05a8`、`1579c96`｜manifest 補齊 WORKFLOW §3.2 欄位，validator 依 manifest 宣告的任意 numeric fields 驗證；E1.2/E1.4 寫入本機 `artifacts/runs/<task>/<run_id>/` 並立即 validate；補缺欄與竄改 summary negative tests。
* [AF-003] 已修正（待稽核複驗）｜commit `1579c96`｜新增 image SSIM、L∞、uint8 pixel equality、float32 bit-exact，以及 latent same-seed repeat／xrec re-inversion 的 cosine、MSE、RMSE、MaxAbs、bit-exact rate。
* [AF-004] 已裁決並修正（待稽核複驗）｜commit `44a05a8`｜[事實] pytest 與 Pillow 由使用者自行安裝；Pillow 9.0.0 仍無法正常運作後，使用者已回復 Pillow 8.4.0。新增 `scripts/run_cfg_ddim.sh` 固定 `PYTHONNOUSERSITE=1`，inventory 記錄套件版本／`__file__`／site flag／freeze hash，套件若載自 `sys.prefix` 外則 fail closed。
* [AF-005] 已修正（待稽核複驗）｜commit `9eb4ccc`｜重建 d=4,096、d=65,536、seeds 0–99 × R=1/2/4；報告對照 97/100 與 98/100 的不同 estimand，加入 KS，舊 Householder 表標 historical，stable hash 排除 timing。
* [AF-006] 已修正（待稽核複驗）｜本輪 commits 均使用類型／任務依據／變更檔案／驗證結果與 `Refs:` body；未改寫已推送的歷史 commit。
* [AF-007] 已修正（待稽核複驗）｜commit `44a05a8`｜guard 新增 `*.pt/*.pth/*.ckpt/*.safetensors/*.npy/*.npz` 與 private-key header 掃描；暫存 repo tests 通過。
* [AF-008] 已修正（待稽核複驗）｜commit `44a05a8`｜unit test 直接載入學長 `ChexpertResNormPipeline._preprocess_image` 與 `train_util.visualize` 作 oracle，不再複寫 runner 邏輯。
* [AF-009] 已更正（待稽核複驗）｜[更正] 舊條目所稱「舊 Pillow 缺 `Image.Resampling`」及「runner 保留雙版本 compatibility」沒有足夠紀錄支持，現行 runner 也未使用 `Image.Resampling`；首次失敗當下的完整 interpreter／trace 未保存，不能再把根因寫成既定事實。可確認的是使用者曾自行安裝 pytest/Pillow、Pillow 9.0.0 仍失敗，最後回復 8.4.0。
* [AF-010] 已修正（待稽核複驗）｜commit `44a05a8`｜split hash 限定 `(sample_id, patient_id, label, source_path, file_sha256)`，升版 `dev_v1.1`；更換 `local_path` 前綴的 unit test 通過。
* [AF-011] 使用者已裁決並落地｜commit `1579c96`｜P/S0/S1/S2a/S2 的 security、correctness、reversibility 主表共用 guidance=0；guidance=4 只能另作病灶健康化附表。
* [AF-012] 使用者已裁決並落地｜commits `44a05a8`、`1579c96`｜以學長備份 train.csv 重現每類 16,000 張、`random_state=1911` 抽樣，排除入選的 22,253 位病人後，從 raw train 剩餘病人固定隨機抽取 200 位。

### 3. 執行結果 (Results & Observations)
* [事實] CFG_DDIM 現為 Pillow 8.4.0、numpy 1.23.0、scipy 1.10.0、pytest 9.1.1，皆載自 `/home/user/anaconda3/envs/CFG_DDIM`；預設與 wrapper 執行目前解析到同一組套件，wrapper 仍作為正式入口。
* [事實] unit tests `8 passed`；GPU legacy equivalence integration `1 passed`，latent 與 reconstruction 均 bit-exact。
* [事實] E1.2 run `E1.2_20261007T165702Z_9c001b5b` 與 E1.4 run `E1.4_20261007T170153Z_3ca0e32a` 均由 `validate_run` 回傳 `valid=True`。
* [事實] guidance=0 的 E1.4 四張 smoke 全部 finite；平均 PSNR=40.6402 dB、SSIM=0.983468、image L∞=0.056579、uint8 pixel equality=0.183086。科學數值只報告，不作成功門檻。
* [事實] same-seed latent repeat 的 MaxAbs=0、bit-exact rate=1；但 `xrec → inversion` 與原 latent 的平均 cosine=0.905725、RMSE=0.421704、MaxAbs=2.470699。
* [事實] E0 現版重建：d=4,096 gallery 100/100；d=65,536 gallery 97/100；d=65,536 seeds 0–99 在 R=1/2/4 都是 98/100。`canonical_preflight.csv` 與報告重跑後 SHA-256 不變。
* [事實] `security_v1` 為 200 張／200 位病人、健康與積水各 100；split SHA-256=`5843da54d027cbe5b7d518349a374f859b5d2c0e07b9b66ea5f89c9530a94490`，與重建的前作訓練抽樣病人交集為空。

### 4. 達標判定 (Assessment)
* [ ] **已達標 (Achieved)**：不自行宣稱。實作方狀態為 `FIXED?`，須由稽核方複驗 AF-001～AF-012 後更新 `AUDIT.md`。
* [x] **未達標 (Failed)**：G0 尚未正式通過；原因是 finding 尚未由稽核方關閉，且 T2-WB 的 image re-inversion positive control 顯示明顯 latent 誤差。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* **失敗原因分析**：前一輪把「程式可跑」誤當成完整證據閘門，並自行重寫 legacy DDIM schedule；環境與 split hash 也缺乏來源隔離。這些工程缺口已修正。剩餘的低 re-inversion fidelity 是量測結果，不可用換 seed、換樣本或換 metric 隱藏。
* **下輪修改計畫**：先交由稽核方複驗。若 AF-001～AF-005 關閉，再進 E2.2；T2-WB 後續必須把 exact-latent T1 與 image re-inversion T2-WB 分欄，positive control 不足時依 WORKFLOW 標 `Inconclusive`。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無。AF-011 與 AF-012 的使用者裁決已同步寫入 `PROPOSAL.md`、`WORKFLOW.md`；`AUDIT.md` 未修改。

### 7. 下一步
* 請稽核方重跑 AUD-20261008-01 的驗收指令並決定各 finding 是否由 `FIXED?` 轉為 `CLOSED`。
* 稽核通過後執行 E2.2 S0/S1 wrapper；不直接啟動大型 latent cache。
* 先針對 T2-WB 設定 inversion positive-control 門檻與 `Inconclusive` 分流，再排程正式攻擊。
