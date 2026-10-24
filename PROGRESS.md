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


## 2026-10-08 · W1 · Track E · E2.2 S0/S1 legacy wrapper regression

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E2.2。
* 回應稽核：AUD-20261008-02 已判定 PASS 並允許進入 E2.2；AF-016 依使用者裁決先固定 formal N=200，小實驗跑通後才評估另建 `security_v2`。
* 以 guidance=0、noise level=500 確認 S0/S1 wrapper 與學長 legacy anonymization sampler 一致，且不記錄 raw key material。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/e2_legacy_wrapper.py::legacy_anonymize`、`verify_scheme`，直接呼叫 `ddim_sample_loop_anonymization`，沒有複寫 legacy DDIM schedule。
* 新增 `tests/test_legacy_repro.py`，檢查 S0 Rademacher、S1 Signed Permutation 的 transform round-trip 與 wrapper/direct-call bit-exact。
* 產生 `results/E2.2.json`；同步更新 `README.md` 與 `WORKFLOW.md` 的狀態、檔案索引及 formal N 裁決。
* 驗證指令：`scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/test_legacy_repro.py`；`scripts/run_cfg_ddim.sh python -m scripts.e2_legacy_wrapper --checkpoint past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt --noise-level 500 --guidance-scale 0 --output results/E2.2.json`。

### 3. 執行結果 (Results & Observations)
* [事實] unit 與 E2.2 regression 共 `12 passed`；S0/S1 的 wrapper 與 direct legacy call 在 output、latent、input 的 MaxAbs 全為 0。
* [事實] S0/S1 transform round-trip MaxAbs 均為 0；所有 output/latent 皆 finite。
* [事實] 使用單張 `dev_v1.1_000`、noise level=500、guidance=0；結果檔只記錄 tensor SHA-256 與公開 legacy parameter profile，未寫入 raw key。
* [事實] 執行時 GPU 另有程序高使用率，因此本輪不報告或解讀 runtime；數值一致性不受此項效能干擾。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：E2.2 指定的 S0/S1 wrapper、legacy 一致性測試與 `results/E2.2.json` 均完成；工程閘門與小型證據閘門通過。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 本輪未失敗。E2.3 不直接長跑；先完成 AF-014 的真實 stdout/stderr/exit code、非空 numeric fields、status/heartbeat/lock 與失敗控制，再做 1–4 張 cache smoke。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無。formal N=200 為 2026-10-08 使用者裁決；未改動既有 `security_v1` 成員。

### 7. 下一步
* 修正 AF-014 並建立可恢復的 E2 background runner。
* 執行 E2.3 的 1–4 張 latent cache smoke，驗證 shape、finite 與抽樣重算 MaxAbs ≤ 1e-5。
* smoke 通過後才對 `security_v1` N=200 排程 formal cache；不啟動 T2-BB。


## 2026-10-08 · W1 · Track E · AUD-20261008-02 未關閉 finding 修復

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md`：E1.1、E1.2、E1.4、E1.5 與 E2.3 長任務前置條件。
* 回應稽核：AUD-20261008-02 的 AF-006、AF-013、AF-014、AF-015、AF-016；`AUDIT.md` 保持唯讀，關閉狀態仍由稽核方裁定。

### 2. 執行動作 (Actions Taken)
* [AF-006] 已修正（待複驗）｜commit `625f16d` 的 message 逐一列出 14 個實際變更路徑、任務依據、驗證與 `Refs`。
* [AF-013] 已修正（待複驗）｜commit `625f16d`｜新增 `scripts/env_guard.py::enforce_active_prefix`，並接入 `e1_ddim_runner.py`、environment inventory、dev/security split builders；驗證移除 `PYTHONNOUSERSITE` 的直接 conda run 會在載模前拒絕 `~/.local` 套件。
* [AF-014] 已修正（待複驗）｜commit `625f16d`｜`artifact_schema.py::validate_run` 要求非空 numeric fields、欄位存在、exit-code 檔一致及完整 packages；E1 runner 保存真實 stdout/stderr/exit code；新增 `managed_run.py` 的 atomic status、PID、heartbeat、task/GPU lock 與失敗狀態。
* [AF-015] 已修正（待複驗）｜commit `625f16d`｜`run_smoke` 拆分 `runtime_seconds_primary_cycle`、`runtime_seconds_repeat`、`runtime_seconds_reinversion` 並於每段 GPU synchronize。
* [AF-016] 已回覆（待複驗）｜commit `625f16d`｜新增 `reports/compute_budget.md`；依使用者裁決 formal N=200，擴大只能另建 `security_v2`。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration/test_legacy_ddim_equivalence.py tests/test_legacy_repro.py` 得 `20 passed in 52.68s`。
* [事實] `env -u PYTHONNOUSERSITE conda run ... e1_ddim_runner.py --help` 以 exit 1 中止，列出越界 NumPy、SciPy、Pillow 路徑；正式 wrapper 正常。
* [事實] AF-014 intentional failure control 留下真實 traceback、`exit_code=1` 與 `status.state=failed`；正常 E1.2/E1.4 新 run 均由 validator 判定 `valid=True`。
* [事實] E1.4 四張 primary cycle 平均 `23.149s`；repeat `29.606s`；re-inversion `28.869s`。後兩者受同卡其他程序競爭，不作效能結論。
* [事實] N=200 每個完整 cycle 預估 `0.922 GPU-hours`；14-cycle core 約 `12.91 GPU-hours`，加 25% operational reserve 約 `16.14 GPU-hours`，再加兩 cycle contingency 上限約 `18.45 GPU-hours`。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：五筆 finding 的實作方驗收條件均完成；狀態為 `FIXED?`，等待稽核方重跑並在 `AUDIT.md` 關閉。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 本輪未失敗。正式 E2.3 仍先執行 1–4 張 managed cache smoke；只有 smoke 的 schema、重算與 lifecycle controls 全過才排程 N=200。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無。N=200 與 `security_v2` 規則是使用者裁決；未修改 `security_v1` 成員或 hash。

### 7. 下一步
* 請稽核方複驗 AF-006、AF-013～AF-016 並決定是否 CLOSED。
* 複驗通過後建立 E2.3 latent cache runner 與 1–4 張 smoke，不直接啟動 N=200。

## 2026-10-09 · W1 · Track E · AF-017／AF-019／AF-020／AF-021 修復與加密流程 P 基線

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E1.4、E2.2，以及 E2.3／E2.5 開始前的前置條件。
* 回應稽核：AUD-20261008-03 的 AF-017、AF-019；AUD-20261009-01 的 AF-020 與 §8-Q2；AUD-20261009-02 的 AF-021。
* [決定] 使用者 2026-10-09 指派 Claude 擔任實作與測試方，並同意依 loop spec §6 逐子任務 commit；稽核方另行驗證。`AUDIT.md` 保持唯讀。

### 2. 執行動作 (Actions Taken)
* [AF-021] 已修正（待稽核複驗）｜commits `15e5a62`、`05ae398`｜`scripts/e2_legacy_wrapper.py`：生成固定 `LEGACY_GUIDANCE_SCALE=-1`，移除 `--guidance-scale`，新增 `legacy_deanonymize`、`legacy_model_kwargs`；manifest 記 `guidance_scale=-1`、`generation_conditioning`、`inversion_conditioning`、`pipeline`；`tests/test_legacy_repro.py` 新增 −1 對 0 的 bit-exact test；`reports/compute_budget.md` 改用加密流程實測成本；README／WORKFLOW E2 協定裁決／PROPOSAL §7.3 用語改為「y=0 健康類條件生成、單次模型呼叫、無 CFG 混合」｜驗證：`scripts/run_cfg_ddim.sh python -m pytest -q tests/test_legacy_repro.py`；`scripts/run_cfg_ddim.sh python -m scripts.e2_legacy_wrapper --checkpoint <ckpt> --noise-level 500 --output results/E2.2.json`。
* [AF-017] 已修正（待稽核複驗）｜commits `faf2fa9`、`7689239`｜`build_legacy_components` 新增 `P`＝恆等 Rademacher 金鑰（全 +1），P／S0／S1 共用 `ddim_sample_loop_anonymization`，只差在金鑰；新增 `scripts/e2_anonymization_runner.py`（manifest `pipeline=legacy_anonymization:ddim_sample_loop_anonymization`）與 P smoke；新增 test：恆等金鑰 P 與直接呼叫學長函式 bit-exact，且等同無金鑰的 legacy forward→backward；E1 runner manifest 標 `pipeline=legacy_inference:ddim_sample_loop_known_progressive`，README 將 E1.4 標為推論流程｜驗證：同上 pytest；`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id AF017_P_SMOKE --validate-artifacts -- python -m scripts.e2_anonymization_runner p-smoke --checkpoint <ckpt>`。E2.3 latent cache 將以 `ddim_anonymization_forward` 產生（尚未開始）。
* [AF-020] 已修正（待稽核複驗）｜commits `cf1879f`、`7689239`、`4fcb7ca`｜`scripts/managed_run.py` 新增 `--validate-artifacts`：子程序成功後以 `validate_run` 驗證，未過即 `state=failed`、exit 3；自身設定改存 `runner_config.json`。新增 `scripts/run_artifacts.py`，E1 runner 在 managed 模式下直接寫入受控 run 目錄。以 committed runner 經 managed_run 重跑 E1.2／E1.4；四個舊 run 目錄未刪除，已加 `SUPERSEDED` 標記｜驗證：`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id E1.4 --validate-artifacts -- python scripts/e1_ddim_runner.py smoke --checkpoint <ckpt>`（E1.2 同理，`benchmark`）。
* [AF-019] 部分修正｜commits `7689239`、`4fcb7ca`、`faf2fa9`、本條 docs commit｜E1.4 的 re-inversion 改名為 `latent_reinversion_shared_noise_*`（0.906 標為共用雜訊、樂觀上限），並新增 `latent_reinversion_unknown_noise_*`（seed+1,000,000）；加密流程 P smoke 報告無雜訊 re-inversion 的 latent 指標與低／高頻、\|z\| 診斷，並以 M1 端到端影像指標為主；新增 `reports/t2wb_protocol.md` 定義 T2-WB 正控制（P）、負控制與 `Inconclusive` 判定提案。驗收條件中的 E2.5 表格與 A4 T2-WB 結果尚未產生，**本 finding 不能在本輪關閉**。
* [AUD-20261009-01 §8-Q2] 已回覆｜commit `05ae398`｜預算改以 forward（F）／generation（G）半週期矩陣計算，P 的匿名生成與 T2-WB 正控制皆已計入。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py` → `33 passed in 60.61s`。`git diff --stat b94a4a9..HEAD -- past/SourceCode` 為空。
* [事實] E2.2 CLI（dev_v1.1_000、T=500，`results/E2.2.json`）：P／S0／S1 的 wrapper 對 direct、wrapper 對 guidance 0、deanonymize wrapper 對 direct、transform round-trip，以及 P 對無金鑰 forward→backward，MaxAbs 全為 0。S0／S1 output 與 latent SHA-256 與舊 guidance 0 版本相同。
* [事實] 加密流程 benchmark（managed run `AF021_ANON_BENCH_20261009T152410677702Z_a5f3dcdc`，valid）：每張半週期 batch 1 為 F=7.73 s、G(−1)=7.82 s、G(0)=15.40 s；batch 8 為 F=5.54 s、G=5.52 s。batch 1 full cycle −1／0 = 15.55／23.13 s，與稽核方的 15.9／23.0 s 一致。N=200 core 11.67、+25% 14.59、+2 cycles 15.82 GPU-hours（舊版 18.45）。
* [事實] P smoke（managed run `AF017_P_SMOKE_20261009T152434977798Z_df76812e`，valid；4 張平均）：P 輸出 vs 原圖 PSNR 42.35 dB、SSIM 0.9838；同一輸入重算 inversion MaxAbs=0（4/4 bit-exact）；re-inversion（PNG 交接）latent cos 0.275（範圍 0.012–0.739）、低頻 32×32 cos 0.974、\|z\| Pearson 0.430；M1 PNG 端到端 PSNR 38.10 dB、SSIM 0.9760、L∞ 0.136；PNG 儲存本身 vs P 輸出 PSNR 54.58 dB。科學數值只報告，不作成功門檻。
* [事實] E1.2／E1.4 重跑（`E1.2_20261009T153114394206Z_8339c2f5`、`E1.4_20261009T153121667310Z_aa04e79c`）state=succeeded、artifact_validation valid；manifest `script_sha256=a9711eea…` 等於 commit `7689239` 的 runner。E1.4 所有既有欄位與舊 CSV 逐值相同（max diff 0），contact sheet 位元組相同；新的未知雜訊 re-inversion cos 平均 0.3713，與稽核方 0.371 一致。
* [觀察] 加密流程即使沒有 t=0 雜訊，re-inversion 的 latent 仍只保留低頻（高頻 cos 0.24）。逐元素 \|z\| 攻擊在 T2-WB 下的結果必須先看 P 正控制才能解讀，這支持 AF-019 的疑慮。
* [觀察] dev_v1.1_010／011 的 M1 還原圖中可見原圖沒有的細小字樣狀紋理（`image/AF017_P_anonymization_smoke.png`），屬生成模型的局部幻覺；n=4，僅記錄，留待 E2.5／Q8 以較大樣本確認。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-017、AF-020、AF-021 的實作方驗收條件與兩個硬閘門均完成，狀態為 `FIXED?`，待稽核方複驗後在 `AUDIT.md` 關閉。
* [ ] **未達標 (Failed)**：無。AF-019 屬「部分修正」而非失敗，其驗收條件依賴尚未開始的 E2.5 與 A4。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 本輪未失敗。E2.3 必須等稽核方關閉 AF-017／AF-020／AF-021 後才開始；開始時先做 1–4 張 managed smoke，並檢查 batch 8 與 batch 1 的 inversion 是否 bit-exact，再排程 N=200。

### 6. ⚠ 與 WORKFLOW.md 不符
- WORKFLOW §3.4 第 4 點要求 positive control 不足時標 `Inconclusive`，但沒有預先登記「不足」的門檻。`reports/t2wb_protocol.md` §4 提出方案 (a) 統計可分離（建議）與方案 (b) 固定 10 倍隨機基準，**需使用者裁決**；裁決前不得用來判定 A4／D7 結果。
- 依 AUD-20261009-02 使用者裁決（生成 guidance −1），已更新 WORKFLOW E2「協定裁決」與 PROPOSAL §7.3 的用語；通過條件未放寬。WORKFLOW E1.4 列的「主線 `guidance_scale=0`」描述的是推論流程 smoke 的實際設定，未改。

### 7. 下一步
* 請稽核方複驗 AF-017、AF-020、AF-021，並確認 AF-019 的部分修正方向；請使用者裁決 T2-WB `Inconclusive` 門檻。
* 稽核通過後執行 E2.3：以 `ddim_anonymization_forward` 對 `security_v1` 建立 latent cache（先 1–4 張 smoke，含 batch 組成的 bit-exact 檢查）。
* 接著完成 E2.4 metric semantics 報告與 E2.5 dev 20 張 P／S0／S1 的 T1 與 T2-WB（M1 PNG 交接）round-trip，用以關閉 AF-019。

## 2026-10-10 · W1 · Track E · E2.3–E2.5 完成：latent cache、指標語意、P/S0/S1 dev round-trip

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E2.3、E2.4、E2.5，即 G0 剩餘項目。
* 回應稽核：AUD-20261009-03 的 AF-022；AUD-20261008-03 的 AF-019（E2.5 部分）；AUD-20261010-02 使用者裁決（T2-WB `Inconclusive` 規則）落地。AUD-20261010-01 已關閉 AF-017、AF-020、AF-021，並建議可開始 E2.3。

### 2. 執行動作 (Actions Taken)
* [裁決落地] commit `e6051ff`｜`reports/t2wb_protocol.md` §3／§4 改為各方案專屬正控制：攻擊成功照實報告，未成功才看該方案以正確金鑰解密後的 latent linkage；新增 §5 latent 方向敏感度診斷。
* [E2.3] commits `a3a9a60`、`068d75b`、`b12ca19`｜`scripts/e2_legacy_wrapper.py` 新增 `legacy_invert`、`apply_legacy_key`、`legacy_generate`，test 證明 cached x_T 路徑與完整 legacy sampler 在 P/S0/S1 anonymize、deanonymize 下都 bit-exact；`scripts/e2_anonymization_runner.py` 新增 `cache` 模式；`scripts/run_artifacts.py` 改為在 run 開始時記錄 git 狀態｜指令：`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id E2.3 --validate-artifacts -- python -m scripts.e2_anonymization_runner cache --checkpoint <ckpt> --split splits/security_v1.csv --batch-size 1 --recompute-count 20 --done-output results/E2.3_DONE.json`（dev 另以 `--split splits/dev_v1.1.csv --recompute-count 5 --done-output results/E2.3_dev_v1.1_cache.json`）。
* [AF-022] 已修正（待稽核複驗）｜commit `905580a`，§5 於本條 docs commit 補齊｜`scripts/e2_metric_semantics.py`、`results/E2.4_metric_semantics.json`、`reports/metric_semantics.md`。由實作方指派的 subagent 撰寫，實作方審閱並修正一處通過條件用語｜指令：`scripts/run_cfg_ddim.sh python -m scripts.e2_metric_semantics`。
* [E2.5／AF-019] commits `4156be7`、`5590937`｜新增 `scripts/e2_roundtrip_runner.py`：從 E2.3 dev cache 出發，分報 transform-only、T1、T2-WB／M1（學長 PNG 交接為主、float 交接為上限）；影像對 x0 與對 P 輸出並列；latent 另報攻擊者視角與解密後保真度；以 bootstrap 95% CI 彙整｜指令：`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id E2.5 --validate-artifacts -- python -m scripts.e2_roundtrip_runner --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/E2.3_20261009T190608331743Z_6edaf199`。
* [加速診斷] commit `3a53335`｜`scripts/perf_probe.py`、`results/perf_probe_batch1.json`，回應使用者的加速詢問；非正式實驗。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py` 在各 commit 前通過；本條時點 unit 23 passed、`tests/test_legacy_repro.py` 20 passed。
* [事實] E2.3：managed runs `E2.3_20261009T183736363217Z_84cf642a`（security_v1，200 張）與 `E2.3_20261009T190608331743Z_6edaf199`（dev_v1.1，20 張）皆 state=succeeded、artifact_validation valid；shape (N,1,256,256)、全部 finite；seed 1911 抽樣以 batch 1 重算 20/20 與 5/5 逐位元相同（MaxAbs=0 ≤ 1e-5）。
* [事實] batch 組成會改變結果：smoke 以 batch 4 建 cache、batch 1 重算，MaxAbs=6.93e-4、0/4 bit-exact，runner 正確判 failed；batch 1 建 cache 則 4/4 bit-exact。
* [注意] E2.3 兩個正式 run 的 manifest `git_commit` 記為 `905580a`：run 期間提交了 E2.4，而舊版 `run_artifacts` 在結束時才讀 git。runner 的 `script_sha256=2906b142…` 與 run 開始時的 `a3a9a60` 相同。`068d75b` 已改為開始時記錄。
* [事實] E2.4：dev_v1.1 不同病人像素 cosine 中位數 0.8844（190 對）、去均值後 0.5323；同圖加雜訊到 PSNR 30.6／26.2 dB 時 cosine 0.998697／0.996424；`y=0.5·x` 時 cosine=1 但 PSNR 10.79 dB。與稽核方 `audit/out/e2_legacy_cosine_semantics.json` 比對差 ≤ 1.3e-5（協定差異已記錄）；連跑兩次除 created_at 外相同。
* [事實] E2.5（managed run `E2.5_20261009T191319376995Z_f990c7d7`，60 rows valid；20 張平均 [95% CI]）：
  * transform round-trip 60/60 逐位元相同；T1 還原對 x0 為 42.23 dB，P/S0/S1 逐位元相同。
  * M1 PNG 還原對 x0：P 37.71 [37.25, 38.07]、S0 31.47 [30.60, 32.30]、S1 31.30 [30.75, 31.89] dB；SSIM 0.975／0.927／0.898；L∞ 0.232／0.558／0.416。
  * 對 P 輸出：43.09／30.75／30.20 dB，影像 cosine 0.9999／0.9986／0.9985。
  * M1 float 還原對 x0：37.62／38.70／39.40 dB。
  * 解密後 latent cos（PNG）0.299／0.990／0.990，\|z\| Pearson 0.434／0.976／0.976。
  * 與稽核方 AUD-20261010-01 §4 的獨立量測在四捨五入內一致。
* [事實] 對正交的 P/S0/S1 而言，攻擊者視角（ẑ_ano 對 z_ano）與合法解密（ẑ 對 z）的全域 latent 指標逐值相同，因此 S0/S1 的 T2-WB 攻擊者能取得與解密者同等保真度的 \|z\|（Pearson 0.976）。
* [觀察] `image/E2.5_roundtrip_grid.png` 中，dev_v1.1_000 的 S0 匿名影像仍可見胸廓與肺野的粗略輪廓（與 S0 保留 \|z\| 一致）；僅目視、n=2，留給 A3.3／P3.1 量化。
* [觀察] P 的 T1 還原中，dev_v1.1_012 的 L∞ 達 0.607（PSNR 40.6 dB），其餘 ≤ 0.204：DDIM 反演＋生成本身也會有局部大誤差，與加密無關。
* [事實] 加速診斷：CUDA Graphs（以快取同一 CPU 運算的 monkey-patch 讓 legacy `timestep_embedding` 可被 capture）與 2／3 個 batch 1 行程並行，latent SHA-256 都與單行程 eager 相同；CUDA graph 約快 8%，並行吞吐約 1.23×／1.30×（保守）。
* [限制] 量測期間 GPU 與 ollama runner（6.3 GB，約 19:32 UTC 起）及稽核腳本 `audit/e2_wht_smoke.py` 共用，單行程 eager 為 37–40 ms/step（獨占時 15.5 ms/step）。上述比例與 E2.5 後半段的 runtime 欄位只能參考。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：E2.3、E2.4、E2.5 的工程閘門（bit-exact test、finite、shape、重算 MaxAbs、cache hash 驗證）與證據閘門（managed run、manifest、per-sample、summary、split／checkpoint／script hash）皆通過。G0 的子任務 E0.1、E1.1–E1.5、E2.1–E2.5 全部有交付物；G0 是否正式通過、AF-019／AF-022 是否關閉，由稽核方判定。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。唯一的負結果是 batch 4 cache 的重算檢查不過；依據是 E2.3 預先登記的 1e-5 判準，因此改用 batch 1，沒有放寬判準。

### 6. ⚠ 與 WORKFLOW.md 不符
- [決定]（實作方，可由使用者推翻）正式 pass 固定 batch 1。N=200 預算上限從 batch 8 估計的 15.82 改為 22.29 GPU-hours（`reports/compute_budget.md`）。若要以 batch 8／並行／CUDA Graphs 加速，需先在獨占 GPU 上重測並經使用者同意。
- WORKFLOW E2.5 的交付物 `table_baseline_correctness.csv` 依 §3.1 建議放在 `paper_assets/tables/`。
- GPU 目前與其他專案（ollama）及稽核腳本共用，WORKFLOW §8 的長任務排程需考慮此競爭；runtime／效能表必須在獨占 GPU 上量測。

### 7. 下一步
* 請稽核方複驗 E2.3–E2.5、AF-019（E2.5 部分）、AF-022，並判定 G0。
* 進入 Week 2：A3.1 key-space audit、A3.2 小空間 brute force、A3.3 S0/S1 不變量、A3.4 T1 gallery（以 `security_v1` cache 做 P 正控制與 shuffle 負控制）。
* A4 T2-WB 依 `reports/t2wb_protocol.md` §4 實作，需先為 security_v1 產生 S0/S1 匿名影像與 re-inversion（N=200，約 2.6 GPU-hours）。

## 2026-10-10 · W1 · Track E · AF-023 用語與 provenance 修正

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：E2.3、E2.4、E2.5（文件與證據鏈收尾）。
* 回應稽核：AUD-20261010-03 的 AF-023（7 點）。AUD-20261010-03 判定 G0 通過，並已關閉 AF-019、AF-022。
* [決定] 使用者 2026-10-10 裁決 AF-023 第 6 點：影像指標維持不 clip 的定義，只在報告中註明，不修改、不重算。

### 2. 執行動作 (Actions Taken)
* [AF-023-1] 已修正（待稽核複驗）｜commit `6db5347`｜`reports/metric_semantics.md` 摘要中「0.9989 等於 PSNR 約 30–32 dB，沒有 PSNR 以外資訊」改標 `[推定]`，並限定在加性、近正交誤差下；§3.4 與 §5 同類敘述一併改為 `[推定]`。
* [AF-023-2] 已修正（待稽核複驗）｜commit `6db5347`｜「還原圖對 P 輸出」一律改稱「前作比較方式的類比」，註明前作實際比較的是兩條路徑的偽健康 `samples`；引用的稽核方 AUD-20261010-01 §4 表格，依稽核方自身更正同步改寫。
* [AF-023-3] 已修正（待稽核複驗）｜commit `6db5347`｜T1 的「S0/S1 與 P 逐位元相同」改寫為依構造成立：transform 逐位元可逆，所以生成輸入與 P 相同，runner 直接沿用 P 的輸出（`t1_reused_p_output=1`，60/60）。
* [AF-023-4] 已修正（待稽核複驗）｜commit `6db5347`｜`reports/t2wb_protocol.md` §1 改為各方案專屬正控制；`README.md` 的 t2wb_protocol 列改為已裁決。
* [AF-023-5] 已修正（待稽核複驗）｜commit `26aed6f`｜`scripts/run_artifacts.py`：`git_dirty` 改用 `git status --porcelain=v1 --untracked-files=no`；manifest 新增 `imported_project_modules`，記錄 repo 內被 import 的 `.py` 及其 SHA-256，含 runtime 載入的 `past/SourceCode` 模組，並略過相對路徑的擴充模組；`tests/unit/test_run_artifacts.py` 新增對應檢查。
* [AF-023-6] 依使用者裁決只註明｜commit `6db5347`｜`reports/metric_semantics.md` §3.3 與 §5 註明：PSNR、SSIM、L∞、MSE、MAE、cosine、bit-exact 都直接用未 clip 的生成輸出計算，只有 uint8 pixel equality 先 clip；與 clip 慣例的差異 ≤ 0.042 dB（稽核方 E2.5 逐張比對）。
* [AF-023-7] 已修正（待稽核複驗）｜commit `26aed6f`｜`results/perf_probe_batch1.json` 補上 9 個行程中每張影像的 latent SHA-256 與參考值；全部與單行程 eager 相同。
* 驗證指令：`scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py`。

### 3. 執行結果 (Results & Observations)
* [事實] 測試 `44 passed in 132.10s`（unit 23、integration 1、legacy repro 20）。
* [事實] 本輪只改文件、provenance 欄位與診斷 JSON，沒有修改任何正式 run、per-sample 結果或主表數值；E1 runner（`scripts/e1_ddim_runner.py`）未改動。
* [更正] 前一條目（2026-10-10 E2.3–E2.5）§3 寫「T1 還原對 x0 為 42.23 dB，P/S0/S1 逐位元相同」。這是依構造成立，不是另外量測（見 AF-023-3）：S0/S1 的 T1 輸出直接沿用 P 的輸出，因為 transform 逐位元可逆（60/60）。舊條目保留不改。
* [注意] `git_dirty` 改定義後，只有已追蹤檔案有未提交修改時才會是 true；本輪之前的 run，其 `git_dirty=true` 可能只是因為有 `AUDIT.md`、`audit/` 等未追蹤檔。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-023 七點的實作方修正完成，測試通過；狀態 `FIXED?`，待稽核方複驗。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。教訓是：報告摘要不得比正文的證據等級更強；依構造成立的等式要寫明機制，不要寫成量測結果。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無。

### 7. 下一步
* 請使用者 push（`git push origin main`）；自動模式擋下了實作方的 push。
* Week 2：A3.1 key-space audit、A3.2 小空間 brute force、A3.3 S0/S1 不變量、A3.4 T1 gallery（security_v1 cache）。

## 2026-10-10 · W1→W2 · Track E/A · AF-024 保存值域的 M1 存檔協定

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A4（T2-WB 匿名影像產生前置條件）、D5（HMAC 部分，延後）、§1.3 M1 定義。
* 回應稽核：AUD-20261010-05 的 AF-024（第 1、2、4、5 點；第 3 點 HMAC 依稽核排程於 D5 完成）。
* [決定] 使用者 2026-10-10 裁決（AUD-20261010-05）：
  * 前作 M1 約 30–32 dB 的損失，歸因於存檔時 per-image min-max 後未保存值域，屬於工程失真，不是翻號／置換加密本身的限制。
  * 之後所有以影像形式保存、且需要再 inversion 或還原的輸出，都必須記錄實際值域並在讀回時還原。

### 2. 執行動作 (Actions Taken)
* [AF-024-1] 已修正（待稽核複驗）｜commit `f5c967e`｜新增 `scripts/m1_storage.py` 的 `range_preserving_png/v1`：
  * 寫入真實 8-bit 灰階 PNG，以 `np.rint` 量化。
  * float32 lo/hi 以 IEEE-754 位元保存在 PNG text chunk 與 storage record。
  * 讀回公式為 `q/255·(hi−lo)+lo`；缺值域或含非有限值時 fail closed。
  * manifest 記錄 `storage_protocol`、`quantization`。
* [AF-024-2] 已修正（待稽核複驗）｜commits `f5c967e`、`a4b0db9`｜`legacy_png_handoff` 與 E2.2／E2.5 結果不變。新增 `scripts/m1_storage_compare_runner.py`，從 E2.3 dev cache 以新協定重跑 P/S0/S1：逐張驗證匿名影像與 E2.5 相同，再唯讀併入 E2.5 的 legacy PNG／float 欄，輸出三欄並列表。
* [AF-024-3] 規格已落地、實作延後至 D5｜commit `fd4f799`｜PROPOSAL §4.4 與 WORKFLOW D5.5／D5.6、§2.3 已要求：
  * HMAC 涵蓋 `storage_protocol`、`quantization`、lo/hi。
  * PNG text chunk 與 record 的 lo/hi 須逐位元一致。
  * T4 須含「只改 lo/hi」且必須被拒絕。
  * 驗證前不得以 lo/hi 讀回。
* [AF-024-4] 已修正（待稽核複驗）｜commits `f5c967e`、`a4b0db9`｜`tests/unit/test_m1_storage.py` 共 9 項：任意值域誤差 ≤ (hi−lo)/510、lo/hi 精確、常數影像、缺 metadata／NaN 拒絕、優於 legacy min-max；GPU 回歸用 dev 全部 20 張（≥ 4 張的要求）。
* [AF-024-5] 已修正（待稽核複驗）｜commit `fd4f799`｜更新 PROPOSAL §2.1／§4.4／§4.5、WORKFLOW §1.3／§2.3／A4.1／A4.2／D5.5／D5.6、`reports/t2wb_protocol.md` §2–§4、`reports/metric_semantics.md` §3.2／§3.3／§4／§5，並新增 §7（含 §7.5 dev 實測）。文件初稿由 subagent 撰寫，實作方審閱，並補上它列出的 7 處未改衝突，例如 t2wb_protocol §4 第 5 點的舊裁決已標示被取代。
* [證據基礎設施] commit `b1a2fea`｜`scripts/run_artifacts.py` 在 run 開始時記錄已 import 專案模組的 SHA-256；finalize 時若任一檔案改變，run 即失敗。這是實作方審查 AF-023-5 修正時發現的缺口：hash 原本在 run 結束時才讀檔。
* 指令：`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id AF024_M1_STORAGE --validate-artifacts -- python -m scripts.m1_storage_compare_runner --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/E2.3_20261009T190608331743Z_6edaf199 --e2-5-run artifacts/runs/E2.5/E2.5_20261009T191319376995Z_f990c7d7`。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py` → `54 passed in 66.52s`。
* [事實] managed run `AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8`：state=succeeded、artifact_validation valid（60 rows）。
  * script／`m1_storage.py` 的 SHA-256 等於 `f5c967e`。
  * `imported_project_modules` 含 12 個 `past/SourceCode` 模組。
  * 60/60 匿名影像的 `anon_vs_x0_psnr_db` 與 E2.5 逐值相同。
* [事實] 還原對 x0 的 PSNR（20 張平均，三欄依序為 legacy PNG／range-preserving PNG／float）：

  | 方案 | legacy PNG | range-preserving PNG | float |
  |---|---|---|---|
  | P | 37.71 | 37.81 | 37.62 |
  | S0 | 31.47 | 38.44 [37.98, 38.85] | 38.70 |
  | S1 | 31.30 | 39.17 [38.98, 39.34] | 39.40 |

  * S0/S1 新協定與 float 的平均差為 0.26／0.23 dB；逐張差 S0 −1.04～+2.14 dB、S1 −0.39～+1.37 dB。
  * SSIM：S0 0.927 → 0.972，S1 0.898 → 0.973。
* [事實] dev 加密匿名影像的 lo 最低 −0.53，hi 最高 1.0。
* [事實] 保存值域也提升了攻擊者能取得的資訊：S0/S1 的 T2-WB 攻擊者視角 \|z\| Pearson 從 0.976 升到 0.999，解密 latent cosine 從 0.990 升到 0.9997。`[推定]` A4 在 range-preserving 欄下的 magnitude 類攻擊至少不弱於 legacy 欄，待實測。
* [結論] dev 20 張的結果支持使用者的歸因裁決。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-024 第 1、2、4、5 點的實作方驗收條件完成（協定與單元測試、manifest 欄位、兩種交接並列的 dev 結果、文件更新），狀態 `FIXED?`。第 3 點（HMAC 涵蓋 lo/hi）依稽核排程留待 D5，finding 需到 D5 後才能完全關閉。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。A4 起，T2-WB 的匿名影像與 re-inversion 必須同時產生 legacy PNG 與 range-preserving PNG 兩欄。

### 6. ⚠ 與 WORKFLOW.md 不符
- 依 AF-024 與使用者裁決，已更新 WORKFLOW §1.3（M1 定義）、§2.3、A4.1／A4.2、D5.5／D5.6 的文字；通過條件只增不減。
- AF-024 未規定 S2a/S2 是否也要跑 legacy PNG 欄；目前文件不禁止也不強制。建議 D7 至少在 dev 上對 S2 跑一次 legacy 欄作對照，需使用者決定。

### 7. 下一步
* 請稽核方複驗 AF-023 與 AF-024（第 1、2、4、5 點）。
* 請使用者 push（實作方的 push 被自動模式擋下；本地領先遠端的 commit 數持續增加）。
* Week 2：A3.1–A3.4；A4 依兩種存檔協定分欄產生 security_v1 的 T2-WB 資料。

## 2026-10-10 · W2 · Track D · 使用者裁決：S2a/S2 也跑 legacy PNG 欄

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：D6.5、D7.1–D7.3、F9（存檔協定欄位）。
* 回應：前一條目 §6 提出的待裁決事項（AF-024 未規定 S2a/S2 是否也跑 legacy PNG 欄）。

### 2. 執行動作 (Actions Taken)
* [決定] 使用者 2026-10-10 裁決：P/S0/S1/S2a/S2 **兩欄都跑**。
  * range-preserving PNG 為對外宣稱的主結果。
  * legacy PNG 欄是與前作相同基準的對照與歸因，放附表：{S0/S1, S2} × {legacy, range-preserving} 的 2×2 用來分離加密與存檔的效果。
  * R／permutation 等設定的選擇只依 range-preserving 欄。
* 同步更新 WORKFLOW §1.3、PROPOSAL §4.5、`reports/t2wb_protocol.md` §2 第 4 點、`reports/metric_semantics.md` §7.1。D6／D7 的 runner 預設兩欄都產生。

### 3. 執行結果 (Results & Observations)
* [事實] legacy 欄的額外成本只有「讀回 → inversion → 解密 → 生成」一段，batch 1 每張約 15.6 s（F 7.73 s＋G 7.82 s，`results/AF021_anonymization_benchmark.json`）；S2 與 S2a 的匿名影像相同，只需算一次。
* [推定] 額外時間約為：dev 20 張每組設定約 5 分鐘；R=1–4 × permutation on/off 共 8 組約 42 分鐘；正式 N=200 只跑最終 S2 設定約 52 分鐘（約佔 22.29 GPU-hours 上限的 4%）。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：裁決已記錄並落地到規格文件。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 無。

### 6. ⚠ 與 WORKFLOW.md 不符
- 無；WORKFLOW 已依裁決更新，通過條件未放寬。

### 7. 下一步
* Week 2：A3.1–A3.4；A4 依兩種存檔協定產生 security_v1 的 T2-WB 資料。

## 2026-10-10 · W2 · Track A · A3.1–A3.4、P3.1：前作金鑰空間與 magnitude 攻擊

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A3.1、A3.2、A3.3、A3.4、P3.1。
* 證明前作 short-seed 與幅值不變量；在精確 latent（T1）上以 gallery linkage 量化 magnitude／sorted-magnitude／norm 攻擊，附正負控制。
* [注意] AF-025（MAJOR）在 A3 開始前已由稽核方確認不影響 A3（A3 使用精確 latent 與金鑰空間分析，不涉及影像存檔）；AF-025 的兩欄存檔與 norm 旁道要求屬 A4／D7。

### 2. 執行動作 (Actions Taken)
* [A3.1] commit `0a1042e`｜新增 `scripts/a3_keyspace_audit.py`、`paper_assets/tables/table_keyspace.csv`、`reports/keyspace_method_note.md`。
* [A3.2] commit `da7d940`｜新增 `scripts/a3_bruteforce.py`、`results/A3.2.json`。
* [A3.3] commit `ab2a221`｜新增 `scripts/a3_invariants.py`、`results/A3.3.json`。
* [A3.4] commit `188eaf9`｜新增 `scripts/a3_t1_gallery.py`、`results/A3.4_T1_gallery.json`、`paper_assets/tables/table_T1_magnitude.csv`、`image/A3.4_T1_cmc.png`。
* [P3.1] commit `91bf0ff`｜新增 `scripts/p3_attack_overview.py`、`image/P3.1_attack_overview.png`。
* 分工：A3.1–A3.3 由實作方指派的 subagent（CPU、唯讀前作程式）完成，實作方獨立複核四項關鍵宣稱；A3.4、P3.1 由實作方完成。
* 指令：A3.1–A3.3 皆 `scripts/run_cfg_ddim.sh python -m scripts.<module>`；A3.4 `scripts/run_cfg_ddim.sh python -m scripts.a3_t1_gallery --latent-cache artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a`；P3.1 經 `managed_run.py`（dev cache）。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py` → `54 passed`；`past/SourceCode` 未改動。
* [事實] A3.1 金鑰空間（實作方獨立複核皆成立）：
  * S0 password 有效位元上界 32；S1 **password 模式上界 64**（符號 seed 取自 `sha256(pw)[:4]`、置換 seed 取自 `sha256(pw+"_permutation")[:4]`，兩個獨立 32-bit 值）；S1 整數 seed 模式才是 32（`perm_seed = seed + 1000000` 耦合）。
  * 宣稱為 S0 2^d=65,536 bits、S1 2^d·d!≈1,019,573 bits。
  * sha256[:4] 碰撞（如 `pw57085`/`pw89039`）產生逐位元相同的 Rademacher 金鑰；torch 整數 seed 只保留低 32 位；password 優先於啟動命令的 `seed=42`（故 seed 42 無作用）。
* [更正] 前作探索腳本 `attack/attack_keyspace.py` 的平直「32 bits」在 S1 password 模式不精確，正確上界為 64 bits。PROPOSAL §2.1 的「sign 與 permutation 由相同短 seed 產生」只在整數 seed 模式成立；在 password 模式是兩個獨立短 seed。兩者都遠小於宣稱，§2.1 的結論（宣稱不成立）不變，但機制描述待使用者／稽核裁決是否修訂 §2.1。
* [事實] A3.2 縮小空間（2^20）暴力搜尋：S0 合成 d=4096 還原 seed 809560、真實 d=65536 預篩還原 580344（無假陽性）；負控制掃完 2^20 找不到；S1 拆開搜尋還原符號/置換兩 seed，並以找回金鑰加密新 latent 逐位元復現。單執行緒吞吐外推到 2^32：S0 64 座標預篩約平均 1.2 GPU-h、S1 簡單搜尋約 8.7 天。
* [事實] A3.3 不變量（5 合成＋5 真實 latent × password/seed 金鑰）：S0 逐座標 |z_enc|=|z| 最大差 0；S1 sorted(|z_enc|)=sorted(|z|) 最大差 0，被置換改變的座標比例 0.99997–1.0；norm：S0 差 0、S1 float64 相對差 ≤8.0e-16／float32 ≤3.7e-7（歸約順序效應，非解析偏離）；negative control sorted 差 0.25–0.53。
* [事實] A3.4 T1 gallery（security_v1 200 張，精確 latent 上限，隨機 Top-1=0.5%）：

  | scorer | P | S0 | S1 |
  |---|---|---|---|
  | abs（逐座標 \|z\|） | 1.00 | 1.00 | **0.005** |
  | sorted_abs | 1.00 | 1.00 | 1.00 |
  | norm | 0.99 | 0.99 | 0.985 |

  * 負控制（shuffle）Top-1 ≈ 0.005–0.010。符合解析預期：S0 保留逐座標幅值，S1 打散逐座標但保留排序後多重集合與 norm。
  * norm 的 0.99（非 1.0）來自兩張影像 norm 幾乎相同（cache 中 norm 最小間距 0），悲觀 tie-break 下 max_rank=2。
* [事實] P3.1：固定病例 dev_v1.1_000／_010 的 `abs_equal_s0`、`sorted_abs_equal_s1` 皆為 0；sorted-|·| 的 z／S0／S1 三曲線重合，匿名圖視覺打亂。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：A3.1–A3.4 與 P3.1 的工程閘門（確定性、file:line 驗證、正負控制、tie-break、CI）與證據閘門（versioned CSV/JSON、script/cache hash、seed）皆完成。科學結果（攻擊 Top-k）只報告。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。A3.4 是 T1 上限；A4 為其 T2-WB（含 inversion 誤差）版本，且依 AF-025 須兩種存檔協定分欄並加 norm-only／lo-hi 旁道。

### 6. ⚠ 與 WORKFLOW.md 不符
- PROPOSAL §2.1 對 S1 金鑰耦合的機制描述（「相同短 seed」）在 password 模式不精確（實為兩個獨立 32-bit seed，上界 64）。結論不變，是否修訂 §2.1 文字需使用者／稽核裁決；本輪未改 PROPOSAL。

### 7. 下一步
* 請稽核方複驗 A3.1–A3.4、P3.1。
* 進入 A4：T2-WB anonymous-image inversion、magnitude gallery、KPA、CPA、tamper baseline；依 AF-025 兩種存檔協定分欄，magnitude 類加 norm-only（含偏差校正）與 lo/hi 旁道。

## 2026-10-11 · W2 · Track A · AF-026～AF-029 修復；A3 唯密文攻擊補完

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A3.1、A3.2、A3.4、P3.1；流程面 AF-029。
* 回應稽核：AUD-20261011-01 的 AF-026、AF-027、AF-028、AF-029；AUD-20261011-02 的唯密文協定與使用者裁決（PROPOSAL §2.1 修訂、兩份週摘要並存）。

### 2. 執行動作 (Actions Taken)
* [AF-029] 已修正（待複驗）｜commit `002427e`｜新增 `scripts/output_guard.py`、`tests/unit/test_output_guard.py`；為 4 支無 argparse 的腳本補上 argparse，為 13 支會寫追蹤檔的腳本加 `--force`，輸出已存在時預設拒絕覆寫。
* [AF-028-1,2] 已修正（待複驗）｜commit `401c4e9`｜`scripts/a3_t1_gallery.py` 的 feature 改用 float64；top-k 改用 Wilson score CI、mAP 仍 bootstrap。
* [AF-027、AF-028-3] 已修正（待複驗）｜commit `deb1c3b`｜`scripts/a3_bruteforce.py` 新增 E5（S0 唯密文）、E6（S1 唯密文兩階段）與負控制；extrapolation 標明 CPU 單執行緒並加高斯尾部假設；E2 補解析假陽性率、E4 改報 2·2^18 對 2^36。
* [AF-026] 已修正（待複驗）｜commit `39b290c`｜`scripts/a3_keyspace_audit.py` 的 CSV 欄位與 method note 改寫（金鑰上界 vs 攻擊工作量分開；補唯密文與密碼寫死位置、第二組設定）；PROPOSAL §2.1 依使用者核准文字修訂。由 subagent 執行、實作方審閱。
* [AF-028-4] 已修正（待複驗）｜commit `257160e`｜`scripts/p3_attack_overview.py` 的 profile 面板改畫兩位病人 sorted |z| 的逐 rank 差。
* 分工：AF-026 與（上一輪）A3.1–A3.3 由 subagent 執行；AF-027 唯密文攻擊、AF-028 數值與圖、AF-029 由實作方執行。AF-027 的四項關鍵數字由實作方以自寫實作獨立重現。

### 3. 執行結果 (Results & Observations)
* [事實] `scripts/run_cfg_ddim.sh python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py` → `57 passed`；13 支腳本 `--help` 皆只印 usage、不寫任何追蹤檔；無 `--force` 重跑 `a3_invariants` 被拒絕且 `results/A3.3.json` 不變。
* [事實] **唯密文攻擊（AF-027，實作方獨立重現）**：單一 T1 密文 latent 即可在 2^14 縮小 seed 空間中辨識正確金鑰——
  * S0：正確金鑰 37.1σ，2^14 個錯誤候選最高 3.7σ，排第 1。
  * S1：置換 seed 12.0σ、符號 seed 37.0σ，皆排第 1；負控制（置換 seed 在範圍外）最高 3.8σ。
  * 分數本身即以 σ 為單位（錯誤金鑰 ~ N(0,1)）；2^32 個錯誤候選的期望最大值約 6.66σ，仍遠低於正確金鑰。10 個正負控制全過。
  * 與稽核方一致：S0 37σ、S1 符號 36.9σ；S1 置換 12.0 對稽核 14.6σ，因單張 latent 的空間相關不同（兩者皆 ≫ 控制）。
* [事實] **金鑰空間表述（AF-026）**：金鑰上界 S0=32／S1=64 bits；攻擊工作量另分三欄：已知一對明密文=0（直接讀出）、唯密文 S0≈2^32、S1≈2^33。密碼 `Ki@13579` 寫死於 `run_*_Crypto.sh` 等腳本（另有 `paper_experiment_key_2024`）。
* [事實] **norm 攻擊更正（AF-028-1）**：A3.4 的 norm scorer 改用 float64 後，T1 Top-1 由 0.99/0.99/0.985 更正為 **1.00/1.00/1.00**（P/S0/S1）。先前 PROGRESS 稱「兩張影像 norm 幾乎相同」不正確，實為 float32 捨入碰撞；security_v1_0031 與 0105 的 float64 norm 差 5.3e-6、latent cosine 僅 0.027，是不同影像。200/200 的 CI 由 bootstrap [1.00,1.00] 改為 Wilson [0.981,1.00]。
* [更正] AF-028-5：上一輪（2026-10-10 W2）PROGRESS 與 method note 引用的 sha256 碰撞例不同，兩者都是正確的碰撞。canonical 以 `reports/keyspace_method_note.md`（由 `a3_keyspace_audit.py` 產生）為準；舊 PROGRESS 條目保留不改。`A3.2.json` 的 `git_commit` 原記為 `91bf0ff`（腳本當時未追蹤），重跑後記錄當前 commit，另有 `script_sha256` 可補足。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-026、AF-027、AF-028、AF-029 的實作方驗收條件完成，狀態 `FIXED?`，待稽核複驗後關閉；關閉後 A3.1 可判 Achieved、A3.2 的 COA 補完（G1 前要求）。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。A3 的教訓：金鑰「上界」與攻擊「工作量」是兩回事；norm 類不變量要用 float64 避免捨入碰撞被誤判為攻擊失敗。
* [注意] `a3_keyspace_audit.py` 以 file:line 錨點斷言 `scripts/e2_legacy_wrapper.py` 的某一行；該檔日後被編輯時錨點會失效（本輪 AF-029 的 import 已造成一次 73→74 位移，由 subagent 修正）。建議改以函式名或內容比對取代硬編行號，留待後續。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。PROPOSAL §2.1 的修訂經使用者 2026-10-11 裁決（AUD-20261011-02 §1）。

### 7. 下一步
* 請稽核方複驗 AF-026～AF-029。
* 尚未處理（屬後續週次，非本輪範圍）：AF-024 剩 HMAC 涵蓋 lo/hi（D5）、AF-025 兩欄存檔與 norm/lo-hi 旁道（A4／D7）、AF-030 S2 金鑰材料不得經短 seed／非密碼學 PRNG（D5／D6）。
* 進入 A4：T2-WB inversion、magnitude gallery（兩種存檔分欄、加 norm-only 與 lo/hi 旁道）、KPA、CPA、tamper baseline。

## 2026-10-12 · W2 · Track A · AF-028(4)、AF-031、AF-032 修復

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A3.1、A3.2、P3.1；可重現性 AF-031。
* 回應稽核：AUD-20261012-03 的 AF-032 與 AF-028 第 4 點殘留；AUD-20261012-01 的 AF-031；AUD-20261012-03 §9 使用者裁決（PROPOSAL §2.1）。

### 2. 執行動作 (Actions Taken)
* [AF-032] 已修正（待複驗）｜commit `1ee1994`｜`scripts/a3_bruteforce.py` 新增 E7（全 200 張 security_v1 latent 的唯密文正確金鑰分數分布，無搜尋）；conclusion 改報全樣本範圍；`a3_keyspace_audit.py` 的 method note／table basis 由「稽核方結果、補測未完成」改為引用 A3.2 E5/E6/E7；PROPOSAL §2.1 依使用者裁決改為「S0 與 S1 僅憑密文即可」。
* [AF-028-4] 已修正（待複驗）｜commit `e8e1ac3`｜`scripts/p3_attack_overview.py` 的 diff 面板改為 between-patient（非零）對 within-patient z−S1（恆 0）兩條，y 軸 symlog，修正被覆蓋的小標題。
* [AF-031] 已修正（待複驗）｜commit `25eece6`｜`scripts/run_artifacts.py` 與 `e1_environment_inventory.py` 記錄 `gpu_driver_version`。
* 驗證指令：`scripts/run_cfg_ddim.sh python -m scripts.a3_bruteforce --force`；`... python -m pytest -q tests/unit tests/integration tests/test_legacy_repro.py`。

### 3. 執行結果 (Results & Observations)
* [事實] 測試 `57 passed`。
* [事實] 全樣本唯密文（E7，實作方獨立重現）：S0 與 S1 符號階段對 **200/200** 張皆高於 2^32 門檻 6.66σ（中位 37.0σ、最小 29.0σ）；S1 置換階段 **199/200**（最弱 security_v1_0018 為 4.52σ，低於門檻）。與稽核方 `a3_coa_all_images_check` 一致（S0 37.0σ、S1 置換最弱同為 index 18）。
* [事實] 範圍修正落地：唯密文結論寫為「S0 200/200、S1 置換 199/200；前作共用一把金鑰故部署層級單一密文即可」，不寫成每張影像皆可（AF-032 第 1 點）。
* [事實] AF-031：`nvidia-smi` 回報驅動 580.178.04；manifest 與 inventory 已加 `gpu_driver_version`；凍結的 E1 環境基準未重新產生，新欄自下次起生效。
* [更正] AF-032 第 3 點：2026-10-11 條目（§3）所引 A3.2 外推「約 1.2 GPU-h」為**單位錯誤**，正確為 **CPU 單執行緒**約 1.23 小時（平均）／2.46 小時（最壞）；`results/A3.2.json` 的 `timing_nondeterministic` 一向標為 CPU，PROGRESS 舊條目用語有誤，於此更正，舊條目不改（只增不改）。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-028 第 4 點、AF-031、AF-032 的實作方驗收條件完成，狀態 `FIXED?`，待稽核複驗。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 未失敗。教訓：「單一密文即可」要標明適用於 200/200 還是 199/200；外推數字要標 CPU／GPU。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。PROPOSAL §2.1 的兩次修訂（AUD-20261011-02、AUD-20261012-03 §9）皆經使用者裁決。

### 7. 下一步
* 請稽核方複驗 AF-028(4)、AF-031、AF-032。
* 尚未處理（屬後續週次）：AF-024 剩 HMAC 涵蓋 lo/hi（D5）、AF-025 兩欄存檔與 norm/lo-hi 旁道（A4／D7）、AF-030 S2 金鑰材料不得經短 seed／非密碼學 PRNG（D5／D6）。
* 進入 A4：T2-WB inversion、magnitude gallery（兩種存檔分欄、加 norm-only 與 lo/hi 旁道）、KPA、CPA、tamper baseline。


## 2026-10-12 · W2 · Track A · A4.1 T2-WB inversion 啟動

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A4.1；同時把 AF-025 對 A4.2 的 norm 校正與 lo/hi 旁道規格凍結，避免正式資料 schema 返工。
* 工程硬閘門：目標 checkpoint 與 E2.3 cache hash 一致；S0/S1 × legacy/range-preserving 兩種存檔逐筆 finite；query cache shape/hash 可重讀；managed artifact validation 通過。
* 證據硬閘門：security_v1 N=200、batch 1、guidance −1、逐筆 cosine/MSE/RMSE/MaxAbs/bit-exact/norm ratio、storage metadata、run/config/source/checkpoint/cache hash 齊全。科學數值不作通過門檻。

### 2. 執行動作 (Actions Taken)
* commit `ed60420`：新增 `scripts/a4_t2wb_inversion.py` 與 3 項單元測試；更新 WORKFLOW A4.1/A4.2、`reports/t2wb_protocol.md` AF-025 規則與 README 檔案地圖。
* AF-025 protocol 凍結：dev_v1.1 只作 norm bias calibration、security_v1 只作 formal evaluation；legacy lo/hi=(0,1) 為常數控制；range metadata 公開；lo/hi linkage 必須有不同影像的 labeled anonymous enrollment，否則標 `not_evaluable`，不得把同一 record 當 enrollment。
* GPU smoke：managed run `A4.1_SMOKE_20261012T083131186112Z_ba5005f8`，security_v1 第一張，4 rows，artifact validation valid。
* 正式 run 已啟動：`A4.1_20261012T083425685874Z_3132fa49`，PID 1012036，`resumable=false`，由 managed runner 提供 heartbeat、task/GPU lock 與最終 artifact validation。

### 3. 執行結果 (Results & Observations)
* [事實] 相關測試 15 passed；完整既有套件共 61 tests，退出碼 0；staged-file safety guard 通過。
* [事實] smoke 的 query cache shape=(4,1,256,256)，逐筆 hash 重讀一致，manifest `git_dirty=false`、commit=`ed60420`、checkpoint SHA-256=`3f61b80a…`。
* [事實] 第一張影像：legacy/range-preserving 的 attacker latent cosine 為 S0 0.990227/0.999736、S1 0.989908/0.999754；RMSE 為 S0 0.13646/0.02247、S1 0.13884/0.02173。這只驗證 pipeline 與先前 dev 趨勢一致，不是正式科學結論。
* [事實] smoke 約 47 秒/張；正式 N=200 預估約 2.6 小時，query cache 約 100 MiB；啟動前磁碟可用 712 GiB。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：A4.1 的 protocol/schema、runner、單元測試與 1-sample smoke 硬閘門通過。
* [ ] **進行中 (In Progress)**：A4.1 formal N=200；完成並通過 artifact validation 前不得宣稱 A4.1 Achieved。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* range-preserving 讓合法端與攻擊端同時取得更準的 inversion；正式結果必須兩欄並列，安全結論以部署用 range-preserving 欄為準。
* 本 runner 依規格明確標成不可續跑；若中斷，保留失敗 run，另建全新 run 以完全相同 config/split/seed 從頭重跑，並在 PROGRESS 記錄前一 run（不得誤稱 resume）。下一版長任務可把逐影像 checkpoint 與 managed progress API 納入共用 runner。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。A4.1/A4.2 的通過條件依 AF-025只增不減；lo/hi enrollment 前提另行明確化，避免把 T2-WB+enrollment 誤寫成純 T2-WB。

### 7. 下一步
* 監控正式 A4.1 run；成功後驗證 800 rows、query cache/hash、summary 與 `table_T2_WB_inversion.csv`，追加完成條目。
* 再跑 dev_v1.1 N=20 calibration cache，之後實作 A4.2 gallery、方案專屬正控制、shuffle 負控制與 Inconclusive 判定。
* 等待 GPU 時可進行 CPU 的 A4.3 KPA 與 A4.4 CPA，但各自維持原子 commit。


## 2026-10-12 · W2 · Track A · A4.1 formal 完成

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` A4.1：確認 security_v1 N=200 的 S0/S1 × legacy/range-preserving T2-WB inversion 正式結果完整且可供 A4.2 使用。

### 2. 執行動作 (Actions Taken)
* 驗證 managed run `A4.1_20261012T083425685874Z_3132fa49` 的 status、artifact schema、800 筆 per-sample、query index、query latent 逐筆 hash、summary 與正式表。
* 獨立重算 `query_latents.npy` 800 筆 SHA-256，並核對 scheme/storage 各 200 筆、sample ID 唯一、所有指標 finite。

### 3. 執行結果 (Results & Observations)
* [事實] status=succeeded、exit=0、artifact validation valid；query cache shape=(800,1,256,256)，檔案 SHA-256=`a8437677…`，逐筆 hash 800/800 一致。
* [事實] 平均 attacker latent cosine（legacy/range）：S0 0.990040/0.999745；S1 0.989944/0.999767。平均 RMSE：S0 0.137581/0.022037；S1 0.138497/0.021071。
* [事實] range-preserving 讓攻擊端 inversion 顯著更接近精確 ciphertext；是否足以提升 linkage 由 A4.2 實測，不在 A4.1 先下結論。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：A4.1 工程與證據硬閘門全部通過。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* AF-025 不在此時選保護方案；先以 dev-only calibration 與 formal A4.2 實測 norm/lo-hi 攻擊效果，再把效果量與處置選項交由使用者裁決。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。

### 7. 下一步
* 完成 dev_v1.1 calibration run，執行 A4.2；之後依序完成 A4.3 KPA、A4.4 CPA、A4.5 tamper baseline。

## 2026-10-12 · W2 · Track A · A4.2–A4.5 完成（T2-WB gallery、KPA、CPA、tamper）

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A4.2、A4.3、A4.4、A4.5。
* 以 A4.1 的 re-inversion query cache 完成 T2-WB magnitude gallery 攻擊，並補齊 KPA、CPA、tamper baseline。

### 2. 執行動作 (Actions Taken)
* dev 校正 run（AF-025）：managed run `A4.1_DEV_CAL_20261012T133002259066Z_0fa671c3`（dev_v1.1 20 張、80 rows、valid），提供 norm 偏差 median(||zhat_ano||/||z_ano||)：S0 legacy 1.00755／range 0.99678；S1 legacy 1.01116／range 0.99676。
* [A4.2] commit `859dc8e`｜新增 `scripts/a4_t2wb_gallery.py`＋5 單元測試；gallery=E2.3 security cache、query=A4.1 formal cache；scorer abs／sorted_abs／norm／lowfreq_abs；方案專屬解密正控制、seed 1911 shuffle 負控制、Wilson CI、AF-025 norm 校正與 lo/hi 旁道、Inconclusive 判定。
* [A4.3] commit `6213025`｜[A4.4] commit `7b7b9b8`｜[A4.5] commit `9323734`：由實作方指派的 subagent（CPU、唯讀前作程式、不寫金鑰陣列）完成，實作方審閱並獨立驗證確定性與 script_sha256 provenance。
* 驗證指令：`scripts/run_cfg_ddim.sh python -m scripts.a4_t2wb_gallery --gallery-cache <E2.3 security> --query-run <A4.1 formal> --dev-calibration-run <A4.1_DEV_CAL>`。

### 3. 執行結果 (Results & Observations)
* [事實] 測試 `66 passed`；`past/SourceCode` 未改動。
* [事實] **A4.2 T2-WB magnitude gallery（security_v1 N=200，Top-1，legacy／range-preserving）**：

  | scorer | S0 | S1 |
  |---|---|---|
  | abs（S0 指定） | **1.00／1.00** | 0.005／0.005 |
  | sorted_abs（S1 指定） | 0.065／0.930 | **0.19／0.975** |
  | norm raw | 0.005／0.005 | 0.005／0.005 |
  | norm 校正後（AF-025） | 0.005／**0.255** | 0.005／**0.285** |
  | lowfreq_abs | 1.00／1.00 | 0.005／0.005 |

  * 負控制 Top-1≈0.005；方案專屬解密正控制 Top-1=1.00（故無 Inconclusive 格）。
  * [結論] T2-WB 下 magnitude 攻擊成立；**保存值域的部署協定（range-preserving）同時放大攻擊**：S1 sorted 0.19→0.975，校正後 norm-only 由隨機升到 0.26–0.29 且僅在 range-preserving 達顯著（AF-025 得到 security_v1 正式驗證）。
  * [注意] lo/hi 旁道：每病人單筆 anonymous record，依 AF-025 標 `not_evaluable`（需不同影像的具標籤 enrollment）。
* [事實] **A4.3 KPA**：S0 一對即逐位元還原金鑰、held-out 解密 bit-exact。S1 合成 d=4096（量值互異）一對精確；真實 d=65536 因 float32 量值重複（114–150 座標）單對不精確（held-out 錯位 7e-4–1e-3），第 2 對即精確。
* [事實] **A4.4 CPA**：固定金鑰下 d=4096 次基底查詢重建 S0（對角）/S1（簽名置換）解密映射，全新明文 max-abs error 0.0；每查詢換金鑰（模擬 per-image nonce）則 S0 錯誤率 0.506／S1 1.000。
* [事實] **A4.5 tamper baseline**：24 案例全部 accepted=True（無完整性檢查、不拒絕），竄改 PNG 與記錄 `png_sha256` 不符仍被接受；影響 cosine 0.992–0.99999、max_abs 0.098–4.98。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：A4.2–A4.5 的工程閘門（確定性、控制、hash、tie-break、CI）與證據閘門（versioned 表/JSON、run/cache/script hash、seed）完成。A4.1 已於先前條目 Achieved。科學 Top-k 只報告。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* AF-025 在 security_v1 得到正式數據：range-preserving 讓合法還原與攻擊端同時更準，norm-only 旁道因此可用。效果量已量化，處置方向（lo/hi 入 AEAD payload、金鑰導出 norm 擾動、或接受為限制）仍待使用者裁決（AF-025 OPEN）。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。

### 7. 下一步
* 請稽核方複驗 A4.1–A4.5。
* AF-025 的處置方向待使用者裁決。
* G1（10/18）：Week 2 baseline evidence 檢視。之後進入 Week 3 D5（key pipeline、HKDF、CSPRNG、nonce、container、HMAC；AF-024/AF-030）。
* 尚未處理（屬 D5/D6/D7）：AF-024 HMAC 涵蓋 lo/hi、AF-030 CSPRNG 金鑰材料、AF-025 處置與 D7 結構探測。

## 2026-10-13 · W2 · Track A · AF-033～AF-036 修復

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：A4.2（scorer）、A4.4、A4.5、P4.1。
* 回應稽核：AUD-20261012-05 的 AF-033、AF-034、AF-035、AF-036；並落實該輪 §2 使用者裁決（AF-025 依 S2a/S2 於 D7 決定，前作部署欄為 legacy）。

### 2. 執行動作 (Actions Taken)
* [AF-033] 已修正（待複驗）｜commit `a5b8896`｜`scripts/a4_t2wb_gallery.py` 新增尺度不變 scorer `abs_unit`／`sorted_abs_unit`（特徵先除自身 L2 norm）；選擇規則以 dev_v1.1 凍結；`reports/t2wb_protocol.md` §3 記錄規則與 dev 證據；重跑 formal A4.2。
* [AF-035] 已修正（待複驗）｜commit `b1423f2`｜`scripts/a4_4_cpa.py` 新增 1 次查詢攻擊（d=4096 與 d=65536，fresh max-abs 0）、執行 `attack/verify_cpa.py` 作 structured_transform_reference、deviation_from_workflow 說明、修 f-string、補 provenance。
* [AF-036] 已修正（待複驗）｜commits `3208e8d`、`764b432`｜`a4_5_tamper.py` 補 legacy 存檔竄改列；`a4_3/a4_5` 補 git_dirty 與 imported_project_modules；新增 `scripts/a4_5_deployment_tamper.py`（GPU managed run `A4.5_DEPLOY_20261012T182847866867Z_acb4e5a1`）補部署路徑。
* [AF-034] 已修正（待複驗）｜commit `341e319`｜`scripts/build_attack_package.py` headline 修正，C7 以 legacy 欄與尺度不變 scorer 為主、C8 移除 AF-025 已驗證框定。
* 分工：AF-033／AF-034／部署路徑 GPU 由實作方；AF-035／AF-036 CPU 部分由 subagent（中途因用量上限中止），實作方重建驗證確定性與 provenance 後完成並 commit。

### 3. 執行結果 (Results & Observations)
* [事實] 測試 `66 passed`；`past/SourceCode` 未改動。
* [事實] **A4.2 尺度不變 scorer（AF-033）**：選擇以 dev 凍結（尺度不變版一律 ≥ 原始：S1 legacy sorted 0.75→0.80、S0 legacy 0.40→0.50）。formal（security_v1，Top-1，legacy／range）：S1 sorted 0.465／0.995、S0 sorted 0.145／0.975；abs 與 norm 不變。
* [更正] 先前「S1 sorted 0.19→0.975」把存檔效果誇大了，其中一部分是 scorer 弱點。更正後前作部署（legacy）下 S1 T2-WB 為 **0.465**，保存值域為 0.995，即 0.465→0.995。
* [更正] AF-025 框定：依 AUD-20261012-05 §2 使用者裁決，**S0/S1 的存檔對比不作 AF-025 裁決依據**（前作部署本就不保存值域，對應 legacy 欄）；AF-025 的處置依 S2a/S2 於 D7 的跨存檔攻擊決定。先前 PROGRESS／attack package 寫「AF-025 得到 security_v1 正式驗證」在此更正。
* [事實] **A4.4 CPA（AF-035）**：前作實際 CPA 成本為 **1 次查詢**（S0 全一向量讀 k；S1 相異明文讀 (k,π)），d=4096 與 d=65536 皆 fresh max-abs 0；另執行 WORKFLOW 指名的 `verify_cpa.py`（SOT-WHT，D=1024，重建 Q、unseen max-abs 3.6e-15）作參考。
* [事實] **A4.5 tamper（AF-036）**：container/latent 層新增 legacy 存檔竄改（共 30 列，皆 accepted=True）；部署路徑（匿名影像→存檔→inversion→逆金鑰→生成）2 張亦 accepted=True、輸出有限且與未竄改不同——前作管線無任何驗證步驟。
* [事實] A4.3–A4.5 結果檔已補 git_dirty 與 imported_project_modules；script_sha256 與 HEAD 一致、連跑兩次確定。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：AF-033～AF-036 的實作方驗收條件完成，狀態 `FIXED?`，待稽核複驗。AF-033 的尺度不變 scorer 已凍結供 D7.2 沿用。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 教訓：T2-WB 下攻擊者反推 latent 帶共用尺度偏差，magnitude scorer 必須尺度不變，否則低估前作部署的攻擊；此 scorer 已凍結，D7 評估 S2a/S2 時沿用。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。AF-025 框定與 scorer 選擇皆依使用者裁決與 dev 證據。

### 7. 下一步
* 請稽核方複驗 AF-033～AF-036。
* G1（10/18）後進入 Week 3 D5（key pipeline、HKDF、CSPRNG、nonce、container、HMAC；AF-024/AF-030，CPU 為主）。
* AF-025 的處置待 D7 S2a/S2 跨存檔攻擊結果與使用者裁決。

## 2026-10-13 · W2 · Track A · A4.6 只有影像的金鑰還原（使用者指示的延伸實驗）

### 1. 當前目標 (Objective)
* 使用者裁決（2026-10-13，對話中）：確認前作架構在「攻擊者只有原圖與匿名圖」下的危害程度（單張與多張），作為架構是否安全與何處該修的依據；屬實驗室內部評估。
* 補 A4.1（T2-WB 重新反演）與 A4.3（KPA）之間的缺口：A4.3 用的是精確 latent 對；A4.6 改用「影像反演所得的近似密文 latent」。對應 WORKFLOW A4 攻擊章的延伸（非既有子任務；記為 A4.6）。
* 子項：(a) 單一已知對的危害基線；(b) K=1,2,4,8 對時攻擊者能力的進展。(c) 端到端影像還原另記。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/a4_6_imageonly_keyrec.py`（CPU、確定性）：重用 A4.3 的 `s0_kpa`／`s1_decrypt`／`legacy_truth` 等原語；明文 latent 取 E2.3 cache（原圖反演為確定性，攻擊者可精確取得），密文 latent 取 A4.1 `query_latents.npy`（匿名影像反演，近似）。
* S0 多對：以 |z| 加權的符號投票（近零明文座標的符號不可靠，降權）。S1 多對：以每座標 K 維幅值簽章做最近鄰匹配（cKDTree）還原置換（A4.3 的 lexsort 需位元相等幅值，近似 latent 下崩潰）。
* 已知對取 index 0–7，受害者取 index 100–149（50 張，與已知池不相交）；S0/S1 × legacy／range × K∈{1,2,4,8}。
* 不寫出任何原始金鑰材料，只記準確度與相似度。

### 3. 執行結果 (Results & Observations)
* [事實] 產出 `results/A4.6.json`、`paper_assets/tables/table_A4.6_imageonly_keyrec.csv`；連跑兩次表格位元相同；script_sha256 與 HEAD 一致。
* [事實] **S0（符號金鑰）單張即大致可破**：legacy 單對符號準確度 0.956、受害者解密 latent 對正常使用者 cosine 中位數 0.914；range 單對 0.992／0.985。多對快速逼近：K=4 近 1.0，**K=8 時 50/50 受害者皆位元相同解密**（legacy 與 range 皆然；range K=4 已位元相同）。
* [事實] **S1（簽章置換）單張無用、多張可破**：單對置換準確度 ~0（cosine ~0）；range K=4 置換 0.905／cosine 0.904，**K=8 置換 1.000、50/50 位元相同**；legacy 較慢，K=8 僅置換 0.645／cosine 0.647（部分還原）。
* [事實] **保存值域顯著加速 S1 金鑰還原**：range 在 K=4 已達 0.905，legacy 到 K=8 才 0.645。S0 兩存檔都快，range 略快（單對 0.992 vs 0.956）。
* [推定] S0 單對殘差來自「已知者該座標近零、符號判不出」；這些座標在別的受害者身上可能很大，故單對 L∞ 仍大（中位數約 6.6–7.7）。多對投票以不同人補齊 → 近零座標被解決。
* [結論] 前作金鑰與影像無關（一把全域金鑰）、且幅值／排序為不變量，使攻擊者只要蒐集少數幾組（原圖,匿名圖）對，即可還原整把金鑰，之後同金鑰下所有影像都可解密。這直接支持 D5 的 per-image nonce、CSPRNG 金鑰材料與 authenticated container。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：a、b 完成，結果確定性、含正負面向（S0 易、S1 需多對）、provenance 齊全，待稽核複驗。
* 科學結論只報告：S1 legacy 多對仍未完全還原屬如實結果，不調參湊高。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 「認出配對」≠「還原金鑰」：A4.2 的 S1 linkage（只需幅值集合相近）成立，但還原置換需逐位對應，單對在近似 latent 下崩潰；多對的 K 維簽章才足以辨識。
* 下一步：以 b 的還原金鑰做 c（端到端影像還原對照，GPU），比較正常使用者與攻擊者重構圖的 PSNR／SSIM／L∞。

### 6. ⚠ 與 WORKFLOW.md 不符
* A4.6 非 WORKFLOW 既有子任務，是使用者指示的 A4 攻擊章延伸；不改動既有子任務的定義與結論。

### 7. 下一步
* 實作 c：`scripts/a4_6_reconstruct.py`（GPU，重用 A4.1 快取密文 + b 的還原金鑰做生成），少數受害者 × S0/S1 × 兩存檔，報影像級相似度。

## 2026-10-13 · W2 · Track A · A4.6c 端到端影像還原（GPU）

### 1. 當前目標 (Objective)
* 用 A4.6b 的還原金鑰（K=8 已知對）實際生成受害者重構圖，比較「正常使用者（真金鑰）」與「攻擊者（還原金鑰）」對同一張存檔匿名影像解密後的重構圖。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/a4_6_reconstruct.py`（GPU，managed run `A4.6_RECON_20261013T052324279030Z_165666a1`，validate 通過、8 樣本）。受害者 index 100、101 × S0/S1 × legacy／range。
* 流程：A4.1 快取 ẑ_ano →（真金鑰／還原金鑰）de-anon → 生成；x0 以 preprocess 取得、匿名圖另生成供顯示。指標用 skimage PSNR／SSIM、L∞、RMSE；完全相同時 PSNR 以 100 dB 封頂。輸出面板 `image/A4.6_panels/*.png`（原圖、匿名圖、正常重構、攻擊重構、差異）。不寫出原始金鑰材料。

### 3. 執行結果 (Results & Observations)
* [事實] **S0（兩存檔）與 S1（range）：攻擊者重構圖與正常使用者重構圖位元相同**（PSNR 封頂 100 dB、SSIM 1.000、L∞ 0）。即 8 組（原圖,匿名圖）對還原的金鑰，解其他受害者可得與合法持金鑰者**完全相同**的重構。對原圖保真度：S0 legacy 31.2／33.1 dB、S0 range 35.2／38.7 dB、S1 range 39.4／39.2 dB（攻擊者＝合法者）。
* [事實] **S1（legacy）仍部分**：legit~attacker PSNR 14.3／14.6 dB、SSIM 0.35／0.39、L∞ ~1.0；攻擊者重構圖為「貌似胸片但解剖錯誤」，攻擊者對原圖僅 14 dB，合法者為 29.7／30.6 dB。K=8 legacy 置換僅 0.645，故攻擊尚未成功。
* [結論] 影像級證實 A4.6a/b：對 S0（兩存檔）與保存值域的 S1，少數（~8）配對即可讓攻擊者完整重現任一受害者原圖；前作部署（legacy）的 S1 需更多配對。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：c 完成，managed run 驗證通過、provenance 齊全。影像面板與數值表供論文／內部評估。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 這組 a/b/c 的總結論：前作的全域、與影像無關的金鑰＋幅值不變量，使「少數已知（原圖,匿名圖）對 → 竊得整把金鑰 → 解所有同金鑰影像」成立。保存值域會加速（尤其 S1）。這是 D5（per-image nonce、CSPRNG 金鑰、authenticated container）要解的核心弱點。

### 6. ⚠ 與 WORKFLOW.md 不符
* A4.6 為使用者指示的 A4 延伸（同前條目）；GPU 用量 <30 分鐘，使用者已於對話中授權本實驗。

### 7. 下一步
* 交付影像面板與數值表給使用者；A4.6 可供稽核複驗。後續進入 D5。


## 2026-10-13 · W3 · Track D · D5.1–D5.7 實作與驗證中

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` D5.1–D5.7：完成不依賴 WHT 的 key pipeline、CSPRNG、nonce registry、authenticated container、HMAC 與 no-pickle key format。
* 回應 open findings：AF-024 的 lo/hi HMAC／verify-before-decode 部分；AF-030 的 D5 部分（完整 256-bit key→HKDF→CSPRNG，不經短 seed 或非密碼學 RNG）。AF-025 留待 D7 實測後由使用者裁決。
* 使用者規則已版本化於 `AGENTS.md`（commit `a09d591`）：本專案是實驗室自有模型的防禦性弱點驗證；subagent 可用；GPU 預估超過 30 分鐘必須先裁決。本輪 D5 全部為 CPU。

### 2. 執行動作 (Actions Taken)
* 安裝並鎖定成熟 primitive provider `cryptography==50.0.2`；沒有自行實作 SHA、ChaCha 或 HMAC。
* D5.1/D5.2：`src/crypto/kdf.py` 提供 OS CSPRNG 32-byte master key、明確 scrypt password mode、RFC 5869 HKDF-SHA256、每輪 sign-pre/perm/sign-post 與 mac domain labels。
* D5.3：`src/crypto/prng.py` 以完整 HKDF subkey 直接驅動 ChaCha20；sign 取 keystream bits；permutation 使用 uint32 rejection sampling Fisher–Yates；禁止 manual_seed／NumPy／Torch／Python RNG seed 路徑。
* D5.4：`src/crypto/nonce.py` 以 SQLite UNIQUE(scope,nonce) 做持久化 atomic registry；scope 為 master-key keyed fingerprint，不存 raw key。
* D5.5/D5.6：strict canonical JSON container，float 禁止進 canonical 欄；M1 lo/hi 使用 float32 bit hex；HMAC-SHA256 覆蓋 nonce、params、ciphertext bytes、AAD、storage/quantization/lo-hi；驗證及 M1 一致性檢查成功前不呼叫 opener。
* D5.7：32-byte raw key file、0600、拒絕覆寫與 trailing bytes；AST guard 禁止 pickle/torch.load/np.load key loader。
* 稽核 subagent 僅唯讀列出 D5 DoD／風險，實作與整合由主代理完成。

### 3. 執行結果 (Results & Observations)
* [事實] D5 security tests 26/26 passed：RFC 5869、scrypt、ChaCha fixed vector、強制 rejection、65,536 signs、permutation inverse、跨 thread/restart nonce reuse、canonical schema、M1 lo-hi bit consistency、tamper-before-opener、no-pickle。
* [事實] 完整回歸命令退出碼 0；現行套件共收集 92 tests。
* [事實] 初次 evidence smoke：100,000/100,000 random nonce 唯一、collision=0、顯式 reuse rejected；12-case tamper matrix 全過。其後腳本新增 `png_and_record_lo` 與 `ciphertext_byte` 兩例，因此目前追蹤候選 evidence 是舊 12-case 版本，必須在 source commit 後以 clean provenance 重跑成 14 cases，不能先宣稱正式完成。
* [釐清] 本輪沒有「資安政策拒絕」。遇到的是本機 filesystem sandbox 的 `bwrap: loopback: Failed RTM_NEWADDR`，使內建 `apply_patch` 偶發失敗；已用精確斷言替換繼續工作。權限提示是執行環境核准，不是研究內容被阻擋。

### 4. 達標判定 (Assessment)
* [x] **程式硬閘門已通過**：D5.1–D5.7 第一版與 26 項 security tests 通過。
* [ ] **證據硬閘門進行中**：尚須先提交 source/tests，再 clean rerun 100k nonce＋14-case tamper evidence、完整 92 tests、staged guard，才可將 D5 判為 Achieved。
* [ ] **未達標 (Failed)**：無。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 最大風險是 canonical serialization、驗證順序、短 seed 回歸與跨程序 nonce reuse；目前分別由 strict schema、internal M1 verify-before-opener、AST/fixed-vector tests、SQLite UNIQUE constraint 控制。
* AF-030 尚不能完全關閉，因 D6 的 SOT 整合仍需證明這些 CSPRNG 產物被直接使用，且 wrong-key/wrong-nonce sanity control 落在錯誤分布；D5 只完成其 key-material 部分。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。D5 不使用 GPU。`cryptography` 是成熟函式庫，符合「不得自行實作 primitive」。

### 7. 下一步
* 完成 source/test 原子 commit；以 clean HEAD 重跑 14-case evidence 與完整 92 tests，再提交 results／README／本條完成補記。
* 請稽核方複驗 AF-024 D5 部分與 AF-030 D5 部分；之後進入 D6。任何預估超過 30 分鐘的 GPU D6 工作先交由使用者裁決。


## 2026-10-13 · W3 · Track D · D5.1–D5.7 clean evidence 完成

### 1. 當前目標 (Objective)
* 對上一條「實作與驗證中」完成證據硬閘門：以 source commit `9e578dd` 的 clean HEAD 重跑正式 nonce/tamper evidence 與完整回歸。

### 2. 執行動作 (Actions Taken)
* 執行 `python -m scripts.d5_security_evidence --force`：重新產生 D5.4 100,000 nonce 與最新版 D5.6 14-case tamper matrix。
* 分組回歸：security 26 tests、既有 unit 45 tests、integration/legacy 21 tests；共 92 tests。
* 獨立核對兩份 JSON 的 git commit、tracked dirty state、script SHA-256，並核對所有 rejected case 的 opener calls 與外部錯誤字串。

### 3. 執行結果 (Results & Observations)
* [事實] D5.4：100,000/100,000 random 128-bit nonce 唯一、collision=0；同 master scope 的顯式 reuse 被拒絕；相同 nonce 在不同 master scope 可登記；artifact 不含 raw nonce/key。
* [事實] D5.6：未竄改 case 接受且 opener=1；wrong key、tag、nonce、version、parameters、AAD、storage protocol、quantization、record-only lo/hi、PNG-only lo、PNG+record lo、ciphertext byte 共 13 個負控制全部拒絕，opener=0，外部錯誤統一為 `authentication failed`。
* [事實] evidence provenance：git commit=`9e578dd`、`git_dirty_tracked_files=false`；recorded script SHA-256 `711d578f…` 與檔案重算一致。
* [事實] 92-test 回歸未留下 lastfailed；security 26/26、unit 45/45，integration/legacy 21 項完成。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：D5.1–D5.7 的工程正確性與證據完整性硬閘門全部通過。
* [ ] **未達標 (Failed)**：無。
* [注意] AF-024 的剩餘 D5 部分已修復，狀態 `FIXED?` 待稽核；AF-030 的 D5 部分完成，但 finding 必須等 D6 SOT 直接使用 CSPRNG material 與 wrong-key/wrong-nonce sanity test 後才能完全關閉。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* verify-before-open gate 已由控制流測試證明，而非只驗證「最後有報錯」；這可避免 tag 失敗前就使用 lo/hi、inverse 或輸出部分 plaintext。
* D5 不處理 AF-025 的 confidentiality trade-off；依使用者裁決，必須等 D7 實測 S2 norm/lo-hi attack 效果後，才選擇與風險相稱的保護。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。全部使用 CPU；沒有未經裁決的 >30 分鐘 GPU 任務。

### 7. 下一步
* 提交 D5 results／table／README／本完成條目，請稽核方複驗。
* 進入 D6 前先估算各 GPU 工作；任何預估超過 30 分鐘者先請使用者裁決。

## 2026-10-13 · 專案協作 · 安全提示與背景程序回報規則

### 1. 當前目標 (Objective)
* 釐清先前 `This content can't be shown` 提示的性質，並避免介面顯示 `worked` 後，使用者無法判斷本機程序是否仍在執行或工作停在哪裡。

### 2. 執行動作 (Actions Taken)
* 查閱 OpenAI 官方文件對 cybersecurity safeguard 與 false positive 的說明。
* 以本機 process table 檢查 `pytest`、`managed_run`、D5 evidence 與專案實驗 Python 程序。
* 在 `AGENTS.md` 加入持久化規則：未完成、受阻或仍執行中的工作，結束回合前必須寫入 `PROGRESS.md`；仍在跑的程序必須回報 run ID/PID、狀態、log/status 路徑、預估時間及是否需監看。

### 3. 執行結果 (Results & Observations)
* [事實] 引用的提示代表該則回覆被平台 cybersecurity safeguard 隱藏、阻擋、改道或限制；先前稱「沒有資安政策拒絕」不夠精確。
* [事實] 平台回覆是否顯示，與已啟動的本機 shell/GPU 程序是否仍執行是兩個獨立狀態，不能由 `worked` 或提示文字推斷。
* [事實] 本次檢查沒有發現仍在執行的專案 `pytest`、`managed_run`、D5 evidence 或攻擊實驗程序；D5 已於上一條完成紀錄為 Achieved。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：事件性質已更正、目前背景程序已核對、後續回報義務已寫入版本化專案規則。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 問題不是研究授權說明不足，而是先前把「平台內容層 safeguard」與「本機 sandbox／程序執行層」混為一談，且沒有在回覆不可見後立即留下狀態交接。
* 後續一律分開報告三層狀態：平台內容提示、工具權限／sandbox、實際本機 run；只要 run 未結束就同時寫進 `PROGRESS.md`。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。此項為協作與可追溯性規則補強，不改變研究方法、D5 結論或八週工作內容。

### 7. 下一步
* 目前沒有需要等待或監看的程序；下一個實驗可從 D6 前置估時與小型 smoke test 開始，若預估 GPU 超過 30 分鐘則先請使用者裁決。

## 2026-10-13 · 環境維護 · sandbox helper 根因確認（待主機管理權限）

### 1. 當前目標 (Objective)
* 修復內建寫檔／`apply_patch` 反覆出現的 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，避免後續再以 sandbox 外的精確替換繞道。

### 2. 執行動作 (Actions Taken)
* 依 OpenAI Docs 的 Linux sandbox 指引檢查 OS、Codex、bubblewrap、user namespace、AppArmor、套件與核心 audit log。
* 以直接 `bwrap --unshare-user ... --unshare-net ... /bin/true` 做最小重現。
* 嘗試安裝官方建議的 `apparmor-profiles` 與 `apparmor-utils`；`sudo` 因需要互動式密碼而立即退出，沒有安裝或修改系統。
* 檢查背景程序：沒有 `apt-get`、`apparmor_parser`、pytest、managed run 或 D5 evidence 仍在執行。

### 3. 執行結果 (Results & Observations)
* [事實] 主機為 Ubuntu 24.04.4；Codex CLI 0.153.0；發行版 `bubblewrap 0.9.0` 已安裝。
* [事實] `kernel.apparmor_restrict_unprivileged_userns=1`，但 `apparmor-profiles`／`apparmor-utils` 與 `bwrap-userns-restrict` profile 均不存在。
* [事實] 核心 audit 明確拒絕 `bwrap` 的 `setpcap` 與 `net_admin`；最小重現得到完全相同的 `Failed RTM_NEWADDR`。當日紀錄共有 75 個 bwrap deny events、涉及 39 個 PID；一個 sandbox 呼叫可能衍生多個 PID／deny，不能解讀成 39 次寫檔，但證明問題廣泛且反覆發生。
* [根因] Ubuntu 24.04 AppArmor 的 unprivileged-userns 限制正在套用 generic `unprivileged_userns` profile，而主機缺少允許 bubblewrap 必要能力的專用 profile。

### 4. 達標判定 (Assessment)
* [ ] **尚未修復**：根因與官方修復路徑已確認，但系統套件／profile 需要使用者在主機終端輸入 sudo 密碼。
* [x] **沒有殘留程序或半套系統變更**：安裝在取得權限前即退出。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 採最小權限修復：保留 `kernel.apparmor_restrict_unprivileged_userns=1`，安裝並載入專用 bwrap AppArmor profile；不採官方列為 fallback 的全域關閉限制。
* 使用者需在自己的終端執行：
  1. `sudo apt-get install -y apparmor-profiles apparmor-utils`
  2. `sudo install -m 0644 /usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/bwrap-userns-restrict`
  3. `sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict`

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。此為實驗基礎環境維護，不改變研究方法或結果。

### 7. 下一步
* 使用者完成三條 sudo 指令後，重新執行 direct bwrap smoke、內建 `apply_patch` no-op／temporary-file smoke，並確認 kernel audit 沒有新增 bwrap DENIED；通過後再將本項標為修復完成。

## 2026-10-13 · 環境維護 · sandbox helper 修復複驗完成

### 1. 當前目標 (Objective)
* 複驗 Ubuntu 24.04 AppArmor 專用 bubblewrap profile 是否已修復 Codex 內建 sandbox helper 與 `apply_patch`。

### 2. 執行動作 (Actions Taken)
* 核對 `apparmor-profiles`、`apparmor-utils`、`bubblewrap` 與 `/etc/apparmor.d/bwrap-userns-restrict` 已安裝；確認全域 `kernel.apparmor_restrict_unprivileged_userns=1` 仍保持開啟。
* 在 sandbox 外以 `set -e` 執行 host-level bubblewrap user/network namespace smoke。
* 在正常 workspace-write sandbox 執行一般讀寫，並以內建 `apply_patch` 完成暫存檔建立、內容更新、讀回驗證與刪除。
* 檢查 kernel audit，區分修復前 generic `unprivileged_userns` 的 `setpcap`／`net_admin` 拒絕，與 sandbox 內再次啟動巢狀 bubblewrap 時 `unpriv_bwrap` 對 `sys_admin` 的預期拒絕。

### 3. 執行結果 (Results & Observations)
* [事實] host-level direct bubblewrap 成功，退出碼 0；沒有重現 `loopback: Failed RTM_NEWADDR`。
* [事實] 內建 `apply_patch` 的 Add／Update／Delete 三種操作全部成功；內容讀回一致，暫存檔已清除，Git tracked working tree 無測試殘留。
* [事實] 一般 sandbox `exec` 的 workspace 寫入與清理成功。
* [釐清] sandbox 內再啟動第二層 bubblewrap 會因 `unpriv_bwrap` 禁止 `sys_admin` 而失敗；這是避免巢狀 namespace 的限制，不是原始 helper 故障，頂層 Codex sandbox 與 `apply_patch` 均不受影響。

### 4. 達標判定 (Assessment)
* [x] **已修復 (Achieved)**：原始 RTM_NEWADDR 問題已由專用 AppArmor profile 修復，核心寫檔路徑與命令 sandbox 均通過。
* [x] 全域 unprivileged-userns 保護仍啟用，未使用降低全機安全性的 sysctl fallback。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 後續正常使用內建 `apply_patch`；不再預設使用 sandbox 外 Python 精確替換。
* 若再次失敗，先區分頂層 `profile="unprivileged_userns"` 的 `setpcap`／`net_admin`（回歸）與刻意巢狀測試的 `profile="unpriv_bwrap"`／`sys_admin`（預期限制）。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。此為實驗基礎環境修復，不改變研究方法或結果。

### 7. 下一步
* sandbox helper 無待辦或背景程序；可恢復 D6 前置估時與 smoke test。

## 2026-10-13 · W3 · D5 每週摘要與論文素材彙整

### 1. 當前目標 (Objective)
* 依 `PROGRESS_summary.md` 最新規範整理 Week 3 D5.1–D5.7，並建立所有數字可回溯的 `appendix/week3/` 素材。

### 2. 執行動作 (Actions Taken)
* 依 AUD-20261013-03 更新 D5 稽核結論、AF-024／AF-030 狀態及 AF-037／AF-038 待辦。
* 擴充 `scripts/build_weekly_appendix.py` 的 `WEEKS[3]`，產生 W3-T1 key pipeline、W3-T2 nonce registry、W3-T3 tamper matrix。
* 以 CFG_DDIM 連續重建 Week 3 appendix 兩次，逐檔比較 SHA-256。
* 在 `PROGRESS_summary.md` 新增固定八節 Week 3 摘要，並同步 `README.md` 的狀態與 appendix 索引。

### 3. 執行結果 (Results & Observations)
* [事實] 兩次重建的 `MANIFEST.csv` 與三份表格 SHA-256 全部相同；素材直接衍生自已追蹤的 D5 source／result／table。
* [事實] 摘要將 D5 正確標為 Achieved／`PASS WITH FINDINGS`：AF-024 CLOSED、AF-030 留至 D6、AF-037／AF-038 須在 D6 前修正，AF-025 留至 D7 裁決。
* [事實] 本摘要只涵蓋 D5；A4.6、協作規則與 sandbox 維護未混入 D5 研究成果。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：Week 3 D5 摘要、可追溯素材、雙重建確定性與檔案索引均完成。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* D5 的主要工程結論可直接用於論文 Method／完整性結果，但尚未與 SOT transform 整合；摘要明確避免把元件測試外推成 S2 端到端安全。
* D6 開始前先處理 AF-037／AF-038，避免錯誤 IV/counter test vector 與未凍結環境進入下一階段證據。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。Week 3 原排 D5；僅因整體進度提前於 2026-10-13 完成並彙整。

### 7. 下一步
* 請使用者檢視 `PROGRESS_summary.md` Week 3；確認後先修復 AF-037、AF-038，再開始 D6 smoke 與估時。

## 2026-10-13 · W3 · AF-037／AF-038 修復

### 1. 當前目標 (Objective)
* 在 D6 接入 D5 PRNG 前修復 AF-037 的 ChaCha20 counter overflow 可用性問題，並完成 AF-038 的 D5 套件鎖定、Week 3 環境快照與 run-manifest 套件紀錄。

### 2. 執行動作 (Actions Taken)
* [AF-037] `src/crypto/prng.py` 改為：per-image nonce 只作 HKDF salt；每個 domain-separated subkey 的 ChaCha20 IV 固定為 `counter=0 || 96-bit zero`，不再把隨機 image nonce 前四 bytes 當初始 counter。
* [AF-037] 更新 frozen sign／permutation vectors；新增 image nonce 前四 bytes 為 `0xFFFFFFFF` 時，仍可完成 65,536 維 sign 與 permutation／inverse 的 regression。
* [AF-038] `requirements-d5.txt` 精確鎖定 `cryptography==50.0.2`、`cffi==2.1.1`、`pycparser==3.11`；新增、不覆寫 Week 1 的 `artifacts/environment_freeze_w3.txt`。
* [AF-038] `scripts/run_artifacts.py::package_versions()` 新增 `cryptography`；unit test 驗證 manifest packages 欄，另新增 requirements 與 Week 3 freeze 一致性測試。
* 原子 source commit：`e22f01c`；其後以 clean tracked HEAD 重新執行驗收。

### 3. 執行結果 (Results & Observations)
* [事實] AF-037／AF-038 targeted tests：7 passed；極端 nonce regression 不再拋 counter overflow。
* [事實] 完整 CPU 回歸：81 passed、13 skipped；skip 為既有條件性測試，無 failure。
* [事實] 現行 CFG_DDIM `pip freeze --all` 與 `environment_freeze_w3.txt` 逐行相同；requirements SHA-256=`6db863da…`、freeze SHA-256=`63c45e4e…`。
* [事實] `package_versions()` 回報 `cryptography=50.0.2`，因此後續 run manifest 會記錄 provider 版本。
* [事實] 測試前後 tracked working tree 均為 clean；全程 CPU，沒有 GPU 任務或背景程序。

### 4. 達標判定 (Assessment)
* [x] **AF-037 已修復待稽核複驗**：fixed counter、docstring、極端 nonce regression 與 frozen vectors 均完成。
* [x] **AF-038 已修復待稽核複驗**：三個版本鎖定、獨立 Week 3 freeze、manifest package 欄與 regression 均完成。
* [ ] finding 的正式 `CLOSED` 狀態仍由稽核方裁定；未擅自修改 `AUDIT.md`。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 安全理由：不同 image nonce／round／purpose 已透過 HKDF 產生不同 stream key，因此同一固定 ChaCha20 IV 不造成 keystream reuse；反而可保證每條 stream 從 counter 0 開始並保留完整 counter 空間。
* D6 必須直接使用此 API，不得自行重新拼 ChaCha20 IV 或把 image nonce 當 raw cipher IV；AF-030 的剩餘驗收仍在 D6。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。兩項都是 D5 稽核後、D6 整合前的必要修正；沒有改變 S2 威脅模型或實驗樣本數。

### 7. 下一步
* 請稽核方依 AUD-20261013-03 驗收 AF-037、AF-038；複驗通過後進入 D6 smoke。任何預估超過 30 分鐘的 GPU 工作仍先請使用者裁決。

## 2026-10-14 · W4 · Track D · D6.1–D6.7 structured transform 與 S2 dev 整合

### 1. 當前目標 (Objective)
* D5 稽核 `PASS`、AF-037／AF-038 `CLOSED` 後，完成 `WORKFLOW.md` D6.1–D6.7：torch FWHT、顯式正交性、Gaussian／真實 latent 分布、R/P 消融、結構診斷及 authenticated S2 dev pipeline。
* 直接使用 D5 HKDF／ChaCha20 sign/permutation、nonce registry、container 與 HMAC；補齊 AF-030 剩餘的 S2 整合證據。

### 2. 執行動作 (Actions Taken)
* 實作 `src/crypto/sot.py`：正規化 FWHT、`D_post·P·H·D_pre`、多輪 `Q_R…Q_1` 與逆向輪序；單張 latent flatten 後保持 shape/dtype/device，另提供 explicit-Q 診斷批次 API。
* 實作 `src/crypto/s2.py`：M2 latent 與 M1 range-preserving PNG authenticated container；open 路徑固定為 tag 驗證成功後才 decode/inversion/inverse；seal 強制使用持久化 `NonceRegistry`，M1/M2 跨模式 nonce reuse 亦拒絕。
* 新增 `tests/security/test_sot.py`、`test_s2.py`：d=65,536 R=1–4/P on-off、d=1,024 顯式 Q、wrong key/nonce、CUDA、M1/M2 ordering 與 nonce reuse regression。
* `scripts/d6_sot_evidence.py` 由 dev E2.3 cache 產生 D6.1–D6.6 CPU 證據；公開 deterministic fixture 只用於可重現實驗，不是部署 secret，結果明列 p-value／結構指標不是安全證明。
* `scripts/d6_s2_dev_runner.py` 經 managed runner 完成 20 張 × 8 組 generation-quality 與固定 4 張 M1/M2 integration；正式 run 分別為 `D6.4_QUALITY_20261013T162058881536Z_c2c4465a`、`D6.7_INTEGRATION_20261013T164724711647Z_ecec4bf5`，皆 `succeeded` 且 artifact validation 通過。
* 原子 commits：`9577ed5`（transform/S2 core）、`b442b5a`（CPU evidence）、`abba1bc`（GPU runner）、`4c99e2f`（nonce registry enforcement）、`173226e`（dev 選型與索引）。

### 3. 執行結果 (Results & Observations)
* [事實] D6.1：d=65,536 float64 inverse MaxAbs 最差 `2.66e-15`（門檻 `1e-12`）；float32 bit-exact rate `8.60%–17.22%`，只記錄、不作 fail gate。
* [事實] D6.2：d=1,024 顯式 Q 的 orthogonality MaxAbs=`0`、spectral norm error=`8.88e-16`、`Q_2Q_1` formula MaxAbs=`0`，通過。
* [事實] D6.3：Gaussian 變換後 mean=`-0.00202`、variance=`0.99799`、skew=`-0.00201`、excess kurtosis=`-0.00115`、KS p=`0.121`；只支持與 N(0,1) 相容的診斷，不能推出安全性。
* [事實] D6.4：真實 latent 經變換後每張 mean/variance 確有位移（八組 mean absolute shift 約 `0.139–0.141`，variance absolute shift 約 `0.0197`），但 norm ratio 最大偏差僅 `5.96e-7`。八組匿名圖對 x0 的平均 PSNR=`5.21–5.35 dB`、SSIM=`0.0076–0.0128`，沒有設定顯著改善匿名圖的醫療可讀性。
* [事實] D6.5：20 張 dev 中，R=1/P off 的 original-|·| 與 sorted-|·| Top-1 都是 `0.05`（1/20 random baseline）；八組 norm-only Top-1 全為 `1.0`。R=1 CPU transform median 為 P off `1.83 ms`、P on `62.6 ms`；P 未改善這兩種攻擊。
* [事實] D6.6：八組 trace／trace-over-d、same-vector cosine expectation、basis nonzero fraction、matrix zero fraction 與正交誤差已完整記錄；它們是 probe，不是 security proof。
* [事實] D6.7：R=1/P off 的 4 張 authenticated dev，M2 decrypted latent MaxAbs 平均 `7.75e-7`；M1 decrypted latent cosine 平均 `0.999811`；M1 端到端 PSNR 平均 `39.37 dB`、SSIM 平均 `0.9720`。最終 GPU 全套回歸 `115 passed`，包含 S0 legacy regression 與 CUDA FWHT。
* [決定] D7 暫用最小的 `R=1、permutation off`；若 D7 adaptive／structured probe 顯示額外弱點則重開選型。這項決定不處理 AF-025：norm-only 仍 100%，須依 D7 實測再裁決保護強度。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：D6.1–D6.7 的 correctness、distribution、ablation、structural diagnostics 與 authenticated dev integration 均完成；G3 可送稽核。
* [x] **AF-030 實作方修復完成待複驗**：S2 直接使用完整 HKDF/ChaCha20 material，wrong-key/wrong-nonce control、per-image nonce 與 persistent reuse rejection 均有測試。
* [ ] 稽核狀態仍由稽核方裁定；未修改 `AUDIT.md`，AF-025 依既定裁決留到 D7。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 一輪 WHT 已足以破壞前作的逐座標與排序幅值等式；增加 R 或 P 對這兩個 dev attack 沒有可見收益，反而增加 runtime 與 float32 累積誤差，因此依預先規則選最小 R。
* 真實 latent 不符合精確 N(0,I)，所以正交變換雖保 norm，仍會改變座標動差；匿名輸出依然不是可診斷胸片。這是 D6 實測 trade-off，不把 Gaussian 理論外推到真實資料。
* M2 誤差來自 float32 FWHT；M1 的主要額外損失仍是 image inversion／量化路徑。tag 不改匿名像素，只提供完整性與 verify-before-use。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。D6.4 generation-quality 用固定 dev N=20，D6.7 用預先固定 4 張 integration samples；正式 security N 維持 200 並屬 D7。最長 GPU run 實際 23 分 40 秒，低於需使用者另行裁決的 30 分鐘門檻。

### 7. 下一步
* 請稽核方複驗 D6.1–D6.7、G3 與 AF-030；若無 blocking finding，進入 D7，以相同 scorer 對 R=1/P off 重跑 T1/T2-WB、adaptive norm、cross-nonce KPA/CPA、geometry、structured probe、repeat 與 tamper。
* D7 完成前不裁決 AF-025，也不宣稱 S2 達到 IND-CPA/AES 等級；目前沒有背景程序需要等待。

## 2026-10-14 · W4 · Track D · P／S0／S1／S2a／S2 單圖比較

### 1. 當前目標 (Objective)
* 依使用者要求，以固定一張影像產生五種方法的原圖／匿名圖／重構圖，並並列重構圖對原圖的 PSNR、SSIM、L∞、最大 pixel error 與 cosine。
* 指定 storage：S0/S1 使用前作 legacy PNG（丟棄值域），S2a/S2 使用 range-preserving PNG；P 採 legacy PNG 作與學長流程一致的無加密基準。

### 2. 執行動作 (Actions Taken)
* 固定使用 dev split 第一張積水病例 `dev_v1.1_010`，未依輸出品質挑圖；從 E2.3 的既有 latent cache 重新執行五條 GPU pipeline。
* P/S0/S1：latent transform → generation → legacy min-max/uint8 PNG（值域丟棄）→ target-model inversion → inverse transform → generation。
* S2a/S2：共用 D6 選定的 R=1/P off、相同 nonce 與 range-preserving PNG；S2 另走 nonce registry、HMAC container 與 verify-before-inversion。程式強制檢查未竄改時 S2a/S2 stored anonymous 與 reconstruction 均逐位元相同。
* 新增 `scripts/method_comparison_runner.py`；managed run `D6.COMPARE_20261013T171753598997Z_69d08919` 成功，5/5 rows 通過 artifact validator。
* 輸出 `image/method_comparison.png`、`paper_assets/tables/table_method_comparison.csv`、`results/method_comparison.json` 與 `reports/method_comparison.md`。

### 3. 執行結果 (Results & Observations)
* [事實] P：PSNR=`38.291 dB`、SSIM=`0.9713`、L∞=`0.1026`、uint8 最大誤差=`26`、x0 raw-pixel cosine=`0.999796`。
* [事實] S0：PSNR=`32.558 dB`、SSIM=`0.9428`、L∞=`0.6144`、uint8 最大誤差=`157`、x0 cosine=`0.999182`。
* [事實] S1：PSNR=`31.946 dB`、SSIM=`0.8865`、L∞=`0.4793`、uint8 最大誤差=`122`、x0 cosine=`0.999044`。
* [事實] S2a/S2 均為：PSNR=`38.345 dB`、SSIM=`0.9684`、L∞=`0.4269`、uint8 最大誤差=`109`、x0 cosine=`0.999799`；兩者影像逐位元相同，S2 的差異只在認證與拒絕竄改。
* [事實] 學長 Signed Permutation 原表為 cosine=`0.998978`、PSNR=`30.23 dB`；但其程式比較 Base `samples` 與加解密 `samples`，不是 x0 對 xrec。新表另列 x0→重構與 P-output→重構 cosine，未把不同 estimand 混為一談。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：五方法比較圖、數值表、provenance 與 cosine 語意註解均完成；S2 authentication 路徑實際執行而非以 S2a 複製標籤。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* S0/S1 相對 S2a/S2 的差距同時包含 transform 與 storage protocol 差異；這是使用者指定的完整 pipeline 比較，不可當成只隔離加密方法的因果消融。若論文要做公平歸因，仍應引用既有 2×2 storage comparison。
* 五種方法的 x0 cosine 全在 `0.9990–0.9998`，但 PSNR 橫跨 `31.95–38.35 dB`、L∞ 橫跨 `0.103–0.614`，再次顯示 raw-pixel cosine 在高相似區飽和，不能取代 PSNR／SSIM／局部最大誤差。
* S2a/S2 的平均誤差很小但 L∞ 仍可達 `0.4269`；高 PSNR 不代表每個 pixel 都接近，圖表應保留 L∞ 與差異圖語意。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無研究設計衝突；這是 D6 完成後的額外展示素材。P 的 storage 未由使用者明示，因此採 legacy PNG 作學長流程的無加密基準，並在報告／表格明列，沒有隱藏此假設。

### 7. 下一步
* 將此圖表交付使用者並送稽核方檢查圖說與 cosine estimand；之後依原計畫進入 D7。現在沒有背景程序需要等待。

## 2026-10-14 · W4 · Track D · D6 每週摘要與可追溯附錄

### 1. 當前目標 (Objective)
* 依 `PROGRESS_summary.md` 的八節格式完成 Week 4（D6）摘要；所有量化宣稱須引用由既有結果檔衍生的 Week 4 appendix，並納入最新 D6 稽核結論。

### 2. 執行動作 (Actions Taken)
* 擴充 `scripts/build_weekly_appendix.py` 的 `WEEKS[4]`，加入 D6 correctness、Gaussian／真實 latent、R/P 消融、structure、generation quality、S2 integration 與五方法比較素材。
* 建立 `appendix/week4/`：3 張圖、8 張表及含來源 SHA-256／run ID 的 `MANIFEST.csv`。
* 連續重建 Week 4 appendix 兩次並比較所有檔案 SHA-256，結果逐位元相同。
* 在 `PROGRESS_summary.md` 新增 Week 4 固定八節摘要；不改寫 Week 3，改以「更正」記錄 AF-037／AF-038 已關閉及 D5 最終 PASS。
* 更新 `README.md` 的 appendix inventory。

### 3. 執行結果 (Results & Observations)
* [事實] Week 4 摘要的數字均引用 W4-F1–F3 或 W4-T1–T8；生成素材可回溯到 D6 原始 JSON／CSV、正式 managed run 與五方法比較結果。
* [事實] 最新 AUD-20261014-01 判定 D6.1–D6.7 PASS、G3 PASS、AF-030 CLOSED；摘要未沿用 D6 完成當下「待稽核」的舊狀態。
* [事實] 附錄兩次生成的 SHA-256 清單無差異；本工作只做 CPU 文件／表格整理，未使用 GPU，沒有背景程序。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：Week 4 摘要、可追溯 appendix、重建驗證與檔案索引均完成。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* D6 dev 結果與正式 D7 結論明確分層：R=1/P off 只標為候選，norm leakage 與 AF-025 不在 Week 4 提前裁決。
* 五方法單病例展示明列 storage protocol 不同，避免把 transform 與 storage 的共同差異誤寫成單一演算法效果。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。Week 4 摘要忠實整理已完成並通過稽核的 D6；正式 N=200 仍保留給 D7。

### 7. 下一步
* 依 Week 5 計畫進入 D7 formal security evaluation；AF-025 待跨存檔結果完成後由使用者裁決。

## 2026-10-14 · W5 · Track D · D7 CPU formal evidence 與 T2-WB runner 準備

### 1. 當前目標 (Objective)
* 開始 D7.1–D7.8：先完成不需 diffusion GPU 的 formal N=200 T1／cross-nonce／geometry／probe／repeat／tamper，再建立可驗證的 T2-WB dev/formal runner 與 analyzer。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/d7_metrics.py`，固定 float64 feature、悲觀 tie-break、Wilson CI 與 bootstrap mAP，沿用 A3/A4 修正後口徑。
* 新增 `scripts/d7_cpu_security.py`，由 frozen `security_v1` E2.3 cache 產生 D7.2–D7.8 CPU evidence；使用 D6 暫選 R=1/P off 與 per-image nonce。
* D7.4 同時建立 d=1,024 chosen-basis CPA 與 full-rank known-pair KPA 正控制，分別測 same nonce 與 cross nonce transfer。
* D7.5 對 fixed-Q 與 per-image-Q 分開測 pairwise cosine／distance 及 sorted distance-signature linkage；D7.6 使用 basis、constant 與 sparse probe；D7.7 同一真實 latent 以 100 個 nonce 重複變換；D7.8 對 S2 七種容器竄改與 S2a 無認證正控制並列。
* 新增 `scripts/d7_t2wb_inversion.py`：每張 S2 匿名圖只生成一次，再經 legacy／range-preserving 兩種存檔反演；新增 `scripts/d7_t2wb_analyze.py`：合併 T1/T2-WB、dev norm calibration 與正負控制。
* 新增 `tests/unit/test_d7_security.py`；targeted 回歸 23 passed、1 skipped。
* 使用者於本輪明確預先同意 D7 階段超過 30 分鐘的 GPU 任務，正式 run 不需再次詢問，但仍須回報與監看。

### 3. 執行結果 (Results & Observations)
* [事實] D7.2 T1 N=200：S2 original-|z| Top-1=0.010、sorted-|z| Top-1=0.005，random baseline=0.005；結果完整保留，不以是否顯著作工程 gate。
* [事實] D7.3 T1 norm-only Top-1=1.000（Wilson 95% CI 下界 0.981），確認正交變換的已知限制。
* [事實] D7.4 d=1,024：same-nonce CPA/KPA 正控制 cosine=1.0、MaxAbs ≤7.55e-15；轉移到另一 nonce 後 RMSE=1.412、cosine≈0.00105。
* [事實] D7.5 fixed-Q distance-signature Top-1=1.0；per-image-Q Top-1=0.005。fixed-Q pairwise cosine MaxAbs error=2.39e-9，per-image-Q=0.145。
* [事實] D7.6 四種非 Gaussian probe 皆有完整輸出統計且 norm ratio=1.0；只作診斷、不作安全證明。
* [事實] D7.7 100/100 nonce 與 protected payload hash 均唯一；off-diagonal cosine mean≈3.70e-5、max≈0.0132。
* [事實] D7.8 S2 tag／nonce／ciphertext／version／parameters／AAD／storage 七種竄改全部拒絕；S2a latent 改值不拒絕且輸出改變。
* [事實] CPU evidence 完成時沒有背景程序；T2-WB GPU runner 尚未啟動。

### 4. 達標判定 (Assessment)
* [x] D7.2/D7.3 的 T1 部分、D7.4–D7.8 CPU evidence 工程與資料完整性已通過實作方檢查。
* [ ] D7.1 T2-WB 尚待 dev smoke/calibration 與 formal N=200；其完成後才合併 D7.2/D7.3 的 T2-WB 結果。
* [ ] G4 與 AF-025 尚未裁決，須等 T2-WB 正式結果與稽核。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* cross-nonce 實驗以能成功重建同一 Q 的 full-basis/full-rank 上界攻擊作正控制；跨 nonce 失敗只能支持「此映射不能直接轉移」，不代表一般 KPA/CPA 安全證明。
* S2a/S2 的匿名像素相同，authentication 不改 confidentiality；T2-WB runner 因此共用 query，但 D7.8 另行量測兩者 tamper 差異。
* T2-WB 必須同時報 legacy 與 range-preserving；前者是 AF-025 的不保存值域反事實對照，後者是 S2 M1 正式協定。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。formal N 維持 200；D7.4 的顯式映射重建因矩陣大小限制使用 d=1,024，並明列為 synthetic diagnostic，不冒充 d=65,536 端到端結果。

### 7. 下一步
* 先提交 source＋CPU evidence 的 clean HEAD，執行 D7.1 dev smoke/calibration並驗證 artifact schema；通過後依使用者預先授權啟動約 80 分鐘的 formal N=200 managed GPU run。
* formal 完成後執行 `d7_t2wb_analyze.py`，更新兩張主表、整理 P7.1 defense package、更新 README／PROGRESS，再送稽核。

## 2026-10-14 · W5 · Track D · D7.1 smoke 與 dev evidence-role 修正

### 1. 當前目標 (Objective)
* 以 managed GPU smoke 驗證 D7.1 雙存檔 inversion runner；再建立 dev-only norm calibration artifact，正式 N=200 前先通過 schema 與角色檢查。

### 2. 執行動作 (Actions Taken)
* sandbox 內第一次 smoke 在載入模型前因 CUDA 不可見而 fail，隨即在 GPU 可見環境以完全相同命令重跑；沒有重用失敗 artifact。
* `D7.1_SMOKE_20261013T183021510503Z_77120b9f` 完成 2 張 × 2 storages，artifact validator valid（4 rows）。
* `D7.1_DEV_20261013T183135566676Z_1b415cd1` 完成 20 張 × 2 storages，40 筆 query 均有限且 hash 驗證通過；但檢查 manifest 發現 `formal=true`。
* 根因為 runner 原本以 `--limit == 0` 判 formal，錯把「完整跑 dev split」視為 formal。已改為同時要求 `security_v1.csv`、N=200、無 limit，並新增 `evidence_role=formal/dev_calibration/smoke`；analyzer fail closed 檢查兩個角色。

### 3. 執行結果 (Results & Observations)
* [事實] smoke 每張約 23.5 秒；legacy attacker latent cosine 約 0.990，range-preserving 約 0.9998，與既有 A4 量級一致。
* [事實] 第一個 dev run 的數值與 query cache 正確，但角色 metadata 錯誤，因此不作正式 calibration 輸入、不修改後重用。
* [事實] 目前沒有背景程序；formal N=200 尚未開始。

### 4. 達標判定 (Assessment)
* [x] GPU smoke 與 artifact validator 通過。
* [ ] dev calibration 須以修正版 clean HEAD 重跑，確認 `formal=false`、`evidence_role=dev_calibration` 後才能進入 formal。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* `--limit` 是取樣控制，不是 evidence role；未來角色由 split contract 與明確 manifest 欄共同決定，analyzer 不再接受推測或錯標 artifact。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。錯標 artifact 已在 formal 前攔截，未進入統計或論文表格。

### 7. 下一步
* 提交 role 修正，重跑 dev calibration；通過後啟動預先授權的 formal N=200 managed run。

## 2026-10-14 · W5 · Track D · D7.1–D7.8 formal 完成與 P7.1 defense package

### 1. 當前目標 (Objective)
* 以 frozen `security_v1` N=200 完成 S2a/S2 的 T1/T2-WB、adaptive norm、cross-nonce KPA/CPA、geometry、structured probe、repeat 與 tamper，並建立 claim→evidence 可追溯的 defense package。

### 2. 執行動作 (Actions Taken)
* 以修正版 clean commit `f365944` 重跑 dev calibration：`D7.1_DEV_20261014T020638335355Z_9223ea81`，20 張 × 2 storage、40 rows，`formal=false`、`evidence_role=dev_calibration`、artifact validator valid。
* 依使用者預先授權執行 formal GPU managed run `D7.1_T2WB_20261014T021454463115Z_7b959db9`：200 張各生成一次匿名影像，再做 legacy／range-preserving inversion，共 400 rows；實際 77 分 06 秒，無失敗或跳過。
* formal query cache 為 `(400,1,256,256)`、全部有限；逐筆 query SHA-256 重讀一致；validator valid，manifest 為 `formal=true`、`evidence_role=formal`、`git_dirty=false`。
* 執行 `scripts/d7_t2wb_analyze.py`，以原先凍結的 abs／abs_unit／sorted_abs／sorted_abs_unit／norm scorer、悲觀 tie-break、dev-only norm bias 與 seed-1911 shuffled negative control，合併 T1/T2-WB 主表。
* 新增 `scripts/build_defense_package.py`，由 D7 表格產生 9 條 claim 的 source SHA-256 索引、`image/D7_attack_summary.png` 與 `paper_assets/defense_package/index.md`。
* defense package 連續重建並比較三個輸出 SHA-256，結果完全一致；正式測試範圍 `pytest -q tests` 為 105 passed、14 skipped。

### 3. 執行結果 (Results & Observations)
* [事實] D7.1 formal inversion：legacy latent cosine mean=`0.990134`（95% CI `[0.989985,0.990285]`）、range-preserving=`0.999798`（`[0.999794,0.999801]`）；兩者正確解密後的 cosine-gallery positive-control Top-1 均為 `1.0`，所以 T2-WB 結果可解釋。
* [事實] D7.2：T1 original-|z| Top-1=`0.010`、sorted-|z|=`0.005`；T2-WB 的 original-|z| 在 legacy/range 都為 `0.010`，其餘 magnitude scorer 多為 `0.005`，random baseline=`0.005`。指定 magnitude leakage 相較 S0/S1 已大幅下降。
* [事實] D7.3：精確 T1 norm-only Top-1=`1.000`（Wilson 95% CI 下界 `0.981`）。T2-WB raw norm 兩種 storage 都是 `0.005`；套用只由 dev 凍結的 median bias 後，legacy 仍為 `0.005`，range-preserving 升至 `0.250`（Top-5=`0.745`、median rank=`3`）。
* [事實] dev norm bias 為 legacy=`1.012647`、range=`0.996747`；formal 沒有用於校正參數。這表示保存值域會讓 inversion norm 更穩定，AF-025 的 storage-dependent leakage 在 S2 上得到正式驗證。
* [事實] D7.4：d=1,024 full-basis CPA 與 full-rank KPA 的 same-nonce 正控制 cosine=`1.0`、MaxAbs≤`7.55e-15`；相同映射跨 nonce 套用後 RMSE=`1.412`、cosine≈`0.00105`。
* [事實] D7.5：fixed-Q 完整保留 pairwise geometry，distance-signature Top-1=`1.0`；per-image-Q Top-1=`0.005`、pairwise cosine MaxAbs difference=`0.145`。
* [事實] D7.6：basis／constant／sparse 非 Gaussian probe 均完整記錄且 norm ratio=`1.0`；只作結構診斷，不作通用安全證明。
* [事實] D7.7：同一真實 latent 以 100 個不同 nonce 加密，nonce 與 protected payload hash 皆 `100/100` 唯一；不同密文 cosine mean≈`3.70e-5`、max≈`0.0132`。
* [事實] D7.8：S2 的 tag／nonce／ciphertext／version／parameters／AAD／storage 七種竄改全部拒絕；S2a 無認證正控制接受竄改且還原輸出改變。
* [事實] P7.1 defense package 共 9 條 claim，每條記錄來源檔與 sha12；GPU 已釋放，目前沒有背景程序。
* [事實] 從 repository root 直接執行未限範圍的 `pytest -q` 會誤收集封存於 `past/SourceCode/` 的學長測試，因其依賴封存樹中不存在的 `guided_diffusion.anonymization_enhanced` 而在 collection 失敗；本專案正式測試目錄 `tests/` 全數通過，這不是 D7 regression。

### 4. 達標判定 (Assessment)
* [x] **D7.1–D7.8 實作方 Achieved**：correctness、positive/negative controls、fixed split、formal per-sample rows、summary、run ID、hash 與 failure accounting 齊全。
* [x] **P7.1 Achieved**：方法圖索引、attack table、norm limitation、tamper matrix 與 claim→evidence package 完成。
* [x] **G4 實作方條件完成，待稽核判定**：S2 T1/T2-WB、norm、geometry、cross-nonce、structured probe、repeat 與 tamper 均已重跑。
* [ ] **AF-025 仍待使用者裁決**：正式數據已足以比較保存／不保存值域，但是否接受 25% norm linkage、加強保護或改部署模式屬設計決策，未擅自關閉。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* SOT 已破壞前作逐座標與排序幅值等式；per-image nonce 也阻止 fixed-Q mapping／geometry 直接跨影像轉移。這些結論只限本次指定攻擊。
* 正交性必然保存 L2 norm。legacy 丟棄值域造成尺度誤差，讓 norm linkage 表面失效；range-preserving 保住重構品質，也讓 dev-calibrated norm attack 在 formal 上達 25%。因此不能把 legacy 的較低攻擊率當成純密碼強度，而是隱私／可逆性的 storage trade-off。
* S2 的 HMAC 解決完整性，不改 confidentiality；S2a/S2 在未竄改時共用相同匿名像素與 linkage 結果。
* 若使用者要求降低 AF-025 風險，下一步應比較：接受並揭露 norm limitation、對 norm/值域 metadata 加入額外 keyed concealment、或改用不需匿名影像可計算性的標準 AEAD/M2 storage；不可只增加 WHT 輪數，因其仍保 norm。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。formal N=200、T1/T2-WB 分開、兩種 storage 並列、positive/negative controls 與科學結果非 pass gate 均符合規格。D7.4 顯式 full-map 重建使用 d=1,024 synthetic diagnostic，已在素材包明列限制。

### 7. 下一步
* 請稽核方複驗 D7.1–D7.8、P7.1 與 G4；尤其獨立重算 range-preserving dev-calibrated norm Top-1=`0.25` 與 cross-nonce 正控制。
* 請使用者依 AF-025 正式結果裁決是否接受 norm leakage，或在 Week 6/7 加入 keyed norm concealment／其他部署對照；裁決前不更動 S2 protocol。
* 稽核通過且 AF-025 裁決後，進入 Q8 品質、wrong-key、runtime 與 storage pilot。

## 2026-10-15 · W5 補驗 · AF-025 固定 norm S2 v2 實作與 CPU formal

### 1. 當前目標 (Objective)
* 落實 AUD-20261014-04 的使用者正式裁決：S2 transform 前固定 norm，真實 norm 與 M1 lo/hi 改為 private AEAD metadata；完成 formal N=200 的 T1、structured probe、repeat，並準備 T2-WB GPU 補驗。

### 2. 執行動作 (Actions Taken)
* 新增 `src/crypto/s2_fixed_norm.py`：固定公開 `C=248.261`（float64 bits 凍結）、原 norm 恢復、M1/M2 wrapper，以及獨立 HKDF domain 的 ChaCha20-Poly1305 private metadata；外層 HMAC 仍先於 metadata decrypt／decode／inversion／inverse。
* 擴充 `scripts/m1_storage.py` 與 `src/crypto/container.py`：`range_preserving_png_private_range/v2` 沿用相同量化，但 public PNG/text/storage record 不含 lo/hi；合法端只從 AEAD plaintext 取得值域。
* 新增 `tests/security/test_s2_fixed_norm.py`，涵蓋 M2 固定 norm 與往返、M1 lo/hi 不公開、合法原 norm 恢復，以及 private metadata tamper 在 inverter 前拒絕。
* 新增 `scripts/af025_fixed_norm_cpu.py`、`af025_fixed_norm_t2wb.py`、`af025_fixed_norm_analyze.py`；CPU runner 已對 frozen `security_v1` N=200 執行，GPU runner 尚未啟動。
* 使用者同意在確認與既有 vLLM 可共存後執行正式 GPU 補驗。檢查 RTX 5090：總 32,607 MiB、vLLM 14,342 MiB、RustDesk 537 MiB、當時 free 17,201 MiB、utilization 0%；先以 N=2 smoke 實測再決定啟動 formal。
* 第一次 N=2 smoke `AF025_FIXED_NORM_SMOKE_20261014T160133624188Z_f2ea04c4` 完成兩張科學計算但 managed artifact validation fail：manifest 遺漏必填的 `container_version`／`nonce_mode`。已補欄；該 run 保留為 failed，不修改後重用。
* 修正版 N=2 smoke `AF025_FIXED_NORM_SMOKE_20261014T160317566922Z_8e4ca5a2` succeeded／valid；dev calibration `AF025_FIXED_NORM_DEV_20261014T160426684716Z_911608a9` 完成 20/20，`formal=false`、`evidence_role=dev_calibration`、clean commit `202c7fc`。
* formal run `AF025_FIXED_NORM_FORMAL_20261014T161327001754Z_c4eda689` 完成 security_v1 200/200；每張只生成一次 anonymous image，再分別做 attacker public-PNG inversion 與 legitimate private-range inversion。run succeeded、artifact validator valid、query `(200,1,256,256)` 全部有限且 hash 重讀一致。
* 執行 `scripts/af025_fixed_norm_analyze.py`：dev 只凍結 median attacker-norm/C=`1.013018`；formal 同時報 raw 與 dev-calibrated norm linkage，未用 formal 標籤調參。

### 3. 執行結果 (Results & Observations)
* [事實] AF-025 CPU formal N=200：固定 norm T1 norm-only Top-1=`0.005`、Top-5=`0.025`，與 random=`0.005` 一致；原 S2 v1 的 T1 Top-1 為 `1.0`。
* [事實] float32 protected norm mean=`248.2610001`，相對 C 最大絕對偏差=`1.49e-5`；四種 float64 structured probes 的 protected norm 與 C 最大差約 `1.93e-12`。
* [事實] 同一 normalized latent 以 100 個 nonce 變換，nonce hash 與 payload hash 均 `100/100` 唯一；off-diagonal cosine mean≈`1.64e-5`、max≈`0.0152`。
* [事實] 全套正式測試範圍 `pytest -q tests` 為 108 passed、14 skipped；新增／相依 crypto focused tests 27 passed。
* [事實] 目前沒有仍在執行的 AF-025 GPU managed run 或其他本專案背景程序；vLLM 為使用者既有程序，未停止或修改。
* [事實] 第一次 smoke 執行中 DDIM Python 約占 1,712 MiB；含 vLLM/RustDesk 的 GPU 總使用 16,626 MiB，仍有 15,484 MiB free，未發生 OOM。兩張約 50 秒；正式 N=200 因每張一次生成＋攻擊者/合法端各一次 inversion，預估約 80–90 分鐘。
* [事實] formal 實際耗時 77 分 10 秒，無 OOM／失敗／跳過；vLLM 未停止或修改。GPU 已釋放，目前沒有本專案背景程序。
* [事實] AF-025 formal N=200：T2-WB raw norm-only Top-1=`0.005`、Top-5=`0.025`；dev-calibrated Top-1=`0.005`、Top-5=`0.035`；負控制與 random Top-1 均=`0.005`。原 S2 v1 的 range-preserving dev-calibrated Top-1=`0.250`。
* [事實] legitimate private-range inversion 的 latent cosine mean=`0.999797`（95% bootstrap CI `[0.999794,0.999800]`）、min=`0.999729`；public PNG／storage record 200/200 均無 lo/hi。
* [事實] 平均 runtime／張：anonymous generation=`7.744 s`、attacker inversion=`7.679 s`、legitimate inversion=`7.679 s`；三者合計約 `23.103 s`，不含一次性載入與 artifact validation。
* [事實] defense package 已擴充為 10 條 claim，新增 D7-C10（fixed norm T1/T2-WB 均 `0.005`）；連續重建的 claims/index/figure SHA-256 完全一致。最終 `pytest -q tests` 為 108 passed、14 skipped。

### 4. 達標判定 (Assessment)
* [x] AF-025 (a)(b)(e) 實作與 CPU formal T1／D7.6／D7.7 已完成實作方驗證。
* [x] AF-025 (c)(d) formal N=200 T2-WB、repeat、structured probe 與 legitimate positive control 已完成實作方驗證。
* [x] **AF-025 實作方 FIXED?**：裁決 (a)–(e) 均已落地，指定 norm attack 由 v1 的 T1=`1.0`／T2-WB=`0.25` 降至 v2 的 `0.005`／`0.005`；finding 仍須稽核方複驗後才 CLOSED。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 固定 norm 不是增加 WHT 輪數，而是在正交核心前移除已確認的病人間 norm 指紋；真實尺度只存在於經 AEAD 加密的 recovery metadata。
* public PNG 刻意只提供 q/255；合法端經驗證後用 private lo/hi 恢復實際值域。T2-WB 必須分開量測攻擊者 public view 與合法 private-range positive control，避免把攻擊失敗誤判成整體 pipeline 無法反演。
* vLLM 預配置約 14.3 GiB，剩餘約 17.2 GiB；仍須用 smoke 實測 DDIM 共存峰值。vLLM 有請求時可能互搶算力並增加延遲，即使不 OOM。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。依使用者最新裁決新增 §2.5 並更新 S2 v2 定義；formal sample size 維持 N=200，舊 S2 v1 結果保留作修補前基準而不覆寫。

### 7. 下一步
* 請稽核方獨立複算 fixed-norm T1/T2-WB norm Top-k、確認 public PNG/record 不含 lo/hi、private metadata tamper ordering，以及 repeat／probe 結果；複驗後裁決 AF-025 是否 CLOSED。
* AF-025 關閉後進入 Q8；影像端合法重構品質沿用稽核 phase-B n=40 作診斷，Q8 formal 另以 PSNR／SSIM／L∞ 完成主表，不以 latent cosine 取代影像品質。

## 2026-10-15 · W5 · D7／P7／AF-025 每週摘要與可重建 appendix

### 1. 當前目標 (Objective)
* 依 `PROGRESS_summary.md` 規範，彙整 Week 5 的 D7.1–D7.8、P7.1 與 AF-025 裁決／補驗；所有量化宣稱須回指同週可重建素材。

### 2. 執行動作 (Actions Taken)
* 擴充 `scripts/build_weekly_appendix.py` 的 `WEEKS[5]`，由已提交的 D7 與 AF-025 正式結果建立 2 張圖、13 張表與來源／run ID／SHA-256 manifest。
* 新增 v1 對 fixed-norm v2 的 norm-only Top-1 對照圖；同時保留 v1 攻擊成功與 v2 修補結果，未覆寫負結果。
* 依固定八節格式在 `PROGRESS_summary.md` 追加 Week 5，並更新根目錄 `README.md` 的 appendix inventory 與 AF-025 狀態。
* 連續執行兩次 Week 5 builder，逐檔比較 16 個輸出雜湊。

### 3. 執行結果 (Results & Observations)
* [事實] `appendix/week5/` 已包含 `W5-F1`～`W5-F2`、`W5-T1`～`W5-T13` 與 `MANIFEST.csv`；第二次重建與第一次的 SHA-256 清單無差異。
* [事實] 摘要清楚區分：D7／G4 已由 AUD-20261014-02 判定 PASS；AF-025 fixed-norm v2 為實作方 `FIXED?`，仍待稽核方複驗才可 CLOSED。
* [事實] 本輪未啟動 GPU 或背景任務，目前沒有需要監控的本專案程序。

### 4. 達標判定 (Assessment)
* [x] Week 5 八節摘要已完成，D7、P7 與 AF-025 均有範圍、結果、決策、稽核、限制與下週計畫。
* [x] 每個科學量化結果均附 W5 圖表編號；appendix 可由 tracked source 重建。
* [x] v1 limitation 與 v2 remediation 並列，宣稱未擴張為通用密碼學安全。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* Week 5 的核心論述應分成「D7 發現 v1 norm leakage」與「使用者裁決後 v2 指定攻擊下降」兩階段，避免把修補前後混成一次成功實驗。
* Q8 應補齊 formal 影像品質、錯誤 key／nonce、crypto 與 diffusion 分項 runtime、storage overhead；不可用本週 latent cosine 取代影像端品質。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。Week 5 覆蓋 D7.1–D7.8、P7.1 與 AF-025 v2 補驗；formal N 維持 200，科學結果未作工程 pass gate。

### 7. 下一步
* 送稽核方複驗 AF-025 fixed-norm T1／T2-WB、public metadata、tamper ordering、repeat 與 probes；通過後才把 AF-025 改為 CLOSED。
* 進入 Week 6 Q8.1–Q8.5 與 P8.1 pilot，先凍結 formal 設定再執行 Week 7。

## 2026-10-15 · W5 · AF-025 稽核關閉同步

### 1. 當前目標 (Objective)
* 讀取稽核方最新 `AUD-20261015-01`，將 AF-025 的正式狀態同步到實作方進度、每週摘要與 README；不修改 auditor-owned 文件。

### 2. 執行動作 (Actions Taken)
* 核對 `AUDIT.md` 的本輪範圍、獨立驗證項目、finding 狀態及附帶限制。
* 更新 `PROGRESS_summary.md` Week 5 的 P7.1／AF-025 狀態、稽核段落與限制段落；更新 `README.md` 專案現況。

### 3. 執行結果 (Results & Observations)
* [事實] AUD-20261015-01 判定 PASS、新 finding 0，AF-025 CLOSED；目前沒有未關閉的 MAJOR／MINOR finding。
* [事實] 稽核方獨立確認 fixed-norm T1／T2-WB norm linkage、public lo/hi 隱藏、HKDF domain separation、private metadata tamper 在 inversion 前拒絕，以及合法 M1／M2 還原。
* [事實] 本輪只更新文件狀態，未啟動 GPU 或背景程序；目前沒有需要監控的本專案程序。

### 4. 達標判定 (Assessment)
* [x] AF-025 已由唯一有權關閉 finding 的稽核方正式標記 CLOSED。
* [x] Week 5 摘要不再顯示過期的 `FIXED?`／待複驗狀態。
* [x] 稽核關閉沒有被誤寫成一般密碼學安全證明。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* AF-025 的工程與稽核流程已終止；後續研究焦點轉為 Q8 的影像端品質、wrong-key／nonce、runtime 與 storage，而不是繼續調整 fixed norm。
* 稽核仍指出正式影像 PSNR／SSIM／L∞ 主表屬 Q8 範圍；本輪關閉只代表 AF-025 驗收條件完成。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。AF-025 已依 WORKFLOW §2.5 以 formal N=200 補驗並經獨立稽核關閉。

### 7. 下一步
* 開始 Week 6 Q8.1–Q8.5 與 P8.1 pilot；正式長時間 GPU 任務仍依既有授權與 managed-run／監控規範執行。

## 2026-10-15 · W6 · Q8／P8 Claude 委派前置授權阻擋

### 1. 當前目標 (Objective)
* 開始 Q8.1–Q8.5 與 P8.1 pilot；依使用者指示，由主代理規格化工作並委派 Claude 實作，再由主代理獨立驗證。

### 2. 執行動作 (Actions Taken)
* 主代理重新核對 `AGENTS.md`、`CLAUDE.md`、`WORKFLOW.md` Week 6、`loop_engineering_spec.md`、最新 `PROGRESS.md` 與既有 D6／D7／AF-025 runner。
* 準備以 Claude Sonnet、high effort 委派 bounded implementation：只做 Q8/P8 pilot 程式與 CPU 測試，不跑 GPU、不 commit、不修改稽核方檔案。
* 外部執行審核拒絕啟動，理由是尚缺「允許將完成 Q8/P8 所需的本地研究程式與文件內容提供給 Anthropic Claude」的明確資料外傳授權。

### 3. 執行結果 (Results & Observations)
* [事實] Claude session 未建立，Claude 沒有讀取任何專案檔案，也沒有產生或修改 Q8/P8 程式。
* [事實] 本輪未啟動 GPU、managed run 或其他背景程序；目前沒有 run ID／PID 需要監控。

### 4. 達標判定 (Assessment)
* [ ] 未達標：Q8/P8 實作尚未開始，阻擋點是外部資料分享授權，不是程式或 GPU 錯誤。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 使用 Claude CLI 的最小 `OK` 測試不涉及 repository 內容；實作 Q8/P8 則必須把必要程式與文件內容傳給 Anthropic，兩者資料範圍不同。
* 使用者明確授權後，依原 bounded prompt 啟動 Claude；若不授權，改由本地 Codex／內建 subagent 完成，不對外提供專案內容。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。尚未更改 Q8/P8 規格或啟動實驗。

### 7. 下一步
* 等待使用者明確決定是否允許 Anthropic Claude 讀取完成 Q8/P8 所需的 `/data2/paper` 程式碼與研究文件；取得授權後再啟動委派。

## 2026-10-15 · W6 · Q8.1–Q8.5／P8.1 pilot 基礎設施完成

### 1. 當前目標 (Objective)
* 在不先執行長時間 GPU 任務的前提下，建立 Q8 品質／匿名性／wrong-key、runtime、storage 與 P8 固定選圖的可追溯 pilot runner、CPU analyzer 與 regression tests。

### 2. 執行動作 (Actions Taken)
* 依使用者授權，以 Claude Sonnet 5.5 high effort 在隔離 worktree 實作；主代理另用內建 subagent 與本地檢查做獨立審查，未直接接受第一版。
* 新增 `scripts/q8_pilot_runner.py`、`q8_metrics.py`、`q8_analyze.py`、`tests/unit/test_q8_pilot.py` 與 `reports/q8_pilot_infrastructure.md`，並更新 `README.md` 檔案地圖。
* 第一輪審查發現 authenticated rejection 未作工程硬閘、cache provenance 未完整綁定、runtime 名稱過度宣稱、公開影像視角不一致、pilot/formal fixture nonce policy、cross-patient 前提與 LPIPS clipping 揭露等問題；全部退回修正後再驗。
* 修正版強制 wrong key／nonce 在 inversion 前拒絕、逐樣本核對 E2.3 cache/x0/設定、要求病人唯一、區分 `public_available` 與 diagnostic views、pilot/formal 使用每 run CSPRNG key/nonce，並明列 cached-latent 計時邊界及 LPIPS clipping fraction。

### 3. 執行結果 (Results & Observations)
* [事實] 主工作目錄 `py_compile` 通過；`tests/unit/test_q8_pilot.py` 為 64 passed、1 個 Pillow 相容性 deprecation warning。
* [事實] 隔離 worktree 的 3-sample synthetic managed run succeeded／artifact validator valid；CPU analyzer 產生 `table_fidelity.csv`、`table_anonymity.csv`、`table_wrong_key.csv`、`table_runtime.csv`、`table_storage.csv` 與 P8 grid。
* [事實] LPIPS backbone 目前不在本機 cache，因此 analyzer 依規格標 `deferred` 並保留空值及 clipping disclosure，未下載權重、未代換其他指標。
* [事實] 本段工作未啟動 GPU；Claude 工作 `98c2641c`／`aa6b901a` 均已 done，錯誤續接工作 `782c7317` 為 blocked 且未執行專案修改，沒有需監控的本專案程序。

### 4. 達標判定 (Assessment)
* [x] Q8.1–Q8.5／P8.1 pilot 基礎設施的 CPU plumbing、schema、fail-closed controls 與 deterministic analyzer 已完成實作方驗證。
* [x] S2 主方法固定為 `S2-fixed-norm/v2`；P/S0/S1/S2 主 fidelity cell 均走 range-preserving storage，legacy 只作可選 attribution cell。
* [ ] 真實 GPU smoke 與 dev N=20 pilot 尚未在本條目執行；目前沒有科學 Q8 結果，synthetic 數字不得引用。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 匿名性的公平比較不能只看 generator float；實際公開資訊不對稱，因此主視角定義為 `public_available`：P/S0/S1 可用公開 lo/hi 還原，S2 只能看到 q/255。
* runtime 從既有 E2.3 latent cache 起算，故使用 `cached_latent_to_m1` 與 `m1_to_reconstruction`，明確不把原圖初始 inversion 偽稱為本次端到端時間。
* 先以 dev 前 2 張執行預估約 5 分鐘的 GPU smoke；通過後 N=20 pilot 預估約 45 分鐘，依 AGENTS 規則須另取得使用者裁決。

### 6. ⚠ 與 WORKFLOW.md 不符
* LPIPS 尚未實際計算，因本機缺少已核對的 AlexNet backbone 權重；目前是明確 deferred，不以替代指標填值。Week 7 formal 前必須補齊本機權重並用 `--lpips require`，或由使用者／稽核方正式裁決延後。

### 7. 下一步
* 提交經驗證的 Q8/P8 基礎設施後，以 clean commit 執行 N=2 GPU smoke（預估約 5 分鐘，低於 30 分鐘門檻），再執行 CPU analyzer 並檢查五表、P8 圖與 provenance。
* smoke 通過後，向使用者報告 N=20 dev pilot 的目的、約 45 分鐘估時與 CPU/smaller-smoke 替代方案，取得明確裁決才啟動。

## 2026-10-15 · W6 · Q8／P8 真實 N=2 GPU smoke

### 1. 當前目標 (Objective)
* 以真實 checkpoint、dev E2.3 latent cache 與 clean Q8/P8 commit 驗證 GPU runner、artifact schema、五張 Q8 表與 P8 視覺格可完整執行；本 smoke 不作正式科學結論。

### 2. 執行動作 (Actions Taken)
* 將基礎設施以 commit `e0723aa` 提交；commit 前只 stage 7 個明確路徑，`scripts/check_staged_files.sh` 與 cached diff check 均通過。
* 一般沙箱首次啟動 `Q8_PILOT_SMOKE_20261015T030314601820Z_955298c1`，因無法看見 NVIDIA driver 在 1 秒內 fail closed；未載入模型、未處理樣本，failed run 保留不覆寫。
* 以相同設定、樣本與 commit 在 GPU 權限下重跑 `Q8_PILOT_SMOKE_20261015T030330332601Z_955298c1`：dev_v1.1 前 2 張、noise level 500、range-preserving 主 cell、smoke fixture key/nonce。
* 對成功 run 執行 `scripts.q8_analyze --lpips auto`，輸出到本地未追蹤的 `paper_assets/q8_pilot_smoke/`；大型／逐樣本或 smoke 產物未加入 Git。

### 3. 執行結果 (Results & Observations)
* [事實] 成功 run 為 2/2、artifact validator valid、無 failure/skip；總牆鐘約 4 分 43 秒，每張完整四方法比較約 132 秒。
* [事實] smoke M1 對 x0（N=2，僅工程診斷）mean PSNR：P=`38.314`、S0=`37.943`、S1=`39.370`、S2=`38.239` dB；mean SSIM：P=`0.98020`、S0=`0.97715`、S1=`0.97824`、S2=`0.97671`。S2 未觸發「比 S0 低超過 1 dB」檢查條件。
* [事實] S2 wrong-key 與 wrong-nonce authenticated open rejection rate 均=`1.0`，inverter calls 均=`0`；artifact/analyzer 的 fail-closed consistency checks 通過。
* [事實] cached-latent→M1 median 約 `7.75–7.78 s`，M1→reconstruction median約 `15.42–15.49 s`；名稱明確排除 E2.3 初始 x0→z inversion。
* [事實] analyzer 產生 fidelity 228 rows、anonymity 285、wrong-key 94、runtime 61、storage 106；LPIPS 因 AlexNet backbone 不在 cache 而標 `deferred`，未下載或替代。
* [事實] GPU run 已結束、session `35258` exit 0；目前沒有需監控的本專案程序。

### 4. 達標判定 (Assessment)
* [x] 真實模型 API、E2.3 cache provenance／x0 hash、S2 fixed-norm v2 M1/M2、auth controls、managed artifacts 與 CPU analyzer 已在 N=2 跑通。
* [x] Q8/P8 pilot 基礎設施可進入 dev N=20；目前結果角色仍為 `smoke`、`formal=false`，不得取代 pilot 或 formal。
* [ ] P8 固定病例尚不完整：dev 前 2 張皆 label 0，因此 smoke 沒有 fixed label-1 row；完整 dev N=20 才涵蓋預先固定的健康與積水病例。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 真實 smoke 未出現 CPU synthetic 測不到的 model/API 錯誤，且實測約 132 秒／張；依線性外推，N=20 主 cell 約 44 分鐘，加載與分析後估 45–50 分鐘。
* N=2 的 CI 與平均數不具穩健科學解釋，只用來確認工程管線；下一輪不得因 smoke 數值調參或改樣本。
* dev pilot 使用 `evidence_role=pilot`，因此會自動改用每 run 隨機 256-bit master key 與唯一 random nonce，不沿用 smoke fixture。

### 6. ⚠ 與 WORKFLOW.md 不符
* LPIPS 仍 deferred；其他 Q8.1–Q8.5／P8.1 smoke 路徑均依凍結規格執行。N=2 是 smoke，不是 WORKFLOW 的完整 dev pilot。

### 7. 下一步
* N=20 dev pilot 預估 GPU 45–50 分鐘，超過 30 分鐘門檻，需使用者明確裁決後才啟動；CPU 替代只能重跑 synthetic plumbing，不能驗證 diffusion 科學結果。
* 若獲准，啟動 managed run、低頻監控至完成，再執行 analyzer；若暫不跑，可先只保留本次 N=2 smoke 作工程驗證。

## 2026-10-15 · W6 · LPIPS 本機權重啟用與測試去環境相依

### 1. 當前目標 (Objective)
* 補齊 Q8 所需的真實 LPIPS(alex) backbone，確認既有 smoke arrays 可用 `--lpips require` 計算，並維持測試在有／無本機權重時皆可重現。

### 2. 執行動作 (Actions Taken)
* 透過 `CFG_DDIM` 的 torchvision 官方 weights API 下載 `alexnet-owt-7be5be79.pth` 到本機 torch hub cache；未安裝額外 `lpips` 套件。
* 執行 `q8_metrics.lpips_status()` 驗證 torchmetrics 線性權重與 AlexNet backbone SHA-256。
* 對既有 N=2 run 以 `q8_analyze --lpips require --force` 重建本地 `paper_assets/q8_pilot_smoke/`，不重跑 GPU。
* 權重出現後，兩個原先依賴「本機必定缺權重」的測試失敗；將 deferred-path fixture 改為顯式注入 `LpipsUnavailable`，computed path 仍由獨立 injected-metric 測試覆蓋。

### 3. 執行結果 (Results & Observations)
* [事實] `lpips_status.available=true`；官方 backbone SHA-256=`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`，torchmetrics LPIPS 線性權重 SHA-256=`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`。
* [事實] smoke analysis manifest 的 `lpips.state=computed`；M1 mean LPIPS（N=2 工程診斷）為 P=`0.04893`、S0=`0.05701`、S1=`0.04921`、S2=`0.05851`。
* [事實] 修正後 `tests/unit/test_q8_pilot.py` 為 64 passed、1 個既有 Pillow deprecation warning；有權重與無權重路徑不再依賴開發主機狀態。
* [事實] 沒有啟動 GPU 或背景程序；本次只下載公開模型權重並以 CPU 重算 LPIPS。

### 4. 達標判定 (Assessment)
* [x] Q8/P8 的 LPIPS 真實計算路徑已啟用並由 `--lpips require` 驗證。
* [x] 缺權重時的 fail-closed/deferred 與有權重時的 computed 兩種路徑都有環境獨立測試。
* [x] Week 7 formal 可要求 LPIPS，不需重新執行 diffusion；只要保留 hash-verified arrays 即可重算。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 原本的失敗不是 Python module 缺少，而是 torchvision backbone cache 缺檔；額外安裝 standalone `lpips` 反而會增加相依衝突。
* 測試不得用開發機 cache 狀態作固定前提；deferred/computed 必須由注入或隔離 cache 明確控制。
* LPIPS 會將影像 clip 至 [0,1]，PSNR 等指標仍使用未裁切 float；表格與 manifest 已揭露 clipping fraction，不混稱同一輸入。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。LPIPS 已補齊，且 N=2 數值仍只標 smoke 診斷，不作 formal claim。

### 7. 下一步
* 依使用者最新明確授權，以 clean commit 啟動 N=20 dev GPU pilot（預估 45–50 分鐘），使用 subagent 低頻監控；完成後以 `--lpips require` 產生完整 Q8/P8 pilot 結果。

## 2026-10-15 · W6 · Q8.1–Q8.5／P8.1 真實 N=20 dev pilot

### 1. 當前目標 (Objective)
* 依使用者對超過 30 分鐘 GPU 任務的明確裁決，以凍結的 dev_v1.1 20 張／20 病人、真實 checkpoint 與 E2.3 latent cache 完成 Q8/P8 pilot；再以 `--lpips require` 產生五張 aggregate 表與固定規則視覺格。

### 2. 執行動作 (Actions Taken)
* 從 clean commit `820cd80` 啟動 managed run `Q8_PILOT_20261015T044834956998Z_a03d6951`；角色為 `pilot`、`formal=false`、真實 backend、每 run 隨機 256-bit master key 與唯一 nonce，主 storage cell 為 range-preserving。
* 低頻監控 heartbeat/status；監控 subagent 中途觸及其用量限制，但 GPU 子程序與 managed run 未中斷，主代理改以 run status／log 驗證完成狀態。
* run 成功後以 CPU 執行 `scripts.q8_analyze --lpips require`，產生 fidelity、anonymity、wrong-key、runtime、storage 五表及 P8 fixed-label／median／worst grid；主代理另目視核對 grid。

### 3. 執行結果 (Results & Observations)
* [事實] managed run 為 20/20、exit code 0、artifact validation valid、stderr 0 bytes；牆鐘約 44 分 32 秒，每張約 132.2–132.8 秒。PID `997744` 已結束，目前沒有需監控的 GPU／背景程序。
* [事實] M1 reconstruction 對 x0 的 mean PSNR／SSIM／LPIPS／L∞：P=`37.809/0.97529/0.06252/0.20464`、S0=`38.439/0.97199/0.06459/0.26657`、S1=`39.168/0.97247/0.06258/0.15501`、S2=`39.275/0.97271/0.06233/0.15722`。S2 比 S0 高 `0.836 dB`，未觸發「低超過 1 dB」review trigger；這是 pilot 數據，不是 pass gate。
* [事實] `public_available` cosine NN（gallery 20）Top-1／Top-5：P=`1.00/1.00`、S0=`0.05/0.25`、S1=`0.10/0.45`、S2=`0.00/0.15`；random expectation 為 `0.05/0.25`。N=20 很小，僅供 pilot 診斷，不宣稱通用匿名安全。
* [事實] S2 wrong-key 與 wrong-nonce authenticated open 均 `20/20` 拒絕，rejection rate=`1.0` 且 inverter calls=`0`；bypass-auth counterfactual 仍只作錯誤條件品質示意。
* [事實] cached-latent→M1 median 為 P/S0/S1/S2=`7.798/7.797/7.808/7.802 s`，M1→reconstruction=`15.515/15.506/15.516/15.518 s`；S2 container seal/open median=`7.96/3.78 ms`，顯示本 pilot 的 wall time 主要由 diffusion 支配。
* [事實] mean storage total bytes（M1／M2／M2+preview）：P=`32,814/262,144/294,958`、S0=`53,334/262,144/315,478`、S1=`54,172/262,144/316,316`、S2=`74,625/350,561/405,645`；S2 額外包含 nonce、tag、manifest、private metadata 與編碼成本。
* [事實] LPIPS state=`computed`，backbone 與 linear weights hashes 均吻合；manifest 明列 LPIPS clip [0,1]、其他影像指標使用未裁切 float 及各陣列 clipping fraction。
* [事實] P8 固定選樣為 `dev_v1.1_000`（label 0）與 `_010`（label 1）；S2 M1 PSNR lower-median 為 `_018`（39.373 dB），worst 為 `_007`（38.390 dB）。視覺格欄位、公開視角與固定 [0,1] display range 均正常。
* [事實] 唯讀 subagent 獨立重算 8 個分析輸出、3 個來源檔與 latent cache hashes，全部吻合；另確認 nonce registry 為 40 筆。唯一 metadata 小瑕疵是 managed `status.json.progress` 未隨子程序更新，最終 `state=succeeded`、exit 0 與 artifact validator 不受影響。

### 4. 達標判定 (Assessment)
* [x] Q8.1–Q8.5 dev pilot 的五張 aggregate 表、bootstrap CI、LPIPS、authenticated controls、runtime/storage 拆分與 provenance 已完整產生。
* [x] P8.1 使用預先固定規則而非手挑最佳案例，健康／積水 fixed cases 與 median／worst 均有輸出。
* [x] runner/artifact/analyzer 工程管線可進入 Week 7 formal；科學結果仍標 `pilot`、`formal=false`。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* P/S0/S1/S2 的 M1 reconstruction 品質接近，且 S2 未出現相對 S0 超過 1 dB 的品質下降；正式 N=200 仍須保留 paired statistics 與 effect size，不能由 N=20 宣告等效。
* 四方法總 runtime 接近是因每張都執行相同次數的 DDIM generation/inversion；毫秒級 transform/KDF/MAC/container 成本被每段約 7.8 秒的 diffusion 成本淹沒。正式表仍須分項呈現，而非只報總時間。
* `public_available` 對 S2 使用 q/255，對 P/S0/S1 使用公開 lo/hi 還原，這是部署資訊不對稱而非相同像素域；正式論文須保留該註解並把 generator-float 視角列為診斷。
* managed wrapper 目前只知道子程序存活與最終狀態，沒有接收 Q8 runner 的逐樣本 progress；若 Week 7 需要 UI 精確進度，應在 formal 前補 heartbeat/progress bridge，但不需為此重跑本 pilot。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。Q8/P8 本輪按 Week 6 定義執行 dev pilot；正式樣本數仍維持 N=200，將留到 Week 7，且本輪結果未被用作工程 pass/fail gate。

### 7. 下一步
* 先由稽核方複驗 N=20 run provenance、五表 estimand、auth ordering、LPIPS disclosure 與 P8 選樣；若無新 finding，凍結 Q8 formal 設定。
* Week 7 以 security_v1 N=200 執行 formal Q8/F9，補 paired test、effect size、multiple-comparison 規則與 claim-evidence matrix；超過 30 分鐘 GPU 任務仍須在啟動前取得新的明確裁決。

## 2026-10-15 · W6 · Q8/P8 mixed-storage 主部署修正與 N=20 補跑

### 1. 當前目標 (Objective)
* 依使用者最新裁決，將 Q8/P8 主端到端比較由「四方法皆 range-preserving」改為 **P/S0/S1 使用學長 `legacy_png_handoff`，S2 使用 authenticated private-range preserving**；保留舊 N=20 結果作 storage-controlled attribution，不覆寫歷史證據。

### 2. 執行動作 (Actions Taken)
* 更新 `WORKFLOW.md`、`PROPOSAL.md` 與 Q8 infrastructure 規格，明列 2026-10-15 決定取代舊主欄口徑；P 亦採 legacy，作學長流程的 identity baseline。
* 新增 `scripts/q8_legacy_supplement.py`：嚴格驗證 parent manifest/config/per-sample/array-index、checkpoint/sampler、E2.3 cache 與逐 latent hashes，重用 parent 的 anonymous float arrays，只補 P/S0/S1 legacy encode/decode→DDIM inversion→inverse→reconstruction，並以 legacy inverted latent 重算 S0/S1 wrong-key；S2 完全不重算，也不需要其未保存 raw key。
* 修改 `scripts/q8_analyze.py`：以逐方法 deployment storage map 合併 parent/supplement，主 fidelity/anonymity/wrong-key/runtime/storage 與 P8 grid 使用 mixed deployment；P/S0/S1 原 range 結果另標 `range_attribution`。
* 先執行 N=2 真 GPU supplement smoke `Q8_LEGACY_SUPPLEMENT_SMOKE_20261015T082923364032Z_ff340c06` 與 mixed analyzer；修正 warmup 唯讀 NumPy warning 後，67 個 Q8 unit tests 全通過。
* 執行 N=20 managed supplement `Q8_LEGACY_SUPPLEMENT_20261015T083637049107Z_cf941408`，再以 `--deployment-supplement ... --lpips require` 產生 `paper_assets/q8_pilot_deployment/`；舊 `paper_assets/q8_pilot/` 保留不覆寫。

### 3. 執行結果 (Results & Observations)
* [事實] N=20 supplement 為 20/20、exit code 0、artifact validation valid、stderr 0 bytes；起訖 `08:36:37Z–08:57:55Z`，牆鐘約 21 分 18 秒。PID `1360098` 已結束，目前沒有需監控的 GPU／背景程序。
* [事實] mixed-deployment M1 reconstruction 對 x0 的 mean PSNR／SSIM／LPIPS／L∞：P=`37.708/0.97486/0.06227/0.23219`、S0=`31.467/0.92721/0.11697/0.55780`、S1=`31.305/0.89795/0.16593/0.41613`、S2=`39.275/0.97271/0.06233/0.15722`。
* [事實] S2−S0 paired difference：PSNR=`+7.808 dB`（95% patient-bootstrap CI `[+6.980,+8.658]`）、SSIM=`+0.04550`、LPIPS=`−0.05464`、L∞=`−0.40059`。差異包含「新加密設計＋新 storage protocol」，不可歸因為 SOT/AEAD 單獨造成。
* [事實] parent P/S0/S1 全 range-preserving 的 PSNR 仍為 `37.809/38.439/39.168 dB` 並保留於 attribution rows；由此可見 S0/S1 mixed 主欄降至約 31 dB 的主要原因是 legacy handoff 丟失原值域，與 AF-024 的既有歸因一致。
* [事實] `public_available` cosine NN Top-1／Top-5：P=`1.00/1.00`、S0=`0.05/0.25`、S1=`0.10/0.45`、S2=`0.00/0.15`；N=20 random expectation=`0.05/0.25`，僅作 pilot 診斷。
* [事實] deployment cached-latent→M1 median 為 P/S0/S1/S2=`7.797/7.796/7.807/7.802 s`，M1→reconstruction=`15.645/15.671/15.670/15.518 s`；M1 total mean bytes=`32,728/53,248/54,083/74,625`。S2 auth wrong-key/wrong-nonce 仍 20/20 拒絕、inverter calls=0。
* [事實] supplement merge 前後，S2 的 `anonymous_public_q`、`anonymous_public_available`、`recon_m1`、`recon_wrong_key` arrays 逐值完全相同；P8 selection JSON 亦完全相同，只替換 P/S0/S1 panel 為 legacy public/reconstruction。
* [事實] LPIPS=`computed`；mixed analysis 產生 fidelity 285 rows、anonymity 360、wrong-key 98、runtime 93、storage 115，output manifest 綁定 parent 與 supplement hashes。
* [事實] 唯讀 subagent 獨立重算 parent／supplement／analysis 全部 file 與 slice hashes、檢查 20 unique patients/IDs、13,060 筆 final long rows及 finite，均通過；其指出的 manifest 頂層 storage 語意已修成 mixed deployment 清單，另保留 `source_parent_storages=[range_preserving]`，67 個 unit tests 再次全通過。

### 4. 達標判定 (Assessment)
* [x] Q8/P8 N=20 主端到端比較已符合使用者最新 storage 口徑：P/S0/S1 legacy、S2 private-range preserving。
* [x] 舊全 range 結果未刪除，已降級並明標為 storage-controlled attribution，可分離 storage 與方法效應。
* [x] bounded supplement 避免重跑匿名生成與 S2；hash binding、synthetic bit-exact regression、N=2 真 GPU smoke、N=20 artifact validation 與 mixed analyzer 均通過。
* [ ] 目前仍是 `pilot/formal=false`；N=200 與正式 paired statistics 屬 Week 7，尚未執行。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* mixed-deployment 主欄最貼近「學長原部署 vs S2 新部署」，但它同時改變加密與 storage；論文不能用 `+7.8 dB` 宣稱 S2 加密本身提升品質，必須同列 range-attribution 說明主要增益來自保存值域。
* P 的 legacy PSNR 只比其 range cell低約 0.10 dB，但 S0/S1 低約 6.97/7.86 dB，顯示逐 key 產生的匿名輸出值域使 legacy min-max 丟尺度問題特別嚴重；這也是使用者希望主欄呈現的真實舊流程成本。
* bounded supplement 的來源是 parent hash-verified anonymous arrays；正式 N=200 可沿用「parent＋deployment supplement」分段設計，先凍結 parent，再補 legacy，避免把所有 control 重跑兩次。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。`WORKFLOW.md` 已依使用者 2026-10-15 最新裁決同步；既有 N=20 全 range 主結果沒有被竄改，而是保留並重新標示為 attribution。

### 7. 下一步
* 由稽核方複驗 supplement 的 parent/source/cache hash binding、P/S0/S1 legacy bit-exact 路徑、mixed table storage labels、S2 不變性、range attribution 與 P8 panel 來源。
* 稽核通過後凍結 Week 7 formal 設定；N=200 預估仍超過 30 分鐘，啟動前另向使用者報告正式 parent/supplement GPU 預估與 CPU/smaller alternative，取得明確裁決。

## 2026-10-15 · W6 · Q8／P8 週摘要與論文 appendix

### 1. 當前目標 (Objective)
* 依最新 mixed-deployment 結果與 AUD-20261015-04／-05 稽核結論，完成 `PROGRESS_summary.md` Week 6 八段式摘要，並建立所有數字可追溯的 `appendix/week6/`。

### 2. 執行動作 (Actions Taken)
* 在 `scripts/build_weekly_appendix.py` 登記 Week 6 素材：1 張 P8 fixed visual grid、Q8 fidelity／anonymity／wrong-key／runtime／storage 五張原始 aggregate 表，以及由 `P8_selection.json` 確定性產生的 selection CSV。
* 由已提交的 `paper_assets/q8_pilot_deployment/` 建立 `appendix/week6/`；以相同命令連續重建兩次並比較 MANIFEST、圖與表的 SHA-256。
* 在 `PROGRESS_summary.md` 追加 6.1–6.8，明確分開 mixed deployment 主比較與全 range-preserving attribution，並更新根目錄 `README.md` 的檔案索引。

### 3. 執行結果 (Results & Observations)
* [事實] Week 6 appendix 共 1 張圖、6 張表；每項含來源 hash、run ID、caption、claim scope 與輸出 hash，兩次重建逐位元相同。
* [事實] 摘要記錄 Q8／P8 N=20 pilot 的 fidelity、public-view linkage、authenticated rejection、runtime、storage 與 deterministic selection；所有數值均回指 W6-F1 或 W6-T1～T6。
* [事實] AUD-20261015-04 與 AUD-20261015-05 均為 PASS、新 finding 0；摘要未把 N=20 pilot 寫成 formal 結論，也未把 S2−S0 `+7.808 dB` 誤歸因為加密演算法單獨效果。
* [事實] 本輪只做 CPU 文件／素材建置；沒有啟動 GPU 或背景程序，目前沒有需監控的 PID/run。

### 4. 達標判定 (Assessment)
* [x] Week 6 固定八段式摘要已完成並附論文素材索引。
* [x] mixed deployment、storage attribution、pilot/formal 邊界與稽核狀態均明確揭露。
* [x] appendix builder 可重現，且沒有納入 per-sample long output、dataset、checkpoint、cache、secret 或 auditor-owned 檔案。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* Week 6 最重要的寫作風險是把部署差異當成 cipher 消融；摘要用 W6-T1 同列 mixed 與 range-attribution，將主要品質改善歸因到值域保存。
* Week 7 formal 可沿用同一素材 schema，待 N=200 結果提交後另建 `appendix/week7/`，不得以 Week 6 表格覆寫正式結果。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。Week 6 完成的是 Q8／P8 pilot；formal N=200、paired inference 與 claim matrix 仍按 WORKFLOW 留在 Week 7。

### 7. 下一步
* 啟動 Week 7 formal 前，先提出 parent／legacy supplement 的 GPU 用途、預估時間及小型替代方案供使用者裁決；獲准後執行 N=200 並建立 Week 7 appendix。

## 2026-10-15 · W7 · Track F · F9 CPU 部分（Q8 formal parent 執行期間）

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md`：F9.3（adaptive norm limitation note）、F9.4（statistical comparison 機制）、F9.5（claim matrix v1）。
* 使用者裁決（2026-10-15）：先背景執行 Q8 品質 formal **parent**（N=200，~7.5h），**停在 legacy supplement 前等下一次同意**；期間並行做不需 GPU 的 F9 部分。Week 7 sampler 維持現狀（DDIM + 非對稱條件）。

### 2. 執行動作 (Actions Taken)
* 啟動背景 GPU managed run `Q8_FORMAL`（`--evidence-role formal --split security_v1.csv`，security_v1 既有 cache，不重跑 inversion）；僅 parent，supplement 待裁決。
* 新增 `scripts/build_claim_matrix.py` → `paper_assets/claim_matrix/{claims.csv,index.md}`（F9.5）：由已凍結、已稽核的 `attack_package/claims.csv`（A3/A4/A4.6）與 `defense_package/claims.csv`（D5/D6/D7/AF-025）彙整，headline 不手抄；RQ1–RQ3 的攻擊與 S2 防禦假設對應證據，H3c/RQ4 品質假設標 `pending`（待 Q8_FORMAL）。
* 新增 `reports/f9_3_norm_limitation.md`（F9.3）：T1 vs T2-WB norm-only linkage 與固定 norm remediation 的機制分析，數字引自 `table_norm_leakage.csv`、`table_AF025_fixed_norm_norm_leakage.csv`、`table_D7_T2WB_inversion.csv`。
* 新增 `scripts/f9_statistics.py`（F9.4 機制）：paired Wilcoxon signed-rank、matched-pairs rank-biserial 效果量、Holm-Bonferroni 固定多重比較（alpha=0.05）；在 N=20 pilot long rows 上驗證並確認確定性，待 formal 輸出重跑產生正式 `table_statistics.csv`。

### 3. 執行結果 (Results & Observations)
* [事實] `Q8_FORMAL` parent 已啟動並進入 running（通過 formal guard：security_v1、N=200、無 limit）；預估約 7.5 小時，完成後通知。
* [事實] claim matrix v1：10 條假設，**8 條已由 N=200 formal 證據支持**（H1a/b/c、H2a/b/c、H3a/b），2 條（H3c、RQ4 品質）`pending`；連續重建兩次位元相同。
* [事實] F9.3 note 彙整：S2 v1 T1 norm Top-1=**1.0**、T2-WB range dev-calibrated=**0.25**（legacy=0.005）；AF-025 固定 norm v2 後 T1／T2-WB 皆**0.005**（=random），合法還原不變。明載 legacy 低攻擊率是存檔 trade-off、非密碼強度。
* [事實] F9.4 在 N=20 pilot（mixed deployment）驗證：S2 對 S0/S1 的 PSNR 中位差 +8.11/+8.13 dB、SSIM、LPIPS、cosine、rmse 經 Holm 後皆顯著（rank-biserial |r|≈1）；S2−P 的 PSNR +1.62 dB 顯著、LPIPS/L∞ 不顯著。**此為 mixed deployment（含存檔效應），JSON 已註明不可歸因為加密本身**。腳本確定性通過。
* [事實] 本輪為 CPU 文件／機制建置（parent 在背景跑）；新增檔案皆為 additive，未改既有模組。

### 4. 達標判定 (Assessment)
* [x] F9.5 claim matrix v1 完成（security/attack 面已凍結，品質面標 pending）。
* [x] F9.3 norm limitation note 完成。
* [x] F9.4 統計機制完成並在 pilot 驗證；正式 `table_statistics.csv` 待 formal。
* [ ] F9.1 formal security freeze（複用 D7/AF025 N=200）、F9.2 formal bootstrap CI、F9.4 formal 表：待 parent（+使用者同意後的 supplement）與 `q8_analyze` 落地。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* F9.2 的 patient-bootstrap CI 與 paired difference 已內建於 `q8_analyze.py`（percentile bootstrap、seed 1911、10000 resamples；Wilson for Top-k），formal 輸出即自動產生；F9.4 的顯著性檢定／效果量／多重比較則由新 `f9_statistics.py` 補上。
* claim matrix 已標出品質假設為唯一 pending 項，精準界定 Q8_FORMAL 要填的內容，避免把部署差異誤當加密消融。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。F9.4 以新腳本補「顯著性檢定＋效果量＋固定多重比較」，WORKFLOW 僅指定 `table_statistics.csv` 交付物與規則固定，未指定腳本名。

### 7. 下一步
* 等 `Q8_FORMAL` parent 完成（通知後）→ 向使用者報告 parent 結果與 supplement 的 ~3.5h GPU 估計，取得同意後再跑 legacy supplement。
* supplement 完成後：跑 `q8_analyze`（mixed deployment）產生 formal fidelity/anonymity/wrong-key/runtime/storage 與 bootstrap CI（F9.2），再跑 `f9_statistics` 產生正式 `table_statistics.csv`（F9.4），更新 claim matrix 的品質列（F9.5），建立 `appendix/week7/`。

## 2026-10-16 · W7 · Track F9+附錄 · Q8_FORMAL 完成、啟動 legacy supplement、規劃 BDIA 2×2 安全+品質附錄

### 1. 當前目標 (Objective)
* `WORKFLOW.md` 子任務：F9（收尾 formal 主結果）＋規劃附錄 BDIA ablation（含安全背書）。

### 2. 執行動作 (Actions Taken)
* 確認 `Q8_FORMAL` parent 完成：`artifacts/runs/Q8_FORMAL/Q8_FORMAL_20261015T173907560215Z_7e4c3368`，state=succeeded、exit 0、artifact_validation valid、sample_count=200，耗時約 7h19m（17:39Z→00:58Z）。
* [使用者同意] 啟動 A＝legacy supplement（formal）：`scripts/run_cfg_ddim.sh python -m scripts.q8_legacy_supplement --task-id Q8_LEGACY_SUPPLEMENT --parent-run artifacts/runs/Q8_FORMAL/Q8_FORMAL_20261015T173907560215Z_7e4c3368`，背景、~3.5h；run dir `Q8_LEGACY_SUPPLEMENT_20261016T064241Z_964cb3e0`（已通過 parent/checkpoint/sampler/E2.3-cache/逐 latent hash 閘門）。
* BDIA P/S0/S1 smoke（n=4 dev、legacy 不保值域）：`/tmp/.../scratchpad/bdia_pss_nolabel_smoke.py`，比 DDIM 原設定 vs BDIA+對稱null 的 PSNR/SSIM/L∞（scratchpad，未 commit）。
* 規劃附錄 BDIA 2×2（S2-only）＋安全背書，設計文件：`/tmp/.../scratchpad/bdia_2x2_appendix_design.md`。

### 3. 執行結果 (Results & Observations)
* [事實] Q8_FORMAL（N=200，全 private-range＝range-attribution 臂）M1 重構 PSNR：P 37.58／S0 38.28／S1 38.77／S2 **38.82** dB；latent round-trip cosine 皆 ≈0.9998（`xform` 加解密不經擴散 cosine=1.0、max_abs 8.3e-7，證明缺口全來自 sampler+8-bit 量化，非密碼）。**同存檔下 S2−S0 僅 +0.54 dB**，坐實 pilot mixed +7.8 dB 幾乎全來自存檔差異而非加密。
* [事實] S2 認證控制（N=200）：錯金鑰拒絕 200/200、錯 nonce 拒絕 200/200，inverter_calls=0；錯金鑰/錯 nonce latent cosine≈0。摘要 `summary.json`。
* [事實] BDIA smoke（n=4 dev，Δ=BDIA對稱null−DDIM原設定）：P **+4.10 dB**、S0 +0.28、S1 +0.14。→ **BDIA+對稱null 的品質增益集中在 P（恆等），對有加密的 S0/S1（legacy 不保值域）幾乎無效**，因 legacy 8-bit min-max 量化（丟值域）才是 S0/S1 主誤差源。增益只在「保值域、sampler 成瓶頸」時顯現（即 S2）。[注意] 此對比綁了 sampler+條件兩變更；稽核方 S2 both-null 診斷顯示 BDIA 本身僅約 +0.5 dB，其餘來自對稱條件。

### 4. 達標判定 (Assessment)
* [x] Q8_FORMAL parent 完成並驗證。
* [ ] A（legacy supplement）背景執行中，完成後解鎖 F9.2/F9.4 formal 表與 F9.5 品質列。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* A 跑完即可收尾 F9：`q8_analyze`（mixed）→ fidelity/anonymity/wrong-key/runtime/storage + bootstrap CI（F9.2）；`f9_statistics` on formal mixed long → `table_statistics.csv`（F9.4）；更新 claim_matrix H3c/RQ4（F9.5）；F9.1 複用 AF025/D7/A4.2 既有 N=200 證據做組裝（無新 GPU）；建 `appendix/week7/` + PROGRESS_summary W7 段。BDIA 附錄**不屬** F9/G5 凍結範圍。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。

### 7. 下一步 ── [待辦] BDIA 2×2 附錄 GPU run 的參考資訊（有時間才跑）
* [決定/使用者 2026-10-16] 附錄要**同時宣稱品質更好與安全性相同**，背書 ③與④ 兩格；主表仍 DDIM+非對稱（scheme of record）不動。採用為正式方法屬改研究設計，依 skill §9 需回 Cowork 拍板；驗證實驗本身可實作方執行（GPU 需 gate）。
* 2×2 格：① DDIM+非對稱（=Q8_FORMAL S2，**重用不跑**）／② DDIM+對稱null／③ BDIA+非對稱（關鍵格）／④ BDIA+對稱null。「對稱」=兩邊都 null（無條件）。拆因：①→③ 純 BDIA（語義不變、論文主張用此）；①→② 純對稱條件；②④/③④ 交互。
* 反演快取：四格反演條件皆 null，**只隨 sampler 變**→ ①②用現有 E2.3 `E2.3_20261009T183736363217Z_84cf642a`（DDIM-null）；③④ 需**新增一份 BDIA-null 反演快取**（同 200 張 security_v1、seed 1911、新 run_id、hash 登錄）。
* 安全臂分類（以選定 cell 的 S2-BDIA 重驗）：**僅 T2-WB（重反演再識別）+ norm scorer + dev 校正隨 sampler 變**，須各 cell（③、④ 匿名影像不同）各做一次（~0.5h/cell，重用品質 run 的匿名影像）；T1/norm/geometry(QᵀQ=I)/cross-nonce/structured-probe/repeat/tamper(HMAC) 皆 latent/容器層、與 generate 條件無關，③④ 共用、**只算一次**（免擴散，分鐘級）。fixed-norm 已強制 ‖·‖=C，預期 T1/T2-WB 在 BDIA 下仍 random，重跑為實證坐實「同樣安全」。
* GPU 盤點（N=200，③④ 都背書安全，可分三段獨立跑）：BDIA-null 快取 ~0.5h → 品質 2×2 跑 ②③④（①重用）~3.8h → T2-WB×2 + dev 校正 + latent 臂 ~1h；**合計 ≈ 5.5–6h**。加 ④ 相對只背書③ 的增量 ≈ +40 min。**不需重跑**：E2.3 DDIM 快取、Q8_FORMAL ①、整個 D7/AF-025 DDIM 安全電池、F9 主表。
* CPU 腳手架（開發中、不佔 GPU、不 commit 結果）：`scripts/bdia_backend.py`（γ=1 BDIA generate/invert + NFE 計數 + both-null 可逆性單測）、`scripts/bdia_2x2_runner.py`（重用 Q8 S2 封裝，`--cells/--sampler/--conditioning`）、`scripts/bdia_security_verify.py`（T2-WB 重反演 + latent 臂重驗，重用品質 run 匿名影像）；分析沿用 `q8_analyze.py`＋`f9_statistics.py`。BDIA 參考實作見稽核方唯讀 `audit/bdia_vs_ddim_s2fix.py`（AUD-20261015-02，γ=1）。

## 2026-10-16 · W7 · 附錄 · BDIA CPU 腳手架完成並 smoke 驗證（A 仍背景執行中）

### 1. 當前目標 (Objective)
* [使用者同意 2026-10-16] 在 A（GPU）背景執行期間，完成 BDIA 附錄的 CPU 腳手架與 smoke test（不佔 GPU、不 commit 結果）。

### 2. 執行動作 (Actions Taken)
* `scripts/bdia_2x2_runner.py`（commit `b4100ef`）：2×2 品質 ablation runner。重用 `q8_pilot_runner` 的 S2 crypto（`normalize_to_fixed_norm`+`transform`+`seal/open_anonymous_image`）與量測助手（`image_columns`/`latent_columns`/`Meter`）；每 cell 換 sampler+generate 條件，fixed-norm/SOT/HMAC/AEAD 與凍結 S2 一致（ROUNDS=1、USE_PERMUTATION=False、private range）。managed run、fail-closed、per-slice-hash 陣列（x0/anon/public/recon_m1）。方法 token 固定 `S2`，另加 `cell/sampler/conditioning/base_sample_id` 識別欄；`sample_id` 以 `<id>__cell<N>` 保唯一（過 artifact validator）。
* `scripts/bdia_security_verify.py`（commit `fca5ccd`）：單 cell 的 T2-WB 安全重驗。= `af025_fixed_norm_t2wb` 把 generate/invert 換成該 cell 的 sampler（`bdia_2x2_runner.build_backend`），**FIXTURE_KEY／nonce／E2.3 latents／fixed-norm／SOT／容器全部與 AF-025 相同，只差 sampler**；輸出 AF-025 同 schema 的 per_sample + `query_latents.npy`，沿用 `af025_fixed_norm_analyze`/`d7_t2wb_analyze` 評分。
* 兩者皆加 `--synthetic-selftest`（CPU、無模型、SyntheticBackend）走完整 crypto+量測+artifact 線。

### 3. 執行結果 (Results & Observations)
* [事實] 2×2 runner CPU smoke：`--synthetic-selftest --cells 2 3 4 --limit 3 --warmup 0`，9 列（3 cell×3 樣本）、sample_id 唯一、欄位含 anon/anonpub/m1/m1lat/runtime/nfe，managed run artifact_validation 通過。輸出於 scratchpad（未 commit）。
* [事實] security verify CPU smoke（cell 3）：`--synthetic-selftest --cell 3 --limit 4`，per_sample + query_latents.npy 存檔並逐筆 hash 驗證、run 驗證通過；legit_cos≈0.99998（synthetic 近無損）。表頭與 AF-025 一致（method/storage/metric/n/mean/ci95/...）。
* [事實] `build_backend` 逐 cell 構造正確：② DDIM generate_null=True；③ BDIA invert_null=True/gen_null=False（純非對稱）；④ BDIA 兩邊 null。invert 恆 null（DDIM `ddim_anonymization_forward` 強制；BDIA invert_kwargs 帶 null=True）。
* [事實] DDIM-null 生成路徑＝`ddim_anonymization_backward(model_kwargs{...,null:True}, guidance_scale=-1)`（guidance=-1→無 CFG 混合、單次無條件 model call）；BDIA 非對稱生成＝`_gr` 不帶 null（y=0 條件、guidance=-1 單次），與學長語義一致。

### 4. 達標判定 (Assessment)
* [x] 三件 CPU 腳手架（`bdia_backend` 4296965、`bdia_2x2_runner` b4100ef、`bdia_security_verify` fca5ccd）皆 smoke 通過、原子 commit、無結果檔入庫。
* [ ] 真實 GPU run 待使用者 go（「有時間才會跑」）；A（Q8_LEGACY_SUPPLEMENT）仍背景執行中，完成後優先收尾 F9。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* [待辦] GPU run 指令模板（使用者 go 後；>30min 需 gate）：
  * 品質 2×2（②③④，①重用 Q8_FORMAL）：`scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id BDIA2X2 --validate-artifacts -- python -m scripts.bdia_2x2_runner --evidence-role formal --cells 2 3 4 --split splits/security_v1.csv --checkpoint <ckpt> --ddim-latent-cache artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a`（cell 2 用 E2.3；③④ 線上 BDIA-null 反演）。
  * 安全重驗（每 cell 一次）：`... python -m scripts.bdia_security_verify --cell 3 --split splits/security_v1.csv --latent-cache <E2.3> --checkpoint <ckpt>`（③、④ 各跑；dev 校正用 `--limit`）。
  * 分析：`q8_analyze`（品質，bootstrap CI）＋ `f9_statistics`（paired ①→③、①→④，另開 family 不混主表 Holm）＋ `af025_fixed_norm_analyze`/`d7_t2wb_analyze`（T2-WB 再識別）。
* [決定] 主表仍 DDIM+非對稱；BDIA 升為正式方法＝改研究設計，需回 Cowork（skill §9）。附錄只背書「品質更好＋安全相同」。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。附錄非 WORKFLOW 指定交付物，為 F9 外的選配附錄（使用者「有時間才會跑」）。

### 7. 下一步
* 等 A（`Q8_LEGACY_SUPPLEMENT_20261016T064241Z_964cb3e0`）完成通知 → 驗 run dir/validation → 收尾 F9（F9.2 `q8_analyze` mixed、F9.4 `f9_statistics`、F9.5 claim matrix、F9.1 組裝、`appendix/week7/` + PROGRESS_summary W7）。
* BDIA 附錄 GPU run 待使用者另行 go。

## 2026-10-16 · W7 · Track F9 收尾 · legacy supplement 完成 + formal mixed 主結果凍結

### 1. 當前目標 (Objective)
* A（`Q8_LEGACY_SUPPLEMENT`）完成後收尾 F9：產生 formal mixed-deployment 品質主表、paired 統計、填滿 claim matrix；確認 F9.1 formal security 證據就位。

### 2. 執行動作 (Actions Taken)
* 驗 A：`artifacts/runs/Q8_LEGACY_SUPPLEMENT/Q8_LEGACY_SUPPLEMENT_20261016T064241Z_964cb3e0`，exit 0、evidence_role=formal、expected_samples=200、parent_run_id=Q8_FORMAL_...7e4c3368、failures=0、summary sample_count=200、artifact validation 通過。
* F9.2：`scripts/run_cfg_ddim.sh python -m scripts.q8_analyze --run-dir <Q8_FORMAL parent> --deployment-supplement <上述 supplement> --output-dir paper_assets/q8_formal --lpips auto`（LPIPS computed）→ table_fidelity/anonymity/wrong_key/runtime/storage（285/360/98/93/115 列）+ patient-cluster bootstrap CI。
* F9.4：`python -m scripts.f9_statistics --long-csv /data2/paper/paper_assets/q8_formal/per_sample_long.csv --manifest .../analysis_manifest.json --out /data2/paper/paper_assets/tables/table_statistics_formal.csv --json-out ....json --force`（絕對路徑避開 relative_to bug）→ 35 檢定、Holm 後 35 全顯著、role=formal。
* F9.5：改 `scripts/build_claim_matrix.py` 讓 H3c/RQ4 headline **從凍結表讀入**（`q8_formal/table_fidelity.csv` + `tables/table_statistics_formal.csv`，數字不手抄），status→supported；`python -m scripts.build_claim_matrix --force`。
* F9.1：確認 formal security（N=200）證據已凍結且 git-tracked：`paper_assets/tables/table_AF025_fixed_norm_{T1,T2WB_inversion,norm_leakage,probe,repeat}.csv`、`table_D7_T2WB_inversion.csv`、cross_nonce/geometry/tamper/A4.6 等，並由 `paper_assets/defense_package/claims.csv`（已稽核）彙整；claim matrix 經 defense_package 指向。無需新 GPU。

### 3. 執行結果 (Results & Observations)
* [事實] formal mixed deployment（P/S0/S1=legacy PNG、S2=保值域私有 v2）M1 recon vs x0：P PSNR 37.46/SSIM 0.973/LPIPS 0.067；S0 31.15/0.918/0.134；S1 31.42/0.895/0.167；**S2 38.82/0.970/0.067**。來源 `paper_assets/q8_formal/table_fidelity.csv`（comparison=marginal）。
* [事實] 配對（formal，N=200，Wilcoxon+rank-biserial+Holm）：S2−S0 中位 +7.68 dB、S2−S1 +7.73 dB（r=1.0、p_holm≈5e-33）；S2−P 中位 +1.18 dB（r=0.988，S2 微贏明文 P，因保值域容器避開 PNG min-max 損失）。來源 `table_statistics_formal.csv`。
* [事實] S2 latent：transform-only cosine=1.000000、max_abs 8.3e-7（結構變換本身無損）；M1 latent round-trip cosine 0.999794、max_abs 0.089；t1 精確 latent cosine 1.0。→ 影像殘差來自 sampler+8-bit 量化，非 transform（H3c）。
* [事實] claim matrix：10 假設全 supported、0 pending。
* [注意] **部署差 ~+7.7 dB 幾乎全來自保值域存檔，非加密**；同存檔下 S2−S0 僅約 +0.5 dB（Q8_FORMAL range-attribution 臂）。index.md 已明載此限縮，不得歸因為加密本身。

### 4. 達標判定 (Assessment)
* [x] F9.1（frozen security N=200 就位）、F9.2（formal 五表+CI）、F9.3（adaptive limitation note，先前 ed99ce6）、F9.4（table_statistics_formal）、F9.5（claim matrix 全 supported）皆完成。commit `0ab73a0`。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* F9 數值收尾完成。**剩餘屬週末/G5 的步驟**（本輪未做，避免週中提早定稿）：`scripts/build_weekly_appendix.py --week 7`（需先在 WEEKS registry 加 week7 條目，納入 q8_formal 五表 + table_statistics_formal + claim_matrix）＋ `PROGRESS_summary.md` Week 7 段；之後進 G5 凍結主表。
* BDIA 2×2 安全+品質附錄 GPU run 仍待使用者 go（「有時間才會跑」），與 F9/G5 無相依。

### 6. ⚠ 與 WORKFLOW.md 不符
* WORKFLOW 排程 Week 7＝11/16–11/22、G5＝11/22，但實際執行日為 2026-10-16（既有行程差，非本輪造成）；F9 交付物內容與規則均依 WORKFLOW §F9 達成。

### 7. 下一步
* 週末/G5：建 `appendix/week7/`（加 WEEKS[7] registry）+ PROGRESS_summary Week 7 段 → G5 凍結主表數字。
* 待使用者對 BDIA 附錄 GPU run 下 go（指令模板見前一條目 §5）。
* 使用者手動 push 本地 commits。

## 2026-10-17 · W7 · Track F9 · 處理 AF-039（formal 統計表加存檔欄 + 同存檔 family）

### 1. 當前目標 (Objective)
* 回應稽核 `AUD-20261017-01` 新開的 `AF-039`（MINOR/EVIDENCE/OPEN）：`table_statistics_formal.csv` 僅含 mixed deployment 配對檢定且無存檔欄，單看會把「S2−S0 +7.68 dB」誤讀成加密增益；且「加密本身成本（同存檔 S2−S0 ≈+0.5 dB）」缺配對顯著性背書。全部用已凍結資料處理，不重跑 GPU。

### 2. 執行動作 (Actions Taken)
* 改 `scripts/f9_statistics.py`：
  - 每列新增 `method_storage`／`baseline_storage`（自 long 表 `storage` 欄讀出，不手填）、`family`／`comparison_kind`／`holm_family_size` 欄，使每列自我說明。
  - 新增第二個 family `same_storage`（comparison_kind=`encryption_only`）：S2 取私有保值域臂（group `m1`/`m1lat`，`range_preserving_private_range_v2`），P/S0/S1 取全 range 臂（group `m1range`/`m1rangelat`，`range_preserving_public_range_v1`），兩者皆保值域 → 隔離加密本身成本。新增 `group_for()`／`storage_for()`，`paired_values()` 改成可分別指定 method/baseline 的 group。
  - **兩 family 各自獨立 Holm 校正，絕不合併**（依 AF-039 §修正建議）。`same_storage` 需全 range 臂（`m1range`/`m1rangelat`）存在才產生，否則 skip 並於 summary 記 `families_skipped`。summary schema→v2，note 說明兩 family 用途。
* 改 `scripts/build_claim_matrix.py`：`_paired()` 加 `family` 參數（因同一 metric/method/baseline 現有兩列）；RQ4 headline 同時引【部署對照，含存檔差異】與【加密-only，同為保值域】兩個 S2−S0，並明寫 +7.7 dB 幾乎全來自存檔協定；index.md 說明段補兩 family 定義與存檔欄。
* 指令（絕對路徑避開 relative_to bug）：
  - `scripts/run_cfg_ddim.sh python -m scripts.f9_statistics --long-csv /data2/paper/paper_assets/q8_formal/per_sample_long.csv --manifest /data2/paper/paper_assets/q8_formal/analysis_manifest.json --out /data2/paper/paper_assets/tables/table_statistics_formal.csv --json-out /data2/paper/paper_assets/tables/table_statistics_formal.json --force`
  - `scripts/run_cfg_ddim.sh python -m scripts.build_claim_matrix --force`

### 3. 執行結果 (Results & Observations)
* [事實] `table_statistics_formal.csv` 現 70 列：`mixed_deployment`=35、`same_storage`=35，每 family 獨立 Holm。role=formal。
* [事實] `mixed_deployment`（S2 private-range vs P/S0/S1 **legacy**）主數字不變：S2−S0 psnr 中位 +7.682、S2−S1 +7.734 dB，r=1.0、Holm p≈5e-33（與凍結前一致，向後相容）。
* [事實] `same_storage`（全保值域，加密-only）：psnr S2−S0 中位 **+0.301**（mean +0.536）dB、r=0.742、Holm p=1.4e-18（顯著但小效果）；**S2−S1 中位 +0.035 dB、p_holm=0.296 不顯著**（加密下 S2≈S1 品質）；S2−P 中位 +1.08 dB。SSIM/LPIPS/L∞ 多為 honest null（p_holm→1），即加密本身對這些指標幾乎無成本。來源同表。
* [事實] 稽核自算同存檔 S2−S0 mean=+0.536 dB，與本輪一致。
* [事實] claim matrix RQ4 headline 已區分：「【部署對照，含存檔差異】S2−S0 +7.68 dB」vs「【加密-only，同為保值域】S2−S0 +0.30 dB」，10 假設仍全 supported、0 pending。

### 4. 達標判定 (Assessment)
* AF-039 三項驗收條件全達成：(1) 統計表每列標存檔 ✓；(2) 同存檔配對檢定存在且 Holm 另成 family ✓；(3) claim matrix RQ4 區分「部署對照(含存檔)」與「加密-only」✓。
* [事實] 本修正僅用已凍結 `q8_formal/per_sample_long.csv`（hash 已釘於 analysis_manifest.json），未動 GPU、未改任何 run artifact；mixed 主數字逐格不變。
* [回覆稽核] AF-039 的 EVIDENCE 補強已完成，請稽核方複驗 `table_statistics_formal.csv` 兩 family 與 claim matrix RQ4；本輪不碰 BDIA 附錄。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 根因是 F9.4 原設計只輸出部署對照一個 family，雖 PROGRESS/index 有文字限縮，但統計表自身不自我說明。加存檔欄 + 同存檔 family 後，「加密幾乎不損品質」有了配對統計背書，審稿人不易誤讀。
* [注意] same_storage 的 S2 臂為 `private_range_v2`、基準為 `public_range_v1`，差別僅在 lo/hi 是否加密入 AEAD metadata；兩者皆保值域，故此對照隔離的是「結構變換+容器」成本，非存檔。此點已寫入 f9_statistics docstring 與 note。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。AF-039 為論文定稿前 EVIDENCE 補強，稽核明示不阻擋 G5；本輪依其修正建議以已凍結資料完成。

### 7. 下一步
* 等稽核複驗 AF-039 是否可轉 CLOSED。
* 週末/G5 其餘步驟（build_weekly_appendix --week 7、PROGRESS_summary W7 段）與 BDIA 附錄 GPU run 維持原狀，未受本輪影響。
* 使用者手動 push 本地 commits。

## 2026-10-17 · W7 · Track Q10 · 誤差歸因、數值精度、AEAD-ref 與 M1/M2 對照（Q10.1–Q10.4）

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` Q10（誤差歸因、M1/M2 對照與緩衝）。大目標：正確解讀前作 0.9989，不把 pixel bit-exact 當 structured transform 的必要條件。本輪完成 Q10.1–Q10.4（全 CPU，由已凍結 q8_formal 陣列 + 確定性 crypto 組裝，無新 GPU）。Q10.5（optional partial-key）暫略。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/q10_numeric_precision.py`（Q10.2）：transform-only round-trip 精度掃描，d∈{4096,65536}×dtype∈{float32,float64}×rounds∈{1,2,4}×perm∈{F,T}=24 組，每組 16 樣本；指標 MaxAbs/RMSE/cosine_min/bit-exact/forward_norm_dev。用 `d7_cpu_security.FIXTURE_KEY`/`nonce_for`（公開重現 fixture，非金鑰材料）、seed 1911；僅寫 aggregate 指標，不寫金鑰。→ `paper_assets/tables/table_numeric_precision.csv`(+json)。
* 新增 `scripts/q10_aead_reference.py`（Q10.4）：以**部署同原語** ChaCha20-Poly1305（HKDF 子金鑰 `PRIVATE_METADATA_LABEL`）對 6 種 raw payload（random 32/4096 B、latent d4096/d65536 × float32/float64）做 round-trip + 5 類竄改（ciphertext/tag/nonce/key/aad）。→ `table_aead_reference.csv`(+json)。
* 新增 `scripts/q10_error_attribution.py`（Q10.1）：**從凍結 `q8_formal/table_fidelity.csv` 讀**（不手抄）組裝誤差階梯：stage1 crypto_transform_only → stage2 m2_exact_latent → stage3 m1_sampler_range → stage4 m1_deploy_legacy → stage5 conditioning（deferred_gpu，交叉引用 BDIA 附錄，不杜撰）。→ `table_error_attribution.csv`(+json)。
* 新增 `reports/deployment_modes.md`（Q10.3）：M1 image-only vs M2 latent-payload 的 fidelity/security/storage 對照，數字引自 q8_formal `table_fidelity`/`table_storage` + Q10.4。
* 三個 CPU 腳本的最後 print 加 `_rel()`（out 路徑在 repo 外時不崩潰，檔案本已先寫出）。
* 指令（絕對路徑）：`scripts/run_cfg_ddim.sh python -m scripts.q10_numeric_precision|q10_aead_reference|q10_error_attribution --out <table> --json-out <json> --force`。

### 3. 執行結果 (Results & Observations)
* [事實] Q10.2：transform-only round-trip float32 worst max_abs=1.55e-6、float64=3.05e-15、cosine_min 全 1.000000000、forward_norm_dev≤8e-7（正交 ‖Qz‖=‖z‖）。bit-exact rate ~0.09–0.20 為浮點捨入、非缺陷。→ **結構變換到浮點精度無損**，0.9989 不是 transform 性質。
* [事實] Q10.4：6 payload 全 100% byte-exact 往返；5 類竄改（ciphertext/tag/nonce/key/aad）對每個 payload 全部 fail-closed。→ M2 latent-payload 的 bit-exact 可逆 + 竄改拒絕下限。
* [事實] Q10.1 誤差階梯（PSNR vs x0）：crypto 無損（latent cosine 1.0）；M2 精確 latent 天花板 41.87 dB（全方法相同，殘差=VAE+8bit）；sampler 反轉 drop S0 −3.59／S1 −3.10／S2 −3.05 dB（honest 主殘差，latent cosine ~0.9998）；legacy 存檔 drop S0 −7.14／S1 −7.35 dB（純存檔損失），S2 保值域 not_applicable。
* [事實] Q10.3 儲存：M1 total S2 74,526 B（2.26× P）；M2 total S2 350,565 B（含 private_metadata 65＋tag 32＋nonce 16 B）。M1≈M2 的 1/4.7。
* [事實] 三腳本重跑 bit-identical（scratchpad 比對 np/aead/ea 全 IDENTICAL）→ 確定性通過。

### 4. 達標判定 (Assessment)
* [x] Q10.1 table_error_attribution.csv、[x] Q10.2 table_numeric_precision.csv、[x] Q10.3 deployment_modes.md、[x] Q10.4 table_aead_reference.csv 皆完成並確定性驗證。
* [ ] Q10.5 optional partial-key（derived-key Hamming / ciphertext correlation）：WORKFLOW 標「有餘裕才跑」，本輪暫略。
* [限制] 所有 magnitude/precision/AEAD 結果為指定量測，非通用安全證明；conditioning 列為 deferred（GPU BDIA 附錄）。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* Q10 核心結論（可直接入論文）：**加密層（transform+AEAD）可逆性是精確的（float 精度 / bit-exact）；M1 影像可逆缺口全來自 sampler 反轉 + 8-bit 量化，大部署差則來自 legacy 存檔**。前作 0.9989 屬影像空間殘差，不可寫成 latent 一致性 / 零失真 / 無損（呼應 `reports/metric_semantics.md`）。
* conditioning 軸（y=0 條件 vs 無條件 null）對 M1 保真的影響，若使用者對 BDIA 附錄下 GPU go 再補正式數字並回填 stage5。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。WORKFLOW §Q10「不加入新 sampler 或新資料集」已遵守；「宣稱原始 bytes bit-exact 只能引用 AEAD-ref」已於 deployment_modes.md 落實。

### 7. 下一步
* 稽核方可 spec-first 複驗：Q10.2 自寫 transform round-trip、Q10.4 自寫 AEAD、Q10.1 從 table_fidelity 重算階梯、Q10.3 核對引用。
* Q10.5 視餘裕再做；conditioning 正式數字待 BDIA 附錄 GPU go。
* 使用者手動 push 本地 commits。

## 2026-10-17 · W7 · Track Q10 · Q10.5 金鑰雪崩診斷（optional partial-key）

### 1. 當前目標 (Objective)
* 補 `WORKFLOW.md` Q10.5（optional partial-key）：derived-key Hamming distance 與 ciphertext correlation。使用者 2026-10-17 指示補上。全 CPU、不載入 checkpoint、確定性。定位為 KDF 擴散的**診斷**，非通用安全證明，不以金鑰空間作安全論證（skill §8）。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/q10_partial_key.py`：翻轉 32-byte master key 的每一個 bit（256 次），量測 (a) 衍生 sign-subkey 的 bit Hamming fraction、sign-pre/post flip fraction、permutation change fraction；(b) 固定 latent（d=65536、rounds=1、use_permutation=True、N=8、seed 1911）在原金鑰 vs 翻 1-bit 金鑰下的 ciphertext |cosine|。含同金鑰正控制。用 `d7_cpu_security.FIXTURE_KEY`/`nonce_for`（公開 fixture，非金鑰材料）；僅寫 Hamming/correlation aggregate，不寫金鑰 bytes。
* 指令：`scripts/run_cfg_ddim.sh python -m scripts.q10_partial_key --out paper_assets/tables/table_partial_key.csv --json-out paper_assets/tables/table_partial_key.json --figure image/Q10.5_partial_key_avalanche.png --force`。
* 產物：`paper_assets/tables/table_partial_key.csv`(+json)、`image/Q10.5_partial_key_avalanche.png`（兩直方圖：Hamming、ciphertext |cos|）。

### 3. 執行結果 (Results & Observations)
* [事實] 單 bit 金鑰翻轉的雪崩：derived sign-subkey Hamming mean **0.505**（min 0.426/max 0.613，理想 0.5）；sign-pre flip mean **0.50001**、sign-post **0.50001**；permutation change mean **~1.0000**（幾乎全位置改變）。
* [事實] ciphertext |cosine|（原金鑰 vs 翻 1-bit）mean **3.12e-3**、max over flips **1.36e-2**（趨近 0，近乎無相關）。同金鑰正控制 cosine **1.0**（sanity 通過）。
* [事實] 執行時間 2m39s wall（CPU，≤30 分）。
* [限制] 此為 KDF 擴散診斷，非通用安全證明；只刻畫 S2/S2a 的金鑰排程。

### 4. 達標判定 (Assessment)
* [x] Q10.5 完成（table_partial_key.csv + figure）。至此 Q10.1–Q10.5 全部完成。
* 確定性：重跑 CSV 比對（scratchpad）—見下一步/commit 說明。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 結論：KDF（HKDF→ChaCha20 sign/perm）對 master key 具良好雪崩，單 bit 改動即約 50% sign bit 翻轉、密文近乎無相關。支持「金鑰材料與影像無關且良好擴散」的構造論述（與 D5 per-image nonce、CSPRNG 金鑰一致），但不作通用安全宣稱。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。Q10.5 原為 optional，依使用者指示補上。

### 7. 下一步
* 稽核可 spec-first 複驗（自寫 bit-flip Hamming 與 ciphertext cosine）。
* conditioning 正式數字仍待 BDIA 附錄 GPU go。
* 使用者手動 push 本地 commits。

## 2026-10-19 · W7 · Track Q11 · 下游病灶定位 parity demo（S0/S1/S2 換密碼透明性）+ contribution.md

### 1. 當前目標 (Objective)
* 使用者 2026-10-19 指示：(a) 將「by-construction 下游定位保留」與本專案全部貢獻整理成 `paper_assets/contribution.md`；(b) 跑 GPU 實驗，在 CheXpert 影像上讓 S0/S1/S2（均 fixed-norm + range-preserving）各跑完整像素空間 M1 流程（加密→解密→CFG-DDIM 偽健康→heatmap），比較三者還原圖/偽健康圖/heatmap 差異，驗證「更換加密流程不影響下游病灶定位」。
* 背景：前作 11 張醫師標註外部測試集（台北關渡醫院 Dr. Wei-Hsin Yuan 提供，pleural effusion + 專家 bbox）原始影像與座標不在 repo（僅論文縮圖 `figures/experiments/external_test_set_preview_with_mask.png`），醫師合作已結束 → 無法重跑量化定位 benchmark；且前作定位評估本身為質化 diff map（`05_experiments.tex` §5.4）。故改以 by-construction 論證 + parity demo。

### 2. 執行動作 (Actions Taken)
* 新增 `paper_assets/contribution.md`：整合 A（證偽 0.9989=無損 / 雙空間報告）、B（T1–T4 系統化密碼分析）、C（S2/S2a 有原則修補）、D（誠實分層統計，分離 cipher 與存檔增益）、E（下游定位 by-construction 保留）。E 段依使用者釐清修正：CFG 接「還原影像」非還原 latent（pixel-space diffusion、`ddim_sample_loop_anomaly_detection_with_deanonymization` Step 2 `ddim_sample_loop_known(img=(x_rec,))`）；M2 恆等、M1 等價到 sampler 殘差（不宣稱逐 byte 相同）。
* 新增 `scripts/q11_localization_parity.py`：唯讀 import 前作 `guided_diffusion`（`create_model_and_diffusion` + checkpoint `modelchexpert050000.pt`，114M params），用兩個 helper `ddim_anonymization_forward`/`ddim_anonymization_backward`（noise_level=500、CFG gs=4.0、eta=0）自建 M1 全流程，在 x_T（65536=2^16）注入密碼。三密碼共用 fixed-norm(C=248.261，私存 per-image norm 無損還原)+range-preserving(in-memory float，不做 legacy min-max) 包絡，唯一變因為正交核心：S0 sign、S1 signed-perm、S2 `src.crypto.sot`（Hadamard-Rademacher, rounds=1, perm）。輸入對上模型 `visualize()` 的 **[0,1]** 範圍（非 [-1,1]）。密碼鑰由 `d7_cpu_security.FIXTURE_KEY`/`nonce_for`（公開 fixture）；只寫影像 metric 與渲染圖，不寫金鑰 bytes。
* 選圖：`train_visualCheXbert.csv` 篩 Frontal + Pleural Effusion==1，跨 4 個 batch 資料夾解析實體路徑，取前 10 張。
* 指令：smoke `scripts/run_cfg_ddim.sh python -m scripts.q11_localization_parity --n 1`；正式 `... --n 10 --force`。產物：`paper_assets/localization_parity/<img>/`（original/encrypted/restored/healthy/heatmap PNG）、`paper_assets/tables/table_localization_parity.csv`(+json)、`image/localization_parity_grid.png`。

### 3. 執行結果 (Results & Observations)
* [事實] smoke（1 張）163s；正式 N=10 共 1626s（162s/張）。recovery vs 原圖 mean PSNR：S0 39.1、S1 38.1、S2 38.0 dB（SSIM ~0.965，同量級，確認 [0,1] 範圍正確、還原圖為正常 CXR）。
* [事實] cross-method parity（N=10，mean PSNR dB / mean SSIM）：
  - 還原圖：S2–S0 **46.0 / 0.996**、S2–S1 51.8 / 0.9995、S0–S1 46.4 / 0.996。
  - 偽健康圖：S2–S0 **43.6 / 0.995**、S2–S1 50.7 / 0.999、S0–S1 43.5 / 0.995。
  - heatmap：S2–S0 **34.7 / 0.890**、S2–S1 40.4 / 0.952、S0–S1 36.6 / 0.959。
* [事實] 目視（10 張 grid）：還原圖與偽健康圖三法逐列幾乎無法區分；heatmap 多數列一致，少數列（如第 6、7 列）熱點**位置相同、強度**在 S0/S1/S2 間有差異。
* [結論] **S2–S0 的下游差異與前作自身 S0–S1 同一量級**（restored 46.0 vs 46.4、healthy 43.6 vs 43.5、heatmap 34.7 vs 36.6 dB）→ 引入 S2 未比前作兩方案之間引入更多下游變動。heatmap 一致性數字偏低係 |還原−偽健康| 差分放大 + per-image min-max 正規化所致，非定位區域不同。

### 4. 達標判定 (Assessment)
* [x] contribution.md + Q11 parity demo 完成，實證支持 E 段「更換加密流程對下游定位透明、等價到既有 sampler 殘差帶、不劣化」。
* [限制] 非定位準確度 benchmark（無 ground-truth mask）；只能宣稱「保留/不劣化」，不能宣稱 S2 改善定位。
* [注意] 確定性：pipeline 設計為確定性（eta=0、seed 1911），但此為 GPU 浮點實驗，**不保證跨次 bit-identical**（有別於 Q10 的 CPU 任務）；數字為單次 N=10 量測，趨勢穩健。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 殘差來源：M1 把 DDIM invert/regen 夾在加密外層跑兩次，三密碼的加密影像不同 → 還原圖有 sampler 殘差級別差異（非恆等），與 Q10.1 歸因一致。此差異落在前作 S0/S1 之間既有的變動帶內。
* 可選加強：若未來取得定位 ground-truth，可補量化 IoU/Dice；目前資料不可得，於 limitations 誠實交代。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* Q11 為使用者 2026-10-19 當面指示的 appendix 式 utility 示範，**不在 WORKFLOW.md 既列子任務內**（供稽核方知悉）。判定非研究設計變更（未改 method of record、未換資料集——CheXpert 為既有專案資料集、未改方法），比照 skill「appendix ablation 不需回 Cowork」。未修改前作原始碼（`past/SourceCode/` 僅唯讀 import）。

### 7. 下一步
* 待使用者裁決是否 commit（`contribution.md` + `scripts/q11_localization_parity.py` + `paper_assets/tables/table_localization_parity.*` + `image/localization_parity_grid.png` + 本 PROGRESS 條目；per-image PNG 目錄體積較大，commit 前以 `scripts/check_staged_files.sh` 檢視是否納入）。
* conditioning 正式數字仍待 BDIA 附錄 GPU go。
* 使用者手動 push 本地 commits。

## 2026-10-19 · W8 · Track Q · 回應 AUD-20261019-01：AF-040 精度註記更正

### 1. 當前目標 (Objective)
* 回應稽核：AUD-20261019-01 的 AF-040（MINOR／REPRO／Q10.2）。
* 核心問題：`table_numeric_precision.json` 的 `note` 把 transform-only round-trip 誤差寫成 float32「~1e-4」、float64「~1e-12」，與同檔計算欄位及 CSV（float32 1.55e-6、float64 3.05e-15）矛盾，誇大約 60–1000×。順帶修稽核方指出（非 finding）的 `table_error_attribution` stage-1 註記：P/S0/S1 實為 bit-exact（max_abs=0.0，純 sign/perm），只有 S2（WHT）為 ~8.3e-7。

### 2. 執行動作 (Actions Taken)
* [AF-040] 已修正｜修改 `scripts/q10_numeric_precision.py::main`：`note` 改為由計算出的 `f32_worst`／`f64_worst` 以 f-string 格式化（`~{:.1e}`／`~{:.0e}`），prose 不再能與資料漂移；並明載「這些值與上方 roundtrip_max_abs_worst 欄位一致」。
* 順帶精確化（非 finding，稽核方 §AF-040 建議）：`scripts/q10_error_attribution.py::ladder` stage-1 attribution 依該列實際 `t_max` 分流——`==0.0` → 「bit-exact (max_abs=0.0, pure sign/perm)」；否則 → 「lossless (max_abs~8.3e-07, WHT core)」；docstring 第 1 階說明同步精確化。
* 以 `scripts/run_cfg_ddim.sh python -m scripts.q10_numeric_precision --force` 與 `... scripts.q10_error_attribution --force` 重新產生 `paper_assets/tables/table_numeric_precision.{csv,json}`、`table_error_attribution.csv`（均經 `output_guard` 覆寫保護）。

### 3. 執行結果 (Results & Observations)
* `table_numeric_precision.json` note 現為「float32 max_abs worst ~1.5e-06（d=65536, rounds=4）、float64 ~3e-15」，與計算欄位 `float32_roundtrip_max_abs_worst=1.5497e-06`、`float64_roundtrip_max_abs_worst=3.0531e-15` 一致；稽核方獨立重算 1.45e-6／2.90e-15（GPU 浮點容差內一致）。
* `table_error_attribution.csv` stage-1：P/S0/S1 → 「cipher round-trip bit-exact (max_abs=0.0, pure sign/perm)」（`latent_max_abs_vs_z=0.0`、`bit_exact_rate=1.0`）；S2 → 「lossless (max_abs~8.3e-07, WHT core)」（`latent_max_abs_vs_z=8.29e-07`）。無任何 CSV 數字或下游結論改變。
* 殘留 grep `~8e-7／~1e-4／~1e-12`：僅剩 CSV S2 列的正確值 8.3e-07。

### 4. 達標判定 (Assessment)
* [x] **已達標 (Achieved)**：JSON note 與自身計算欄位、CSV 一致；attribution 文字逐列精確。純註記／文字更正，不影響任何量化結論。
  - 下一步：原子 commit（`fix:` + `Refs: AF-040`），納入兩支 generator 腳本與其重生的 artifact。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* （已達標，略）根因：note 為早期手寫經驗值、未隨實測值更新。改為從計算值格式化後不會再漂移。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。未修改 `past/SourceCode/`、`AUDIT.md`；PROGRESS 只增不改。

### 7. 下一步
* 待稽核方下一輪複驗 AF-040（驗收：note 數量級與 CSV 一致）後關閉。
* conditioning 正式數字仍待 BDIA 附錄 GPU go。
* 使用者手動 push 本地 commits。

## 2026-10-20 · W8 · Track T2-BB · Inner-risk 匿名圖 linkage（pilot 定稿：前作透明但弱、S2 回隨機、學習式過擬合）

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md`：T2-BB（PROPOSAL §3，P2 選做／另表）。protocol：`reports/t2bb_inner_linkage_protocol.md`。
* 核心問題：多張匿名圖之間（anon↔anon）是否可連結到同一病患？前作 S0/S1 與我們的 S2 各如何？

### 2. 執行動作 (Actions Taken)
* split：`scripts/build_t2bb_inner_split.py`→`splits/t2bb_inner_v1.csv`（20 病患×2 victim）；`scripts/build_t2bb_shadow_split.py`→`splits/t2bb_shadow_v1.csv`（1000 病患×2 shadow，固定全域 permutation 前綴、與訓練/security/victim disjoint）。
* 反演快取：`scripts/t2bb_invert_cache.py`（file_sha256-keyed、prefix 可增量重用、gitignored）。背景 run 反演 victim 40 + shadow 前 200 = 440 latent。
* 步驟 1：`scripts/t2bb_step1_image_control.py`（影像層 positive control）。
* 步驟 2：`scripts/t2bb_inner_linkage.py`——新增 `cos`（原始帶符號 cosine，前作正交不變量的最強線性探針）scorer，並改為 cache-reuse（40 張全命中、CPU 0.1s）。
* 學習式：`scripts/t2bb_learned_attack.py`（SupCon metric net，shadow 訓練、victim 評估；channels s0_abs/s2_anon）。
* 指令（皆 `scripts/run_cfg_ddim.sh`）：`python -m scripts.t2bb_step1_image_control --split splits/t2bb_inner_v1.csv`；`python -m scripts.t2bb_invert_cache --split splits/t2bb_shadow_v1.csv --max-patients 200`；`python -m scripts.t2bb_inner_linkage --split splits/t2bb_inner_v1.csv --force`；`python -m scripts.t2bb_learned_attack --shadow-split splits/t2bb_shadow_v1.csv --max-patients 200 --victim-split splits/t2bb_inner_v1.csv`。

### 3. 執行結果 (Results & Observations)
* [事實] 步驟1 影像層：DenseNet121(ImageNet) Top-1=0.550、AUC=0.811、neg-ctrl 0.100 → 同病患 biometric 訊號在影像層存在。
* [事實] 步驟2 latent/anon（decrypt 正控制 S0/S1 max_abs=0.0、S2=9.5e-7）：
  - plaintext_latent cos/abs/norm AUC=0.659/0.657/0.680（Top-1 0.30/0.20/0.15）。
  - S0=S1 range_preserving cos AUC=0.657 ≈ plaintext cos → **透明性證實**（正交映射保內積）；minmax 降至 0.512。
  - S2 cos/abs 降至 ≈0.499（minmax）／0.433（range cos）→ **per-image nonce 回隨機**。
* [事實] 學習式 pilot(200)：train loss→0.0048 但 shadow-val AUC=0.48、victim AUC=0.468 < 固定 0.657，`efficacy=false`（過擬合；合成 smoke 曾達 Top-1=1.0 證非 trainer bug）。

### 4. 達標判定 (Assessment)
* [x] **已達標**：三結論成立——(i) 前作對 inner-risk 透明但訊號弱（AUC≈0.66，cos(anon)=cos(plaintext)）；(ii) 最弱～學習式攻擊者皆無法顯著連結（Top-1≤~0.3、AUC≤~0.68，學習式過擬合）；(iii) S2 by-construction 回隨機。
* [決定] 依 §10 決策點**結案**；不補跑 1000（~3.5h GPU，期望值低：cos≈abs≈norm≈0.66 上界 + 學習式過擬合皆指向 latent linkage 本就弱）。
* artifact：`table_t2bb_step1_image_control`、`table_t2bb_inner_linkage`、`table_t2bb_learned_attack`（各 csv+json）；protocol §15 定稿。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* [限制] N=20(victim)/200(shadow pilot) 小；記憶化/訓練集重疊以 disjoint split 控制（manifest 記錄）。結論限於固定統計 + pilot 學習式攻擊者，非通用安全證明。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。

### 7. 下一步
* 整理原子化 commit（scripts/splits/tables/protocol/PROGRESS；latent 快取 gitignored 不入庫）。
* BDIA 2×2 附錄 GPU run 仍待使用者 go（與本 track 無相依）。

## 2026-10-20 · W8 · Track T2-BB-M1-Frozen · 實際公開匿名圖的黑箱 linkage pilot

### 1. 當前目標 (Objective)
* 補強既有 T2-BB：舊 `t2bb_inner_linkage.py` 直接把 encrypted latent 正規化後評分，能回答正交結構透明性，但不是攻擊者實際拿到的 DDIM-generated M1。新目標是以真正公開 PNG、無 checkpoint／key／latent／原圖的 frozen encoder，驗證跨 study anon↔anon 同病患 linkage。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/t2bb_m1_generate.py`：沿用 `t2bb_inner_v1`（20 病患×gallery/query）與既有 40 latent cache，生成 P/S0/S1/S2 共 160 張真正 M1。P/S0/S1=predecessor legacy range-discarding PNG；S2=fixed-norm+per-image random nonce+SOT R=1/no permutation，公開 private-range container 的 q/255。
* 新增 `scripts/t2bb_m1_frozen_linkage.py`：只讀 public PNG，以 frozen ImageNet DenseNet121 與 CXR RAD-DINO embedding 算 cosine retrieval；報 Top-1/Top-5/AUC、fixed-gallery conditional-query bootstrap CI、5000 次 label permutation null/p-value。
* 攻擊/真值硬隔離：public manifest v2 改 opaque 隨機 ID/檔名並打亂順序，不含 sample/patient/pair/role；private truth map 位於 export 外，embedding/similarity 完成後才 join。加入 SHA containment、tamper、truth-leak、path-traversal 負控制；`--checkpoint` 現在綁定實際 model loader。
* 安裝 `transformers==4.46.3` 於 CFG_DDIM，RAD-DINO resolved revision `110cbc...`（weights SHA `dbfb9f...`）；完整版本與 SHA 寫入 `artifacts/t2bb_m1_environment.txt`，scoring 強制 `local_files_only=True`。
* 2-patient smoke：16 generation + 兩 encoder通過。正式 run：`T2BB_M1_GEN_20261020T023232092216Z_5340a90f`（PID 2636751，已退出），160/160、1257 s、managed state=succeeded/exit 0；之後把 v1 truth-bearing manifest 無損遷移成 v2（PNG bytes 不變）。
* 結果：`paper_assets/tables/table_t2bb_m1_frozen.{csv,json}`。單元測試 `tests/unit/test_t2bb_m1_frozen_linkage.py` 4 passed；另含 q8/preprocess regression 共至少 5 passed。

### 3. 執行結果 (Results & Observations)
* [事實] P positive control：DenseNet Top-1=0.50/AUC=0.810；RAD-DINO Top-1=0.75/AUC=0.976；兩者 permutation p=0.0002。→ DDIM-generated、legacy-stored P M1 確實保留足以 linkage 的病患訊號，null 有可解讀分母。
* [事實] S0：DenseNet 0.00/0.434（AUC p=0.940），RAD-DINO 0.10/0.532（p=0.161）。S1：0.05/0.479（p=0.642）、0.05/0.498（p=0.512）。S2：0.10/0.558（p=0.098）、0.00/0.510（p=0.374）。random Top-1=0.05；所有 encrypted cells 未顯著。
* [事實] public export 160 artifacts 經 SHA 驗證，manifest `f8ee6b...`；禁止欄位集合為空。結果 8 列皆 finite 且落在 [0,1]。
* [觀察] managed wrapper 的 status progress 未 ingest 子程序 `progress.json`，執行中維持 0/null，但 stdout 有逐張 1–160，最終 state/exit/manifest/PNG 數量均成功。初始 Visdom localhost 拒絕為前作 import side effect、非 run failure；新版 generator 已用 headless no-op 避免後續連線雜訊。

### 4. 達標判定 (Assessment)
* [x] **pilot 達標**：actual-M1 positive control 強，frozen-encoder BB attack 對 S0/S1/S2 均回到近隨機且不顯著。
* [修正舊結論] 不能再把 direct-latent 的 `cos` 透明性直接寫成「公開 S0/S1 圖可被 BB 顯著連結」。正確分層是：S0/S1 對可接觸 latent 統計的介面透明；在目前 PNG-only frozen-encoder T2-BB 下未成功。
* [限制] N=20、單一隨機 S2 key、兩個 frozen encoders，屬 exploratory pilot；null 不是通用安全證明，不能宣稱 S2 顯著優於 S0/S1。S2 DenseNet AUC=0.558/p=0.098 需誠實保留。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* direct-latent 與 actual-M1 結論不同的主因是公開介面不同：DDIM generation + 8-bit per-image public view 破壞了可直接利用的 latent 幾何，而 P control 證明不是 encoder 完全失效。
* 若要繼續提高攻擊上界，優先做 shadow-trained **image-space** encoder（輸入必須是 actual public M1），或擴大 victim/key repetitions；既有 latent SupCon 不應直接當作此介面的上界。任何預估 >30 分鐘 GPU 工作仍先回使用者裁決。

### 6. ⚠ 與 WORKFLOW.md 不符
* 無。T2-BB 仍為 P2 選做／另表；未改主方法、正式 N=200 主結果或 frozen claims。這一輪是修正既有 T2-BB 測量介面，結果不併入主表。

### 7. 下一步
* 交由稽核複驗 public/private truth isolation、checkpoint binding、encoder provenance 與 `table_t2bb_m1_frozen` 重算。
* 由使用者裁決是否把 T2-BB 往 shadow-trained actual-M1 image encoder／多 key／更大 victim N 擴展；目前 pilot 可作誠實的附錄結果。
* 無背景程序仍在執行；正式 PID 2636751 已退出，smoke/analyzer 也都完成。

## 2026-10-20 · W7 · Track 方法比較 · 外部 re-id 攻擊者（Packhäuser 架構重寫）+ 定位表

### 1. 當前目標 (Objective)
* 使用者 2026-10-20 指示：補強方法比較（目前只對前作 S0/S1）。找醫學匿名化相關論文，試著實作或借用數據比較我們方法好壞。選定 (1) 定位表（借發表數字）+ (2) 借外部 re-id 攻擊者；先查可行性。
* 交付：`reports/method_comparison.md`（定位表）、`scripts/reid_external_attacker.py` + `table_reid_external`（外部攻擊者）。

### 2. 執行動作 (Actions Taken)
* 文獻盤點：refpaper/ 已有關鍵 CXR 基線 112（PriCheXy-Net, MICCAI 2023）、111（CXR-Patient-ReID, Sci Rep 2022）、114、RiDDLE、G2Face；web 補 secret-key diffusion（arXiv 2510.01031, 2025）、CheXGenBench（2505.10496）。可行性：兩 repo 皆 **GPL-3.0**（不 vendor，改依論文重寫架構）；re-id 架構 = Siamese ResNet-50、256²、ImageNet norm、emb 128、BCE。
* `reports/method_comparison.md`：§0 誠實界線（可逆/金鑰 vs 不可逆去識別，不同威脅模型、不可單一數字比高低）、§1 九列定位總表、§2 讀表要點、§3 借外部攻擊者設計 + §3.1 結果。
* `scripts/reid_external_attacker.py`（license-clean 重寫，非他們 ChestX-ray14 checkpoint）：在 `splits/t2bb_shadow_v1.csv`（CheXpert 1000×2）訓練 Siamese ResNet-50（early-stop on held-out shadow pair-AUC），重用 `t2bb_inner_linkage.encrypt/attacker_view/evaluate` 與 victim 快取 latent 評 plaintext / S0/S1/S2 密文。指令：`scripts/run_cfg_ddim.sh python -m scripts.reid_external_attacker --max-patients 1000 --epochs 60 --patience 6`（背景 GPU，~12 分）。

### 3. 執行結果 (Results & Observations)
* [事實] 攻擊者學習有效：held-out shadow pair-verification AUC **0.911**（非過擬合，對照 T2-BB 學習式 0.48）。
* [事實] victim（N=20×2）AUC：plaintext **0.764**（舊 DenseNet 0.811、Packhäuser 報告 CheXpert 0.987）；S0 密文 0.616；S1 0.541；**S2 0.445 ≈ 隨機**。來源 `table_reid_external.{csv,json}`。
* [結論] **即使換成專用發表架構、且確實學會再識別的攻擊者，S2 匿名視圖仍降至隨機（0.445）、前作 S0 仍殘留洩漏（0.616）**。與 T2-BB（線性 cos/abs + 弱學習式）互補，此為影像級強學習式攻擊者。

### 4. 達標判定 (Assessment)
* [x] Opt 1 定位表、Opt 2 外部攻擊者皆完成；S2 擊退強攻擊者之證據入庫。
* [限制] GPU 浮點單次訓練、不保證跨次 bit-identical（同 Q11）；victim N=20 小；license-clean 重寫架構非原始 checkpoint；§1 外部數字部分取自摘要頁，定稿前再核對正文。
* [決定] Opt 3（CheXpert 重跑 PriCheXy-Net head-to-head）暫不做：數個 GPU-day、CP 值低、需謹慎框定威脅模型差異。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 學長論文（03_related_works §GAN-based Privacy Protection、02_background §Advantages of Diffusion）已明載不選 GAN 之由：GAN 不可逆（原始像素永久丟棄）、幻覺改變病灶、訓練不穩/mode collapse/記憶化；diffusion 的 DDIM 確定性反演才能精確還原。[注意] 匿名圖「像噪聲」源於釋放加密潛碼 x_T（by design 的安全特性），非「少了 GAN」。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。方法比較為使用者當面指示的補強；未修改 past/SourceCode/、未改主方法/資料集。GPU run 已取得使用者裁決。

### 7. 下一步
* §1 外部數字核對正文頁碼；（可選）Packhäuser 公開權重 out-of-tree 跨集 sanity。
* 使用者手動 push 本地 commits。

## 2026-10-21 · W7 · Track 收尾 · BDIA/DDIM 2×2 保真消融 + Packhäuser 原始權重跨集 sanity

### 1. 當前目標 (Objective)
* 使用者 2026-10-21 指示：(a) 同意 Packhäuser 公開權重 out-of-tree 跨集 sanity；(b) 仍要跑 6 小時 BDIA 實驗，雖對 PSNR 無大幅進步，但可寫入論文當做過的實驗、與 DDIM 做消融、供後續研究者參考。
* 交付：`paper_assets/tables/table_bdia_2x2.{csv,json}` + `scripts/bdia_2x2_table.py`（BDIA 2×2 凍結表）；`method_comparison.md` §3.2（原始權重 sanity，已 commit afd1821）。

### 2. 執行動作 (Actions Taken)
* **正文核對**（commit 2bf109f/4e04de7）：§1 外部數字對 refpaper 112 Table 1、111 摘要+Table 1/2 逐一核對；修 PriCheXy-Net 分類為 80.5→75.8%（µ=0.01 點）。
* **原始權重 sanity**：clone `kaipackhaeuser/CXR-Patient-ReID`（GPL-3.0）至**樹外** `/data2/external_gpl/`（不 `git add`）；權重為 Git LFS，經 raw 端點下載並核對 sha256=`331efaed…`（95.4 MB）。樹外 adapter `reid_original_weights_sanity.py` 重用我們的輸入建構/encrypt/attacker_view/evaluate，只換模型為他們原始 `SiameseNetwork`（embed=forward_once, head=fc_end），CPU 執行（不碰 GPU，不干擾 BDIA 計時）。
* **BDIA 2×2 GPU run**：CPU self-test 先過；`scripts/run_cfg_ddim.sh python -m scripts.bdia_2x2_runner --evidence-role formal --cells 2 3 4 --checkpoint …/modelchexpert050000.pt --limit 0`（N=200，random key，與 vLLM 共存 GPU peak ~0.8 GB）。run_id `BDIA2X2_20261020T182731Z_efbf6f85`，exit 0、無 failures.json、stderr 空、validated。`scripts/bdia_2x2_table.py` 聚合 cell 2/3/4 並從凍結 Q8_FORMAL fidelity 表注入 cell 1（reference=x0/z 消歧）。

### 3. 執行結果 (Results & Observations)
* [事實] **原始權重 sanity**（victim N=20×2，樹外 CPU）：plaintext Top-1 0.850/AUC **0.982**（≈發表跨集 0.987）；S0 0.560；S1 0.493；**S2 0.477 ≈ 隨機**。來源 `/data2/external_gpl/reid_original_weights_sanity.json`。
* [事實] **BDIA 2×2**（m1 vs x0 PSNR / m1lat-z cosine，N=200，NFE 全 **1996** cost-parity）：

  | | asymmetric（匿名條件） | symmetric-null |
  |---|---|---|
  | DDIM | cell1 38.82 / 0.999794（凍結 Q8） | cell2 39.90 / 0.999787 |
  | BDIA | cell3 39.17 / 0.999698 | cell4 40.17 / 0.999679 |

* [結論] **sampler 軸（BDIA−DDIM，同條件）**：m1 PSNR 僅 +0.35/+0.27 dB、latent cosine 不變（~0.9997）、NFE 相同 → **T=500 單發加密下 BDIA 的 exact-inversion 買不到實質改善，DDIM 已足夠**（乾淨的 negative/robustness 結果）。
* [結論] **conditioning 軸（symmetric-null−asymmetric）**：+1.08/+1.00 dB —— 這 ~1 dB 是**匿名化（asymmetric）條件的品質代價**，非免費增益；symmetric-null 不具匿名性，本表僅報保真/可逆，**不重新宣稱安全性**（安全邊界見 `bdia_security_verify.py`）。補回 Q10.1 stage-5 一直 deferred 的 conditioning 軸。
* [結論] 原始 checkpoint plaintext 0.982≈發表 0.987 → 坐實 (i) 我們 CheXpert 影像/前處理無誤、(ii) §3.1 reimpl 0.764 僅因只訓 1000 病患（架構/資料皆正確）。**即使原作者本尊近乎完美的權重，S2 仍被壓到隨機（0.477）**。

### 4. 達標判定 (Assessment)
* [x] BDIA 2×2 凍結表入庫、sanity 入庫、正文核對完成。
* [限制] BDIA 為 GPU 浮點單次量測、runtime/peak 非跨次 bit-identical（同 Q11）；sanity 為跨資料集套用（原權重訓於 ChestX-ray14）+ 我們 bicubic/grayscale 前處理（非其 bilinear/RGB 原協定），plaintext AUC 不必等於 0.987；N=20 小。
* [決定] 外部 GPL code/權重/adapter 全留樹外、不入庫、不散布 → 本 repo 保持 license-clean；論文僅引用+報告數字。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* BDIA 消融定位為「完整性/擋審稿/robustness」次要貢獻，非賣點：措辭須為「sampler 選擇對保真度影響有限」，不得寫成「BDIA 改善本方法」。
* [待辦]（可選）把 `table_bdia_2x2` 併入 Week-7 appendix（會改動已 commit 的 appendix MANIFEST，待使用者裁示）。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。BDIA 6h GPU run 已取得使用者裁決（CLAUDE.md §3）；未改 past/SourceCode/、未改主方法/資料集；GPL 外部工具樹外執行。

### 7. 下一步
* 收尾 P11/Z12；（可選）appendix 併入 BDIA 表。
* 使用者手動 push 本地 commits。

## 2026-10-22 · W7 · Track 稽核回應 · 修 AF-041 / AF-042（方法比較 null 的措辭與控制）

### 1. 當前目標 (Objective)
* 稽核方 AUD-20261022-01 開立 AF-041（MINOR, DOC）、AF-042（MINOR, METHOD），皆為 null 結果的**措辭／控制**問題（把介面-／表徵-特定的 null 寫得像 scheme-level 安全）；底層數字全部可重現且誠實。使用者 2026-10-22 指示修復。

### 2. 執行動作 (Actions Taken)
* **AF-041**（`reports/t2bb.md §6` headline）：改綁兩前提（(a) 通用 image-encoder 介面、(b) legacy range-discarding/minmax 儲存）；明示與 §5 minmax 列一致（S0/S1 minmax cos 0.512）**非推翻透明性**；保留 range_preserving `cos` 旁道仍洩漏 0.657（scheme-aware latent 攻擊者仍可連結）；註明正控制 P 0.976 只證 encoder 能讀 image-domain 特徵、非 scheme-level 安全。
* **AF-042**（`reports/method_comparison.md §3.1/§3.2`）：補一個 **identity-cipher latent-view 控制**隔離「密碼效果 vs 表徵 OOD」——樹外 adapter `reid_original_weights_sanity.py` 加 `plaintext_latent` 條件（未加密 latent 的同表徵 attacker-view），原始 checkpoint CPU 重跑。
  - 結果：plaintext 真實-CXR **0.982** → plaintext_latent（未加密、換 latent-view 表徵）**0.550** → S0 0.560 → S1 0.493 → **S2 0.477**。
  - 改寫：S2 安全論述錨在**同表徵 S0 基線**（0.560→0.477）而非 plaintext 真實-CXR 天花板；「擊退強攻擊者」軟化為「現成／發表級 CXR re-id 網路無法連結 S2 的 noise-like 密文」；§4 兩項標 [x]。

### 3. 執行結果 (Results & Observations)
* [事實] identity 控制坐實稽核方質疑：**0.982→0.550 的降幅大半來自 latent-view 對 CXR 網路 OOD（未加密即近隨機），非密碼**。密碼淨效果在同表徵內：plaintext_latent 0.550 ≈ S0 0.560（S0 透明正交變換）→ S2 0.477（per-image nonce 移掉殘留 ~0.06）。
* [結論] S2 相對前作 S0 的改善**在同表徵下成立且乾淨**；但不得宣稱「S2 擊退 plaintext 0.76/0.98 的強攻擊者」（跨表徵、混淆 OOD）。與 §5（S2 range cos 0.433、S0 0.657）內部一致。
* [事實] 來源 `/data2/external_gpl/reid_original_weights_sanity.json`（樹外 GPL 工具，code/權重/json 不入庫；文件僅報數字）。

### 4. 達標判定 (Assessment)
* [x] AF-041、AF-042 的措辭／控制修正皆入庫；數字未變（原本就正確），僅收斂宣稱範圍並補控制列。
* [限制] identity 控制為單次 CPU、N=20；跨資料集 + 我方前處理。BDIA 2×2（稽核無 finding）不動。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 教訓：null/安全類結論務必標清「在什麼介面、什麼表徵、什麼儲存下」成立；跨表徵的天花板不能當同一攻擊者的基線。後續 P11 claim-audit 一併套用此檢查。
* 待稽核方複驗 AF-041/042；Q11 parity、reinvert ablation、decrypt flow 文件尚待稽核（AUD-20261022-01 §5 [待辦]）。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。僅改文件措辭 + 補樹外控制；未改主方法/資料集、未改 past/SourceCode/、未動 F9/G5 凍結範圍。

### 7. 下一步
* 收尾 P11/Z12。
* 使用者手動 push 本地 commits。

## 2026-10-22 · W7/W8 · Track 收尾 · P11（重現/表圖/claim audit/freeze）+ Z12（測試/索引/交接）

### 1. 當前目標 (Objective)
* 使用者 2026-10-22 指示開始 P11/Z12 收尾；先確認 GPU 實驗範圍，P11.2 重跑深度經 AskUserQuestion 選**小切片 N=10 + 確定性驗證**。

### 2. 執行動作 (Actions Taken)
* **GPU 盤點**：P11/Z12 僅兩處碰 GPU 且皆非長 run——P11.2（N=10 ≤30min，已核准直接跑）與 Z12.1 的 3 支 `skipif(cuda)` 短測試（~2 分）。其餘全 CPU。
* **P11.1**（`d41e2b0`）：`scripts/p11_experiment_matrix.py` → `final_experiment_matrix.{csv,md}`，34 cells、每個有 run id/provenance、缺值原因；n 從實表讀取。
* **P11.2**（`bc87369`）：`scripts/p11_reproduce.py` 新 GPU run 反演 N=10，對 frozen E2.3 cache **10/10 bit-exact**（max_abs=0、cos=1）。
* **P11.3**：確認所有交付表圖由 build script 讀 frozen CSV 生成（`build_claim_matrix/weekly_appendix/attack/defense_package`、`bdia_2x2_table`、`p11_*`），無手抄。
* **P11.6**（`21c9b0b`）：`scripts/p11_claim_audit.py` 掃論文面向文件 30 處命中，20 同行否定/限縮、10 手檢皆合法 → `claim_audit_p11.md`，0 無證據 over-claim。
* **P11.7/Z12.4**（`c5a7078`）：`scripts/p11_checksums.py` → `SHA256SUMS`（335 檔）+ `freeze_manifest.json`（combined fingerprint、git HEAD、split hashes、env）。
* **Z12.1**：`pytest tests/` → **193 passed**（含 3 GPU 測試實跑）、0 fail；`reports/final_test_report.md`。（註：根目錄 `pytest` 會誤收 `past/SourceCode/scripts/test_*`，已改 scope 到 `tests/`。）
* **Z12.2**：`reports/results_index.md`——RQ1–4 + 4 個 P2 track + freeze 保證的對照索引。
* **Z12.3**：`reports/december_handoff.md`——thesis 章節 outline + 插圖位置 + 禁止事項。

### 3. 執行結果 (Results & Observations)
* [事實] 主表確定性輸入（E2.3 latent cache）**bit-exact 可重現**；193 測試全綠；0 無證據 over-claim；335 檔 checksum 凍結。
* [事實] results_index 已把 AF-041/AF-042 的限縮併入對應 track 的「限制」欄，確保論文引用時不漏。

### 4. 達標判定 (Assessment)
* [x] P11.1/P11.2/P11.3/P11.6/P11.7 完成；Z12.1/Z12.2/Z12.3/Z12.4(freeze manifest) 完成。
* [待辦] **P11.4**（每圖 caption：take-away+限制、PDF/SVG 300dpi）、**P11.5**（§5–§10 可引用草稿彙整成連續 notes）、威脅模型/pipeline 示意圖（12 月繪）。皆非阻擋。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 收尾階段守 WORKFLOW Week 8 禁令：不新增方法/資料集、不換 seed/sample、不覆蓋舊 run、不手改表格數字；定稿前重跑 `p11_checksums.py` 更新 fingerprint。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。P11.2 GPU 已取得使用者裁決（小切片）；未改 past/SourceCode/、未動凍結範圍。

### 7. 下一步
* P11.4 caption / P11.5 notes 彙整（可選，寫作期）；使用者手動 push。

## 2026-10-23 · W8 · Track P11 · P11.4 機械性向量匯出（PDF，無損）

### 1. 當前目標 (Objective)
* `WORKFLOW` 子任務：P11.4 的機械性部分（圖檔向量匯出）。使用者裁定 caption 文字與示意圖留 12 月寫作期、現在先做零風險的向量匯出。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/p11_figure_export.py`：自建最小 PDF、以 `FlateDecode` **無損**嵌入原始像素（規避 Pillow 8.4 PDF writer 的 `DCTDecode`/JPEG 有損再編碼），300 dpi、確定性（無時間戳）、可重跑；內含 zlib round-trip 斷言。
* 匯出 `paper_assets/q8_formal/fig_P8_visual_grid.pdf`（2560×1178 RGB，8.53×3.93 in）、`paper_assets/reinvert_ablation/grid_gs2.pdf`（1300×2600 L，4.33×8.67 in）。
* 重跑 `scripts/p11_checksums.py` → `SHA256SUMS`（337 檔）+ 更新 `freeze_manifest.json`（combined `325c4b62…`）。
* 更新 `reports/december_handoff.md` 插圖清單（標記 PDF 已出、localization 仍待合成、不做 SVG 的理由）。

### 3. 執行結果 (Results & Observations)
* [事實] 腳本內 zlib 解壓 == 原 PNG 像素（bit-exact），`_verify` 斷言通過；`pdfinfo` 頁面尺寸符合 300 dpi；`pdftoppm` 回render shape 完全一致（方向正確），mean 差 3.3/7.8 於 255 為 Poppler 邊緣重取樣，非嵌入有損。
* [結論] 兩張 grid 是真實 CXR 像素的點陣圖，PDF（點陣無損嵌入）才是正解；**不產 SVG**（只會 base64 包同一份點陣）。真正該向量化的是 §3/§4 示意圖，12 月繪。

### 4. 達標判定 (Assessment)
* [x] P11.4 向量匯出完成（兩張 grid PDF，bit-exact）。
* [待辦] P11.4 剩 caption 文字、localization 合成圖、§3/§4 示意圖；P11.5 notes 彙整。皆 12 月寫作期、非阻擋。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 守 Week 8 凍結：未改表格數字、未動既有 PNG（原圖保留）、未換 seed/sample；定稿前照常重跑 `p11_checksums.py`。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。純 CPU、無 GPU；未改 past/SourceCode/。

### 7. 下一步
* caption 文字 / P11.5 notes / 示意圖留 12 月跟正文同步寫；使用者手動 push。

## 2026-10-24 · W8 · Track P11 · 修 AF-043（exact-recovery 詞表缺口 + 一處影像端過度宣稱）

### 1. 當前目標 (Objective)
* 回應稽核 AUD-20261023-01 開立之 **AF-043**（MINOR, DOC）：P11.6 claim-audit 詞表漏「精確還原/exact-recover」，致 `method_comparison.md:10` 影像端 exact-recovery 過度宣稱漏網。

### 2. 執行動作 (Actions Taken)
* `scripts/p11_claim_audit.py`：BANNED 加入 `精[確準](還原|恢復)`、`exact(ly)?[- ]recover`；明列**不**納入 `bit-exact`（本專案正確區分詞、非過度宣稱）。
* `reports/method_comparison.md`：改 3 處影像端 exact-recovery 宣稱——:10（+:20 表格「可逆?」欄）、:35。一律改為「金鑰控制高保真還原（latent/crypto 精確 cos=1、max_abs≈8.3e-7；影像端 ~38–42 dB，非 bit-exact；逐 byte 需 M2 AEAD-ref）」；:35 另把「密文期零洩漏」改為「密文期不可連結（T2-BB ~chance）」。
* `paper_assets/claim_audit_p11.md`：更新 verdict（50 hits）、記錄 AF-043 修正、改以**逐項人工處置**為準（稽核指出同行否定 heuristic 兩方向皆會誤判，僅作排序）；新增 README:72（還原金鑰≠還原影像）、method_comparison:27（外部論文自述）、PROPOSAL:252（反例列舉）、results_index/本報告（meta）之處置列。
* 重跑 `scripts/p11_checksums.py` → `SHA256SUMS`（337 檔）+ `freeze_manifest.json`（combined `3f2d6836…`，HEAD=186df46）。

### 3. 執行結果 (Results & Observations)
* [事實] 擴充詞表重掃：影像端 exact-recovery 過度宣稱 **0 處**。剩 exact-recovery 命中 = README:72（KPA 還原**金鑰**）、method_comparison:27（外部 arXiv 2510.01031 自述）、claim_audit 報告自身 meta —— 皆合法。
* [事實] 修正與自身凍結證據一致：M2 exact-latent 天花板 41.87 dB、Q11 recovery ~38 dB，皆非影像 bit-exact；bit-exact 僅成立於 latent/crypto 層或 M2 AEAD-ref。
* [限制] `method_comparison.md` 在 reports/（草稿、非凍結範圍），不進 checksum；claim_audit_p11.md 在 paper_assets/ 已重算入 fingerprint。

### 4. 達標判定 (Assessment)
* [x] AF-043 兩項建議（(a) 詞表 (b) :10 改寫）皆完成並擴及 :20/:35 同類句；重掃 0 影像端過度宣稱。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 詞表缺口的教訓：claim-audit 的價值在「區分詞」而非一般褒義詞；已把專案最關鍵的 latent-vs-image 區分詞補上。自動 same-line 否定僅排序、不作判準。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。純 docs/scanner，未動凍結表數字、未改 past/SourceCode/、無 GPU。

### 7. 下一步
* 待稽核複驗 AF-043。caption 文字/P11.5/示意圖仍留 12 月寫作期；使用者手動 push。

## 2026-10-24 · W8 · Track P2 · latent 誤差「方向≠大小」demo（獨立重現 + 影像並排圖）

### 1. 當前目標 (Objective)
* 使用者要求產生 demo，驗證 `重要.md`「cos 不代表還原程度好壞」：同樣大小的 latent 誤差、方向不同，影像還原差天共地。

### 2. 執行動作 (Actions Taken)
* 新增 `scripts/latent_direction_demo.py`（GPU，確定性 seed 1911）：重用 `e2_legacy_wrapper.legacy_invert/legacy_generate`（guidance −1、T=500），對 dev_v1.1 idx {0,1,10,11} 四張，於 z 注入 4 種誤差（白雜訊小/大、S0 實際解密誤差、P 實際 re-inversion 誤差），報 latent cos/rmse 與影像 PSNR vs G(z)；**額外輸出實際生成影像並排圖** `paper_assets/latent_direction_demo/grid.png`（稽核版只有長條圖）。
* 是稽核診斷 `audit/e2_latent_direction_sensitivity.py` 的實作方獨立重現（未改稽核檔）。

### 3. 執行結果 (Results & Observations)
* [事實] 四條件 mean 與 `重要.md`／稽核圖 w01_fig03 **逐一吻合**：white_small cos 0.990/31.3 dB；S0_actual cos 0.991/31.8 dB；white_large cos 0.654/6.4 dB；**P_actual cos 0.275/43.9 dB**。
* [事實] 更強的個案：P_actual 於 idx 010/011 latent cos **0.04/0.01（近正交）**，影像仍 44.0/43.4 dB → latent 距離與影像品質可完全脫鉤。
* [事實] 範圍確認 x0 [0,1]、base [−0.006,1]，佐證稽核以 [0,1] 計 PSNR 的慣例正確。
* [結論] 同樣大小的誤差，隨機/金鑰打散方向（生成器敏感）→ 影像崩；P 的 re-inversion 方向（生成器不敏感）→ 影像近乎無損。可逆性必看影像端指標。

### 4. 達標判定 (Assessment)
* [x] demo 完成、數字重現稽核診斷、並排圖交付；確定性（seed 固定）。GPU <30 分、使用者指示授權。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
* 可作論文 fidelity 討論的插圖（latent cos 不可作可逆性指標的視覺證據）；若要寫進正文，另擬一段。

### 6. ⚠ 與 CLAUDE.md/WORKFLOW.md 不符
* 無。未改 past/SourceCode/ 與 audit/；GPU <30 分。

### 7. 下一步
* 視使用者是否要把此圖與論述併入論文 §7 fidelity 討論；使用者手動 push。
