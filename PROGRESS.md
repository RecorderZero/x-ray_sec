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
