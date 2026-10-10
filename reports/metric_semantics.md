# 指標語意：前作「Cosine Sim 0.9989」的實際意義與本專案的指標定義（E2.4／AF-022）

**重現指令**（CPU，不載入 checkpoint；須經 wrapper 才會通過環境守衛）：

```bash
cd /data2/paper && scripts/run_cfg_ddim.sh python -m scripts.e2_metric_semantics
```

* 腳本：`scripts/e2_metric_semantics.py`（seed 1911，確定性；獨立實作，未匯入或複製稽核方的 `audit/e2_legacy_cosine_semantics.py`，該檔的輸出只在 `cross_check` 區塊被讀取比對）
* 產物：`results/E2.4_metric_semantics.json`（`script_sha256 = 60afa8da…`、`split_sha256 = 7b3e77a8…`、`git_commit = a3a9a60`；腳本本身在產生時尚未 commit，以 `script_sha256` 為準）
* 資料：`splits/dev_v1.1.csv`，20 張、20 位不同病人（腳本檢查 `patient_id` 唯一），共 190 個不同病人配對
* 標記規則：`[事實]` 已讀過原始碼／實際量測，並附來源；`[推定]` 由事實推論，尚未直接驗證；`[假設]` 未驗證的前提。沒有直接證據的敘述不寫成 `[事實]`。

**摘要**

1. `[事實]` 前作評估程式的「Cosine Sim」計算的是**輸出影像（像素空間）**的 cosine，不是 latent 的 cosine（§1）；論文的 0.9989 確實由該程式產生則是 `[假設]`（§1.3）。
2. `[事實]` 在本專案 dev 的 20 張影像上，兩張**不同病人**的 X 光 cosine 中位數就有 0.884；同一張圖加雜訊到 PSNR 30.6 dB，cosine 為 0.9987（§2）。`[推定]` 在加性、與影像近似正交的誤差下（§2.3），0.9989 這個數字只對應「PSNR 約 30–32 dB」，沒有提供 PSNR 以外的資訊；誤差不屬此類時不成立（§3.4 有 cosine = 1 但只有 10.8 dB 的反例）。
3. `[事實]` cosine = 1 只代表兩個向量平行，不代表相等；本報告給出 cosine = 1 但 PSNR 只有 10.8 dB 的例子（§3）。
4. 論文用語不得把 0.9989 寫成 latent 一致性、零失真或無損（§4）。

---

## 1. 前作論文描述 vs 程式實作的差異

所有行號都已實際開檔核對（`past/` 唯讀）。

### 1.1 論文怎麼寫

| 說法 | 出處（已核對） |
|---|---|
| Cosine Similarity 是「解密後 latent 雜訊向量 $Z^{rec}$」與「原始 latent 雜訊向量 $Z$」在高維特徵空間的方向一致性 | `past/Documents/Thesis/原碼_Overleaf(Latex)/05_experiments.tex:212`；公式 `:214`；「Cosine Sim ≈ 1.0000 represents perfect reversibility」`:216` |
| 解密雜訊 $Z^{rec}_t$ 與原始雜訊 $Z_t$ 的相似度「stably remains at 1.0」 | `04_methodology.tex:144` |
| 12 個組合的 Cosine 皆 > 0.997，CFG-DDIM＋Signed Permutation 為 0.998978（MaxAbsError 0.71417） | `05_experiments.tex:292`、表格列 `:308` |
| 「0.9989」「zero-distortion lossless restoration」「defending against inversion attacks」 | `00_abstract.tex:3`（中文摘要）、`:10`（英文摘要）；`05_experiments.tex:390`；`06_conclusion.tex:7` |

* `[事實]` 摘要的 `PSNR 30.23 dB` 出自 Signed Permutation 表（`05_experiments.tex:275`，該列 SSIM 為 0.888），但同句的 `SSIM 0.912` 出自 Rademacher 表（`:246`，該列 PSNR 為 29.88）。摘要把兩個方案的數字併在一起。這不影響本報告結論，只是引用「30.23 dB／0.9989」時要知道兩者同屬 Signed Permutation。

### 1.2 程式實際算什麼

| 項目 | 程式實作 | 出處（已核對） |
|---|---|---|
| 被比較的量 | NPZ 的 `samples` 欄（輸出影像，形狀 N×1×256×256）；**沒有任何 latent** | `past/SourceCode/scripts/evaluation_metrics.py:119-137`（`load_samples_raw` 取 `samples`／`orgs`，不正規化）、`:140-202` |
| cosine 的算法 | 每張影像 `samples[i].flatten()`（65,536 維），`np.dot / (norm·norm)`，**不減均值**，逐張算完取平均 | `evaluation_metrics.py:175-192` |
| 比較哪兩份資料 | Base 的 `samples` 對加解密的 `samples`，以檔名配對（不是 x0 對 x_rec） | `:626-627`（呼叫 `match_samples_by_name`）、`:680-752`、`:637`（呼叫 `calculate_reversibility_metrics`）；`run_encryption_reversibility_eval_sample3.sh` 的 `--base_npz_paths`／`--signed_perm_npz_paths` |
| 縮放 | cosine、MAE、MSE、MaxAbs 用**原始** `samples`（`:167-190`）；PSNR、SSIM、FID 則先對每張影像做 min-max（`visualize`，`:158-164`） | `evaluation_metrics.py:64-68`（`visualize`）、`:158-190` |
| Base 的 `samples` | 標準病灶定位流程輸出：`sample, x_noisy, org = sample_fn(...)`，存成 `samples=arr` | `past/SourceCode/scripts/cfg_image_sample.py:152-154`、`:245` |
| 解密流程的 `samples` | `deanonymize` 分支回傳 `sample, x_noisy, x_rec`，其中 `sample` 是**第 3 步偽健康（CFG）重建**，`org = x_rec` 才是還原原圖；NPZ 的 `samples` 存 `sample`，`orgs` 存 `x_rec` | `cfg_image_sample_anonymization.py:221-236`（`:223` 呼叫、`:236` `org = x_rec`）、`:380-386`、`:428-429`；`guided_diffusion/gaussian_diffusion.py:1721-1752`（docstring 步驟 1–5 與 Returns）、`:1775-1788`（`return sample, x_noisy, x_rec`） |

* `[事實]` 該程式算出的 cosine 是「兩份輸出影像（像素空間）的 cosine」，**不是** latent cosine，也不是 x0 對 x_rec。論文的 0.9989 由此程式產生，則為 `[假設]`（見 §1.3）。
* `[事實]` 輸入資料以 min-max 縮放到 [0,1]（`past/SourceCode/guided_diffusion/bratsloader.py:216`，與本專案 `preprocess()` 的輸出範圍相同）；`clip_denoised` 只把 x0 預測夾在 [−1,1]（`gaussian_diffusion.py:337-342`）。
* `[推定]` 原始 `samples` 的數值大致落在 [0,1]（論文表中 MaxAbsError 0.676–0.928 也符合此尺度；uint8 尺度會到 255 等級），但前作 NPZ 不在本 repo（`find past -name '*.npz'` 無結果），**輸出實際值域無法直接驗證**。本報告的像素基準用 [0,1] float 計算。

### 1.3 這代表什麼

| 面向 | 論文敘述 | 程式實作 |
|---|---|---|
| 空間 | latent（雜訊向量） | 輸出影像（像素） |
| 比較對 | 解密 latent vs 原始 latent | Base 流程輸出 vs 加解密流程輸出 |
| 解密端的影像 | （未提） | `samples`＝偽健康重建；還原圖 `x_rec` 在 `orgs`，**沒有被拿來算 cosine** |
| 和 x0 的關係 | 暗示與原圖一致 | 完全沒有和 x0 比較 |

* `[假設]` 論文實際使用的 NPZ 正是上述程式路徑產生的。理由與限制：`run_encryption_reversibility_eval_sample3.sh` 引用的檔名（如 `samples_cfg_signed_perm_chexpert050000_3x1x256x256.npz`）和 `cfg_image_sample_anonymization.py:425` 的輸出命名（`samples_deanonymize_…npz`）不同，是事後改名或另行放置，原始 NPZ 不在 repo 內。見 §6 的未決事項。

---

## 2. 像素空間 cosine 基準

**量測設定**（`[事實]`，腳本與 JSON）：影像為 `scripts.e1_ddim_runner.preprocess()` 的輸出，float32、值域 [0,1]（逐張 min-max，每張恰好含 0 與 1；平均值 0.501，平均平方 0.334），攤平成 65,536 維，在 float64 計算；**不減均值、不轉 uint8、不再 min-max**，與前作 `evaluation_metrics.py:175-190` 相同。與前作逐字相同的 float32 `np.dot` 版本，與 float64 的最大差為 4.8e-6（JSON `legacy_float32_vs_float64_max_abs_diff`）。PSNR 為 −10·log10(MSE)，峰值 1。雜訊為 i.i.d. N(0,1)，種子 `[1911, round(10·PSNR), 影像序號]`。

### 2.1 不同病人（190 對）與去均值後

| 比較 | min | p05 | 中位數 | p95 | max |
|---|---|---|---|---|---|
| 不同病人，像素 cosine | 0.7392 | 0.8076 | **0.8844** | 0.9383 | 0.9675 |
| 不同病人，PSNR（dB） | 7.60 | 8.92 | 11.13 | 13.86 | 16.63 |
| 不同病人，**各自減去平均亮度**後 cosine | −0.0635 | 0.2179 | **0.5323** | 0.7508 | 0.8685 |

* `[事實]` 兩個完全不同的人，像素 cosine 中位數就有 0.884，最高 0.967。X 光值全為非負，cosine 被整體亮度與大面積的明暗結構主導；減去平均亮度後中位數降到 0.532。

### 2.2 同一張圖加雜訊

| 目標 PSNR | 實測 PSNR | cosine 平均（20 張；範圍） | 夾回 [0,1] 的版本 | 前作值 |
|---|---|---|---|---|
| 30.6 dB | 30.600 | **0.998697**（0.998691–0.998702） | 0.998696 | 0.998978（PSNR 30.23 dB） |
| 26.2 dB | 26.200 | **0.996424**（0.996409–0.996437） | 0.996414 | — |

### 2.3 cosine 隨 PSNR 的變化（雜訊不夾回，20 張平均）

| 雜訊 PSNR | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | 35 dB | 40 dB |
|---|---|---|---|---|---|---|---|
| cosine 平均 | 0.8771 | 0.9558 | 0.9853 | 0.9953 | 0.9985 | 0.9995 | 0.9999 |
| cosine 最小值（20 張） | 0.8764 | 0.9554 | 0.9852 | 0.9953 | 0.9985 | 0.9995 | 0.9998 |

* `[事實]` 20 dB 到 40 dB 的畫質差了 20 dB（均方誤差差 100 倍），cosine 只從 0.985 動到 0.99985；10 dB 的雜訊 cosine 仍有 0.877，和「兩個不同病人」（0.884）相當。cosine 在高於約 0.99 之後幾乎飽和。
* `[事實]` 對加性雜訊（與影像近似正交），cosine ≈ 1 / √(1 + MSE / S)，S 為影像的平均平方（此資料集 0.334）。JSON 的 `orthogonal_noise_theory_*` 顯示此式與實測平均的差：10、15 dB 約 3e-5，20 dB 以上小於 2e-6；逐張最大差在 30.6 dB 為 1.0e-6、26.2 dB 為 5.6e-6。也就是說 cosine 在這類誤差下只是 PSNR 的單調函數，沒有額外資訊。
* `[推定]` 前作的 0.998978 代入上式，對應 MSE ≈ 6.8e-4（約 31.7 dB 等級），與論文同列 MSE 0.0007（`05_experiments.tex:275`）同數量級；PSNR 30.23 dB 是逐張 min-max 後取平均，略有差異。前提是前作輸出影像的平均平方與 dev 影像接近，這點未驗證。所以 `[推定]` 0.9989 只是「PSNR 約 30–32 dB 的兩張圖」本來就會得到的數值。

### 2.4 與稽核方數字比對

`[事實]` 與 `audit/out/e2_legacy_cosine_semantics.json`（AUD-20261009-03）比對，結果寫在 JSON 的 `cross_check`：

| 量 | 本腳本 | 稽核方 | 絕對差 |
|---|---|---|---|
| 不同病人 cosine 中位數 | 0.884416 | 0.884416 | 0 |
| 最小值／最大值 | 0.739230／0.967468 | 0.739230／0.967468 | 0／0 |
| 去均值後中位數 | 0.532335 | 0.532335 | 0 |
| 加雜訊 ≈30.6 dB，cosine | 0.998697（20 張平均）；0.998690（僅第 0 張，夾回） | 0.998685（僅第 0 張，σ=0.03，夾回，30.579 dB） | 1.3e-5；5.7e-6 |
| 加雜訊 ≈26.2 dB，cosine | 0.996424（20 張平均）；0.996398（僅第 0 張，夾回） | 0.996414（僅第 0 張，σ=0.05，夾回，26.221 dB） | 1.0e-5；1.6e-5 |

* 不同病人與去均值兩項與稽核方**完全一致**（差 0）。加雜訊兩項差異在 2e-5 以內，來源是協定不同：稽核方只用第 0 張、固定 σ（PSNR 30.579／26.221 dB）、夾回 [0,1]；本腳本取 20 張平均，並把雜訊縮放到目標 PSNR。結論相同。

---

## 3. 本專案的指標定義

### 3.1 四種比較，名稱分開

WORKFLOW E2.4 要求「x0 vs xrec 與前作舊定義分開」。本專案的結果檔與論文一律用下列名稱，**不得只寫「cosine」而不註明是哪一種**：

| 名稱 | 比較對象 | 空間 | 用途 | 與前作關係 |
|---|---|---|---|---|
| 前作舊定義（legacy-style image cosine） | Base 輸出 vs 加解密輸出的影像 cosine（前作為偽健康 `samples`，見 §1） | 影像 | 只作與前作對照的診斷，不作通過條件 | 沿用前作公式，但**不沿用其語意** |
| 對 x0 | x0 vs 端到端還原圖 x_rec | 影像 | 可逆性主指標（端到端保真度） | 新增 |
| 對 P 輸出 | P 輸出 vs 方案輸出（P＝恆等金鑰，與 S0/S1 走同一條加密流程） | 影像 | 把「金鑰造成的損失」與「DDIM 反演＋生成＋PNG 本身的損失」分開 | 對應前作「對 Base」的比較方式，但 P 與方案走同一流程 |
| latent | 原始 latent z vs 解密後 latent ẑ（或 re-inversion 結果） | latent | 診斷、攻擊特徵分析 | 新增（前作未量測） |

### 3.2 latent 指標

* 主指標：**MaxAbs**、**RMSE**、**float32 bit-exact rate**（`scripts/e2_anonymization_runner.py:72-98` `latent_pair_metrics`；`scripts/e1_ddim_runner.py:296-310` `pair_metrics`；WORKFLOW §3.3、PROPOSAL §7.2）。前作 S0/S1 的 transform-only 路徑在 float32 下為 MaxAbs 0（E2.2）；S2 structured transform 的通過條件是 float64 MaxAbs ≤ 1e-12（PROPOSAL §7.4、WORKFLOW D6.1），bit-exact rate 如實記錄、不用重跑 seed 隱藏。
* cosine 只作診斷，與 norm ratio、低頻（8×8 區塊平均，32×32）cosine、高頻 cosine、|z| Pearson 一起報告（`e2_anonymization_runner.py:64`、`:72-98`）。`[事實]` latent 近似零均值高斯向量，cosine 約等於相關係數，對誤差很敏感；它**不能**和 §2 的像素 cosine 比較。
* **雜訊處理**（AF-019，`reports/t2wb_protocol.md` §1）：
  * 推論（病灶定位）流程 `ddim_sample_loop_known_progressive` 在 inversion 前有 t=0 的 `q_sample` 雜訊。E1 因此分開報告 `latent_reinversion_shared_noise_*`（第二次 inversion 重用同一份雜訊，樂觀上限）與 `latent_reinversion_unknown_noise_*`（獨立雜訊，種子偏移 `UNKNOWN_NOISE_SEED_OFFSET = 1_000_000`，`e1_ddim_runner.py:73`、`:460-462`）。`[事實]` 稽核方與實作方的未知雜訊平均 cosine 約 0.371（AUDIT.md AUD-20261010-01 §3 AF-019 列），共用雜訊的 0.906 只能標為「共用雜訊、樂觀上限」。
  * P/S0/S1 使用的**加密流程** `ddim_sample_loop_anonymization` 沒有 t=0 雜訊，inversion 是確定性的，「共用／未知雜訊」的區分不適用；E2 起的 latent 指標都在這條流程上量測（`e2_anonymization_runner.py`）。
* 每個 latent 數字必須註明：流程（推論流程或加密流程）、雜訊條件（共用／未知／不適用）、交接方式（float、legacy PNG：per-image min-max＋uint8 截斷、丟棄值域，`legacy_png_handoff`；或 range-preserving PNG：保存 lo/hi、`np.rint` 量化，`scripts/m1_storage.py`，見 §7）。

### 3.3 影像指標

* 主指標：**PSNR**（峰值 1，−10·log10 MSE）、**SSIM**（`skimage` `data_range=1.0`）、**L∞**、**uint8 pixel equality**（`rint(clip(x,0,1)·255)` 相等的比例）、**float32 bit-exact rate**（`e2_anonymization_runner.py:101-120` `image_pair_metrics`；`e1_ddim_runner.py:313-335`）。MSE、RMSE、MAE 照常記錄；影像 cosine 只作診斷。
* **不 clip**：PSNR、SSIM、L∞、MSE、MAE、cosine 與 float32 bit-exact 都直接用生成輸出計算，**沒有**先 clip 到 [0,1]；只有 uint8 pixel equality 先 `clip(·,0,1)` 再量化。生成輸出會略微超出 [0,1]（`clip_denoised` 只限制 x0 預測在 [−1,1]），所以與「先 clip 再算」的慣例相比，PSNR 會有小差異（稽核方在 E2.5 逐張比對中觀察到 ≤ 0.042 dB，AUD-20261010-03 §3）。本專案維持不 clip 的定義（使用者 2026-10-10 裁決：只註明、不修改），E2.5 與之後的結果都沿用此定義。
* **一律並列兩種參照**：「對 x0」與「對 P 輸出」。只報其中一種不得下可逆性結論。
* 存檔交接分兩欄：前作重現的主路徑是 legacy PNG（`legacy_png_handoff`）；本專案的 M1 協定是 range-preserving PNG（AF-024，§7）。float 交接只作上限對照。可逆性判定以影像端端到端指標為準（`reports/t2wb_protocol.md` §2）。
* **存檔協定必須標明**（AF-024）：每個影像指標數字都要註明匿名影像的交接／存檔協定，只能是 legacy PNG（前作：per-image min-max＋uint8 截斷，丟棄值域）、range-preserving PNG（`range_preserving_png/v1`，本專案 M1 協定）或 float（診斷用上限）三者之一；沒有標明者不得放入論文。上一條的「主路徑」僅指「前作重現」；本專案 M1 協定與兩種 PNG 協定的分欄報告見 §7。
* 影像 cosine 若出現，必須同列 PSNR 與 L∞。前作表格中的 MaxAbsError 0.676–0.928（`05_experiments.tex:306-317`）和 cosine 0.998 同時存在，正是 cosine 看不到局部大誤差的例子（見下）。

### 3.4 cosine = 1 不等於 equality

`[事實]`（數學）對非零向量，cos(x, y) = 1 若且唯若 y = a·x 且 a > 0（Cauchy–Schwarz 的等號條件）。所以 cosine = 1 只表示「方向相同」，不表示 y = x。反方向：x = y 一定推出 cosine = 1，但反過來不成立。

`[事實]` 建構性反例（dev_v1.1，20 張；`results/E2.4_metric_semantics.json` 的 `d_cosine_one_without_equality`）：

| y 的構造 | cosine | PSNR | MaxAbs | uint8 相等比例 | float32 bit-exact | 是否相等 |
|---|---|---|---|---|---|---|
| y = x（對照） | 1 | ∞ | 0 | 100% | 100% | 是 |
| **y = 0.5·x** | **1**（\|1−cos\| < 1e-15；20 張範圍 0.9999999999999999–1.0000000000000002） | **10.79 dB**（10.77–10.81） | **0.500** | 0.35% | 0.07% | 否（20 張皆否） |
| y = 0.9·x | 1（同上） | 24.77 dB | 0.100 | 1.84% | 0.07% | 否 |
| x 中央 12×12 區塊（0.22% 像素）換成 1−x | 0.99879（0.99706–0.99984） | 32.01 dB（27.08–39.62） | **0.707**（最大 0.992） | 99.78% | 99.78% | 否 |

* 第一列說明 cosine 對正的整體縮放完全無感（亮度、對比被改了一半，cosine 仍是 1）。
* 最後一列說明另一個方向的盲點：99.78% 的像素逐位元相同，只有 0.22% 的像素錯得離譜（MaxAbs 0.71，和前作表中 0.71417 同量級），cosine 仍是 0.9988，和前作的 0.9989 同等級。`[事實]` 區塊大小是**事後選定**（讓平均 cosine 約 0.9989），僅作說明，不是前作誤差的模型。
* `[事實]` 前作式的 float32 `np.dot` 在 y = 0.9·x 上算出 1.0000039（大於 1），在 y = 0.5·x 上算出 0.99999988（JSON `legacy_float32_cosine`）。cosine 的小數第 6 位以後在 float32 下沒有意義，「接近 1.0000」不能當作逐位元相等的證據。
* `[事實]` 對照真實的 S1 量測（§5）：影像 cosine 0.9985 對應 30.2 dB，和 §2.3 的雜訊基準（30 dB → 0.9985）一致。`[推定]` 在這組實際的 DDIM／PNG 誤差上，cosine 也沒有給出 PSNR 以外的資訊。

### 3.5 各方向等權的 latent 距離無法預測影像可逆性

以下是**稽核方的量測**（AUDIT.md AUD-20261010-02 §2；腳本 `audit/e2_latent_direction_sensitivity.py`，輸出 `audit/out/e2_latent_direction_sensitivity.json`；dev_v1.1 的 000／001／010／011 四張，學長加密流程，guidance −1，T=500；PSNR 對 P 輸出 G(z)）。本報告已把表中四個平均值與該 JSON 的 `summary` 逐項核對相符，但並非本專案實作方的量測。

| 對 z 加上的誤差 | latent RMSE | latent cos | 生成影像 PSNR |
|---|---|---|---|
| 隨機白雜訊（小） | 0.135 | 0.990 | 31.3 dB |
| S0 實際的解密 latent（PNG 交接） | 0.132 | 0.991 | 31.8 dB |
| 隨機白雜訊（大） | 1.12 | 0.654 | 6.4 dB（影像崩壞） |
| P 實際的 re-inversion latent（PNG 交接） | 1.13 | 0.275 | 43.9 dB |

* `[事實]`（稽核方量測）P 的 re-inversion 誤差與大幅白雜訊一樣大（RMSE 約 1.12），前者影像幾乎不變（43.9 dB），後者崩壞（6.4 dB）。S0 的解密誤差與同大小白雜訊的影像影響相當（31.8 對 31.3 dB）。
* `[推定]`（稽核方推論，本專案尚未驗證機制）生成器是「多對一」的，誤差落在生成器不敏感的方向時，latent 差很多、影像幾乎一樣；金鑰反運算會把誤差打散成近似隨機方向，生成器對這種方向敏感。
* `[結論]` 任何各方向等權的 latent 距離（cosine、RMSE、L2）都無法預測影像可逆性。可逆性必須以影像端指標判斷；latent 指標只用於診斷與攻擊特徵分析。

---

## 4. 論文用語規範

**禁止**把前作 0.9989（或其任何 cosine）稱為：

| 不可使用 | 理由 |
|---|---|
| latent consistency／latent cosine／decrypted latent vs original latent | 程式算的是輸出影像的 cosine（§1）；稽核方量到的真實 latent cosine 是另一個量（約 0.990 PNG／0.9999 float，見 §5），不能拿來替前作背書 |
| zero-distortion、lossless、perfect／strict reversibility、完全可逆、無損還原 | 同一流程的端到端 PSNR 約 30–31 dB、L∞ 0.4–0.6（§5），不是無損；cosine ≈ 1 不蘊含相等（§3.4） |
| 「cosine 0.9989 證明加密保持線性代數性質／抵禦反轉攻擊」 | cosine 與安全性無關；本專案的安全宣稱依 PROPOSAL §5 另有協定 |
| 把 cosine 當作可逆性的通過條件 | 通過條件是影像端 PSNR／SSIM／L∞／pixel equality，並同時報告「對 x0」與「對 P 輸出」 |

**建議寫法**（可直接套用）：

* 中文：「前作報告的 Cosine Sim 0.9989 是 Base 流程輸出影像與加解密流程輸出影像在像素空間的 cosine（未減均值），其對應的 PSNR 為 30.23 dB；在本專案 dev 集上，同一張圖加雜訊到 30.6 dB 即有 0.9987，不同病人的中位數為 0.884。該數值未量測 latent，也不蘊含像素或位元相等。」
* English: "The predecessor's reported 'Cosine Sim 0.9989' is the pixel-space cosine between the base-pipeline output image and the encrypt/decrypt-pipeline output image (no mean subtraction), at a PSNR of 30.23 dB. It is not a latent cosine and does not imply pixel- or bit-level equality. On our dev set, adding noise to a PSNR of 30.6 dB already gives 0.9987."
* 本專案的可逆性敘述格式：「（流程）下，（交接方式）的 M1 端到端還原，對 x0 的 PSNR／SSIM／L∞ 為……，對 P 輸出為……；latent MaxAbs／RMSE 為……（診斷）。」

**每個指標數字都要標註**：空間（latent／影像）、比較對象（x0／P 輸出／前作舊定義）、流程（推論／加密）、交接（float／legacy PNG／range-preserving PNG）、雜訊條件（共用／未知／不適用）、縮放（[0,1] float；是否去均值）。缺任何一項，不得放入論文。

---

## 5. E2.5 dev 實測（本專案實作方量測）

* 來源：`results/E2.5_roundtrip_per_sample.csv`、`paper_assets/tables/table_baseline_correctness.csv`；managed run `E2.5_20261009T191319376995Z_f990c7d7`（`scripts/e2_roundtrip_runner.py`，commit `4156be7`）；輸入為 E2.3 dev cache `E2.3_20261009T190608331743Z_6edaf199`。dev_v1.1 20 張；學長加密流程；生成 guidance −1；T=500；batch 1。數值為平均與 bootstrap 95% CI（B=10,000，seed 1911）。影像指標未先 clip 到 [0,1]（uint8 pixel equality 除外，見 §3.3）。
* 標記：`[事實]`。科學數值只報告，不是通過條件。

| 路徑／量測 | P（恆等金鑰） | S0 Rademacher | S1 Signed Perm |
|---|---|---|---|
| transform-only：D(E(z)) MaxAbs／bit-exact | 0／100% | 0／100% | 0／100% |
| 匿名影像對 x0：PSNR／SSIM | 42.23 dB／0.985（= P 輸出） | 5.34 dB／0.018 | 5.43 dB／0.012 |
| T1（精確 latent）還原對 x0：PSNR／SSIM／L∞ | 42.23 [41.90, 42.53] dB／0.985／0.102 | 同 P（逐位元相同） | 同 P（逐位元相同） |
| T2-WB／M1 PNG 還原對 x0：PSNR | 37.71 [37.25, 38.07] dB | 31.47 [30.60, 32.30] dB | 31.30 [30.75, 31.89] dB |
| 同上：SSIM／L∞／uint8 相等比例 | 0.975／0.232／15.3% | 0.927／0.558／8.7% | 0.898／0.416／8.0% |
| M1 PNG 還原對 P 輸出：PSNR | 43.09 dB | 30.75 dB | 30.20 dB |
| 同上：影像 cosine（前作 Cosine_Sim 公式的類比量） | 0.9999 | 0.9986 | 0.9985 |
| M1 float 還原對 x0：PSNR（上限對照） | 37.62 dB | 38.70 dB | 39.40 dB |
| 解密後 latent 對 z（PNG）：cosine／RMSE | 0.299／1.123 | 0.990／0.135 | 0.990／0.137 |
| 同上：低頻 32×32 cosine／\|z\| Pearson | 0.974／0.434 | 0.996／0.976 | 0.996／0.976 |
| 解密後 latent 對 z（float）：cosine | 0.292 | 0.9999 | 0.9999 |

* `[事實]` 「還原圖對 P 輸出」只是前作比較方式的**類比**：前作實際比較的是 base 與加解密兩條路徑第 3 步的偽健康 `samples`（§1.2、§6 第 2 點），本專案沒有重現那一對影像。這個類比量在本專案得到 S1 30.20 dB／影像 cosine 0.9985，與論文 30.23 dB／0.998978 同等級；但同一批還原圖**對原圖 x0** 只有 31.3 dB、L∞ 0.42，uint8 相等比例 8%。
* `[事實]` T1（保存精確 latent）時，S0/S1 的 transform 逐位元可逆（60/60），生成器的輸入與 P 完全相同，所以輸出也相同（42.23 dB）。這是依構造成立：runner 在 transform 逐位元可逆時直接沿用 P 的輸出（`scripts/e2_roundtrip_runner.py` 的 `t1_reused_p_output=1`，60/60 列），不是另外生成後比對的結果。T1 的損失因此全部來自 DDIM 反演＋生成本身。M1 的額外損失主要來自 legacy PNG 存檔（per-image min-max 後未保存值域；S0/S1 float 交接 38.7–39.4 dB，legacy PNG 交接 31.3–31.5 dB）；保存值域後的結果見 §7。
* `[事實]` 對正交的 S0/S1/P 而言，攻擊者視角（ẑ_ano 對 z_ano）與合法解密（ẑ 對 z）的全域 latent 指標相同（兩者差一個相同的正交變換），因此 S0/S1 的 T2-WB 攻擊者能取得與解密者同等保真度的 \|z\|（Pearson 0.976）。這是 A4 T2-WB magnitude 攻擊的前提，攻擊本身待 A4 實測。
* 本節數字與稽核方以獨立程式量測的結果（下表）在四捨五入內一致。

### 稽核方獨立量測（交叉驗證用）

* 來源：AUDIT.md「AUD-20261010-01 · §4」；原始數據 `audit/out/e2_legacy_decrypt_latent_cosine.json`（腳本 `audit/e2_legacy_decrypt_latent_cosine.py`；dev_v1.1 全部 20 張；直接呼叫學長函式；guidance −1；T=500；20 張平均）。本報告已把表中各列的平均值與該 JSON 的 `summary` 逐項核對，除 P 的 PNG 交接 latent cos（JSON 0.2985，表寫 0.299）與低頻 cos（JSON 0.9735，表寫 0.974）有 0.001 的四捨五入差異外，其餘相符；下表照 AUDIT.md 原文引用。
* 標記：`[事實]`（稽核方量測）。

| 量測 | P（恆等金鑰） | S0 Rademacher | S1 Signed Perm |
|---|---|---|---|
| 精確 latent 路徑 D(E(z)) 的 MaxAbs | 0 | 0 | 0 |
| latent cos(z, ẑ)：float 交接 | 0.292 | 0.9999 | 0.9999 |
| latent cos(z, ẑ)：PNG 交接（學長實際做法） | 0.299 | 0.990 | 0.990 |
| 低頻 cos／\|z\| Pearson（PNG 交接） | 0.974／0.434 | 0.996／0.976 | 0.996／0.976 |
| 還原圖對 x0 PSNR：float 交接 | 37.6 dB | 38.7 dB | 39.4 dB |
| 還原圖對 x0 PSNR／SSIM：PNG 交接 | 37.7 dB／0.975 | 31.5 dB／0.928 | 31.3 dB／0.898 |
| 還原圖對 x0 L∞：PNG 交接 | 0.23 | 0.56 | 0.42 |
| 還原圖對 P 輸出 PSNR：PNG 交接（前作比較方式的類比；前作實際比較的是偽健康輸出） | 43.1 dB | 30.8 dB | 30.2 dB |
| 同上，影像 cosine（前作 Cosine_Sim 公式的類比量） | 0.9999 | 0.9986 | 0.9985 |

* `[事實]`（稽核方）以學長的程式與參數，在「還原圖對 P 輸出」這個類比量上得到 **S1 30.2 dB、影像 cosine 0.9985**，與前作論文的 30.23 dB、0.9989 同等級。稽核方已在 AUD-20261010-03 AF-023 第 2 點更正：這不是重現前作比較的同一對影像（前作比的是偽健康輸出）。
* `[事實]`（稽核方）S0/S1 解密後 latent 與原始 latent 的真實 cosine 約 0.990（PNG）／0.9999（float）；對應影像只有約 31 dB，所以「零失真」不成立。影像劣化主要來自 PNG 儲存（float 交接 38.7–39.4 dB，PNG 交接 31.3–31.5 dB）。〔註：稽核方其後在 AUD-20261010-04 細分，損失幾乎全部來自 per-image min-max 未保存值域，而非 8-bit 量化；此處「PNG」指 legacy PNG。〕
* `[事實]` 本專案 §2.3 的雜訊基準（30 dB → 0.9985、30.6 dB → 0.9987）與 S1 的 0.9985／30.2 dB 一致。`[推定]` 在這類誤差下，這個影像 cosine 的大小可由 PSNR 預測。

---

## 6. 與 AF-022 描述的差異與未決事項

驗收條件（AF-022）：報告存在且含四點，數字可由腳本重現。四點分別為 §1、§2、§3、§4；§2 的數字由 `scripts/e2_metric_semantics.py` 重現。核對 AUD-20261009-03 敘述與原始碼時發現：

1. `[事實]` AF-022 引用 `cfg_image_sample_anonymization.py:220-233` 描述 `deanonymize` 分支；實際該分支為 `:221-236`，決定「`org` 是 x_rec」的 `org = x_rec` 在 `:236`，不在所引用的範圍內。`:380-429`（收集與存檔）、`evaluation_metrics.py:140-202`、`:175-190` 的引用正確。
2. `[事實]` AF-022 說「`samples` 是第 3 步偽健康影像」在程式路徑上成立（§1.2），但稽核方在 AUD-20261010-01 §4 重現 30.2 dB／0.9985 時，比較的是 **x_rec 對 P 輸出**（`audit/e2_legacy_decrypt_latent_cosine.py` 的 `png_rec_vs_Pout_*`），不是偽健康對偽健康。所以：(a) 兩處說的不是同一對影像；(b) 前作 NPZ 不在 repo，無法確定論文的 0.9989 究竟是哪一對。`[推定]` 不論哪一對，它都是輸出影像的像素 cosine，AF-022 的結論不受影響；30.2 dB 與 30.23 dB 的吻合只證明「該量的大小與約 30 dB 的影像對一致」，不能用來斷定是哪一對。
3. `[事實]` AUDIT.md AUD-20261009-03 的基準表把前作 cosine 寫為 0.9990；論文表格值是 0.998978（`05_experiments.tex:308`），論文文字截斷為 0.9989。四捨五入到四位是 0.9990。
4. `[事實]` AUDIT.md 的雜訊基準（0.9987、0.9964）來自**單張影像、單次雜訊、固定 σ**（`audit/out/e2_legacy_cosine_semantics.json`）；本報告為 20 張平均。兩者差 2e-5 以內（§2.4）。
5. `[推定]` AF-022 把前作 MaxAbsError 的值域稱為 [0,1]；輸入值域已驗證為 [0,1]（§1.2），輸出實際值域因 NPZ 不在 repo 而未驗證。
6. `[事實]` AF-022 未提到：前作 PSNR／SSIM 是對逐張 min-max 後的影像計算，cosine／MaxAbs／MAE／MSE 則用原始 `samples`（`evaluation_metrics.py:158-190`），兩組指標的縮放不一致；摘要的 30.23 dB 與 SSIM 0.912 來自兩個不同方案的表（§1.1）。
7. `[事實]` AUDIT.md AUD-20261010-01 §4 表中 P 的 PNG 交接 latent cos（0.299）與低頻 cos（0.974），對照 `audit/out/e2_legacy_decrypt_latent_cosine.json` 的 `summary` 為 0.2985 與 0.9735，是 0.001 的四捨五入差異，不影響結論。

**限制**：N = 20、單一資料集（CheXpert dev），像素基準用的是預處理後的輸入影像而非前作的模型輸出；`[推定]` 兩者的影像統計相近，故基準可作為量級參考，但不能用來重建前作的確切數值。雜訊為高斯 i.i.d.，真實的 DDIM 誤差有空間結構，其 cosine 與 PSNR 的關係會略有不同（稽核方的 S1 量測與高斯基準的吻合支持量級一致，見 §5）。

**確定性檢查**：同一指令連續執行兩次，`results/E2.4_metric_semantics.json` 在剖析後除 `created_at` 外完全相同（種子 1911、numpy 1.23.0）。

---

## 7. M1 存檔協定與值域保存（AF-024）

> 依據：使用者裁決 2026-10-10（AUDIT.md AUD-20261010-05）；稽核方診斷 AUD-20261010-04。實作：`scripts/m1_storage.py`（`range_preserving_png/v1`）。標記規則同本報告開頭。

### 7.1 兩種存檔協定的定義

| 項目 | legacy PNG（前作原樣交接） | range-preserving PNG（本專案 M1 協定） |
|---|---|---|
| 實作 | `legacy_png_handoff`（`scripts/e2_anonymization_runner.py:123`） | `range_preserving_png/v1`（`scripts/m1_storage.py`：`encode_range_preserving_png`、`decode_range_preserving_png`、`range_preserving_png_handoff`） |
| 角色 | P/S0/S1「前作重現」的主路徑（E2.2／E2.5 的既有結果不改寫）；S2a/S2 也跑此欄，作為與前作相同基準的對照與歸因（附表；使用者 2026-10-10） | 本專案的 M1 協定與對外主結果；P/S0/S1/S2a/S2 都以此協定跑，使方案間的比較只差在加密方法，不差在存檔方式；設定選擇只依此欄 |
| 正規化 | per-image min-max：`stored = ((x − low)/(high − low)·255).to(uint8)` | `lo = min(x)`、`hi = max(x)`（float32，每張影像各一組） |
| 量化 | 截斷（`.to(uint8)`），不是四捨五入 | `q = clip(np.rint((x − lo)/(hi − lo)·255), 0, 255)`（round-half-to-even）；`hi = lo` 時 `q` 全為 0；8-bit 灰階 PNG |
| 值域 | **丟棄**；讀回時對 `stored/255` 再做一次 min-max，輸出值域恆為 [0,1] | `lo`、`hi` 以 IEEE-754 float32 的位元樣式原值保存（PNG text chunk `x-range-lo-float32-bits`／`x-range-hi-float32-bits`，storage record 的 `lo_float32_bits`／`hi_float32_bits`），不經十進位字串 |
| 讀回 | `x̂ = (stored/255 − r_low)/(r_high − r_low)`，`r_low`、`r_high` 為讀回影像自己的最小、最大值 | `x̂ = q/255·(hi − lo) + lo`，值域回到原值；PNG 缺少 `lo`／`hi` 時拒絕讀回，不猜測值域 |
| 記錄 | 前作未記錄值域 | manifest／storage record 記 `storage_protocol`、`quantization`（量化方式字串）、`lo`、`hi` 與其位元樣式、PNG 的 SHA-256 與位元組數 |

* `[事實]`（程式讀取）兩者的程式出處如上表；legacy 的描述同 `legacy_png_handoff` 的 docstring：前作存 `(visualize(sample)·255).astype(uint8)`，載入時 min-max 正規化。
* `[事實]`（數學）round-half-to-even 的量化使逐像素讀回誤差不超過半個量化階 `(hi − lo)/510`（另加 float32 運算誤差）。`tests/unit/test_m1_storage.py` 檢查這個上界（含負值、超過 1 的值域）；legacy 協定沒有這個保證，因為它讀回後的值域被強制成 [0,1]。

### 7.2 歸因（使用者裁決 2026-10-10）

* `[決定]` 前作 M1 還原品質（約 30–32 dB）的主要損失，來自存檔流程的 per-image min-max 正規化後沒有保存原值域，屬於**工程失真**，**不是**翻號／置換加密本身的限制。論文與報告須如此歸因（AUD-20261010-05 §1）。
* `[決定]` 之後所有以影像形式保存的輸出（匿名影像，以及任何需要再 inversion 或還原的影像），都必須記錄影像的實際值域，讀回時還原回原值域。
* `[事實]` 本報告 §5 及其稽核方交叉驗證表中的「M1 PNG」「PNG 交接」都是 legacy PNG（E2.5 與稽核方都使用 `legacy_png_handoff` 的 per-image min-max＋uint8 截斷）。
* `[推定]` §5 表中 S0/S1 與 float 交接的差距（legacy PNG 約 31 dB 對 float 38.7–39.4 dB）依本節歸因於未保存值域的 min-max，不應讀成「8-bit PNG 本身造成」或「加密造成」。稽核方的分解只在 4 張影像上做（§7.3）；dev 20 張的實測見 §7.5，結果支持此歸因。
* `[決定]` §3.2 與 §4 要求的「交接方式」標註，由原來的「float 或前作 PNG」擴充為三種：float、legacy PNG、range-preserving PNG（見 §3.3 新增條目）。

### 7.3 稽核方診斷（AUD-20261010-04）

以下是**稽核方的量測**，不是本專案實作方的量測。來源：AUDIT.md AUD-20261010-04 §2；腳本 `audit/e2_png_handoff_decomposition.py`，輸出 `audit/out/e2_png_handoff_decomposition.json`；4 張 dev 影像（與 `e2_wht_smoke` 同一組影像與金鑰），PSNR 對 x0。以下數字依 AUDIT.md 原文引用，並已用該 JSON 的 `per_sample` 重算核對（`[事實]`，本報告重算）：四個方案各 4 張的平均與 AUDIT.md 表一致；min-max 交接 RMSE 與 legacy PNG 還原 PSNR 的 Pearson／Spearman 重算為 −0.884／−0.803（n=16，與 AUDIT.md 相符）。

* `[事實]`（稽核方）加密方案的「只做 min-max」與 legacy PNG 的還原 PSNR 相差 ≤ 0.3 dB；「只做 8-bit 量化、保留 lo/hi」與 float 交接相差 ≤ 0.6 dB。因此 legacy PNG 交接的損失幾乎全部來自 per-image min-max，8-bit 量化本身影響很小。
  * `[事實]`（本報告重算）上述兩個界限都是**逐方案的 4 張平均**：只做 min-max 與 legacy PNG 的差為 0.02–0.25 dB（逐張最大 0.25 dB）；只做量化與 float 的差為 0.21–0.46 dB（P 0.46、S0 0.31、S1 0.45、WHT R=1 0.38、WHT R=2 0.21），但**逐張最大達 0.78 dB**。「≤ 0.6 dB」不可寫成逐張的保證。
* `[事實]`（稽核方）加密後生成的匿名影像值域約為 [−0.4, 0.7–1.0]，不在 [0,1]；min-max 讀回等於對整張圖做一次亮度／對比的仿射改變（交接 RMSE 約 0.3），再經 inversion 與金鑰反運算放大成還原誤差。P 的輸出值域本來就接近 [0,1]，所以不受影響。
* `[事實]`（稽核方）16 個加密樣本中，min-max 交接 RMSE 與 legacy PNG 還原 PSNR 的 Pearson 相關為 −0.884（Spearman −0.803）。
* `[事實]`（稽核方）R=2 減 R=1 的逐張差不顯著（配對 t 檢定 p=0.27）。`[推定]`（稽核方）legacy PNG 下 R=1 看起來較差，是各金鑰剛好生成了值域不同的匿名影像，與 R 無關。
* 稽核方當時的 `[假設]`「range-preserving PNG 下 S0/S1 的還原 PSNR 會回到接近 float 交接」，已由 §7.5 的 dev 20 張實測確認（平均差 0.26／0.23 dB）。科學數值只報告，不作門檻。

### 7.4 完整性要求（HMAC 涵蓋值域）

* `[決定]` `lo`、`hi` 會直接決定讀回數值，只改它們而不動像素資料，也會改變還原結果。S2 的 HMAC／authenticated metadata 必須涵蓋 `storage_protocol`、`quantization`、`lo`、`hi`（PROPOSAL §4.4）。
* `[決定]` 這些欄位的 MAC 驗證成功之前，不得以 `lo`／`hi` 讀回影像，也不得進行 inverse transform、inversion 或 diffusion generation（WORKFLOW §2.3）。
* `[決定]` T4 竄改測試必須包含「只修改 lo/hi」的情境，且必須被拒絕；在 Week 3 D5.5／D5.6 的 authenticated container 與 tamper matrix 實作。
* `[事實]` 目前 repo 沒有 `src/` 目錄，也沒有任何 HMAC 程式（本報告撰寫時以 `grep -il hmac` 檢查 `.py`，排除 `audit/`、`past/`、`refpaper/`），所以本節的完整性要求尚未實作，只有 `scripts/m1_storage.py` 產生的 storage record 可供日後納入認證。
* `[決定]` `lo`／`hi` 是公開的儲存 metadata，不是秘密；T2-WB 攻擊者取得匿名影像時同時取得它們，因此 range-preserving PNG 欄的 T2-WB 攻擊以含值域還原的讀回評估（`reports/t2wb_protocol.md` §2 第 4 點、§3）。

### 7.5 dev 實測：三種存檔交接並列（本專案實作方量測）

* 來源：`results/AF024_m1_storage_per_sample.csv`、`paper_assets/tables/table_m1_storage_comparison.csv`；managed run `AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8`（`scripts/m1_storage_compare_runner.py`，commit `f5c967e`）。range-preserving 欄由本 run 產生；legacy PNG 與 float 欄唯讀併入 E2.5 run `E2.5_20261009T191319376995Z_f990c7d7`。本 run 重新生成的 60 張匿名影像，`anon_vs_x0_psnr_db` 皆與 E2.5 逐值相同（`anon_matches_e2_5=1`），所以三欄比較的是同一批匿名影像。
* 設定同 §5：dev_v1.1 20 張、學長加密流程、guidance −1、T=500、batch 1；平均與 bootstrap 95% CI（B=10,000，seed 1911）；影像指標未 clip（§3.3）。標記：`[事實]`；科學數值只報告，不作門檻。

| 還原對 x0 | legacy PNG | range-preserving PNG | float 交接（上限） |
|---|---|---|---|
| P：PSNR／SSIM／L∞ | 37.71 dB／0.975／0.232 | 37.81 [37.32, 38.19] dB／0.975／0.205 | 37.62 dB／0.975／0.254 |
| S0：PSNR／SSIM／L∞ | 31.47 dB／0.927／0.558 | **38.44 [37.98, 38.85] dB**／0.972／0.267 | 38.70 dB／0.973／0.264 |
| S1：PSNR／SSIM／L∞ | 31.30 dB／0.898／0.416 | **39.17 [38.98, 39.34] dB**／0.973／0.155 | 39.40 dB／0.973／0.135 |
| S0／S1 還原對 P 輸出：PSNR | 30.75／30.20 dB | 43.13／45.42 dB | 43.97／46.29 dB |
| S0／S1 解密 latent：cosine | 0.990／0.990 | 0.9997／0.9998 | 0.9999／0.9999 |
| S0／S1 攻擊者視角 \|z\| Pearson | 0.976／0.976 | **0.9993／0.9994** | 0.9998／0.9998 |

* `[事實]` 保存值域後，S0/S1 的 M1 還原從 legacy PNG 的約 31.3–31.5 dB 回到 38.4–39.2 dB，與 float 交接的平均差 0.26 dB（S0）／0.23 dB（S1）；逐張差範圍 S0 −1.04～+2.14 dB、S1 −0.39～+1.37 dB。P 幾乎不受存檔協定影響（37.6–37.8 dB）。
* `[事實]` dev 20 張中，加密匿名影像的值域下界 lo 最低到 −0.53，上界 hi 最高為 1.0，與稽核方 4 張的觀察（約 [−0.4, 0.7–1.0]）一致。
* `[結論]` 本實測支持 §7.2 的歸因：前作約 31 dB 的 M1 損失，主要是 legacy PNG 存檔未保存值域造成的工程失真，不是翻號／置換加密本身。
* `[事實]` 保存值域同樣提升了 T2-WB 攻擊者看到的 latent 保真度：S0/S1 攻擊者視角的 \|z\| Pearson 由 0.976 升到 0.999。`[推定]` 在 range-preserving 協定下，A4 的 magnitude／sorted-magnitude 攻擊至少不會比 legacy PNG 弱；這正是 AF-024 要求兩欄分開報告的原因，待 A4 實測。
