# T2-WB 可逆性指標與正控制協定（AF-019）

> 狀態：§1–§3 為量測規則；§4 為使用者 2026-10-10 裁決的 `Inconclusive` 規則（取代原提案），A4／D7 依此實作。
> 依據：WORKFLOW §3.4 第 4 點、PROPOSAL §5.4、AUD-20261008-03（AF-017、AF-019）。

## 1. 為什麼不能用 latent cosine 0.906

- [事實] E1.4 的 re-inversion cosine 0.906 來自**推論流程**（`ddim_sample_loop_known_progressive`，inversion 前有 t=0 `q_sample` 雜訊），且第二次 inversion 重用了同一份雜訊。稽核方實測不知道雜訊時只有 0.29–0.37（`audit/out/e2_inversion_error_attribution.json`）。0.906 一律標為「共用雜訊、樂觀上限」。
- [事實] 自 AF-020 重跑起，`results/E1.4_ddim_smoke.csv` 同時列出 `latent_reinversion_shared_noise_*` 與 `latent_reinversion_unknown_noise_*`（後者以 seed+1,000,000 產生獨立雜訊）。
- [事實] P/S0/S1 實際使用的**加密流程**（`ddim_sample_loop_anonymization`）沒有 t=0 雜訊，inversion 是確定性的，因此「已知／未知雜訊」的區分不適用；本專案 E2 起的 latent 指標都在加密流程上量測（`scripts/e2_anonymization_runner.py`）。
- [事實] 加密流程上的 P smoke（dev_v1.1 的 000/001/010/011，`results/AF017_P_anonymization_smoke.csv`，managed run `AF017_P_SMOKE_20261009T152434977798Z_df76812e`）：

  | 量測（4 張平均，括號內為範圍） | 數值 |
  |---|---|
  | 同一輸入重算 inversion：MaxAbs | 0（4/4 bit-exact） |
  | P 輸出 vs 原圖：PSNR／SSIM | 42.35 dB（41.70–43.07）／0.9838 |
  | re-inversion（PNG 交接）latent cosine | 0.275（0.012–0.739） |
  | 同上，低頻 32×32 cosine／高頻 cosine | 0.974／0.244 |
  | 同上，\|z\| Pearson 相關 | 0.430（0.296–0.660） |
  | M1 端到端（PNG 交接）vs 原圖：PSNR／SSIM／L∞ | 38.10 dB／0.9760／0.136 |
  | PNG 儲存本身 vs P 輸出：PSNR | 54.58 dB |

- [結論] 即使沒有 t=0 雜訊，匿名影像的 re-inversion 在 latent 端仍只保留低頻；影像端端到端仍有 38 dB。latent cosine 因此不能代表可逆性，逐元素 \|z\| 攻擊在 T2-WB 下的成敗也必須先看 P 正控制（§3）才能解讀。
- [觀察] 4 張圖都只有 P 恆等金鑰，樣本數很小；正式數字以 E2.5（dev 20 張）與 pilot 為準。dev_v1.1_010／011 的 M1 還原圖中可見原圖沒有的細小字樣狀紋理（`image/AF017_P_anonymization_smoke.png`），屬生成模型的局部幻覺，Q8 視覺格與病灶差異圖需要注意。

## 2. 可逆性的主指標與報告方式

1. **主指標是影像端端到端**：x0 對 M1 還原影像的 PSNR、SSIM、L∞、uint8 pixel equality、float32 bit-exact。
2. latent 指標（cosine、RMSE、MaxAbs、bit-exact、norm ratio）照常報告，但只作診斷；另報低頻（8×8 區塊平均，32×32）與高頻 cosine、|z| Pearson 相關，用來說明 inversion 誤差落在哪個頻段。
3. 三條路徑分開報告（PROPOSAL §7.2）：transform-only、T1（精確 latent）、T2-WB／M1（匿名影像 → 目標模型 inversion → 解密 → 生成）。
4. M1 的主路徑採**學長實際的儲存方式**：匿名影像經 per-image min-max 後以 uint8 PNG 保存（truncation），重新讀入後 inversion（`legacy_png_handoff`）。float 交接只作上限對照。

## 3. T2-WB 攻擊的正控制與負控制（A4／D7 實作）

- Gallery：E2.3 cache 的原圖 latent `z_i = ddim_anonymization_forward(x0_i)`，`security_v1`，N=200。
- Query（方案 s）：攻擊者對保存的匿名 PNG 自行 inversion 得 `ẑ_j^s`；T1 則直接取精確 `z_ano,j^s`。
- 特徵與 scorer 在 T1、T2-WB、所有方案間完全相同：S0 用 `|z|`，S1 用 `sort(|z|)`，另報 norm-only `‖z‖`；可加報低頻 `|z|` 區塊平均作為 re-inversion 穩健特徵（`[假設]` 身分資訊主要在低頻，見 AUD-20261008-03 §3.3）。
- **正控制**（依 §4 裁決）：每個方案各自的正控制，即以正確金鑰把 re-inversion 得到的 `ẑ_ano` 解密成 `ẑ` 後，以 latent 相關對 gallery 做 linkage。P（恆等金鑰）的 re-inversion linkage 只作參考，不用來判定其他方案。
- **負控制**：gallery ID 以 seed 1911 隨機重排後的 Top-1（預期約 1/N）。
- 統計：Top-1、Top-5、mAP／CMC；以 query 為單位 bootstrap（B=10,000、seed 1911）95% CI。

## 4. `Inconclusive` 判定（使用者裁決 2026-10-10，AUD-20261010-02 §1）

原提案（以 P 的 re-inversion 當所有方案的正控制）已被取代：稽核方實測 P 的 re-inversion 保真度（\|z\| Pearson 約 0.43）遠低於 S0/S1 自身（約 0.976，AUD-20261010-01 §4），以 P 判定會低估對 S0/S1 的攻擊可行性。

對每個（方案 s、特徵 f、T2-WB）格子：

1. **攻擊成功**：Top-1 的 95% CI 下界 > 負控制 Top-1 的 95% CI 上界 → 照實報告為攻擊成功，不受正控制影響。
2. **攻擊未成功**時，才看**該方案專屬的正控制**：實驗者以正確金鑰把 re-inversion 得到的 `ẑ_ano` 解密成 `ẑ`，再以 latent 相關（cosine／Pearson）對 gallery `z_i` 做 linkage，取 Top-1 與 95% CI。
   - 正控制的 CI 下界 ≤ 負控制的 CI 上界 → 該格標 `Inconclusive`（受 inversion 誤差限制），不得寫成攻擊失敗或方案安全。
   - 正控制通過 → 只能寫「T2-WB 下此特徵的 linkage 未達顯著（正控制通過）」；依 PROPOSAL §5，這仍不是安全性宣稱。
3. 每個方案都報告解密後 latent 的保真度（cosine、RMSE、低頻 cosine、\|z\| Pearson）。P 的 re-inversion 保真度只作參考，**不再**用來判定其他方案。
4. T1 的正控制（P 精確 latent，Top-1 應為 1）一律報告，作為 scorer 正確性的檢查。
5. 匿名影像維持學長的 per-image min-max uint8 PNG 儲存；改為無損儲存不在本次裁決範圍內（AUD-20261010-01 §5-Q2）。

## 5. 診斷：latent 距離不能預測影像可逆性

- `[事實]`（稽核方 `audit/e2_latent_direction_sensitivity.py`）P 的 re-inversion 誤差（RMSE 約 1.13）與大幅白雜訊同等級，但前者生成的影像對 P 輸出仍有 43.9 dB，後者只有 6.4 dB；S0 的解密 latent 誤差（RMSE 0.132）與同大小白雜訊的影像影響相當（31.8 對 31.3 dB）。
- `[結論]` 任何各方向等權的 latent 距離（cosine、RMSE、L2）都無法預測影像可逆性；可逆性一律以影像端指標判斷，latent 指標只用於診斷與攻擊特徵分析。
