# T2-WB 可逆性指標與正控制協定（AF-019）

> 狀態：§1–§3 為已實作的量測規則；§4 的 `Inconclusive` 判定門檻是**提案，待使用者裁決**，裁決前不得用來判定 A4／D7 的 T2-WB 結果。
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

## 3. T2-WB 攻擊的正控制與負控制（已定義，A4／D7 實作）

- Gallery：E2.3 cache 的原圖 latent `z_i = ddim_anonymization_forward(x0_i)`，`security_v1`，N=200。
- Query（方案 s）：攻擊者對保存的匿名 PNG 自行 inversion 得 `ẑ_j^s`；T1 則直接取精確 `z_ano,j^s`。
- 特徵與 scorer 在 T1、T2-WB、所有方案間完全相同：S0 用 `|z|`，S1 用 `sort(|z|)`，另報 norm-only `‖z‖`；可加報低頻 `|z|` 區塊平均作為 re-inversion 穩健特徵（`[假設]` 身分資訊主要在低頻，見 AUD-20261008-03 §3.3）。
- **正控制**：方案 P（恆等金鑰，與 S0/S1 同一條加密流程）以相同的 re-inversion 與特徵做 linkage。它只衡量 inversion 誤差本身對攻擊的影響。
- **負控制**：gallery ID 以 seed 1911 隨機重排後的 Top-1（預期約 1/N）。
- 統計：Top-1、Top-5、mAP／CMC；以 query 為單位 bootstrap（B=10,000、seed 1911）95% CI。

## 4. `Inconclusive` 判定（提案，待使用者裁決）

對每個（特徵 f，威脅情境 T2-WB）組合：

- **方案 (a)（建議）統計可分離**：若 P 正控制 Top-1 的 95% CI 下界 ≤ 負控制 Top-1 的 95% CI 上界，則 S0/S1/S2a/S2 在 f 上的 T2-WB 結果全部標為 `Inconclusive`。不得寫成「攻擊失敗」或「方案安全」。
- **方案 (b) 固定倍數**：P 正控制 Top-1 未達隨機基準（1/N）的 10 倍即判 `Inconclusive`。好處是直觀；缺點是倍數沒有出現在 WORKFLOW 的預先登記門檻中。
- 正控制通過、但方案 s 的攻擊未達顯著時，只能寫「在 T2-WB 下此特徵的 linkage 未達顯著（正控制通過）」。依 PROPOSAL §5，這仍不是安全性宣稱。
- T1 的正控制（P 精確 latent，Top-1 應為 1）一律報告，作為 scorer 正確性的檢查。
