# new_model.md — 加速 / 輕量化路線規劃

> 用途：規劃「讓加解密 pipeline 更快」的選項、代價與時間。區分**換取樣器（不重訓）**與**模型輕量化（必重訓）**兩條路，並估計後者會連帶讓哪些實驗必須重做。
> 標記：`[事實]` 已查證（附 file:line / 來源）、`[推定]` 由事實推論、`[待辦]`、`[決定]`、`[限制]`、`[風險]`。

## 0. 一句話

加速分三層：**(A) 省冗餘 pass（免費 1.5×）**、**(B) 換快速且可反演的取樣器（不重訓）**、**(C) 模型輕量化（必重訓，且多數實驗要重做）**。建議先榨乾 A+B（低風險、不動 baseline），只有在仍不夠且速度是硬需求時才啟動 C，並當成獨立研究線。

## 1. 現況與瓶頸

* [事實] 單張完整加解密 ≈ **6 × ~7s ≈ 42s**：加密 2 pass（`invert x0→z`、`從 E(z) 生成匿名圖`）＋解密 2 pass（`invert 匿名圖`、`decrypt 後生成 x_rec`）＋2 pass（`x_rec 重新加噪 inversion`、`生成偽健康圖`）。
* [事實] 瓶頸在 DDIM 逐 timestep：`gaussian_diffusion.py:1462`（forward `indices = range(noise_level-1)`）與 `:1525`（backward 反序），**Δt=1、每 pass ≈499 次 model eval**。
* [事實] 固定 pixel-space checkpoint（`modelchexpert050000.pt`，無 VAE）。加密要求 **inversion 往返保真**（latent 要能回推才能解密），所以快取樣器必須能（近）精確反演——這把多數「只為生成設計」的快取樣器排除。
* [事實] 現有 loop **不支援 respacing**：`ddim_sample`/`ddim_reverse_sample` 用 `alphas_cumprod_prev/next`（緊鄰格），硬綁 Δt=1（`:796`、`:845`）。所以「減步」要新寫 respaced loop，不是傳個小數字。

## 2. 三層加速路線

### 層 A — 省冗餘 pass（不改模型、不改取樣器，零保真損失）

* [事實] reinvert 消融（`reports/predecessor_decrypt_flow.md §3`）已證：解密得到 `x_rec_T` 後直接分岔——(a) 無條件 → `x_rec`、(b) CFG → 偽健康圖——省掉「把 x_rec 重新 inversion」那趟，**6→4 pass、穩定 1.50×**，健康圖逐像素幾乎相同（cos≈0.9999）。
* [待辦] 把這條併進主 deanonymize 流程（新寫 loop，不改 `past/SourceCode/`）。**免費的 42s→~28s。**

### 層 B — 快速且可反演的取樣器（不重訓）

| 方法 | 可精確反演 | 每步成本 | 需重訓 | 狀態 |
|---|---|---|---|---|
| **B1 respaced DDIM（減步）** | 近似，隨步數退化 | 1×/步 | 否 | 本回合起手（見 §8） |
| **B2 BELM**（[NeurIPS 2024, 2410.07273](https://arxiv.org/pdf/2410.07273)） | **精確**（雙向顯式線性多步） | 1×/步 | 否 | 未試，最佳升級 |
| **B3 exact-inv DPM-Solver++**（[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.pdf)） | 精確 | 1×/步 | 否 | pixel-space 測過，較複雜 |
| BDIA（[ECCV 2024, 2307.10829](https://arxiv.org/pdf/2307.10829)） | 精確 | 1×/步 | 否 | **已做**：同 NFE、<0.4 dB，被 BELM 取代 |
| EDICT | 精確 | **2×/步** | 否 | 更慢，不採 |

* [決定] 層 B 的**驗證關卡**：任何新取樣器在 full-density（≈499 步）必須重現既有數字（無加密往返 P≈42 dB、full M1≈37 dB）才可信；seed 1911、雙空間報告、不改 `past/SourceCode/`。
* [推定] respaced DDIM 到 ~100 步很可能足夠 → 約 **5×**；不夠再上 B2。

### 層 C — 模型輕量化（必重訓）

* [事實] 蒸餾（progressive / consistency / LCM / guidance distillation）：可達 1–4 步生成，但要訓一顆 student。consistency 模型**非天生可反演**（把任意噪聲塌到原點），latent inversion 仍是開放問題（[CoSIGN](https://arxiv.org/html/2407.12676v1)/MACS 解的是 inverse problem，不是這裡要的 latent 反演）。
* [事實] 剪枝 / 較小 UNet：要 fine-tune / 重訓。
* [風險] 量化（PTQ INT8）看似免訓，但會**破壞密碼的 float 精度**（S0/S1 transform bit-exact、S2 max_abs 8.3e-7 都會垮）並傷反演精度 → **不建議用於需精確反演的路徑**。

## 3. 「輕量化逃不逃得了重訓？」

* [事實] **逃不了。** 凡動到權重（蒸餾／剪枝／量化感知訓練）都要（重）訓；只有換取樣器（層 A/B）不用動權重。
* [風險] 唯一「號稱免訓」的 PTQ 量化對本專案是地雷（見 §2 層 C），等於也不能用。

## 4. 「重訓後哪些實驗要重做？」

分兩類：

* **模型無關、可留**：純密碼不變量——transform-only MaxAbs／bit-exact、正交性 `QᵀQ=I`、AEAD byte-exact（`byte_equality_rate=1.0`）。這些是線性代數／密碼學，與用哪顆 model 無關。
* **必須全部重做（凡經過 generator 或其誘導的 latent 分布）**：
  * 保真/可逆：**RQ4**、error attribution、latent/image 指標、baseline correctness、BDIA/reinvert 消融（往返誤差是**這顆 model** 的性質）。
  * 安全：**RQ1 攻擊、RQ2 S2 防禦成功率、RQ3 norm 旁道、T2-WB inversion、T2-BB linkage、外部 re-id、method comparison**——攻擊成功率依賴 model 產生的 latent 統計，換 model 必重測。
* [結論] 換一顆新 checkpoint ≈ **整套 N=200 formal + attack/defense packages 全部重跑 + 重新 freeze（SHA256SUMS/manifest）+ claim_matrix 重對**。密碼的「可逆性/安全」宣稱都要在新 model 上重新成立，不能沿用舊凍結數字。

## 5. 時間估計（粗估，GPU 可用度需你確認；>30 min 的 run 一律 go/no-go）

| 階段 | 工時 | GPU | 備註 |
|---|---|---|---|
| 層 A 併入主流程 | 0.5–1 天 | 小 | 免費 1.5×，不動 baseline |
| 層 B1 respaced 掃描 | ~0.5 天 | smoke | 本回合起手 |
| 層 B2 BELM 實作+低 NFE 對比 | 2–4 天 | 中 | drop-in，沿用 `bdia_backend` 介面 |
| 層 B3 DPM-Solver++ exact-inv | +2–3 天 | 中 | 選配 |
| **層 C — 蒸餾** | | | |
| ‑ student 訓練實作+除錯 | 3–5 天工 | — | |
| ‑ 蒸餾訓練 compute | **數天**級 | **大（逐次 go/no-go）** | 原 model 50k steps；蒸餾常需數萬 steps + 多輪調參 |
| ‑ 全套 formal 重跑 + 重新 freeze + 重對 claim | **1–2 週** | 大 | 見 §4 |
| **層 C 合計** | **約 3–5 週（樂觀）起** | | 另含「蒸餾後反演保真/安全是否仍成立」的研究風險 |

## 6. 風險與原則

* [風險] **baseline 可比性**：C 會換 model，現有凍結數字不再可直接比較；C 必須自帶 baseline 重跑、論文另立一章，避免污染現有凍結結果。
* [風險] 量化破壞密碼精度 → 禁用於精確反演路徑。
* [決定] 每個新取樣器：full-density 先重現既有數字才信；seed 1911；雙空間（latent + image）報告；不改 `past/SourceCode/`、不動 `audit/`。

## 7. 建議決策順序

1. 層 A 併入（免費 1.5×，42s→~28s）
2. **層 B1 respaced DDIM 掃描**（本回合）——看最少能到幾步
3. 不夠 → 層 B2 BELM（低 NFE 精確反演）
4. 仍不夠且速度是硬需求 → 才評估層 C（獨立研究線、自帶 baseline、逐次 go/no-go）

## 8. 層 B1 執行紀錄

* 腳本：`scripts/respaced_ddim_sweep.py`（新寫 respaced forward/backward，沿用同一顆 model、`p_mean_variance`＋`_predict_eps_from_xstart`＋`alphas_cumprod`，eta=0、guidance −1；不改 `past/`）。
* 掃描 N ∈ {499, 200, 100, 50, 25, 10}；每張圖量 **無加密往返 PSNR(x_rec, x0)**（加密層 bit-exact，與步數正交，故步數效應純看 sampler）＋**每 pass 牆鐘**。
* 驗證關卡：N=499 往返需 ≈42 dB（對齊 baseline P）。
* 結果見 `paper_assets/respaced_ddim_sweep/`（json/csv/grid）與 `PROGRESS.md`。
