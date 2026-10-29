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
| **B1 respaced DDIM（減步）** | 近似，隨步數退化 | 1×/步 | 否 | 基準線（見 §8/§9） |
| **B3 DPM-Solver++(2M)**（[CVPR 2024 exact-inv](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.pdf)；本專案用 2 階近似反演） | 近似（2 階，往返非 bit-exact） | 1×/步 | 否 | **本回合實作 → 最佳**：N=100(5×) 45.6 dB（DDIM 38.9），見 §10 |
| BDIA（[ECCV 2024, 2307.10829](https://arxiv.org/pdf/2307.10829)，γ=1） | 精確（leapfrog） | 1×/步 | 否 | **對齊 backward 重做**：N=200 +2.4 dB、N≤50 崩，見 §10 |
| B2 BELM（[NeurIPS 2024, 2410.07273](https://arxiv.org/pdf/2410.07273)） | **精確**（雙向顯式線性多步） | 1×/步 | 否 | 本設定整數-timestep 下**不穩定/非單調**，擱置（§前述 smoke） |
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

## 9. 加密端到端減步結果（對齊 decode）

> 全程用對齊 respaced decode（§前述修掉 legacy backward 的 decode-grid 偏移）。4-pass 完整流程：`z=F(x0); anon=B(E(z)); [交接]; ẑ'=F(anon); x_rec=B(D(ẑ'))`。dev idx {0,1,10,11}，recon PSNR vs x0。

### 9.1 float 交接（隔離 sampler+cipher；`scripts/aligned_encrypted_sweep.py`）

| N | P | S0 | S1 | S2a | S2a−P |
|---|---|---|---|---|---|
| 499 | 40.23 | 45.10 | 46.35 | 46.26 | +6.04 |
| 100 | 34.56 | 36.34 | 38.41 | 38.22 | +3.65 |
| 50 | 32.27 | 31.33 | 33.32 | 33.09 | +0.82 |

* crypto sanity：P/S0/S1 max|D(E(z))−z|=0；S2a=9.5e-7。三方案 recon 全部 ≥ P（方向與凍結 float 數據一致）。

### 9.2 部署真實（range-preserving PNG；`scripts/deploy_s2_sweep.py`）

| N | P | S2a | **S2fn** | S2fn−P | 整套 4-pass 秒 |
|---|---|---|---|---|---|
| 499 | 40.21 | 45.26 | **45.63** | +5.42 | 33.8 |
| 400 | 39.54 | 43.60 | **45.27** | +5.74 | 27.0 |
| 200 | 37.41 | 42.09 | **41.77** | +4.36 | 13.5 |
| 100 | 34.62 | 37.79 | **37.67** | +3.05 | 6.8 |
| 50 | 32.27 | 32.92 | **33.01** | +0.73 | 3.4 |
| 25 | 28.88 | 27.52 | **27.83** | −1.06 | 1.7 |

* [事實] range-preserving 幾乎不傷 S2（N=499 float 46.26→PNG 45.26），不像 legacy PNG 砸 S0/S1 到 31。
* [事實] **S2fn ≈ S2a 全 N**（差 <1.7 dB）→ fixed-norm 藏 norm 的保真代價 ≈ 0。
* [事實] S2 ≥ P 一路到 N=50；N=25 才跌破 P（~28 dB，不可用）。
* [推定] 操作點：N=200（~13.5s）、N=100（~6.8s,~5×）、N=50（~10×,仍 ≥ P）。**注意**：本表為 **aligned decode**，凍結 Q8_FORMAL（S2 38.8）為 **legacy decode**，兩者相差約 +4.3 dB 的 decode 慣例（見 §9.3），故**不可**把 aligned-N=100 的 37.7「≈ 凍結 38.8」讀成「N=100 重現凍結操作點」——那是跨慣例巧合。要證操作點須在 **同一 decode 慣例**下比（RQ4 重報，§11.2）。
* 「P 46→40」=單輪往返（匿名圖=P 輸出，N=499 單輪 46.62）vs 雙輪 recon（40.21）；非異常。

### 9.3 Decode 慣例警告（aligned respaced vs legacy；AF-044）

> [事實] respaced **forward** 與 legacy forward **逐位元相同**（`backward_offbyone_check.json`：`forward_z_max_abs_mine_vs_legacy=0.0`）→ 本研究線與凍結基準的差異**全在 backward/decode**。
> [事實] 固定同一 z、full-density：aligned decode 47.4 dB vs legacy decode 42.8 dB（+4.6 dB）；dose-response `backward_offset_sweep.json` delta0/1/2/3 = 47.4/47.0/44.7/42.2 → **legacy ≈ delta2–3**（decode eval 比標準對齊慣例偏移 2–3 格）。
> [推定] 「aligned 較正確、legacy decode 有偏移」可信（respaced 用標準 DDIM 生成慣例、forward 又與 legacy bit-identical）；但 **legacy 的確切機制未定位、非單純一格**（C「持續下移一格」反而崩到 6.4 dB）→ 根因維持 [推定]。
> [限制/決定] 凍結基準（Q8_FORMAL／RQ4：P/S0/S2/M2）**全用 legacy decode**。因此本檔所有 §9–§10 的 aligned 數字與凍結數字相比均為**跨慣例**：(i) sampler 之間在**同一 grid 內**的對比 self-consistent、成立；(ii) 與凍結數字比較時，**decode 慣例增益（+4.3 dB）必須與 sampler 增益分列**。RQ4 以新 sampler 重報前（§11.2 item 1），**先在 aligned decode 上重算凍結 DDIM 基準**，否則 +4~7 dB 會把 decode-fix 與 sampler 增益混在一起。
> 診斷腳本：`scripts/backward_offbyone_check.py`、`scripts/backward_offset_sweep.py`（本回合補 commit，Refs AF-044）。

## 10. 層 B 取樣器對比：BDIA（對齊重做）與 DPM-Solver++（本回合）

> 目的：回答「換取樣器能不能在同 NFE 下比 DDIM 更準／更快」。全部用對齊 respaced forward/backward（修掉 legacy decode 偏移）、key-free 往返 PSNR(x_rec,x0)（加密 bit-exact 與步數正交）、seed 1911、dev idx {0,1,10,11}。腳本 `scripts/exact_inv_sweep.py`。
>
> **為何 BDIA 要重做**：前作 BDIA 附錄（`scripts/bdia_backend.py`）在 T=500/Δt=1 且與 **legacy backward**（有 ~4.5 dB decode-grid 偏移）相比，結論「<0.4 dB、被取代」。對齊 backward 後結論改變。

### 10.1 三取樣器 × 步數（4 圖平均；同 NFE＝evals/pass）

| N | evals | DDIM | BDIA | **DPM++** | s/pass | 加速 |
|---|---|---|---|---|---|---|
| 499 | 499 | 46.62 | 47.78 | **48.08** | 9.09 | 1× |
| 200 | 200 | 43.38 | 45.75 | **47.79** | 3.64 | 2.5× |
| 100 | 100 | 38.85 | 39.26 | **45.64** | 1.80 | 5× |
| 50 | 50 | 33.89 | 30.97 | **40.55** | 0.90 | 10× |
| 25 | 25 | 28.65 | 27.05 | **33.64** | 0.46 | 20× |

* 驗證關卡全綠：三者 full-density ≥ legacy anchor(42.35)−1、單調、全有限。
* [事實] **DPM-Solver++(2M) 完勝**：同 NFE 下 **N=100（5×）45.64 dB ≈ full-density**（DDIM 僅 38.85，差 +6.79）；N=50（10×）40.55 dB 仍勝過 DDIM 的 N=100；幾乎不隨步數退化（二階指數積分器每步更準）。
* [事實] **BDIA（γ=1 leapfrog）對齊後**：N=200 +2.36、N=100 +0.41 dB 小勝 DDIM，但 N≤50 二階 leapfrog 在超粗整數格點崩（−2.9/−1.6）。遠優於舊 legacy-backward 的「<0.4 dB」。
* [限制] DPM++ 2M 的 bit-exact 反演有乘多步 buffer 方向障礙（反演往噪方向需尚未算出的 `x0_prev`，同 BELM 風險級），故採**對稱二階近似反演**（diffusers inverse-scheduler 同款），往返非 bit-exact；公平比較的是同 NFE 的往返 PSNR。

### 10.2 非退化確認（`scripts/exact_inv_degeneracy_check.py`，N=100）

「DPM++ +8 dB 太好」必查是否潛碼退化（z 保留影像結構 → decode 虛高還原、無法加密）。結果**洗清**：

| 取樣器 | z mean | z std | z absmax | recon | anon(sign-flip) vs x0 |
|---|---|---|---|---|---|
| DDIM | +0.140 | 0.958 | 5.55 | 38.83 | 5.91 |
| BDIA | +0.140 | 0.964 | 5.67 | 40.96 | 5.93 |
| DPM++ | +0.140 | 0.958 | 5.57 | 46.92 | 5.91 |

* [事實] 三者 z 分布**相同**（~N(0,1)：mean +0.14、std ~0.96）、加密後 decode 偏離 x0 **一致（5.91 dB）** → 潛碼同樣可加密、匿名力道相同。DPM++ 的高還原純來自取樣器精度，非退化。

### 10.3 決定與後續

* [決定] **採 DPM-Solver++(2M) 作為層 B 的加速取樣器**；B1 respaced DDIM 留作基準、B2 BELM/EDICT 擱置。
* [推定] 建議操作點 **N=100（5×，~45.6 dB）**；對速度更硬可到 N=50（10×，~40.6 dB，仍 ≥ legacy baseline）。
* [事實] DPM++ 已接上**部署真實**管線（見 §10.4），端到端同樣拿到 5× 紅利。
* [待辦] 可與層 A（省 reinvert pass）疊加，進一步 1.5×。

### 10.4 部署真實版 DPM++ vs DDIM（range-preserving PNG + S2 加密；`scripts/deploy_dpm_sweep.py`）

> 完整 4-pass 管線（`z=F(x0); anon=B(E(z)); anon_png=PNG(anon); ẑ'=F(anon_png); x_rec=B(D(ẑ'))`），兩取樣器同介面、同 NFE。4 圖平均 recon PSNR vs x0。crypto-sanity 綠（P max|D(E(z))−z|=0、S2fn 9.5e-7）。DDIM/S2fn 逐格**完全重現 §9.2**（45.63/41.77/37.67/33.01/27.83）→ 決定性驗證通過。

| N | 加速 | DDIM/P | DPM++/P | DDIM/S2fn | **DPM++/S2fn** | Δ(S2fn) | s/4pass |
|---|---|---|---|---|---|---|---|
| 499 | 1× | 40.21 | 41.96 | 45.63 | **46.90** | +1.27 | 36.0 |
| 200 | 2.5× | 37.41 | 41.70 | 41.77 | **46.62** | +4.85 | 14.6 |
| 100 | 5× | 34.62 | 40.39 | 37.67 | **44.26** | +6.59 | 7.5 |
| 50 | 10× | 32.27 | 36.57 | 33.01 | **38.62** | +5.61 | 3.9 |
| 25 | 20× | 28.88 | 33.59 | 27.83 | **32.65** | +4.82 | 2.1 |

* [事實] **DPM++/S2fn 在 N=200（2.5×、~14.6s）46.62 dB 已勝過 DDIM/S2fn full-density（45.63）**；N=100（5×、~7.5s）44.26 dB 仍遠勝 DDIM 同步數的 37.67，也勝過 DDIM full-density。
* [事實] N=50（10×、~3.9s）DPM++/S2fn 38.62 dB 仍勝過 DDIM/S2fn 的 N=100（37.67）。端到端 recon 幾乎不隨步數退化。
* [事實] S2 ≥ P 的既有現象在 DPM++ 下維持（DPM++/S2fn > DPM++/P 全 N）。range-preserving PNG 對 DPM++ 幾乎無損。
* [決定] **部署採 DPM-Solver++(2M)、操作點 N=100（5×，~7.5s/4pass，端到端 ~44 dB）**；對速度更硬可到 N=50（10×，~3.9s，~38.6 dB，仍 ≥ 原凍結 DDIM baseline）。
* [限制/AF-045] 此決定**僅就保真/速度**成立；本節全為 key-free 往返/recon，**無任一安全指標在 DPM++ 下重測**。「安全平價」為 [推定]，須待 §11.2 item 3（T2BB-3/4＋MC-2，攻防皆 DPM++ N=100）實測。在該實測前，**不得**於論文宣稱「DPM++ 部署安全不劣於 DDIM」。另：§10.x 與凍結數字的比較受 §9.3 decode 慣例警告約束。

## 11. Pipeline 切換重做計畫（DPM++ N=100 + 4-pass 定案後）

> [決定] 新標準 pipeline：**取樣器 DPM-Solver++(2M)、N=100**；**4-pass**（層 A：最後一趟從解出的 latent 分岔同生「還原圖 + 偽健康圖」，省掉 reinvert 那趟）。
> 本節盤點切換後**哪些實驗要重做**。來源為 `scripts/p11_experiment_matrix.py` 的實際登錄表。

### 11.0 原則
* [事實] 本次只改**取樣器 + flow**，**模型權重與密碼本身未變** → 非 §4「換 checkpoint 全重凍」情境。
* [事實] 潛碼邊際分布實測不變（degeneracy check：DDIM/BDIA/DPM++ 三者 z mean +0.140、std 0.958–0.964 一致）。
* [事實] **發布物是匿名圖 anon = decode(加密後 latent)**，其殘留身分**由加密決定、非取樣器精度**；`anon vs x0` 三取樣器完全一致（5.91 dB）。**還原圖 x_rec 的高保真只有持鑰者可見 → 是效用提升、非隱私風險**。
* [決定] 故安全**預期維持**；但「影像空間」攻擊指標仍須在新設定**重跑實證**（方向上保真↑ 可能利敵，必須用數據堵住，不能以「預期」交差）。攻防雙方重跑時**一致採 DPM++ N=100**。

### 11.1 分類（依登錄表）

| 類別 | cell | 取樣器相依 | 動作 | 理由 |
|---|---|---|---|---|
| 密碼不變量 | RQ2/RQ3 D5/D6/D7、M2 byte-exact、QᵀQ=I、S0/S1 bit-exact、KPA/nonce | 否 | 沿用 | 線代/密碼性質，作用於 z 代數，與 z 來源無關 |
| 分布不變 | RQ2 加密後 ~N(0,1)（KS/動差） | 否* | 便宜重確認 | z 分布實測不變，預期過 |
| latent-space linkage | T2BB-1（直接 latent cos） | 否* | 重確認 | cos 結構依 z 分布，預期不變 |
| 影像空間攻擊 | T2BB-3（M1 frozen linkage）、T2BB-4（image control）、MC-2（外部 re-id）、MC-3、T2-WB inversion | 是 | **重跑** | 指標作用於 anon/重建影像，取樣器會變 → 須證攻擊未變易 |
| RQ1 攻擊 | A3/A4/A4.6 | 視實作 | 查後重跑影像相關者 | latent-view 沿用、image-view 重跑 |
| 保真/可逆主表 | RQ4 Q8_FORMAL（fidelity/runtime/storage） | 是 | **重報** | PSNR/SSIM/latent-cos/runtime 全變（更好），主結果 |
| 層 A 等價 | ABL-2 reinvert ablation | 是 | **重做** | 4-pass 即此條；須在 DPM++ N=100 重證分岔不損、偽健康圖品質不變 |
| sampler ablation | ABL-1 BDIA 2x2 | — | 被 §10 取代 | 新三取樣器對比已涵蓋 |
| 定位 | MC-1（對外文獻） | 否 | 沿用 | 外部數字 |

\* 預期不變，屬「便宜就重確認」級。

### 11.2 必做三項（最小可辯護集）
1. [x] **RQ4 重報**（formal N=200 已交付，見 §11.7）：部署 DPM++ N=100 null、P 45.74 dB / S2fn 46.42 dB（patient-cluster CI + LPIPS）；凍結 legacy DDIM 保留為參考 baseline。
2. [待辦] **層 A 等價 ablation 重做**：DPM++ N=100 證「分岔生成 ≈ 不分岔」（還原圖 cos/PSNR、偽健康圖品質不變）。
3. [待辦] **影像空間安全平價檢查**：T2BB-3/4 + MC-2，攻防皆 DPM++ N=100，證「更快更清晰但攻擊未變強」。

### 11.3 策略與工時
* [決定] **不整套重凍**。保留凍結 DDIM baseline 當參考，DPM++ N=100 以「部署組態」並列，附新 fidelity + 安全平價證據。
* [推定] 工時：RQ4 重報 + 層 A ablation ~0.5–1 天；影像空間安全 3 cell ~1–2 天；latent/分布重確認 ~0.5 天；**合計 ~2–3.5 天**，GPU 需求小（N=100 快），不涉重訓。

### 11.4 Item 1 基礎：decode 慣例 × sampler 分解（回應 AF-044；`scripts/rq4_redo_decode_sampler.py`）

> 在 RQ4 以新 sampler 重報前，先把「換 decode 慣例」與「換 sampler」兩種增益分離。固定 storage=range-preserving PNG 與 scheme，跑完整 4-pass 部署管線，各 backward 慣例量全指標。dev idx {0,1,10,11}。aligned-DDIM 各 N 的 P/S2fn 與 `deploy_dpm_sweep` 逐格一致（決定性交叉驗證）。

| scheme | legacy-DDIM(499)〔凍結慣例〕 | aligned-DDIM(499) | decode 慣例增益 | aligned-DDIM(100) | aligned-DPM++(100) | sampler 增益@100 |
|---|---|---|---|---|---|---|
| P | 38.18 | 40.21 | **+2.03** | 34.62 | 40.39 | **+5.78** |
| S2fn | 39.13 | 45.63 | **+6.50** | 37.67 | 44.26 | **+6.59** |

* [事實] 兩個增益由**各自單變數受控對比**量得（(a) 同 sampler 同 N、只換 backward 慣例；(b) 同慣例同 N、只換 sampler），**非相加**（兩對比在不同操作點）。
* [事實] cosine 全 ~1.0、SSIM 0.96–0.999；aligned 全程優於 legacy 慣例。decode 慣例增益在 S2fn（+6.5）遠大於 P（+2.0）—— S2fn 的保值域+fixed-norm 讓 aligned decode 的改善更能留存。
* [決定/AF-044] **aligned-DDIM 為正確的 re-baseline**；報 DPM++ 的 sampler 增益一律對 aligned-DDIM（同慣例同 N），不對 legacy-decode 凍結數字；與凍結數字相比時明列 decode 慣例分量。
* [限制] 此為 dev n=4 單次；formal N=200（security_v1，patient-cluster CI、LPIPS）為 item 1 的正式交付，待後續。

### 11.5 Item 2：層 A 等價 ablation @ DPM++ N=100（`scripts/layerA_dpm_ablation.py`）

> 4-pass 把最後一趟分岔：從解出的 x_rec_T 同生還原圖（無條件）與偽健康圖（CFG），省掉「把還原圖 re-invert」那趟。驗證 A（re-invert 後生成）≈ B（直接從 x_rec_T 生成）。CFG eps=(1+gs)eps_cond−gs·eps_uncond（cond=y0、uncond=null）。dev idx {0,1,10,11}、N=100、gs∈{0,2}。

| sampler | recovery cos (A,B) | recovery psnr | healthy gs2 cos (A,B) | healthy gs2 psnr |
|---|---|---|---|---|
| aligned-DDIM | 0.99987 | 40.7 | 0.99987 | 40.8 |
| **DPM++** | **0.99997** | **46.7** | **0.99997** | **47.1** |

* [事實] DPM++ 下 re-inversion 完全冗餘：A≈B 的 cos **0.99997**（還原與偽健康皆是），追平並超過前作 legacy DDIM 的 0.9999（ABL-2）。→ **4-pass 層 A 對新 sampler 成立**，可安全省掉一趟 inversion。
* [事實] DPM++ 的 A-vs-B 一致度（cos/psnr）高於 aligned-DDIM，與其較緊的 latent 往返一致。
* [決定] 新標準 pipeline 的 4-pass 分岔（item 2）**已實證**；偽健康圖品質（A vs B）在 gs=0/2 均不受省趟影響。

### 11.6 Item 3：影像空間 linkage 安全平價 @ DPM++ N=100（回應 AF-045；`scripts/t2bb_dpm_parity.py`）

> 防禦端改用 DPM++ N=100 端到端（invert x0→z、加密、anon=backward(E(z))），發布表徵與出貨一致（P/S0/S1→legacy PNG、S2→private-range public-q）。攻擊端為凍結預訓練 encoder（densenet121-imagenet、rad-dino；sampler-agnostic）。anon↔anon 同病患配對 linkage，t2bb_inner_v1 20 對，label-permutation null。

| encoder | 法 | DPM++ top1 / auc | legacy top1 / auc（凍結表） |
|---|---|---|---|
| densenet | P | 0.550 / **0.821** | 0.50 / 0.81 |
| densenet | S0 / S1 / S2 | 0.446 / 0.481 / 0.381（auc） | 0.434 / 0.479 / 0.558 |
| rad-dino | P | 0.750 / **0.975** | 0.75 / 0.976 |
| rad-dino | S0 / S1 / S2 | 0.539 / 0.504 / 0.486（auc） | 0.532 / 0.498 / 0.510 |

* [事實] P 對照在 DPM++ 下幾乎逐格重現凍結 legacy（densenet 0.821 vs 0.81、rad-dino 0.975 vs 0.976）→ 攻擊管線一致、對照有效。
* [事實] **S0/S1/S2 在 DPM++ 下全維持 ~隨機**（auc 0.38–0.54，null_auc≈0.50、random top1=0.05），無一上升；S2 甚至略低於 legacy。→ **換 sampler 未提高釋出匿名圖的可連結性**。
* [結論] AF-045 的 T2BB frozen-encoder linkage（T2BB-3/4 等價）在 DPM++ N=100 下**安全平價成立**：高還原保真只利持鑰者還原、不利凍結 encoder 連結攻擊（與 degeneracy check 的「anon 洩漏由加密決定」一致）。
* [限制] S2 用單次隨機金鑰 + per-image nonce 抽樣（與凍結表同政策，金鑰不落盤只記 SHA-256）；「~隨機」結論穩健、精確 auc 隨金鑰抽樣微動。**MC-2 外部學習式 re-id（Siamese shadow-1000）在 DPM++ 下的重跑仍待做**，為完全關閉 AF-045 的最後一塊。

### 11.7 Item 1 正式交付：RQ4 formal 重報 @ 部署操作點（`scripts/rq4_formal_redo.py`）

> §11.4 的 decode/sampler 分解是 dev n=4 的 re-baseline；本節是 **formal N=200 正式交付**。沿用凍結 Q8_FORMAL 的方法學（security_v1 200 病患 patient-disjoint、patient-cluster 百分位 bootstrap CI seed 1911/10000 resamples、PSNR/SSIM/cosine/L∞/RMSE/MAE + 真 LPIPS(alex)），但操作點換成**部署版 DPM-Solver++(2M) N=100 + null backward + range-preserving PNG**（supersede 凍結表的 legacy DDIM T=500 / y=0 / legacy PNG）。不改已稽核的 `q8_pilot_runner`，另寫自含 runner。產物 `paper_assets/rq4_formal/{table_fidelity,table_runtime,per_sample_long,analysis_manifest}.csv/json`。

| 方案 | recon PSNR dB（CI95） | SSIM | cosine | LPIPS（CI95） | 匿名圖 vs x0（單輪） | 4-pass runtime s |
|---|---|---:|---:|---|---:|---:|
| **P**（恆等金鑰，可逆上界） | **45.74** [45.60, 45.88] | 0.9919 | 0.99996 | 0.029 [0.027, 0.031] | 49.83（≈還原） | 6.15 |
| **S2fn**（部署密文） | **46.42** [46.19, 46.63] | 0.9947 | 0.99996 | 0.019 [0.018, 0.021] | **5.27**（雜訊狀密文） | 6.31 |

* [事實] 部署操作點的 RQ4 保真主表：P **45.74 dB**、S2fn **46.42 dB**（N=200、patient-cluster CI）。對照凍結 Q8_FORMAL 的 **legacy P 37.46 dB** → 同一評估集、換到部署操作點後 **+8.3 dB**（多因：sampler + null 對齊 + range-preserving 儲存，分解見 §11.4，單因不可合併）。
* [事實] **S2fn ≥ P**（46.42 > 45.74）維持（§9/§10 的「加密 ≥ P」現象在 formal N=200 成立）；S2fn 匿名圖對 x0 單輪僅 **5.27 dB**（密文 noise-like，隱私正常），還原後 LPIPS 0.019（感知近乎無損）。
* [決定] 此為 §11.2 item 1 的正式交付：主表以部署操作點（DPM++ N=100 null）報 P/S2fn，凍結 legacy DDIM 保留為參考 baseline。
* [限制] scheme 限 P 與 S2fn（部署 scheme-of-record）；S0/S1 為前作方案，其凍結數字續作前作對照，未在此操作點重跑。runtime 為單 GPU batch=1。
