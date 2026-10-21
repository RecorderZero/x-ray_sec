# 學長解密 + 異常偵測生成流程（含程式碼對照）

> 用途：釐清前作「匿名圖 → 解密 → 重構圖 / 健康圖 / heatmap」的實際程式流程，供論文與後續研究引用。
> 來源一律為 `past/SourceCode/`（唯讀）實際程式碼，標 `file:line`。前綴：`[事實]`（程式碼佐證）、`[評估]`（本人判讀/推論）、`[限制]`。

---

## 0. 一句話

[事實] 前作的「deanonymize」模式其實是**解密 + 弱監督 diffusion 異常偵測**：先把匿名圖解密還原成 `x_rec`（重構圖），再對 `x_rec` 生成一張 CFG 趨健康的反事實 `x_healthy`（健康圖），兩者相減得 `heatmap` 定位病灶。生成器就是這顆 **pixel-space diffusion（CFG-DDIM）**，沒有 GAN。

---

## 1. 完整流程（file:line）

入口腳本 `scripts/cfg_image_sample_anonymization.py`（S2/enhanced 版 `cfg_image_sample_enhanced_anonymization.py` 結構相同）：

- `L194-195` `classes = th.zeros(...)` → **y 強制為 0 = 健康（no_finding）**；`model_kwargs = {y=0, clf_free=True}`
- `L221-235` `deanonymize` 分支呼叫 `ddim_sample_loop_anomaly_detection_with_deanonymization(..., guidance_scale=args.guidance_scale)`
- `L236` `org = x_rec`（之後 heatmap 以 x_rec 為參考）
- `L332-354` 存三種圖：`recovered/`（=x_rec，**重構圖**）、`healthy/`（=sample，**健康圖**）、`heatmaps/`（=|x_rec−x_healthy|）

主編排 `guided_diffusion/gaussian_diffusion.py:1706`
`ddim_sample_loop_anomaly_detection_with_deanonymization` → 回傳 `sample, x_noisy, x_rec`：

```
匿名圖 x_ano
  │
  ├─【解密路徑】ddim_sample_loop_anonymization(mode='deanonymize', guidance_scale=-1)      (L1757 → L1550)
  │     Step1  ddim_anonymization_forward   (L1424)   DDIM 正演(inversion):x_ano → x_ano_T   (noise_level=500, eta=0 確定性)
  │     Step2  施鑰逆運算 (L1664-1689) 於 x_T 空間、僅 mask 區：x = M⊙(k⁻¹·x_T) + (1−M)⊙x_T
  │              · S1  SignedPermutationKey.apply_inverse   (逆號 + 逆置換)
  │              · S0  Rademacher 自逆  x_rec_T = k ⊙ x_ano_T   (k⊙k=1)
  │     Step3  ddim_anonymization_backward  (L1489)   DDIM 反演:x_rec_T → x_rec
  │     └──► x_rec = 重構圖（此路徑 guidance=-1，無 CFG，理論上回到原圖）
  │
  └─【健康圖生成】ddim_sample_loop_known(img=(x_rec,), guidance_scale=w)                    (L1775 → L1005)
        · L1026-1032  t=randint(0,1)=0；x_noisy = q_sample(x_rec, t=0) ≈ x_rec（ᾱ₀≈1，近乎 identity）
        · ddim_sample_loop_known_progressive(time=noise_level)  (L1061)  — 兩個迴圈：
            迴圈A (L1102-1117)  ddim_reverse_sample：對 x_rec 做 DDIM 反演到 t=499（reverse_kwargs.null=True，無條件）→ 得 x_rec_T′
            迴圈B (L1125-1140)  ddim_sample：從 x_rec_T′ 以 CFG 去噪到 t=0（趨 y=0 健康）→ x_healthy
        └──► sample = 健康圖
```

CFG 本體 `condition_clf_free` (L457)：`eps = (1+w)·eps_cond(y=0) − w·eps_uncond(null=True)`。
啟用條件：`ddim_sample` (L768-774) 在 `guidance_scale != -1` 時才掛 CFG。

---

## 2. 兩個容易誤會的點（已對程式碼核實）

1. [事實] **「latent」在這份 code 叫 `x_T` / `x_ano_T`**：是 timestep T=500 的加噪影像（pixel-space UNet，無 VAE）。解密在 `x_T` 空間施鑰，不是另一個潛空間。
2. [事實] **健康圖那步不是「q_sample 加噪」，而是「再做一次 DDIM inversion」**：`ddim_sample_loop_known` 裡 `q_sample` 用 t=0（ᾱ₀≈1）幾乎是 identity；真正回到 latent 的是 `progressive` 的**迴圈A（確定性 DDIM 反演、null 無條件）**，迴圈B 才是 CFG 去噪。所以 `x_noisy`（回傳值）實際上是 x_rec 的再反演латент `x_rec_T′`，不是隨機加噪樣本。

---

## 3. 設計討論：為什麼不直接把 `x_rec_T` 拿去 CFG？（回答此問）

[評估] **這個質疑基本上成立。** 把 §1 的 pass 數攤開看：

| 階段 | 昂貴的 diffusion pass |
|---|---|
| 解密正演 x_ano→x_ano_T | 1（inversion） |
| 解密反演 x_rec_T→x_rec | 1（backward，產生重構圖） |
| 健康圖-迴圈A x_rec→x_rec_T′ | 1（**再一次 inversion**） |
| 健康圖-迴圈B x_rec_T′→x_healthy | 1（CFG backward） |
| **合計** | **4** |

而 `x_rec_T′`（把 x_rec 再反演）在數學上≈ `x_rec_T`（我們解密後早就有的 latent）——DDIM eta=0 反演是確定性的，差別只是 x_rec 的 backward+forward 多累積了一輪數值誤差（本研究量到 m1lat cosine≈0.9997，非 1.0）。**等於把同一個 latent 算了兩次、丟掉第一次。**

你的方案：解密得到 `x_rec_T` 後，直接從它分岔——(a) 無條件 backward → `x_rec`（重構圖，heatmap 仍需要它），(b) CFG backward → `x_healthy`。即 **1 正演 + 2 反演 = 3 passes**，省掉那次重複的 inversion（約 −25% 生成時間），且因少一輪 round-trip，數值上**至少不會更差、通常更準**。

[評估] **那為什麼前作要繞一圈？** 看主編排的註解（`gaussian_diffusion.py:1774` "Now run the normal `ddim_sample_loop_known` with x_rec as input"）——`ddim_sample_loop_known` 是**既有的異常偵測函式，介面吃「影像」**；deanonymize 只是把 `x_rec`（影像）串進這個現成函式。這是**程式重用 / 省事**，不是演算法上的必要。對應到兩種 image-editing 範式：前作等於「影像先還原再重新 encode」，你的建議等於「直接用 inversion latent 編輯」（prompt-to-prompt / null-text 那一類），後者對確定性 DDIM 反演其實更自然。

[限制] 唯二要保留的理由很弱：
- `x_rec` 這張影像**本來就是必要輸出**（重構圖 + heatmap 參考），所以那個 backward 省不掉；能省的只有「再反演」那一次正演。
- `ddim_anonymization_backward` 的 `clip_denoised=True` 會把像素夾回合法範圍，再反演得到的 `x_rec_T′` 可能比帶解密數值殘差的原始 `x_rec_T` 稍「乾淨」——影響極小，未量化。

[評估] 結論：**可以改成重用 `x_rec_T`，省 1 個 inversion pass、品質至少不劣**；前作沒這麼做純粹是接現成的影像介面函式。此結論已以小實驗證實（§3.1）。

### 3.1 實測（2026-10-22，N=10，`splits/security_v1.csv` 前 10 張，gs∈{0,2}）

用前作自己的建構塊（`legacy_invert`→`legacy_generate` 得 x_rec_T / x_rec；唯讀 import，未改 `past/SourceCode/`）比較：
- **A（前作）** `ddim_sample_loop_known(x_rec, gs)`：把 x_rec 再反演 → z_T′ → CFG 去噪
- **B（提議）** `ddim_anonymization_backward(x_rec_T, gs)`：直接從解密 latent CFG 去噪

| 指標 | gs=0 | gs=2 |
|---|---|---|
| psnr(A,B) 平均 | **47.85 dB** | **47.55 dB** |
| cosine(A,B) 平均 | **0.99993** | 0.99991 |
| 健康步驟耗時 A / B | 22.90s / 15.23s | 22.91s / 15.23s |
| **B 相對 A 加速** | **1.50×** | **1.51×** |

- [事實] 兩法健康圖**幾乎逐像素相同**（psnr≈47–48 dB、cos≈0.9999，10 張皆然）；10× 放大的 |A−B| 圖近乎全黑，只剩殘餘數值噪聲 → **再反演確為多餘 round-trip**。
- [事實] **B 穩定快 ~1.50×**（省掉那趟 inversion pass）。
- [限制] 單次 GPU 浮點量測；健康步驟計時不含共用的 x_rec（重構圖/heatmap 參考本就必算）；此 pipeline 在 gs=2 下健康化幅度本就不大（healthy≈x_rec），但不影響 A vs B 結論。
- 來源：`scripts/reinvert_ablation.py`、`paper_assets/reinvert_ablation/{reinvert_ablation.csv,reinvert_ablation.json,grid_gs2.png}`（並排圖欄序：x0 | x_rec | healthy_A | healthy_B | 10×|A−B|）。

---

## 4. 與本研究的關聯

- [事實] 這裡「反演用 null、生成用 y=0 條件」正是本研究 **BDIA 2×2 的 asymmetric conditioning**；也是 asymmetric 比 symmetric-null 低 ~1 dB 的來源（生成端帶健康條件，改了語意，是匿名/健康化的品質代價）。見 `reports/method_comparison.md`、`paper_assets/tables/table_bdia_2x2.*`。
- [事實] 生成「健康圖/重構圖」的生成器就是這顆 diffusion（CFG-DDIM），非 GAN。匿名圖「像噪聲」源於對外釋放加密後的 `x_ano`（其 `x_ano_T` 更是噪聲態），是加密設計的結果，不是缺 GAN。
- [限制] 本研究 S2 的威脅模型是「可逆/金鑰控制加密」；上述異常偵測生成屬前作的「解密後下游用途」示範，兩者不衝突。
