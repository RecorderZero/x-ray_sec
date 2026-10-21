# 方法比較：S2 相對外部醫學/可逆匿名化工作的定位

> 用途：論文 Related Work / Method Comparison 段落的事實底稿（P11.5）。數字一律標來源；外部數字引自 `refpaper/` 的 PDF 或原論文，本研究數字引自已凍結的 `paper_assets/`。
> 前綴遵循專案規範：`[事實]`（附來源）、`[結論]`、`[限制]`、`[決定]`。

---

## 0. 為什麼不能用單一數字宣稱「贏」

- [結論] 我們的 S2 屬於**可逆 / 金鑰控制**匿名化（image encryption）：輸出是噪聲狀密文、持金鑰者**精確還原**原圖、下游在解密後進行；主張是儲存/傳輸期隱私 + 精確還原 + 密碼級保證（AEAD/tamper/KDF）。
- [結論] 主流醫學匿名化（PriCheXy-Net、Privacy-Net、DP-Pix、paper 114）屬**不可逆去識別**：輸出是「看起來仍可診斷的合成/混淆 CXR」、**不需金鑰直接可用**、目標是可分享資料。
- [限制] 兩者威脅模型不同，**不存在共同的單一標量能判誰「好」**。本表的價值在：(i) 把 S2 定位在正確的方法家族（可逆/金鑰控制），(ii) 用共同可比的兩個軸（**再識別抵抗**、**下游效用**）做誠實對照，(iii) 凸顯 S2 獨有的密碼級安全分析。不得寫成「S2 的 PSNR 高於 PriCheXy-Net」這類跨範式比較。

---

## 1. 定位總表

| 方法 | 領域 | 可逆? | 使用需金鑰? | 輸出型態 | 再識別抵抗 | 下游效用 | 密碼/安全分析 | 來源 |
|---|---|---|---|---|---|---|---|---|
| **本研究 S2** | CXR (CheXpert) | **是（精確, AEAD）** | 是（解密） | 噪聲狀密文 | 密文 ≈ 隨機（T2-BB cos/abs/learned 皆 ~chance）；還原後身分保留 | **解密後 by-construction 保留**（Q11 parity） | **形式化 T1–T4 密碼分析 + AEAD/tamper/KDF + fixed-norm** | 本研究凍結表 |
| 前作 S0/S1 | CXR | 是 | 是 | 密文 | **透明**：anon cos = plaintext cos（AUC 0.657，弱） | 解密後保留 | 無（線性自逆，已被本研究破解） | contribution.md、T2-BB |
| PriCheXy-Net | CXR (ChestX-ray14) | 否 | 否 | 擬真合成 CXR | re-id AUC **81.8%→57.7%**（µ=0.01、10 次重訓平均） | 分類 AUC **80.5%→75.8%**（同 µ=0.01 點；弱匿名 µ=0.001 時 ~80%） | 無密碼分析（學習變形場） | refpaper 112 Table 1（MICCAI 2023） |
| Privacy-Net | CXR | 否 | 否 | 混淆影像 | Ver. AUC 49.8（112 基線） | 分類下降較大（57.5%） | — | refpaper 112 Table 1 |
| DP-Pix | CXR | 否 | 否 | DP 像素化 | Ver. AUC 50.0–52.5（112 基線） | 分類下降大（50.0–52.9%） | 差分隱私保證 | refpaper 112 Table 1 |
| RiDDLE | Face | 是（password） | 是 | 擬真去識別臉 | 身分層去識別 | 身分層 | StyleGAN2 latent encryptor | refpaper 112 |
| G2Face | Face | 是 | 是 | 擬真去識別臉 | 身分層去識別 | 高保真還原 | 生成+幾何先驗 | refpaper 113 |
| Secret-key diffusion | Face (CelebA-HQ/LFW) | **是（正確金鑰精確還原）** | 是 | 去識別臉 | 優於 RiDDLE/G2-Face/iFADIT | — | 確定性正逆向 diffusion 注入 secret key | arXiv 2510.01031（2025/10） |
| CXR-Patient-ReID（攻擊工具） | CXR | — | — | — | **verification AUC 0.9940（CXR14 同集）、0.9870（CheXpert 跨集、未微調）** | — | —（再識別網路本身） | refpaper 111 摘要+Table 1/2（Sci Rep 2022） |

---

## 2. 讀表要點

- [結論] **真正的同儕是「可逆/金鑰控制」家族**（RiDDLE、G2Face、secret-key diffusion），但它們都在**人臉**。S2 是這條路在**醫學影像 + 密碼級保證（AEAD/tamper/KDF/形式化 cipher 分析）**的實例——這是我們相對它們的差異點，也是相對前作 S0/S1（線性自逆、無密碼分析）的進步。
- [結論] **相對不可逆醫學去識別（PriCheXy-Net 等）**：它們在「不解密直接可用」這條軸上強（輸出是可用 CXR），而我們在「密文期零洩漏 + 精確還原 + 竄改偵測」上強。兩者互補，不是取代。
- [限制] 外部數字為各自資料集（ChestX-ray14/CelebA）與各自評估協定下的報告值，**不可直接與我們 CheXpert 的數字並列比高低**；只能並列為「不同範式各自的隱私-效用取捨」。

---

## 3. 可借用的外部攻擊者（強化本研究的安全/保真主張）

- [事實] CXR-Patient-ReID 的再識別 Siamese（ResNet-50、256×256、ImageNet normalize、emb 128、BCE）在 CheXpert 跨集達 AUC 0.987（refpaper 111）；我們 T2-BB step-1 的通用 DenseNet 僅 0.811（`table_t2bb_step1_image_control`）→ **借其架構可得更強、發表級的攻擊者**。
- [決定] 因兩 repo 皆 **GPL-3.0**，不把其 code 併入本 repo；改**依論文重寫架構**（license-clean）在我們的 `splits/t2bb_shadow_v1.csv`（CheXpert 1000×2）訓練，當外部架構攻擊者評：
  1. S2 密文 → 預期 re-id AUC ≈ 隨機（安全，外部強攻擊者坐實匿名性）。
  2. S0/S1 密文 → 對照。
  3. **解密還原圖 vs 原圖 → 預期高 AUC（坐實可逆保留身分/效用）**。
  4. plaintext CheXpert → 天花板（預期接近他們報告的 ~0.98）。
- [限制] 重寫架構訓練於 CheXpert，非他們 ChestX-ray14 的原權重；為 license-clean 的等效攻擊者，須於文中註明非原始 checkpoint。可選：另把他們公開權重 out-of-tree 跑一次做跨集 sanity（GPL 外部工具、引用）。

### 3.1 結果（已執行，`table_reid_external`，seed 1911，victim N=20×2）

| 條件 | Top-1 | AUC | 判讀 |
|---|---|---|---|
| shadow-val（訓練有效性） | — | **0.911** | 攻擊者確實學會再識別（對照 T2-BB 學習式過擬合 0.48） |
| plaintext（天花板） | 0.15 | **0.764** | 原圖 biometric 訊號存在；本研究舊 DenseNet 0.811、Packhäuser 報告 CheXpert 0.987 |
| S0 密文（range） | 0.15 | 0.616 | 殘留洩漏，與 T2-BB `cos` 0.657 一致（前作透明性） |
| S1 密文（range） | 0.10 | 0.541 | 近隨機 |
| **S2 密文（range）** | 0.05 | **0.445** | **≈ 隨機；同表徵下 < S0 0.616（per-image-nonce 效果）** |

- [事實] 攻擊者在 held-out shadow 病患上 pair-verification AUC 0.911（非過擬合）；對 victim 原圖達 0.764。來源：`paper_assets/tables/table_reid_external.{csv,json}`。
- [結論] **乾淨的比較是同一 latent-view 表徵內 S0（0.616）vs S2（0.445）**：攻擊者看到的都是密文 attacker-view（min-max 當灰階）、都 noise-like，S0（透明正交變換）仍殘留洩漏、S2 回到隨機 → **隔離出 per-image-nonce 的效果，S2 相對前作 S0 的改善成立**。與 T2-BB（線性 `cos`/`abs` + 弱學習式）互補——此處為影像級學習式攻擊者。
- [限制]（AF-042）plaintext 的 **0.764 是真實-CXR 表徵的天花板，不是 S2 被比較的基線**：CXR re-id 網路對 noise-like 的 latent-view 本就 OOD（見 §3.2 identity 控制 `plaintext_latent`=0.550，未加密即已近隨機），會同時因「密碼強」與「輸入 OOD」而崩潰，故**不**據此宣稱「S2 擊退達 0.76/0.98 的強攻擊者」；更誠實的說法是「現成／發表級 CXR re-id 網路無法連結 S2 的 noise-like 密文」。GPU 浮點訓練、單次量測、不保證跨次 bit-identical（同 Q11）；victim N=20 小；為 license-clean 重寫架構（非原始 ChestX-ray14 checkpoint）。

### 3.2 原始權重跨集 sanity（Packhäuser 公開 checkpoint，樹外執行）

- [事實] 取 Packhäuser et al. 2022 **公開 checkpoint**（`CXR-Patient-ReID`, GPL-3.0；weights sha256-oid `331efaed…`, 95.4 MB，經 LFS raw 端點下載並核對），**原封不動**套在與 §3.1 **完全相同**的 victim 集與輸入視圖上（只換權重）。於**樹外**（`/data2/external_gpl/`，不入庫、不散布）CPU 執行，seed 1911。

| 條件 | Top-1 | AUC | 對照 §3.1 reimpl |
|---|---|---|---|
| plaintext（真實 CXR） | 0.850 | **0.982** | 0.764 |
| plaintext_latent（identity 控制，同表徵、**不加密**） | 0.050 | **0.550** | — |
| S0 密文（range） | 0.100 | 0.560 | 0.616 |
| S1 密文（range） | 0.050 | 0.493 | 0.541 |
| **S2 密文（range）** | 0.100 | **0.477** | 0.445 |

- [結論] 原始 checkpoint 在我們 CheXpert victim 上 plaintext **AUC 0.982**，貼近其發表的跨集 0.987 → **坐實**：(i) 我們的 CheXpert 影像/前處理無誤，(ii) §3.1 reimpl 的 0.764 僅因只在 1000 病患上訓練（原作 800k pairs）、架構與資料皆正確、攻擊者「能」近乎完美。
- [結論] **必須分開兩個效應（AF-042）**：從 0.982 → `plaintext_latent` **0.550**（identity 控制，**未加密**、只是換成 latent-view 表徵）即大幅崩塌 → **大部分降幅來自「latent-view 對 CXR re-id 網路 OOD／讀不動」，不是密碼**。**密碼的淨效果要在同一 latent-view 表徵內看**：`plaintext_latent` 0.550 ≈ **S0 0.560**（S0 為透明正交變換，與未加密 latent 連結相當）→ **S2 0.477**，per-image nonce 把透明基線殘留的 ~0.06 移掉。
- [結論] 因此正確論述是：**現成／發表級 CXR re-id 網路無法連結 S2 的 noise-like 密文；且在同表徵下 S2（0.477）仍優於前作透明的 S0（0.560）**——不把 S2 的安全錨在 plaintext 真實-CXR 天花板（0.982，不同表徵）。原始 checkpoint 另排除「訓練不足造成假性安全」。
- [限制] 為跨資料集套用（原權重訓於 ChestX-ray14），且用**我們的** bicubic/grayscale 前處理（非其 bilinear/RGB 原協定），故 plaintext AUC 不必等於 0.987；單次 CPU 量測；N=20。為 GPL 外部工具，僅引用+報告數字，code/權重不入庫。identity 控制來源同 §3.2 adapter（`reid_original_weights_sanity.py`，plaintext_latent 條件）。

---

## 4. 待產出

- [x] Opt 2：`scripts/reid_external_attacker.py` → `paper_assets/tables/table_reid_external.{csv,json}`（已執行，見 §3.1）。
- [x] 本表（§1）外部數字已對 refpaper PDF **正文表**核對（2026-10-21）：
  - PriCheXy-Net 81.8%→57.7%（µ=0.01）、分類 80.5%→75.8%、Privacy-Net Ver.49.8/Class.57.5、DP-Pix Ver.50.0–52.5/Class.50.0–52.9 → 全部出自 **refpaper 112 Table 1**。
  - Packhäuser verification AUC 0.9940（同集）、0.9870（CheXpert 跨集、未微調） → **refpaper 111 摘要 + Table 1/Table 2**。
- [x] Packhäuser 公開權重 out-of-tree 跨集 sanity（2026-10-21，見 §3.2）：plaintext 0.982（≈發表 0.987）、S2 0.477≈隨機。
  - repo `github.com/kaipackhaeuser/CXR-Patient-ReID` 為 **GPL-3.0**；權重隨 repo 以 Git LFS 附於 `PatientVerification/trained_models/`（經 raw 端點下載並核對 sha256）。
  - 樹外執行（`/data2/external_gpl/`，不 `git add`、不散布權重/code）→ 本 repo 保持 license-clean；論文僅引用 + 報告數字。
- [x] AF-042 identity-cipher latent-view 控制（2026-10-22，見 §3.2 `plaintext_latent`=0.550）：隔離「密碼效果 vs latent-view 表徵 OOD」，S2 安全論述已改錨在同表徵 S0 基線、軟化「擊退強攻擊者」措辭。
- [x] AF-041：`t2bb.md §6` headline 已綁兩前提（image-encoder 介面 + range-discarding 儲存）、明示與 §5 minmax 一致（非推翻透明性）、保留 range_preserving `cos` 旁道仍洩漏 0.657。
- [待辦] Opt 3（在 CheXpert 重跑 PriCheXy-Net 做同 pipeline head-to-head）暫不做（數個 GPU-day、CP 值低、需謹慎框定威脅模型差異）。
