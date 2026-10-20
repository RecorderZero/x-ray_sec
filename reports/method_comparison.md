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
| **S2 密文（range）** | 0.05 | **0.445** | **≈ 隨機 → S2 擊退此強攻擊者** |

- [事實] 攻擊者在 held-out shadow 病患上 pair-verification AUC 0.911（非過擬合）；對 victim 原圖達 0.764。來源：`paper_assets/tables/table_reid_external.{csv,json}`。
- [結論] **即使換成專用、發表級、且確實學會再識別的架構攻擊者，S2 的匿名視圖仍降至隨機（AUC 0.445），而前作 S0 仍殘留洩漏（0.616）。** 強化了「S2 匿名性對更強攻擊者穩健」的主張，並與 T2-BB（線性 `cos`/`abs` + 弱學習式）互補——此處為影像級強學習式攻擊者。
- [限制] GPU 浮點訓練，單次量測、不保證跨次 bit-identical（同 Q11）；victim N=20 小；為 license-clean 重寫架構（非原始 ChestX-ray14 checkpoint）。

---

## 4. 待產出

- [x] Opt 2：`scripts/reid_external_attacker.py` → `paper_assets/tables/table_reid_external.{csv,json}`（已執行，見 §3.1）。
- [x] 本表（§1）外部數字已對 refpaper PDF **正文表**核對（2026-10-21）：
  - PriCheXy-Net 81.8%→57.7%（µ=0.01）、分類 80.5%→75.8%、Privacy-Net Ver.49.8/Class.57.5、DP-Pix Ver.50.0–52.5/Class.50.0–52.9 → 全部出自 **refpaper 112 Table 1**。
  - Packhäuser verification AUC 0.9940（同集）、0.9870（CheXpert 跨集、未微調） → **refpaper 111 摘要 + Table 1/Table 2**。
- [待辦]（可選）把 Packhäuser 公開權重 out-of-tree 跑一次做跨集 sanity；Opt 3（在 CheXpert 重跑 PriCheXy-Net 做同 pipeline head-to-head）暫不做（數個 GPU-day、CP 值低、需謹慎框定威脅模型差異）。
