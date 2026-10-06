# 研究計畫修訂版：基於既有擴散架構的可逆潛空間保護與安全強化

> 修訂自 proposal_claude.md
> 更新日期：2026-10-05
> 對象：Privacy-Preserving and Reversible Diffusion Models for Lesion Localization in Chest X-Rays
> 研究範圍：加解密安全性、可逆正確性、匿名／還原影像品質；不延續病灶定位

---

## 0. 提案摘要

本研究不重新訓練 diffusion model，也不大幅更換學長的 DDIM 架構。研究方式是先對前作的 latent transform 與金鑰管理做密碼分析，再以最小改動完成修補：

1. 以 256-bit master key、HKDF 與密碼學安全 PRNG 取代 32-bit seed。
2. 每張影像使用唯一 nonce，導出不同的 latent transform。
3. 採用文獻既有 Hadamard–Rademacher 原語組成的 key-derived structured orthogonal transform，針對前作的逐元素與排序幅值攻擊進行修補。
4. 加入 HMAC-SHA256，解密前先驗證密文、nonce、參數與必要 metadata。
5. 保留學長原有的 image inversion／generation pipeline，量測安全修補是否造成額外品質損失。

本方案的定位是：

> 一個與既有 diffusion pipeline 相容、低改動、可逆且具完整性保護的 authenticated latent obfuscation baseline。

本研究**不把此結構化變換稱為標準加密，也不宣稱其等同 AES、AEAD 或達成一般 IND-CPA**。它仍是正交線性映射，必然保存 L2 norm；per-image nonce 能避免同一 transform 跨影像重用，但不能消除單一輸出的 norm leakage。HMAC 只提供完整性與來源驗證，不補足 confidentiality。

---

## 1. 為什麼選擇這條路線

### 1.1 對現有架構改動小

前作流程可保留：

\[
x_0
\rightarrow \text{DDIM inversion}
\rightarrow z
\rightarrow \text{keyed transform}
\rightarrow z_{\mathrm{ano}}
\rightarrow \text{DDIM generation}
\rightarrow x_{\mathrm{ano}}
\]

本研究主要替換 keyed transform、key derivation 與密文驗證層。U-Net checkpoint、資料前處理主體與 DDIM 呼叫介面不需重新訓練。

### 1.2 研究問題清楚

前作弱點能形成完整的 attack–repair–retest 結構：

- 先證明 32-bit key-space、幅值不變量、key reuse、KPA／CPA 與無完整性保護。
- 再加入標準 key pipeline、WHT-based structured transform、nonce 與 MAC。
- 最後用完全相同的攻擊重新評估。

### 1.3 11 月底前可完成

工作區已有：

- attack/attack_keyspace.py
- attack/attack_poc.py
- attack/attack_tier1_real.py
- attack/verify_cpa.py
- attack/defense_design_check.py
- attack/validate_direction_candidates.py

結構化變換、KPA 與 CPA 已有 CPU prototype；但既有 gallery Top-1、norm 與 bit-exact 摘要尚未全部具備可由現版腳本直接重現的證據鏈，必須在正式實驗前重新產生 versioned CSV／JSON。主要剩餘風險是與真實 CXR／DDIM pipeline 的整合、攻擊協定與正式統計，而不是重新建立模型。

---

## 2. 前作架構與確認到的問題

### 2.1 前作的重要實作事實

| 項目 | 前作實作 | 影響 |
|---|---|---|
| 空間 | pixel-space diffusion，1×256×256 | latent 維度 d = 65,536 |
| 反演深度 | noise_level = 500 | x_t 仍含約 28% 原圖訊號 |
| Rademacher key | password SHA-256 後只取前 4 bytes | 實際有效空間只有 2^32 |
| Signed Permutation | sign 與 permutation 由相同短 seed 產生 | 宣稱的巨大 key space 不成立 |
| nonce | 無 | 所有影像重用相同 transform |
| 匿名影像 | per-image min-max 後存 uint8 PNG | 遺失尺度與量化資訊 |
| 解密 | 從匿名 PNG 重新 DDIM inversion | 會引入反演與量化誤差 |
| 評估 | reported cosine 0.9989 並非 byte equality | 不能據此宣稱 lossless |

### 2.2 弱點與本研究修補

| 編號 | 弱點 | 攻擊／後果 | 修補 | 修補後仍存在的邊界 |
|---|---|---|---|---|
| W1 | 32-bit seed | 可窮舉或字典搜尋 | random master key／Argon2id、HKDF、CSPRNG | endpoint key theft 不在範圍 |
| W2 | Rademacher 保留逐元素幅值 | magnitude gallery linkage | WHT-based 全域混合 | 必須實測修補幅度；仍保 L2 norm |
| W3 | Signed Permutation 保留幅值 multiset | sorted-magnitude linkage | WHT-based 全域混合 | 不能假設只剩 norm；仍需測 adaptive／structural attack |
| W4 | 無 nonce、重用 transform | 一組 KPA 可危及其他病患 | per-image unique nonce/subkey | nonce reuse 時風險重新出現 |
| W5 | 固定線性 transform | d 次 CPA 可重建 Q | 每張影像使用不同 Q | 單一 Q 仍是線性的 |
| W6 | 無完整性 | protected payload 可被修改後送入反向流程 | transform-then-MAC | 不提供標準加密或完整 CCA reduction |
| W7 | DDIM／PNG 非精確可逆 | 還原誤差、cosine 不等於 lossless | 正確量測、低成本工程消融 | 不保證 pixel／byte exact |
| W8 | 評估只測非適應性攻擊 | 可能錯估安全性 | 加入 norm-only、KPA、CPA、tamper | 結果可能是負面，仍須報告 |

---

## 3. 威脅模型

主線 T1／T2-WB／T3／T4 遵循 Kerckhoffs 原則：目標模型權重、程式、演算法與檔案格式皆公開，只有 master key 保密；T2-BB 是刻意移除目標模型可得性的 P2 補充情境。

### T1：latent ciphertext 洩漏

攻擊者取得精確的 z_ano、nonce、參數與 tag，但沒有 master key。

### T2-WB：只有 anonymous image，但有目標 model

攻擊者取得 PACS／檔案系統中的匿名影像，並持有目標系統實際使用的 checkpoint、程式與 sampler 設定，可自行做 inversion 得到近似的 z_ano。這是本研究的 P0 主情境。

### T2-BB：只有 anonymous image，沒有目標 model

攻擊者只有論文、公開資料或匿名影像，沒有目標 checkpoint；只能自行訓練 surrogate、做通用 embedding／re-ID 或 paired image-to-image attack。surrogate latent 與目標 latent 不得假設逐座標對齊，不能以此直接宣稱恢復目標 key。此情境列為 P2，完成 T2-WB 與全部 P0 工作後有餘裕才執行。

### T3：已知／選擇明文

攻擊者能取得部分明密文對，或可提交自選影像取得匿名結果。

### T4：主動竄改

攻擊者可修改儲存的 anonymous image、latent payload、nonce、header 或 metadata。

不在本研究範圍：

- 已授權解密端已被完整控制。
- master key／KMS／HSM 實體外洩。
- diffusion checkpoint 本身含訓練資料記憶的完整防禦。
- 病灶診斷正確性或臨床可用性。

---

## 4. 修補方案

### 4.1 方案代號

| 代號 | 方法 | 用途 |
|---|---|---|
| P | 不加密 | 正控制 |
| S0 | 前作 Rademacher | 被攻擊基線 |
| S1 | 前作 Signed Permutation | 被攻擊基線 |
| S2a | HKDF＋nonce＋WHT-based structured transform，不含 MAC | transform／leakage ablation |
| S2 | S2a＋HMAC-SHA256 | authenticated obfuscation 完整方案 |
| AEAD-ref | 原始 bytes 的標準 AEAD | 安全與 bit-exact 參考，不取代主架構 |

S2a 只用於分析「混合與 nonce 帶來什麼改變」。真正部署／完整評估以 S2 為準。

### 4.2 Key hierarchy

預設模式：

1. 以 CSPRNG 產生 256-bit master key。
2. 每張影像產生 128-bit random nonce。
3. 使用 HKDF-SHA256 做 domain separation；每輪與每種參數都有獨立 label：
   - info = cxr-transform/v1/round/{r}/sign-pre
   - info = cxr-transform/v1/round/{r}/perm
   - info = cxr-transform/v1/round/{r}/sign-post
   - info = cxr-transform/v1/mac
4. 由 ChaCha20 或 AES-CTR keystream 產生 sign 與 Fisher–Yates permutation 所需亂數。
5. StudyInstanceUID、SOPInstanceUID、shape、dtype、model hash 與 sampler version 作為 authenticated metadata，不直接拿 UID 當唯一 nonce。

nonce 是公開的唯一值，不要求不可預測；程式必須在同一 master key 下拒絕重用。

若使用者只能提供 password：

- 先使用 Argon2id／scrypt 與獨立 salt 導出 master key。
- password mode 是部署選項，不影響實驗主體。

### 4.3 WHT-based structured orthogonal transform

每一輪：

\[
z_{r+1}
=D_{2,r}P_r\frac{H}{\sqrt d}D_{1,r}z_r
\]

其中：

- D 為 per-image keyed sign matrix。
- H 為 Walsh–Hadamard transform。
- P 為 per-image keyed permutation。
- 預設測 R = 1、2、3、4，並對 permutation P 做 on／off 消融；不得因 SORF 使用三個 HD block 就預設 R = 3 較安全。

理論性質：

- 在實數算術中可逆。
- 若 z 精確服從 N(0,I)，正交變換後仍服從 N(0,I)。
- 一輪 WHT 會混合全部座標，理論上不再保留前作的逐元素幅值等式；對排序幅值 linkage 的實際修補程度由攻擊實驗決定。
- 必然保存 L2 norm，因此不可能對任意輸入達成一般 IND-CPA。
- 上述性質不代表只剩 L2 norm；此受限結構化變換族可能仍有尚未識別的結構性洩漏。

多輪總矩陣記為 \(Q=Q_RQ_{R-1}\cdots Q_1\)，不是在各輪參數不同時寫成 \(Q_{round}^R\)。

### 4.4 完整性保護

以獨立 MAC subkey 對下列 canonical bytes 計算 HMAC-SHA256：

\[
\text{version}\parallel\text{nonce}\parallel\text{parameters}
\parallel\text{ciphertext bytes}\parallel\text{authenticated metadata}
\]

規則：

- 驗證 tag 成功前不得進行 inverse transform 或 diffusion generation。
- tag 比對使用 constant-time API。
- latent-payload 模式認證 latent bytes。
- image-only 模式認證實際儲存的 lossless image bytes 與穩定 metadata；若 PACS 轉碼或修改被認證欄位，系統應明確拒絕，而不是靜默解密。
- MAC 提供 tamper detection，不改變此線性變換的 confidentiality 邊界。

### 4.5 部署模式

#### M1：image-only，最接近學長架構

- 儲存 anonymous image、nonce、tag 與 algorithm manifest。
- 解密由 anonymous image inversion 開始。
- 優點：維持既有流程。
- 缺點：受 PNG／DICOM 量化及 DDIM inversion error 影響。

#### M2：latent payload＋anonymous preview

- 精確 z_ano 以受認證 payload 保存，anonymous image 只供瀏覽。
- 優點：避免 image inversion error。
- 缺點：PACS／儲存系統需額外保存 payload；preview 本身不是完整 ciphertext。

本研究以 M1 為主要相容性實驗，M2 作為可逆性上限與部署對照。

---

## 5. 必須限縮的安全與正確性宣稱

### 5.1 不宣稱結構化變換等同標準加密

正確說法：

> S2 修補前作的短 seed 與 transform reuse，針對逐元素／排序幅值攻擊進行強化，加入完整性驗證，並量化 norm 與其他結構性洩漏。

不正確說法：

> S2 已達成 perfect secrecy、一般 IND-CPA 或與 AES 相同的安全性。

### 5.2 nonce 的作用

nonce 的作用是讓每張影像使用不同 subkey／Q，使針對某張影像取得的 transform 資訊不能直接套用到其他影像。

nonce 不會：

- 把線性 transform 變成非線性。
- 隱藏單一 ciphertext 的 L2 norm。
- 自動提供 integrity。

nonce 的安全需求是同一 master key 下唯一，不是保密或不可預測。若 API 允許呼叫者指定 nonce，必須檢查並拒絕 reuse。

### 5.3 結構化變換不保證所有輸入 bit-exact

目前既有 preflight 報告互相不一致：

- d = 4,096：測試樣本可在轉回 float32 後 bit-exact。
- 舊 artifact 曾記錄 31/32；2026-10-05 以現版函式對 seeds 0–99、d=65,536、R=2 快速稽核得到 98/100。
- 原 defense_design_check.py 重新執行時，d = 65,536、R = 1/2/4 皆回報 False。

因此這些數字都只算歷史／稽核觀察，不直接寫入論文結果。正式實驗必須固定程式版本、seed list、硬體與 dtype，輸出 per-sample CSV 後才建立可引用數字。正式判準為：

- 報告 MaxAbsError、RMSE、cosine 與 float32 bit-exact rate。
- float64 MaxAbsError 目標 ≤ 1e-12。
- 不以 bit-exact = 100% 作為 structured transform 必須通過的條件。

若要保證原始 DICOM bytes 完全一致，仍需標準 AEAD encrypted original／residual；這是 AEAD-ref 的用途。

### 5.4 adaptive norm attack 預期可能成功

正交變換滿足：

\[
\|Qz\|_2=\|z\|_2
\]

此外：

- 同一個固定 Q 會完整保留 pairwise inner product、cosine 與 Euclidean distance。
- per-image nonce 使不同影像使用不同 Q，可打破跨影像的直接幾何保持，但必須以實驗確認 cross-nonce linkage。
- trace／cosine diagnostic 只衡量特定平均方向相關性，不是通用安全判準。
- 不得宣稱此變換「只有 norm 一個不變量」。

在精確 latent gallery 中，norm-only nearest-neighbor 可能直接完成 linkage。此結果不是實作失敗，而是方案的理論限制。

研究中必須分開報：

- original magnitude attack。
- sorted-magnitude attack。
- norm-only adaptive attack。
- T1 與帶 inversion error 的 T2-WB。
- 若執行 T2-BB，必須另表報告，不得與 T2-WB 合併平均或用 surrogate 失敗宣稱方案安全。

---

## 6. 研究問題與假設

### RQ1：前作弱點是否能在真實 CXR latent 重現？

- H1a：S0 的 magnitude gallery attack 顯著高於 random baseline。
- H1b：S1 的 sorted-magnitude attack 顯著高於 random baseline。
- H1c：重用 transform 時，一組已知明密文可危及其他影像。

### RQ2：S2 能修補哪些攻擊？

- H2a：S2 顯著降低 original magnitude／sorted-magnitude attack。
- H2b：不同 nonce 下，某張影像取得的 KPA／CPA 資訊不能直接解開其他影像。
- H2c：HMAC 對所有預定 bit／header／metadata tamper case 均拒絕。

### RQ3：S2 還洩漏什麼？

- H3a：T1 下 norm correlation 保持 1。
- H3b：norm-only linkage 在 T1 可能成功；T2-WB 成功率取決於 inversion error。
- H3d（P2）：T2-BB 只評估 surrogate／embedding linkage，不預設可恢復目標 latent key。
- H3c：真實 inversion latent 並非理想 N(0,I)，structured transform 可能改變生成品質。

### RQ4：修補是否影響匿名／還原影像品質？

- 比較 P、S0、S1、S2 的 latent round-trip、PSNR、SSIM、LPIPS、cosine、MaxAbsError 與 runtime。
- 不預設 S2 必須優於 S0；品質下降也是需要報告的 trade-off。

---

## 7. 評估設計

### 7.1 Security experiments

| 實驗 | S0 | S1 | S2a | S2 |
|---|---:|---:|---:|---:|
| effective key-space／dictionary | ✓ | ✓ | ✓ | ✓ |
| magnitude gallery | ✓ | ✓ | ✓ | ✓ |
| sorted magnitude | ✓ | ✓ | ✓ | ✓ |
| norm-only adaptive linkage | ✓ | ✓ | ✓ | ✓ |
| KPA／CPA under reused transform | ✓ | ✓ | ✓ | ✓ |
| cross-nonce transfer | — | — | ✓ | ✓ |
| fixed-Q pairwise geometry | ✓ | ✓ | ✓ | ✓ |
| basis／structured probe | — | — | ✓ | ✓ |
| same-plaintext repeat encryption | ✓ | ✓ | ✓ | ✓ |
| wrong key／wrong nonce | ✓ | ✓ | ✓ | ✓ |
| tamper detection | ✗ | ✗ | ✗ | ✓ |

### 7.2 Correctness and quality

兩個空間分開：

- Latent：cosine、MSE、RMSE、MaxAbsError、float32 bit-exact rate。
- Image：PSNR、SSIM、LPIPS、cosine、L-infinity、pixel equality。

三條路徑分開：

- Transform-only：z → Enc → Dec。
- T1：保存精確 latent payload。
- T2-WB：anonymous image → target-model inversion → decryption。
- T2-BB（P2）：anonymous image → surrogate／embedding attack；結果與 T2-WB 分開報告。

### 7.3 統計與資料

- dev：16–20 張。
- pilot：50 張。
- formal：依 GPU benchmark 決定，最低 200 張，目標 500–1,000 張。
- patient-disjoint split。
- 固定 seed 1911。
- 報 bootstrap 95% CI，不只報平均。
- 所有負結果、outlier、NaN 與失敗案例保留。

### 7.4 方法正確性通過條件

| 項目 | 通過條件 |
|---|---|
| HKDF | RFC 5869 test vectors 全過 |
| nonce | 同 master key 下測試集合無 reuse；程式拒絕已登錄 nonce |
| CSPRNG parameters | 相同 key/nonce 可重現；不同 nonce 輸出不同 |
| structured transform inverse | float64 MaxAbsError ≤ 1e-12；bit-exact rate另報 |
| 正交性 | d=1,024 顯式 Q 的 norm error／orthogonality error ≤ 1e-12 |
| round／permutation ablation | R=1/2/3/4 與 P on/off 均完成；不預設三輪較安全 |
| Gaussian synthetic test | KS 與前四階動差記錄；不以單一 p-value 當安全證明 |
| HMAC | wrong key、tag/header/ciphertext/AAD tamper 全部拒絕 |
| Regression | S0 wrapper 與前作輸出一致 |

攻擊「是否失敗」不是程式正確性的通過條件。每個攻擊須有 positive／negative control；只要 protocol 正確、結果完整，即使 S2 仍被 adaptive attack 擊中，實驗仍有效。

---

## 8. 已完成的前置可行性結果

既有候選數字曾記錄於：

- artifacts/preflight/direction_candidates_cpu.json
- artifacts/preflight/direction_candidates_cpu_d65536.json

但 2026-10-05 稽核發現，現版 defense_design_check.py 只直接執行 bit-exact 測試，沒有重新產生其 docstring 所列的 gallery Top-1、Householder 與 norm 結果；bit-exact 的抽樣結果也依 seed set 而異。因此本節不再把舊數字當成正式證據，只保留以下已由代數或現版腳本支持的 preflight 結論：

- FWHT structured transform 可在 CPU prototype 中完成 forward／inverse。
- 固定線性 Q 可由 d 次 basis CPA query 重建；現版 verify_cpa.py 可重現此 positive control。
- 正交性必然保留 L2 norm，故 norm leakage 是理論限制。
- d=65,536 不保證 float32 bit-exact，正式結果必須重新產生。

所有 gallery、runtime、round-trip 與 distribution 數字，須由 workflow 的 E0 evidence reset 與 Week 4 消融重新建立，並附 run ID、程式 hash、seed list、per-sample rows 與 summary JSON。

---

## 9. 預期論文貢獻

1. 對前作完成實作層密碼分析，區分論文宣稱與真實 key-space／評估指標。
2. 在 CXR diffusion latent 上量化 magnitude、sorted magnitude、norm、KPA、CPA 與 tamper attack。
3. 將文獻既有 Hadamard–Rademacher 結構適配至 CXR diffusion latent，並系統評估 per-image key derivation、round／permutation 消融與 authentication。
4. 清楚區分「修補特定洩漏」與「達到標準加密安全」。
5. 提供 T1／T2-WB、transform／latent／image 三層可逆性評估協定，並界定 T2-BB surrogate 結論的適用邊界。
6. 量化安全強化與生成品質、反演誤差、運算成本的 trade-off。

建議題目：

> Cryptanalysis and Low-Disruption Hardening of Reversible Diffusion-Based Protection for Chest Radiographs

中文可寫：

> 胸部 X 光可逆擴散保護方法之密碼分析與低改動安全強化

---

## 10. 不納入本次主線

- 不重新訓練 diffusion model。
- 不做病灶定位、AUROC／Dice 或醫師診斷評估。
- 不把 CDF nonlinear transform 納入主實驗；列為未來改善 norm leakage 的方向。
- 不完整重現 KCI/O-BELM；列 related work／future work。
- 不同時實作 EDICT、BDIA、O-BELM。
- 不宣稱匿名影像可作診斷影像。
- 不以 cosine = 1 代替 byte／pixel equality。

---

## 11. 主要參考資料

- Labarbarie et al., [Secure and reversible face anonymization with diffusion models](https://arxiv.org/abs/2510.01031).
- Packhäuser et al., [Patient re-identification from chest radiographs](https://www.nature.com/articles/s41598-022-19045-3), Scientific Reports 2022.
- Yang et al., [Gaussian Shading](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.html), CVPR 2024.
- Zhang et al., [Public Diffusion Models, Private Images](https://arxiv.org/abs/2606.22988), 2026 preprint.
- Wang et al., [BELM / O-BELM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html), NeurIPS 2024.
- Ailon and Chazelle, [The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors](https://doi.org/10.1137/060673096).
- Yu et al., [Orthogonal Random Features](https://papers.neurips.cc/paper_files/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html), NeurIPS 2016；僅支持 SORF 的結構與 kernel-approximation 性質，不作安全證明。
- Maekawa et al., [Privacy-Preserving SVM Computing Using Random Unitary Transformation](https://arxiv.org/abs/1908.07915)；其保距／保內積性質同時是 utility 與 leakage 的相關背景。
- [RFC 5869: HKDF](https://www.rfc-editor.org/rfc/rfc5869).
- [RFC 8439: ChaCha20 and Poly1305](https://www.rfc-editor.org/rfc/rfc8439).
- [NIST SP 800-38D: GCM and GMAC](https://csrc.nist.gov/pubs/sp/800/38/d/final).

正式論文引用前仍須逐篇核對作者、版本、頁碼與 publication status。

