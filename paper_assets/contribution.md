# 本研究相對前作的貢獻（Contribution）

> 用途：論文 contribution 段落的事實底稿。數字一律引用已凍結來源（`attack_package/claims.csv`、`defense_package/claims.csv`、`q8_formal/`、`tables/`）。前綴遵循專案規範：`[事實]` 必附來源、`[結論]` 為由事實推得之論斷、`[限制]` 為誠實界線、`[假設]` 為未驗證、`[決定]` 為研究設計決策。交叉參考見 [`threat_model_attack_map.md`](threat_model_attack_map.md)。

---

## 0. 前作（baseline）做了什麼

- [事實] 前作（Ci-Sian Liou 碩論《Privacy-Preserving and Reversible Diffusion Models for Lesion Localization in Chest X-Rays》）在 pixel-space CFG-DDIM 上，對胸腔 X 光做可逆匿名化，提出兩種金鑰方案：S0（Rademacher sign flip, $k\in\{-1,+1\}^d$）與 S1（signed permutation, $P(k\odot z)$）。來源：`past/Documents/Thesis/原碼_Overleaf(Latex)/04_methodology.tex`、`past/SourceCode/guided_diffusion/anonymization.py`。
- [事實] 前作宣稱「zero-distortion lossless reversible」，主要證據為 latent cosine 0.9989；下游用途為病灶定位（CFG-DDIM 生偽健康圖 → 與還原圖做 diff heatmap）。定位評估為**質化** difference map（11 張醫師標註外部測試集，pleural effusion，台北市立關渡醫院 Dr. Wei-Hsin Yuan 提供），**未報告 IoU/Dice 等量化定位指標**。來源：`05_experiments.tex` §5.1、§5.4。

---

## A. 概念層：證偽「0.9989 = 無損」並建立雙空間報告規範

- [事實] 前作的 latent cosine 0.9989 無法等同影像可逆：影像空間 PSNR 僅約 30 dB。本研究每個可逆性數字一律同時報 **latent 空間**（cosine、MaxAbsError）與**影像空間**（PSNR、SSIM、L∞、bit-exact 比例）。來源：skill §6；`q8_formal/table_fidelity.csv`。
- [事實] Q10.1 誤差歸因階梯把 M1 殘差拆成：crypto transform（無損，latent cosine 1.0、max_abs ~8.3e-7）→ M2 exact-latent 影像天花板（~41.9 dB，殘差為 VAE decode + 8-bit）→ M1 sampler invert/regen（主要誠實殘差）→ legacy min-max 存檔（P/S0/S1 的額外損失）。來源：`tables/table_error_attribution.csv`。
- [結論] bit-exact 達不到**不是 cipher 的缺陷**，是 DDIM 取樣與存檔造成；把 0.9989 當作「structured transform 無損」是對兩個不同概念的混淆。

## B. 系統層：對線性自逆構造的系統化密碼分析

- [決定] 形式化威脅模型 T1（已知精確 latent）／T2-WB（影像 + 模型，白盒反演）／T2-BB／T3（KPA/CPA）／T4（竄改），並將每個攻擊對應到目標（再識別 vs 金鑰破解）。來源：[`threat_model_attack_map.md`](threat_model_attack_map.md)、`attack_package/claims.csv`（C1–C14）。
- [事實] **再識別攻擊**（前作 S0/S1，均為 cipher 無法抵抗）：
  - 逐元素幅值：S0 在 T1 再識別 top-1 = 1.0。
  - 排序幅值：S1 在 T1 = 1.0（T2-WB ~0.995）。
  - norm linkage：正交保範數構造在 T1 = 1.0。
  - 來源：`attack_package/claims.csv`。
- [事實] **金鑰破解攻擊**：KPA/CPA full-map 在同 nonce 下 cos = 1.0（整張金鑰圖可還原），跨 nonce cos ≈ 0.001；COA 金鑰空間 S0≈32 / S1≈33 bits。來源：`attack_package/claims.csv`。
- [事實] **完整性**：前作無訊息驗證，30/30 竄改全被接受。來源：`attack_package/claims.csv`。
- [限制]（skill §8）金鑰空間大小**不作為安全論證**；幅值攻擊是「線性自逆變換這個特定構造」的性質，**不**適用於學習式加密器（FIT/RiDDLE/G2-Face）；正交旋轉**未解決 KPA**，真正阻止跨影像擴散的是 per-image nonce。

## C. 修補層：有原則的 remediation（S2 / S2a）

- [決定] S2/S2a 針對 B 的每個失效做對應修補：
  1. **per-image nonce** → 阻斷跨影像金鑰擴散、把 KPA 成本由 1 組明文推高。
  2. **fixed-norm scaling**（公開常數 $C=248.261$，每個 latent 都縮放到固定範數）→ 消滅 norm linkage。
  3. **真正的 AEAD（ChaCha20-Poly1305）+ HMAC 完整性** → 竄改 fail-closed。
  4. **range-preserving 存檔** → 消掉前作 legacy min-max 的存檔損失。
  來源：`src/crypto/`（s2.py、s2_fixed_norm.py、sot.py、mac.py、kdf.py、nonce.py）。
- [事實] 修補後攻擊全數掉到隨機基準：再識別（逐元素/排序/norm）由 1.0 → ~0.005–0.01；norm linkage v1 1.0 → v2 0.005；竄改 7/7 全 reject。來源：`defense_package/claims.csv`（D7-C1…C10）。
- [事實] 金鑰雪崩診斷（Q10.5）：翻 1 bit master key → 衍生 sign-subkey Hamming 0.505、ciphertext |cos| 3.1e-3、permutation 幾乎全變、同金鑰正控制 cos 1.0。定位為 KDF 擴散的**診斷**，非安全證明。來源：`tables/table_partial_key.csv`。

## D. 評估層：比前作誠實的保真度與統計

- [事實] 正式 N=200、mixed-deployment 主數字（AUD-20261017-01 凍結）：P 37.46 dB / SSIM 0.973；S0 31.15 / 0.918；S1 31.42 / 0.895；**S2 38.82 / 0.970**。來源：`q8_formal/table_fidelity.csv`。
- [事實] **把「+7.68 dB」誠實拆開**：S2−S0 mixed median +7.68 dB（r=1.0，Holm p≈5e-33）**大半是存檔差異，不是 cipher**；同存檔（同為保值域）條件下 S2−S0 僅 median +0.30 / mean +0.54 dB（Holm p=1.4e-18 顯著），且 **S2−S1 同存檔 +0.035 dB 不顯著（p=0.296）**，照實記錄。兩個 paired family 各自做 Holm 校正、不混池。來源：`tables/table_statistics_formal.csv`、`build_claim_matrix.py`（AF-039 修補）。
- [結論] 相對前作「只報有利數字」，本研究把 cipher 真實增益與部署/存檔增益分離，並對無顯著差異的比較誠實標注。

## E. 下游層：病灶定位能力「by construction 保留」

> 這是回應「更換加密流程是否影響下游定位」的核心論證。

- [事實] 前作為 **pixel-space** diffusion（無 VAE，$x_T$ 即加噪影像）。定位流程完全發生在**解密之後**：去匿名 → 還原**影像** $x_{rec}$ → 把 $x_{rec}$ **重新 DDIM 正向加噪** → CFG-DDIM 條件生成偽健康圖 → $|x_{rec}-x_{healthy}|$ 做 diff heatmap。**關鍵：CFG 接的是還原「影像」$x_{rec}$，不是去匿名得到的還原噪聲 latent。** 來源：`gaussian_diffusion.py: ddim_sample_loop_anomaly_detection_with_deanonymization`（Step 2 `ddim_sample_loop_known(img=(x_rec,))`）。
- [事實] 本研究的貢獻 S2 只替換**加密層**；在 $x_T$ 的金鑰套用/逆運算為**精確互逆、無損 + 保值域**：transform-only latent cosine = 1.0（max_abs ~8.3e-7）；M1 影像往返 cosine 0.999794；M2 latent payload 逐 byte 相同。來源：`tables/table_error_attribution.csv`、`tables/table_numeric_precision.csv`、`reports/deployment_modes.md`。
- [結論] 偽健康階段是還原影像的確定性函數 $f(x_{rec})$。**M2**（latent payload 直接解碼）下 $x_{rec}$ 逐 byte 相同 → $f(\cdot)$ 輸出相同。**M1**（定位實際採用的模式）把 DDIM invert/regen 夾在加密外層跑兩次，三個密碼的加密影像不同，故 $x_{rec}$ **等價到 sampler 殘差等級**（非逐 byte 相同）；該殘差已由 Q10.1 歸因為 DDIM invert/regen、非 cipher，且同存檔下 S2 與 S0/S1 的還原影像差距僅 +0.3–0.5 dB（S2−S1 不顯著）。
- [結論] 因此**更換加密流程對下游定位是透明的，不引入可觀劣化**；在 M2 為 by-construction 恆等、在 M1 為「等價到可忽略的 sampler 殘差」。後者由 GPU parity demo 端到端實證（見下 [待辦]），**不以恆等式宣稱 M1 逐 byte 相同**。
- [限制] 誠實界線：
  - 只能宣稱「保留 / drop-in 等價 / 不劣化」，**不能宣稱改善定位**。
  - **不能**對醫師 11 張外部測試集產生新的量化定位數字：原始影像與 bounding box 座標不在 repo（repo 僅存論文縮圖 `figures/experiments/external_test_set_preview_with_mask.png`），且醫師合作已結束。此為資料依賴的限制，於 limitations 誠實交代。
- [事實] **Q11 GPU parity demo 已執行**（N=10 Pleural-Effusion frontal CXR，完整像素空間 M1：加密→解密→CFG-DDIM 偽健康→heatmap；三密碼共用 fixed-norm(C=248.261)+range-preserving 包絡，唯一變因為正交核心，S2 用 `src.crypto.sot`）。cross-method 聚合（mean PSNR / mean SSIM）：
  - 還原圖：S2–S0 46.0 dB/0.996、S2–S1 51.8/0.9995、S0–S1 46.4/0.996。
  - 偽健康圖：S2–S0 43.6/0.995、S2–S1 50.7/0.999、S0–S1 43.5/0.995。
  - heatmap：S2–S0 34.7/0.890、S2–S1 40.4/0.952、S0–S1 36.6/0.959。
  - recovery vs 原圖：S0 39.1、S1 38.1、S2 38.0 dB（同量級）。
  來源：`tables/table_localization_parity.csv`(+json)、`image/localization_parity_grid.png`、`scripts/q11_localization_parity.py`。
- [結論] 三密碼的還原圖與偽健康圖近乎相同；localization heatmap 標示相同區域、僅熱點強度有 sampler 殘差級別差異。**關鍵：S2–S0 的下游差異與前作自身 S0–S1 的差異同一量級**（restored 46.0 vs 46.4、healthy 43.6 vs 43.5、heatmap 34.7 vs 36.6 dB），故引入 S2 未比前作兩方案之間引入更多下游變動。**實證支持 E 段論證：更換加密流程對下游病灶定位是透明的（等價到既有 sampler 殘差帶），不劣化。**
- [限制] 非定位準確度 benchmark（無 ground-truth mask）；heatmap 為 per-image min-max 正規化的差分圖，會放大微小殘差，故其 cross-method 一致性數字低於還原/偽健康圖，不代表定位區域不同。

---

## 貢獻定位（一句話）

- [結論] 本研究是對一個已發表醫療匿名化方法的 **break-and-fix**：(i) 證偽其「latent cosine = 無損」宣稱並建立雙空間報告；(ii) 在形式化威脅模型下系統化揭露其再識別、金鑰破解與無完整性等失效；(iii) 提出以 per-image nonce、fixed-norm、AEAD/HMAC、保值域存檔為核心的有原則修補 S2/S2a，使攻擊掉到隨機水準；(iv) 以誠實的分層統計分離 cipher 與存檔增益；(v) 論證下游病灶定位能力在新加密流程下 by construction 保留。
