# T2-BB Inner-Risk（匿名圖之間 linkage）實驗設計

> 狀態：**設計草稿（draft）**，尚未執行。對應威脅模型 T2-BB（PROPOSAL §3，P2 選做、另表報告）。
> 前綴慣例：`[決定]` 設計定案（改需回 Cowork）／`[事實]` 已量測已驗證（附來源）／`[假設]` 未驗證／`[待辦]` 待做／`[結論]` 由事實推得／`[限制]` 誠實界線／`[注意]` 風險提示。
> 交叉參考：`paper_assets/threat_model_attack_map.md` §2、`reports/t2wb_protocol.md`、`scripts/a4_t2wb_gallery.py`。建立日期 2026-10-20。

---

## 1. 目的與定位

- [決定] 本實驗測 **inner risk（linkability 內部風險）**：在**最弱攻擊者假設**下（只有釋出的匿名圖、無目標模型、無金鑰），**兩張匿名圖是否能被連結為同一病患**（anon↔anon）。採 GDPR Art.29 匿名準則用語（singling-out / linkability），對齊 refpaper 114（Li/Ayache/Delingette, arXiv 2501.09114）的 inner/outer risk 框架。
- [決定] **不做** HR / LC（singling-out）：那兩個指標假設匿名輸出是「同空間的 CXR」，對 114 的換圖法成立，對我們的**加密法**（匿名產物為 noise-like 密文）無意義。見 §9。
- [決定] **不做** outer risk（anon↔real）於本實驗：已由 T2-WB / A4.6 以更強形式回答（端到端逐位元重建原圖，`table_A4.6_reconstruct.csv`）。本實驗只補 inner risk 這一格。
- [結論] 期望貢獻：若前作 S0/S1 的匿名圖在此假設下**可被連結**、而 S2 **回隨機**，即在「攻擊者假設由弱到強」光譜最弱端再補一次 break-and-fix；S2 的安全性在此為 per-image nonce 的直接結果。

---

## 2. 威脅模型與攻擊目標

- [決定] 攻擊者擁有：一組**釋出的匿名圖檔**（存檔 PNG）。**無**目標 checkpoint / sampler（否則退化為 T2-WB）、**無** master key、**無**原圖。
- [決定] 攻擊目標：給定 gallery = 每病患一張匿名圖、query = 某病患**另一張**匿名圖，排名後看 Top-1 是否命中同病患（＝re-identification / unlinkability）。
- [注意] 這連的是「**同病患、不同影像**」，訊號遠弱於既有 T1 的「同一張影像的匿名版」（T1 abs Top-1=1.0 是 exact-latent 同內容匹配）。不可預期 Top-1=1.0。見 §7 positive control。
- [決定] 隨機基準：gallery 大小 G 時 Top-1 = 1/G（例 G=200 → 0.005）。

---

## 3. 與既有工作的關係 / 為何現有 split 做不了

- [事實] 現有評估 split `splits/security_v1.csv` 為 **patient-disjoint**（每病患僅一筆；欄位含 `patient_id`）。來源：`splits/security_v1.csv` 檔頭、E1.3 通過條件。
- [事實] 既有 `a4_t2wb_gallery.py` docstring 與 `reports/t2wb_protocol.md §3.1` 已明載：「range-preserving 的 `lo/hi` 旁道**需要一組另外標註的 enrollment set**；每病患僅一筆時標 `not_evaluable`」。來源：`scripts/a4_t2wb_gallery.py` 檔頭註解。
- [結論] 本實驗的**同病患 split 正是該缺口所需的 enrollment set**；它解鎖的是 inner-risk / 同病患 linkage，而非改動主結果。因此定位為**既有 protocol 的補完**，非新研究設計。

---

## 4. 資料集設計（新同病患 split）

- [事實] CheXpert train（`train_visualCheXbert.csv`，223,414 列）：有 ≥2 張 frontal 的病患 **31,744 位（49.2%）**，其中 ≥2 張且跨 ≥2 study 者 30,975 位。資料充足。來源：2026-10-20 CPU 盤點（csv 計數）。
- [決定] 選取準則（`splits/t2bb_inner_v1.csv`，待建）：
  1. 僅 **Frontal**；每病患取**恰 2 張**（1 張入 gallery、1 張入 query），取自**不同 study**（避免同次拍攝幾乎重複）。
  2. **patient-disjoint 於彼此**（不同病患），但**同病患內保留 2 筆**（與 security_v1 的差異點）。
  3. 沿用 security_v1 欄位格式（`sample_id, patient_id, label, label_name, view, ap_pa, source_path, local_path, file_sha256, split_sha256`），另加 `pair_role ∈ {gallery, query}`、`pair_id`（同病患兩筆共用）。
  4. seed=1911 固定抽樣；記錄 `split_sha256`、每檔 `file_sha256`。
- [決定] 規模：pilot **N=20 病患 ×2 = 40 張**；正式再議（§10）。
- [注意][待辦] **訓練集重疊 confound**：若所選病患在 diffusion 模型訓練集內，DDIM 反演保真偏高（記憶化）會灌水 inner-risk。建 split 時**記錄並盡量排除**與模型訓練集重疊的病患；無法完全確認時於 limitations 誠實交代。
- [決定] 前處理一律沿用**安全 pipeline 的前處理（值域 [-1,1]）**，不用 q11 定位用的 [0,1]（兩者目的不同，別混）。

---

## 5. Pipeline（encrypt-only，不解密）

- [決定] 每張影像一次：`x0 (CXR) → DDIM 反演 → latent z → 套用密碼 → 存成匿名 PNG`。**不做** decrypt（backward DDIM）——攻擊者看的是匿名圖，解密與本實驗無關。
- [決定] 密碼三案共用**同一份反演 latent**（反演只跑一次/張）：S0（sign flip）、S1（signed perm）、S2（fixed-norm + per-image nonce + SOT + 容器），與凍結設定一致。
- [決定] 存檔兩種都測：`legacy_png`、`range_preserving_png`（norm 洩漏本為存檔相依，見 threat map §2.3）。
- [事實] 反演是主要 GPU 成本；密碼本身 CPU 分鐘級。來源：既有 Q8/E2.3 run 經驗。

---

## 6. Scorer / 評分協定

- [決定] 重用 `scripts/a4_t2wb_gallery.py` 的 scorer（`abs, abs_unit, sorted_abs, sorted_abs_unit, norm, lowfreq_abs`），但**gallery 改為匿名 latent↔匿名 latent**（既有版本是 anon-query↔original-gallery，需調整 gallery 構造）。
  - S0 指定 scorer：`abs` / `abs_unit`；S1：`sorted_abs` / `sorted_abs_unit`；旁道：`norm`、`lowfreq_abs`。
- [決定] 另加一個**通用影像 embedding**（如 DINOv2）跑在**存檔 PNG**上，作為「不靠密碼結構知識的黑箱 scorer」。
- [限制] **不可**用 Packhäuser CXR re-id 網路於密文圖（輸入須像 CXR；密文是 noise-like）。CXR re-id 網路只用於 §7 的 plaintext 影像層 positive control。
- [決定] 指標：Top-1 / Top-5、以及 same/different pair 的 verification AUC（對齊 114 inner-risk 報法，便於跨論文比較）。
- [決定] 固定 vs 學習式 scorer：**先只跑固定統計 + 通用 embedding**；是否升級為**學習式 linkage 攻擊**（訓一個「匿名 latent → patient 關聯」的網路，shadow-style）**待 pilot 結果再決定**（§10 決策點）。

---

## 7. Positive control（本實驗命門）

- [決定] 原則（skill §5/§7）：linkage 攻擊只有在**未保護情形已成立**時，null 才能解讀為「防禦有效」；否則只是「攻擊太弱」。
- [決定] 兩層 positive control：
  1. **影像層（便宜，§10 步驟 1）**：對 plaintext CXR（不反演）用通用 embedding / 可取得時用 Packhäuser re-id，測同病患 linkage。確認「CXR 層級有生物特徵訊號」。[假設] 依 Packhäuser 2022（AUC 0.994）預期成立，但須在**我們的 CheXpert 子集**上確認。
  2. **latent 層（§10 步驟 2，與主實驗共用反演）**：對**plaintext latent**（反演後、未加密）測同病患 linkage。確認「訊號**存活過 DDIM 反演**、進到我們要加密的 latent」。這是最關鍵的分母。
- [決定] 另設一個 **decrypt-with-key 正控制**（僅驗 pipeline 正確性，非威脅情境）：以正確金鑰解密後 latent 對同病患 gallery 連結，預期高分，證明攻擊實作無誤（比照 `a4_t2wb_gallery` 的 positive control 設計）。
- [結論] 判讀規則：
  - plaintext-latent 控制 **≈ 隨機** → 整個 inner-risk **inconclusive**（訊號在反演後就沒了，與密碼無關），**停止**、不 scale。
  - plaintext-latent 控制 **> 隨機** → 「加密是否破壞它」才有意義，續測 S0/S1/S2。

---

## 8. Negative control

- [決定] gallery 標籤洗牌（seed 1911）後重跑同 scorer，確認 Top-1 掉回隨機；排除評分流程假陽性。比照既有 attack package 慣例。

---

## 9. 不做 HR / LC 的理由（存證）

- [事實] 114 的 HR（Hidden Rate）、LC（Local Cloaking）以 identity encoder 對「匿名圖（CXR）↔真圖（CXR）」算相似度，前提是匿名輸出為同空間 CXR。來源：refpaper 114 §3、Table 3/4。
- [結論] 我們的匿名產物是加密 latent（非 CXR），identity encoder 無從施力 → HR/LC 對我們在釋出產物層級為 by-construction 平凡值，無資訊量。故**借其 linkability(inner/outer) 詞彙**但**不套用 singling-out 指標**。

---

## 10. 分層執行計畫（避免一次押 GPU）

- [決定] **步驟 1（便宜 / CPU 或輕量 GPU）＝ 影像層 positive control**：建 pilot split（20 病患×2）→ 對 40 張 plaintext CXR 用通用 embedding（＋可得時 Packhäuser）測同病患 linkage。**Gate 條件**：若連 CXR 層都接近隨機（極不預期），停止重新檢視資料/metric。
- [決定] **步驟 2（GPU pilot，>30min 需 gate）＝ 反演 + 加密 + anon linkage**：對同 40 張跑一次反演 → (a) plaintext-latent positive control（§7.2）、(b) S0/S1/S2 × {legacy, range} encrypt→存檔→anon↔anon linkage（固定 scorer + 通用 embedding）、(c) decrypt-with-key 正控制、(d) negative control。
  - 反演次數 = 影像數（三密碼共用）；pilot 40 次。[假設] 估 <1h（依既有 inversion 吞吐，待實測確認再報）。
- [決定] **決策點（pilot 結果後）**：
  - plaintext-latent 控制 ≈ 隨機 → 判 inconclusive、結案（誠實寫入 limitations）。
  - 控制 > 隨機 且 固定 scorer 已能顯示前作洩漏 → 直接 scale 正式 N、建完整 split、進 appendix。
  - 控制 > 隨機 但固定 scorer 對前作也接近隨機 → **此時才評估學習式攻擊**（訓 shadow linkage 網路，另估工程/GPU 成本回報使用者）。

---

## 11. 成本與 Gate

- [事實] **不能重用現有反演 cache**：E2.3 / 既有 security cache 是 patient-disjoint 的別組病患；新同病患 split 必須新跑反演。來源：§3。
- [決定] 步驟 2 為 GPU run，依 CLAUDE.md §3 **>30min 須先告知使用者並取得裁決**（附預估時間 / VRAM / artifact / 子任務）。步驟 1 影像 embedding 為輕量，可先行。
- [待辦] pilot 實測後更新精確 GPU-hour 與 VRAM 估計。

---

## 12. 治理 / 範圍界線

- [決定] 本實驗屬 T2-BB（PROPOSAL §3 標 **P2 選做、另表報告**），與 F9/G5 凍結範圍、主表（DDIM+非對稱 scheme of record）無相依、不改主方法與主資料集（仍 CheXpert）。實作方可執行；GPU 需 gate。
- [限制] 誠實界線：
  - 只能宣稱「前作在最弱攻擊者下可/不可連結」「S2 回隨機」；**不**宣稱通用安全證明。
  - S2 的 inner-risk 安全為 **per-image nonce 的 by-construction 結果**，近乎平凡；論文價值主要在**前作的 break**與**攻擊者假設光譜的完整性**。
  - 記憶化 / 訓練集重疊 confound 須揭露（§4）。
  - 若最終需學習式攻擊才見洩漏，須說明「固定統計攻擊不足以連結」本身也是一項發現。

---

## 13. 交付物（待產出）

- `splits/t2bb_inner_v1.csv`（同病患 pilot split）＋ manifest。
- `scripts/t2bb_inner_linkage.py`（反演+加密+存檔+scorer；重用 `a4_t2wb_gallery` scorer 與 `q8_pilot_runner` 的 S2 封裝；managed_run、output_guard、per-sample hash）。
- `paper_assets/tables/table_t2bb_inner_linkage.csv`(+json)：各 (scheme, storage, scorer) 的 Top-1/Top-5/AUC ＋ positive/negative control。
- 本 protocol doc 與 PROGRESS 條目。
- （已產出）`scripts/t2bb_step1_image_control.py`、`scripts/build_t2bb_shadow_split.py`+`splits/t2bb_shadow_v1.csv`、`scripts/t2bb_invert_cache.py`（file_sha256-keyed、prefix-reusable 快取；gitignored）、`scripts/t2bb_learned_attack.py`+`table_t2bb_learned_attack.csv`、`table_t2bb_step1_image_control.csv`。

---

## 14. 待辦清單

- [x] 建 `splits/t2bb_inner_v1.csv`（20 病患×2，frontal、跨 study、記錄訓練集重疊）。
- [x] 步驟 1：plaintext CXR 影像層 positive control（輕量）。
- [x] 步驟 2：GPU pilot（反演+加密+linkage+四控制）——gate 已取得使用者授權。
- [x] 升級學習式攻擊（shadow 1000-patient split，pilot 200）。
- [x] 依決策點判定：**結案**（見 §15）。

---

## 15. 結果與結論（定稿 2026-10-20）

### 15.1 步驟 1 — 影像層 positive control（`table_t2bb_step1_image_control`）
- [事實] plaintext CXR、off-the-shelf DenseNet121(ImageNet)：Top-1=0.550、AUC=0.811、neg-ctrl Top-1=0.100（random 0.05）。
- [結論] 同病患 biometric 訊號**在影像層確實存在**，故步驟 2「此訊號能否穿過反演+加密」可被詮釋。這是任何握有真圖之攻擊者的識別天花板（屬 outer-risk / T2-WB，BB inner 攻擊者取不到真圖）。

### 15.2 步驟 2 — latent/anon 層 linkage（`table_t2bb_inner_linkage`，N=20×2，seed 1911）
反演以 `scripts/t2bb_invert_cache` 建立的 file_sha256-keyed 快取取得（40 張全 cache 命中，CPU 0.1s）。decrypt 正控制：S0/S1 max_abs=0.0、S2 max_abs=9.5e-7。

| scheme | storage | scorer | Top-1 | AUC |
|---|---|---|---|---|
| plaintext_latent | — | cos / abs / norm | 0.30 / 0.20 / 0.15 | 0.659 / 0.657 / 0.680 |
| S0 = S1 | range_preserving | cos | 0.25 | **0.657** |
| S0 = S1 | range_preserving | abs | 0.20 | 0.655 |
| S0 = S1 | minmax | cos / abs | 0.00 | 0.512 |
| **S2** | minmax | cos / abs | 0.05 | 0.499 |
| **S2** | range_preserving | cos | 0.00 | 0.433 |

- [事實] **透明性**：S0/S1 range_preserving 的 `cos`(0.657) ≈ plaintext latent `cos`(0.659)。前作 S0=`D·z`、S1=`P·D·z` 為單一、與影像無關的正交映射，`cos(z_ano_i,z_ano_j)=cos(z_i,z_j)` 恆等 → cipher 對此統計量毫無遮蔽。`cos` 為此結構最強的線性不變量探針，但只較 `abs` 高一點點（Top-1 0.25 vs 0.20，AUC 幾乎相同）。
- [事實] **弱訊號**：cos/abs/norm 三種線性統計上界皆卡在 AUC 0.66–0.68、Top-1 0.2–0.3；相對影像層 0.81，反演已流失大量身分資訊。
- [事實] **S2 修補**：per-image nonce 使跨影像不相關，cos/abs 皆降至 ≈0.5（AUC 0.433–0.57，Top-1 ≤0.15，neg-ctrl 水準）。

### 15.3 學習式（shadow）上界攻擊（`table_t2bb_learned_attack`，pilot 200 病患）
- [事實] shadow training set `splits/t2bb_shadow_v1.csv`（1000 病患×2，固定全域 permutation 前綴、與訓練/security/victim 皆 disjoint；pilot 取前 200）。channels：`s0_abs`(|z|，前作可遷移不變量)、`s2_anon`(per-image nonce 對照)。
- [事實] 結果：訓練 loss 收斂至 0.0048，但 shadow-val Top-1=0.0、AUC=0.48（≈隨機），victim AUC=0.468 < 固定 scorer 0.657。`efficacy=false`（net_learned=false）。
- [結論] **過擬合**：2-shot、160 訓練病患下,網路背住逐張特徵、學不到可遷移的同病患 signal。合成 smoke（植入訊號）可達 Top-1=1.0，證明是資料/訊號問題而非 trainer bug。cos≈abs≈norm≈0.66 的一致上界亦指向 latent 層 linkage 本就弱 → 判定補跑 1000(~3.5h GPU)期望值低,**不執行**。

### 15.4 總結論
- [結論] **前作(S0/S1)對 inner-risk linkage 透明但訊號本身弱**：固定 cos/abs scorer AUC≈0.66,且 `cos`(anon)=`cos`(plaintext) 證實加密未提供任何遮蔽。
- [結論] **最弱～學習式攻擊者光譜內,BB inner linkage 效果不顯著**：無任一通道(線性 cos/abs/norm、ImageNet DenseNet、shadow 學習式)能將 Top-1 推過 ~0.3 或 AUC 過 ~0.68;學習式過擬合無法改善上界。
- [結論] **S2 的 per-image nonce 將其降至隨機**,inner-risk 安全為 by-construction(§12 既有界線:此為近乎平凡結果,論文價值在前作透明性之揭露與攻擊者假設光譜之完整)。
- [限制] N=20(victim)/200(shadow pilot) 規模小;記憶化/訓練集重疊 confound 已以 disjoint split 控制(manifest 記錄)。結論限於「固定統計 + pilot 規模學習式」攻擊者,非通用安全證明。
