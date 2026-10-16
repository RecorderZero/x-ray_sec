# 攻擊 × 威脅模型對照表

> 本檔彙整本研究目前發展的所有攻擊，依**威脅模型**（T1 / T2-WB / T3 / T4）與**攻擊目標**（再識別 / 金鑰破解 / 完整性 / 診斷）分類。
> 數字為實測 `[事實]`，皆附凍結來源表；外推結論標 `[推定]`。威脅模型定義見 `PROPOSAL.md` §3；claim 索引見 `paper_assets/attack_package/claims.csv`（前作 Track A，C1–C14）與 `paper_assets/defense_package/claims.csv`（S2 修補，D7-C1…C10）。
> 產生日期 2026-10-16。

---

## 0. 威脅模型定義（`PROPOSAL.md` §3，Kerckhoffs：模型/程式/格式公開，只有 master key 保密）

| 代號 | 攻擊者擁有 | 本研究定位 |
|---|---|---|
| **T1** | 精確的 `z_ano`、nonce、參數、tag（但無 master key） | latent ciphertext 洩漏 |
| **T2-WB** | 匿名影像 **＋ 目標 checkpoint/程式/sampler**，可自行 inversion 得近似 `ẑ_ano` | **P0 主情境** |
| T2-BB | 僅匿名影像，**無目標模型**，只能 surrogate / embedding / i2i | P2，選做，另表報告 |
| **T3** | 部分明密文對（KPA）或可自選影像提交（CPA） | 已知/選擇明文 |
| **T4** | 可竄改匿名影像 / latent payload / nonce / header / metadata | 主動竄改 |

> T1 與 T2-WB 的差別只在**取得 latent 的方式**：T1 是精確 latent，T2-WB 是「匿名影像→目標模型 inversion→近似 latent」。同一支 scorer（abs / sorted_abs / norm）在兩種威脅下各跑一次，差別來自 inversion error 與存檔協定。

---

## 1. 兩條攻擊線（分析對象不同）

- **Track A（前作 S0/S1）**：證明前作方案**可破**。S0 = Rademacher 符號；S1 = 帶符號置換；金鑰與影像無關、幅值為不變量。
- **D7 / AF-025（S2a/S2 修補）**：對新設計**重跑同一批攻擊**確認已修好。S2 = per-image nonce + 結構變換 + 外層 HMAC；AF-025 v2 另把每張 latent 先縮放到固定公開 norm `C=248.261`，真實 norm/值域以小型 AEAD private metadata 保存。

---

## 2. 再識別類攻擊（gallery linkage，目標＝「認人」，Top-1 越接近隨機越安全）

隨機基準 Top-1 = 0.005（gallery=200）。scorer 皆對匿名後的 latent 統計量做比對。

### 2.1 逐元素幅值 `abs`（per-element magnitude）

| 威脅 | 方案 | 存檔 | Top-1 | 來源表 |
|---|---|---|---|---|
| **T1** | 前作 S0 | exact_latent | **1.000**（可破） | attack_package C4-t1-s0-abs |
| **T1** | S2a/S2 | exact_latent | **0.010** | `table_attack_original.csv` D7.2 T1/abs |
| **T2-WB** | 前作 S0 | legacy / range | **1.000 / 1.000**（可破） | attack_package C6-t2wb-s0 |
| **T2-WB** | S2a/S2 | legacy_png / range_preserving | **0.010 / 0.010** | `table_attack_original.csv` D7.2 T2-WB/abs |

### 2.2 排序幅值 `sorted_abs`（sorted-magnitude，置換不變；尺度不變版 AF-033）

| 威脅 | 方案 | 存檔 | Top-1 | 來源表 |
|---|---|---|---|---|
| **T1** | 前作 S1 | exact_latent | **1.000**（可破） | attack_package C5-t1-s1-sorted |
| **T1** | S2a/S2 | exact_latent | **0.005** | `table_attack_original.csv` D7.2 T1/sorted_abs |
| **T2-WB** | 前作 S1 | legacy / range | **0.465 / 0.995**（可破，值域敏感） | attack_package C7-t2wb-s1 |
| **T2-WB** | S2a/S2 | legacy_png / range_preserving | **0.005 / 0.005** | `table_attack_original.csv` D7.2 T2-WB/sorted_abs |

### 2.3 L2-norm 旁道 `norm`（norm linkage，★ 關鍵旁道）

正交 S2 v1 **必然保存 L2 norm** → norm 本身就是不變量；AF-025 v2 固定 norm 即為針對此旁道的修補。

| 威脅 | 方案 | 存檔 / 校正 | Top-1 | 判讀 | 來源表 |
|---|---|---|---|---|---|
| **T1** | S2 v1（正交，保 norm） | exact_latent | **1.000** | norm limitation 成立（精確 latent） | `table_norm_leakage.csv` D7.3 T1 |
| **T1** | **S2 v2（固定 norm）** | exact_latent | **0.005** | 已回隨機 | `table_AF025_fixed_norm_norm_leakage.csv` |
| **T2-WB** | 前作 S0 / S1 | range-preserving（校正後） | 0.255 / 0.285（顯著） | attack_success | attack_package C8-t2wb-norm |
| **T2-WB** | S2 v1 | legacy_png（raw / dev_median） | 0.005 / 0.005 | legacy 不保值域 → 無洩漏 | `table_norm_leakage.csv` D7.3 |
| **T2-WB** | S2 v1 | range_preserving（dev_median 校正） | **0.250** | **存檔相依**的 norm linkage（觸發 AF-025） | `table_norm_leakage.csv` D7.3 |
| **T2-WB** | **S2 v2（固定 norm）** | private_range_png（raw / dev_median_to_C） | **0.005 / 0.005** | 已回隨機 | `table_AF025_fixed_norm_norm_leakage.csv` |

> `[事實]` v1 range-preserving T2-WB 的 norm Top-1=0.25 需 **dev-only bias 校正**才顯著（raw=0.005）；dev 校正與 formal 評估嚴格隔離。v2 固定 norm 後，同一攻擊（含校正）回到 0.005。

### 2.4 成對幾何 `distance_signature`（pairwise geometry，結構旁道）

跨影像共用固定 Q 時，成對距離/角度保持 → 可用距離指紋認人；per-image nonce 破壞之。

| 設定 | 成對 cos 誤差 | distance-signature Top-1 | 來源表 |
|---|---|---|---|
| fixed_Q（跨影像共用） | 2.4e-9 | **1.000**（可破） | `table_geometry.csv` |
| per_image_Q（S2 實際採用） | 0.145 | **0.005**（已修） | `table_geometry.csv` |

---

## 3. 金鑰破解類攻擊（目標＝還原金鑰 / 解密映射）

### 3.1 唯密文金鑰辨識 COA（T1）— 前作有效金鑰上界

| claim | 威脅 | 結果 | 備註 |
|---|---|---|---|
| C1-keyspace | T1/T3 | 有效金鑰上界 S0=32 bits、S1=64 bits；唯密文 seed 搜尋 S0≈2³²、S1≈2³³ | 金鑰空間大 ≠ 安全 |
| C2-coa-s0 | T1 COA | S0 單一密文 latent 即可辨識金鑰（200/200 高於門檻） | 門檻外推屬 `[推定]` |
| C3-coa-s1 | T1 COA | S1 單一密文辨識 199/200 | 同上 |

### 3.2 已知/選擇明文 KPA/CPA（T3）— 還原完整解密映射

| claim / 攻擊 | 威脅 | 結果 | 來源 |
|---|---|---|---|
| C9-kpa | T3 KPA | 前作 S0 **1 對**、S1 **2 對**即還原金鑰 | attack_package |
| C10-cpa | T3 CPA | 固定金鑰下 d=4096 全基底重建解密映射（前作實際僅需 1 次查詢），fresh max-abs=0 | attack_package |
| cross-nonce（S2） | T3 KPA/CPA | **同 nonce**：full-map 可完整還原（MaxAbs≈2.4e-15、cos=1.0）；**跨 nonce**：失效（cos=0.0011） | `table_cross_nonce.csv` |

> `[事實]` S2 的 full-map KPA/CPA 在**同一 nonce 下成立**（這是 positive control，證明攻擊實作正確），但**per-image nonce 使映射無法跨影像轉移**（cross-nonce cos≈0）。這正是防線所在，而非「映射不可還原」。

### 3.3 影像→金鑰還原（T2-WB → 金鑰，A4.6）— 只憑（原圖,匿名圖）對

| claim | 威脅 | 結果 | 來源表 |
|---|---|---|---|
| C12-imgonly-s0 | T2-WB→金鑰 | S0 單對即還原多數符號（legacy 0.956 / range 0.992）；約 8 對對 50 位受害者**位元相同**解密 | `table_A4.6_imageonly_keyrec.csv` |
| C13-imgonly-s1 | T2-WB→金鑰 | S1 單對無用、多對可破；保存值域顯著加速（range K=4 置換 0.905、K=8 位元相同；legacy K=8 僅 0.645） | `table_A4.6_imageonly_keyrec.csv` |
| C14-recon-e2e | T2-WB 端到端 | 以 8 對還原金鑰重構受害者原圖：S0（兩存檔）與 range 的 S1 達與合法持金鑰者**位元相同**（PSNR 100 / SSIM 1.0） | `table_A4.6_reconstruct.csv` |

> 根因：**金鑰與影像無關 + 幅值為不變量** → 一把全域金鑰被竊後，同金鑰下任一影像皆可還原。直接支持 D5 的 per-image nonce、CSPRNG 金鑰與 authenticated container。

---

## 4. 完整性 / 來源驗證（T4，主動竄改）

| claim / 攻擊 | 威脅 | 結果 | 來源表 |
|---|---|---|---|
| C11-tamper（前作 S0/S1） | T4 | 無完整性檢查，所有竄改皆被接受（30/30，無拒絕） | attack_package |
| D7-C8（S2 HMAC） | T4 | 竄改 tag/nonce/ciphertext/version 全部被拒（S2 rejected 7/7）；S2a 無 MAC（0/1） | `table_tamper.csv` |

---

## 5. 診斷指標（**非**安全證明，只佐證結構/不可連結性）

| 指標 | 內容 | 來源表 |
|---|---|---|
| structured_probe | 正交 S2 下基底探針 `norm_ratio=1.0`（保 norm）；v2 固定 norm 後 norm 區別消失 | `table_structured_probe.csv`、`table_AF025_fixed_norm_probe.csv` |
| repeat（不可連結性） | 同一明文重複加密 → 100/100 payload 唯一，off-diagonal cosine≈3.7e-5（per-image nonce） | `table_repeat.csv`、`table_AF025_fixed_norm_repeat.csv` |
| T2-WB inversion 正控制品質 | inversion 近似品質：legacy latent cosine 0.990、range 0.9998；v2 legitimate cosine 0.9998（確認 inversion 本身夠好，攻擊失敗非因 inversion 太差） | `table_D7_T2WB_inversion.csv`、`table_AF025_fixed_norm_T2WB_inversion.csv` |

> `[事實]` trace/cosine/probe 僅作 diagnostic，不得當通用安全判準（見 `WORKFLOW.md` §A4、skill §8）。

---

## 6. 一句話總結（T1 vs T2-WB 對照）

| 攻擊 | 目標 | T1（精確 latent） | T2-WB（影像+模型 inversion） | S2 修補後 |
|---|---|---|---|---|
| 逐元素 abs | 再識別 | 前作 S0=1.0 | 前作 S0=1.0 | T1/T2-WB 皆 ≈0.005–0.01 |
| 排序 sorted_abs | 再識別 | 前作 S1=1.0 | 前作 S1 range=0.995 | T1/T2-WB 皆 ≈0.005 |
| norm linkage | 再識別 | S2 v1=1.0（保 norm） | S2 v1 range=0.25 | v2 固定 norm → 0.005 |
| 幾何距離指紋 | 再識別 | fixed-Q=1.0 | （同結構） | per-image Q → 0.005 |
| full-map KPA/CPA | 金鑰破解 | — | 同 nonce=1.0 / 跨 nonce≈0 | per-image nonce 阻斷轉移 |
| 影像→金鑰（A4.6） | 金鑰破解 | — | 前作數對即破、端到端位元相同 | nonce+CSPRNG+容器阻斷 |
| tamper | 完整性 | — | 前作 30/30 接受 | HMAC 全拒 7/7 |

**要點**：
1. 再識別攻擊（abs / sorted_abs / norm / geometry）在 T1 與 T2-WB **都各跑一次**，差別只在 latent 來源（精確 vs inversion）與存檔協定（legacy 不保值域 / range 保值域 / v2 固定 norm 私有容器）。
2. norm linkage 是唯一在 S2 v1 仍顯著的再識別旁道（T1=1.0、T2-WB range=0.25），AF-025 v2 固定 norm 已將其壓回隨機。
3. 金鑰破解（KPA/CPA、影像→金鑰）針對的是前作「全域固定金鑰」；S2 的 per-image nonce 使映射無法跨影像轉移，是真正防線——不是宣稱映射數學上不可還原。
4. 以上 norm/geometry/magnitude 結果都是**指定攻擊的實測**，不是通用安全證明（見 defense_package「限制」）。
