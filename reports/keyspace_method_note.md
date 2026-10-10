# 前作金鑰空間審計：方法說明（A3.1）

- 任務：WORKFLOW A3.1 effective key-space audit；seed = 1911；CPU only，未載入擴散模型 checkpoint。
- 產生方式：`scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit`（由腳本寫出本檔與 `paper_assets/tables/table_keyspace.csv`，內容對固定原始碼、torch、numpy 版本為確定性）。
- 被審計原始碼：`past/SourceCode/guided_diffusion/anonymization.py`（sha256 = `5e71401a648a734979432ccf07d92deae02387352ad9ac00a33a89d9d8f38bb1`）；torch 2.14.0+cu130；numpy 1.23.0。
- 產生腳本 sha256：`05e20021af6e8aa02111d3b4de7019c7a05f262c7bfb1ada914a0cb95fa9bc38`。
- 標記說明：[事實] = 逐行讀碼或本腳本實測（附來源）；[推定] = 由 [事實] 推導，未獨立量測；[假設] = 尚未驗證的前提。

## 1. 結論：金鑰上界 S0 = 32／S1 = 64 bits；實際攻擊工作量：已知一對明密文 = 0（直接讀出）；唯密文 S0 ≈ 2^32（AF-027）

- [事實] 金鑰上界（`key_upper_bound_bits`）：S0 = 32 bits，S1 = 64 bits（password 模式；整數 seed 模式為 32）。這是「金鑰由多少個獨立 bits 決定」的上界，**不是攻擊者的工作量**（AF-026）。
- [事實] 實際攻擊工作量：已知一對完整明密文 = 0（金鑰直接讀出，不需搜尋；見第 5 節第 3 點）；唯密文 S0 ≈ 2^32（稽核方 AF-027：單一密文 latent 即可辨識正確 seed）。S1 唯密文 ≈ 2^33（稽核方 AUD-20261011-02：縮小空間下單張密文即可，外推到 2^32 為 [推定]）。後兩者為稽核方結果，本腳本不量測，實作方重現屬 A3.2 補測（AF-027）。
- [事實] 密碼 `Ki@13579` 寫死在多支啟動腳本內（第 5 節第 8 點），取得程式碼即取得密碼，給定程式碼時秘密為 0 bits。另有第二組設定 `paper_experiment_key_2024`（`run_paper_comparison.sh:27`），目前無法確定論文數字出自哪一組（第 5 節第 9 點）。
- [事實] 前作的 Rademacher 金鑰 k（d = 65536 維、值域 ±1）完全由一個 32-bit 整數 seed 經 `torch.Generator.manual_seed` 與 `torch.bernoulli` 決定（anonymization.py:100、anonymization.py:85、anonymization.py:87）。`key_upper_bound_bits` 為 32，宣稱為 65,536，差 65,504 bits（`gap_bits` 是宣稱與金鑰上界的差，不是攻擊工作量）。
- [事實] Signed Permutation 在整數 seed 模式下，置換的 seed 為 `seed + 1000000`（anonymization.py:358），符號與置換由同一個 32-bit 值決定，`key_upper_bound_bits` 為 32，宣稱 2^d·d!（約 1,019,572.9 bits）。
- [事實] 前作實際執行的 S1 設定（`run_cfg_inference_Crypto.sh:7`：`--anonymization_key_password Ki@13579 --use_signed_permutation True`）走 password 模式：符號 seed 與置換 seed 是兩個獨立的 32-bit 值，因此 `key_upper_bound_bits` 是 64，不是 32。這是對「32 bits」說法的重要修正，見第 5 節。
- [事實] 本檔量測的是金鑰上界；攻擊工作量取決於攻擊者持有的資訊與 oracle（第 3、5 節），實測吞吐與外推見 A3.2。本檔不宣稱任何實際攻擊成功率。

## 2. 程式碼路徑（[事實]，每個 file:line 都在執行時逐行驗證）

| 步驟 | 位置 | 內容 |
|---|---|---|
| 宣稱（模組 docstring） | `past/SourceCode/guided_diffusion/anonymization.py:20` | `- Key space expands from 2^d to 2^d × d!` |
| 宣稱（SignedPermutationKey docstring） | `past/SourceCode/guided_diffusion/anonymization.py:305` | `1. Key space expands from 2^d to 2^d × d! (astronomically larger)` |
| 宣稱（取樣腳本日誌） | `past/SourceCode/scripts/cfg_image_sample_anonymization.py:505` | `logger.log(f"Enhanced security: key space = 2^d × d! (astronomical)")` |
| S0：password 分支優先於 seed 分支 | `past/SourceCode/guided_diffusion/anonymization.py:70` | `elif password is not None:` |
| S0：SHA-256 | `past/SourceCode/guided_diffusion/anonymization.py:99` | `hash_digest = hashlib.sha256(password_bytes).digest()` |
| S0：只取前 4 bytes | `past/SourceCode/guided_diffusion/anonymization.py:100` | `seed = int.from_bytes(hash_digest[:4], byteorder='big')` |
| S0：seed 餵給 Generator | `past/SourceCode/guided_diffusion/anonymization.py:85` | `generator.manual_seed(seed)` |
| S0：Bernoulli(0.5) 產生金鑰 | `past/SourceCode/guided_diffusion/anonymization.py:87` | `binary = torch.bernoulli(torch.ones(shape) * 0.5, generator=generator)` |
| S1 置換：password 分支優先於 seed 分支 | `past/SourceCode/guided_diffusion/anonymization.py:164` | `elif password is not None:` |
| S1 置換：password + `_permutation` 後綴 | `past/SourceCode/guided_diffusion/anonymization.py:189` | `password_bytes = (password + "_permutation").encode('utf-8')` |
| S1 置換：SHA-256 | `past/SourceCode/guided_diffusion/anonymization.py:190` | `hash_digest = hashlib.sha256(password_bytes).digest()` |
| S1 置換：只取前 4 bytes | `past/SourceCode/guided_diffusion/anonymization.py:191` | `seed = int.from_bytes(hash_digest[:4], byteorder='big')` |
| S1 置換：randperm | `past/SourceCode/guided_diffusion/anonymization.py:180` | `permutation = torch.randperm(size, generator=generator)` |
| S0 加密：z_ano = k ⊙ z | `past/SourceCode/guided_diffusion/anonymization.py:681` | `z_flipped = key * z_t` |
| S1 加密：先逐元素乘符號 k | `past/SourceCode/guided_diffusion/anonymization.py:394` | `x_signed = key * x` |
| S1 加密：再做置換 P | `past/SourceCode/guided_diffusion/anonymization.py:397` | `x_permuted = self.permutation_key.apply(x_signed)` |
| S1 置換：以索引取值（只搬移位置，幅值不變） | `past/SourceCode/guided_diffusion/anonymization.py:225` | `x_permuted = x_flat[:, perm]` |
| S1 整數 seed：置換 seed 為固定偏移 | `past/SourceCode/guided_diffusion/anonymization.py:358` | `perm_seed = seed + 1000000  # Offset to ensure different permutation` |
| S1：符號金鑰以 seed/password 建構 | `past/SourceCode/guided_diffusion/anonymization.py:344` | `self.rademacher_key = RademacherKey(` |
| S1：置換金鑰以 seed+偏移/password 建構 | `past/SourceCode/guided_diffusion/anonymization.py:359` | `self.permutation_key = PermutationKey(` |
| 取樣腳本：S1 以 seed 與 password 同時建構 | `past/SourceCode/scripts/cfg_image_sample_anonymization.py:506` | `key = SignedPermutationKey(` |
| 啟動命令同時傳 seed 42 與 password | `past/SourceCode/run_cfg_inference_Crypto.sh:7` | `export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
| E2.2 wrapper 以 password 與 seed 同時建構（與前作一致） | `scripts/e2_legacy_wrapper.py:74` | `password=LEGACY_PASSWORD,` |

- [事實] 論文本身的宣稱：§4.3.1 寫「The key space of this scheme is 2^d」，§4.3.2 寫「a key space of up to 2^d × d!」（`past/Privacy-Preserving and Reversible Diffusion Models for Lesion Localization in Chest X-Rays.pdf`，PDF 第 25、26 頁，由 pdftotext 取得）。

## 3. 宣稱 vs 實際（對應 `paper_assets/tables/table_keyspace.csv`）

| 方案 | 模式 | 宣稱 | claim_bits | key_upper_bound_bits（金鑰上界） | seed_search_bits_separable（種子搜尋，可分拆） | kpa_full_pair_search_bits（已知一對完整明密文） | ciphertext_only_seed_search_bits（唯密文） | gap_bits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S0 Rademacher | password (legacy configuration) | 2^d | 65,536.0 | 32 | 32 | 0 | ≈ 32 | 65,504.0 |
| S0 Rademacher | integer seed | 2^d | 65,536.0 | 32 | 32 | 0 | ≈ 32 | 65,504.0 |
| S1 Signed Permutation | integer seed (sign and permutation coupled) | 2^d * d! | 1,019,572.9 | 32 | 32 | 0 | ≈ 32 | 1,019,540.9 |
| S1 Signed Permutation | password (legacy configuration; seed argument ignored) | 2^d * d! | 1,019,572.9 | 64 | 33 | 0 | ≈ 33 | 1,019,508.9 |

- [事實] d = 65,536；claim_bits 的 S0 = d，S1 = d + log2(d!) = 65,536 + 954,036.9（以 `math.lgamma` 計算）。
- [事實] 表中四個 bits 欄是四種不同的量，不可互換：
  - `key_upper_bound_bits`：金鑰可由多少個獨立 bits 決定的上界（由程式路徑得出）；**不是攻擊者的工作量**。
  - `seed_search_bits_separable`：攻擊者只有部分明文或只有密文、必須對 seed 窮舉時的搜尋 bits。S1 password 模式的兩半可分開窮舉（2·2^32 = 2^33，縮小示範為 2·2^18 對聯合 2^36，見 `results/A3.2.json` 的 E4）；**這不是已知明文的成本**。
  - `kpa_full_pair_search_bits` = 0：持有一對完整的已知明文/密文 latent 時不需要搜尋，金鑰直接讀出（第 5 節第 3 點）。
  - `ciphertext_only_seed_search_bits`：唯密文情境下的 seed 搜尋 bits。S0 ≈ 32（稽核方 AF-027）、S1 password ≈ 33（稽核方 AUD-20261011-02）、S1 整數 seed ≈ 32 為 [推定]（金鑰本身只有 32 bits）；皆非本腳本量測（第 5 節第 4 點）。
- [事實] `gap_bits` = claim_bits − key_upper_bound_bits，只比較宣稱與金鑰上界，不是攻擊工作量。

## 4. 數值驗證（皆以前作自己的 `RademacherKey`、`PermutationKey`、`SignedPermutationKey` 實測）

- V1 [事實] `sha256("Ki@13579")` = `5500887a4be187e69bd05984446bd6aec3c24684f1e2e7e4d3e16c81164c6826`，只用前 4 bytes → 符號 seed = 1426098298（位於 [0, 2^32) 的 0.3320 處）；其餘 224 bits 不影響金鑰。`sha256("Ki@13579_permutation")` = `238306c2c1bc5b6a8ae25f69844494fa6531e7f2ad3a580eebc2d26163a942c2` → 置換 seed = 595789506。
- V2 [事實] 只用整數 seed 重建金鑰：`RademacherKey(seed=1426098298).key` 與 `RademacherKey(password=...)` 在 65,536/65,536 個座標相同（torch.equal = 是）；`PermutationKey(seed=595789506)` 與 password 版在 65,536/65,536 個位置相同（torch.equal = 是）；以兩個整數 seed 組成的 `SignedPermutationKey` 對同一個 N(0,1) latent（seed 1911）的密文與 password 版逐位元相同（是）。即整把 65,536 維金鑰只由 32-bit 整數決定。
- V3 [事實] password 優先於 seed：`RademacherKey(seed=42, password="Ki@13579")` 與只給 password 的金鑰完全相同（是），與只給 seed=42 的金鑰不同（符號一致率 0.4989）；`SignedPermutationKey` 同理，符號與置換皆與只給 password 者相同，與只給 seed=42 者不同（置換位置一致率 0.000015）。因此啟動命令的 `--anonymization_key_seed 42` 對實際金鑰沒有任何作用。
- V4 [事實] 整數 seed 模式下 S1 的耦合：seed=0 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=42 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=1911 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=123456789 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）。另 seed = 2^32 − 1000000 = 4293967296 時 perm_seed = 2^32，其置換與 seed 0 的置換相同（是），見 V5。
- V5 [事實] torch 2.14.0+cu130 的 CPU generator 只保留整數 seed 的低 32 bits：seed=0 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=42 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=1234567 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=1426098298 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed = 2^64 會丟出 `ValueError`。所以即使呼叫端傳入 64-bit 整數，金鑰也至多 32 bits（上界與 SHA-256 截斷無關，獨立成立）。
- V6 [事實] 兩個不同密碼得到完全相同的金鑰：`collide-14550` 與 `collide-37896` 的 SHA-256 前 4 bytes 相同（seed = 2874078466，共計算 37,897 次雜湊；32-bit 生日界預期約 82,137 次），完整 digest 卻有 52 個十六進位字元不同；兩者的 `RademacherKey` 在 65,536/65,536 個座標相同（是）。置換同理：`collide-65750` 與 `collide-76760`（後綴 `_permutation`，seed = 4168062789，76,761 次雜湊）的 `PermutationKey` 在 65,536/65,536 個位置相同（是）。

## 5. 對「32 bits」說法的限定與細節：金鑰上界 S0 = 32／S1 = 64 bits ≠ 攻擊工作量（已知一對明密文 = 0；唯密文 S0 ≈ 2^32，AF-027）

- [事實] 摘要：金鑰上界 S0 = 32／S1 = 64 bits；實際攻擊工作量：已知一對明密文 = 0（直接讀出）；唯密文 S0 ≈ 2^32（AF-027）。「64 bits」是金鑰的上界，不是攻擊者的工作量（AF-026）。

1. [事實] 預設 legacy 設定（password 模式）下，S0 的 `key_upper_bound_bits` 是 32；S1 是 64，因為符號與置換各由一個獨立的 32-bit 前綴決定（anonymization.py:100 與 anonymization.py:191，且置換的 password 分支先於 seed 分支，anonymization.py:164，所以 `seed + 1000000` 的耦合在 password 模式不會發生）。
2. [事實] 只有整數 seed 模式才有 `seed + 1000000` 耦合（V4），此時 S1 的上界是 32。本專案 legacy 重現（E2.2，`e2_legacy_wrapper.py:69-74`）與前作一樣同時給 seed 與 password，此時 seed 被忽略（V3），走 password 模式。
3. [事實] 已知一對完整的明文/密文 latent 時不需要任何搜尋。S0 的密文為 `z_ano = mask ⊙ (k ⊙ z) + (1 − mask) ⊙ z`（anonymization.py:681、anonymization.py:682），故對 mask = 1 的座標 `k = sign(c ⊙ z)` 直接讀出；S1 的密文為 `c = P(k ⊙ z)`（anonymization.py:394、anonymization.py:397；置換只是索引取值 anonymization.py:225，不改變幅值），把 |c| 與 |z| 各自排序即可配對出 π，再由符號比對得 k，成本 O(d log d)。`results/A3.2.json` 的 E4 在 d = 4096、幅值互異的 latent 上以此法精確找回 (k, π)（`kpa_recovers_sign_and_permutation_exactly` = true）。所以 `kpa_full_pair_search_bits` = 0。
   [事實] seed 窮舉只在三種情況才有意義：部分明文（稽核方 AF-026：S0 約需 64 個已知座標，[推定]）、magnitude tie，以及唯密文（第 4 點）。float32 latent 並非幅值互異：A3.3 的 10 個 d=65536 latent 的相異幅值數皆小於 65536（`results/A3.3.json` 的 `inputs.*.stats.n_distinct_magnitudes`，即有座標共用相同幅值），僅憑幅值排序時這些座標的配對有歧義，需第二對明密文或對 seed 窮舉（上限為 `seed_search_bits_separable`）才能消除；其處理屬 A4.3，不影響本檔的 bits 上界。
   [推定] 需要 seed 窮舉時，S1 的符號 seed 與置換 seed 可各自獨立窮舉（置換只比對 magnitude 排序，可容忍 tie；符號只比對符號），候選數約 2·2^32 = 2^33，而非 2^64；這是 `seed_search_bits_separable` 欄的意義，不是已知明文的成本。S0 部分與縮小空間 S1 positive control 見 A3.2，完整 S1 KPA 屬 A4.3，不在本任務內。
4. [事實，稽核方結果；本腳本不量測] 唯密文也能對 seed 窮舉，而且 oracle 可分開評估符號與置換，所以 2^64 不是攻擊工作量。S0（AF-027，`audit/a3_s0_coa_oracle_check.py`）：單一密文 latent 對正確金鑰的分數約 37σ（三張真實 latent 為 37.1／36.8／37.0σ），2,000 個錯誤 seed 最高只有 3.3–3.8σ；[推定] 外推到 2^32 個錯誤候選，期望最大值約 6.7σ（高斯尾部假設），仍遠低於 37σ，故 S0 唯密文約 2^32。S1（AUD-20261011-02，`audit/a3_s1_coa_experiment.py`）：在 2^14 縮小空間中，**單一 S1 密文**即讓植入的置換 seed（14.6σ）與符號 seed（置換已知時 36.9σ）都排第 1，錯誤候選與負控制最高 ≤ 4.2σ；[推定] 外推到完整空間總工作量約 2·2^32 = 2^33。AUD-20261011-01 記錄的 subagent 估計是金鑰重用下需 10–50 張密文，AUD-20261011-02 實測更正為單張即可；前作部署本來就讓所有影像共用同一把金鑰（第 7 點）。實作方以自己的實作重現唯密文實驗屬 A3.2 補測（AF-027，尚未完成），在此之前 `ciphertext_only_seed_search_bits` 欄只引用稽核方結果。
5. [事實] 密碼 `Ki@13579` 寫死在啟動腳本內（第 8 點列出位置）；[推定] 取得程式碼的攻擊者不需要猜測，秘密為 0 bits，與 SHA-256 前綴的 32-bit 上界無關。[假設] 若攻擊者沒有程式碼，且密碼是人類可記憶字串，實際熵可能遠低於 32 bits（字典或規則式猜測）；本檔只量測由 SHA-256 前綴施加的 32-bit 上界，不量測密碼熵。
6. [推定] 32 是上界而非精確值：seed → 金鑰的映射在 2^32 個 seed 上是否全為單射本腳本未窮舉驗證（V6 只顯示不同密碼可共用同一 seed）。
7. [事實] 金鑰由 password 確定性導出（V2），且啟動命令寫死單一 password（`run_cfg_inference_Crypto.sh:7`）；[推定] 因此同一 password 下的所有影像共用同一把金鑰，恢復一把即可還原全部——這屬於金鑰重用問題，與位元數是兩個獨立議題。
8. [事實] 密碼 `Ki@13579` 寫死的位置（本腳本逐行驗證）。稽核方列出 `run_{cfg,uncond,clf}_inference_Crypto.sh` 與 `run_comparison_experiments.sh:29`；`.vscode/launch.json` 的兩處不在稽核方清單內，為本腳本額外找到。給定程式碼時秘密為 0 bits（第 5 點）。

   | 位置 | 內容 |
   |---|---|
   | `past/SourceCode/run_cfg_inference_Crypto.sh:7` | `export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_cfg_inference_Crypto.sh:8` | `export DECRYPTO_FLAGS="--anonymization_mode deanonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_uncond_inference_Crypto.sh:7` | `export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_uncond_inference_Crypto.sh:8` | `export DECRYPTO_FLAGS="--anonymization_mode deanonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_clf_inference_Crypto.sh:8` | `export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_clf_inference_Crypto.sh:9` | `export DECRYPTO_FLAGS="--anonymization_mode deanonymize --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
   | `past/SourceCode/run_comparison_experiments.sh:29` | `KEY_PASSWORD="Ki@13579"` |
   | `past/SourceCode/run_comparison_experiments.sh:70` | `--anonymization_key_password "${KEY_PASSWORD}" \` |
   | `past/SourceCode/run_comparison_experiments.sh:83` | `--anonymization_key_password "${KEY_PASSWORD}" \` |
   | `past/SourceCode/.vscode/launch.json:200` | `"Ki@13579",` |
   | `past/SourceCode/.vscode/launch.json:273` | `"Ki@13579",` |

   此外 `past/SourceCode/run_comparison_experiments.sh:46` 會把該密碼 echo 到執行輸出（`echo "Key password: ${KEY_PASSWORD}"`）。
9. [事實] 第二組設定：`run_paper_comparison.sh` 另有一個密碼設定，實際傳入的字串帶後綴（S0 用 `${PASSWORD}_rademacher`，S1 用 `${PASSWORD}_signed_perm`）：

   | 位置 | 內容 |
   |---|---|
   | `past/SourceCode/run_paper_comparison.sh:27` | `PASSWORD="paper_experiment_key_2024"` |
   | `past/SourceCode/run_paper_comparison.sh:62` | `--anonymization_key_password "${PASSWORD}_rademacher" \` |
   | `past/SourceCode/run_paper_comparison.sh:75` | `--anonymization_key_password "${PASSWORD}_signed_perm" \` |

   [推定] 它走同一條 SHA-256 前 4 bytes 路徑（第 1 點），所以 `key_upper_bound_bits` 的結論相同（S0 = 32／S1 = 64）；只有具體的 seed 值不同，本腳本未計算。[假設] 論文數字出自 `Ki@13579`（`run_*_Crypto.sh`、`run_comparison_experiments.sh`）還是 `paper_experiment_key_2024`（`run_paper_comparison.sh`）目前無法確定，尚未查證；本專案 E2.2 legacy 重現使用 `Ki@13579`（`e2_legacy_wrapper.py:74`，`LEGACY_PASSWORD`）。

## 6. 與後續任務的接口

- [事實] A3.2（`results/A3.2.json`）用同一批前作金鑰類別在縮小空間（2^20）上做窮舉 positive control，並量測吞吐後外推到 2^32；本檔不含任何計時。
- [事實] A3.3 驗證 S0 逐元素 |z_enc| = |z| 與 S1 排序幅值不變，這是不需要金鑰就能比對 gallery 的依據；本檔只處理金鑰空間。
- [事實] 唯密文 oracle（AF-027）：稽核方結果與證據檔見第 5 節第 4 點；實作方的 A3.2 補測完成前，本檔不含唯密文量測。
- [推定] 修補方向（D5.1）應使用 256-bit 隨機主金鑰，並由 CSPRNG／KDF 展開到 d 維，使每一位元都來自完整熵；該設計的檢驗不在本任務內。

## 7. 重現

```
scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit
```

- 輸出檔已存在時腳本拒絕覆寫，需加 `--force`（AF-029）：`scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit --force`。
- 驗證項共 42 個 file:line 錨點與 V1–V6 六組數值檢查；任何一項失敗腳本即以非零狀態結束，不寫出檔案。
