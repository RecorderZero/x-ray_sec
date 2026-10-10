# 前作金鑰空間審計：方法說明（A3.1）

- 任務：WORKFLOW A3.1 effective key-space audit；seed = 1911；CPU only，未載入擴散模型 checkpoint。
- 產生方式：`scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit`（由腳本寫出本檔與 `paper_assets/tables/table_keyspace.csv`，內容對固定原始碼、torch、numpy 版本為確定性）。
- 被審計原始碼：`past/SourceCode/guided_diffusion/anonymization.py`（sha256 = `5e71401a648a734979432ccf07d92deae02387352ad9ac00a33a89d9d8f38bb1`）；torch 2.14.0+cu130；numpy 1.23.0。
- 產生腳本 sha256：`7d5a2a6487a3badc348d44fc78281ee32f4c8b2a6181b703de12903b2839ffd4`。
- 標記說明：[事實] = 逐行讀碼或本腳本實測（附來源）；[推定] = 由 [事實] 推導，未獨立量測；[假設] = 尚未驗證的前提。

## 1. 結論

- [事實] 前作的 Rademacher 金鑰 k（d = 65536 維、值域 ±1）完全由一個 32-bit 整數 seed 經 `torch.Generator.manual_seed` 與 `torch.bernoulli` 決定（anonymization.py:100、anonymization.py:85、anonymization.py:87）。`actual_bits` 上界為 32，宣稱為 65,536，差 65,504 bits。
- [事實] Signed Permutation 在整數 seed 模式下，置換的 seed 為 `seed + 1000000`（anonymization.py:358），符號與置換由同一個 32-bit 值決定，`actual_bits` 上界為 32，宣稱 2^d·d!（約 1,019,572.9 bits）。
- [事實] 前作實際執行的 S1 設定（`run_cfg_inference_Crypto.sh:7`：`--anonymization_key_password Ki@13579 --use_signed_permutation True`）走 password 模式：符號 seed 與置換 seed 是兩個獨立的 32-bit 值，因此 `actual_bits` 上界是 64，不是 32。這是對「32 bits」說法的重要修正，見第 5 節。
- [推定] 上述上界均為 bits 數的上限；「搜尋成本」尚須 oracle，見 A3.2（實測吞吐與外推）。本檔不宣稱任何實際攻擊成功率。

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
| S1 整數 seed：置換 seed 為固定偏移 | `past/SourceCode/guided_diffusion/anonymization.py:358` | `perm_seed = seed + 1000000  # Offset to ensure different permutation` |
| S1：符號金鑰以 seed/password 建構 | `past/SourceCode/guided_diffusion/anonymization.py:344` | `self.rademacher_key = RademacherKey(` |
| S1：置換金鑰以 seed+偏移/password 建構 | `past/SourceCode/guided_diffusion/anonymization.py:359` | `self.permutation_key = PermutationKey(` |
| 取樣腳本：S1 以 seed 與 password 同時建構 | `past/SourceCode/scripts/cfg_image_sample_anonymization.py:506` | `key = SignedPermutationKey(` |
| 啟動命令同時傳 seed 42 與 password | `past/SourceCode/run_cfg_inference_Crypto.sh:7` | `export CRYPTO_FLAGS="--anonymization_mode anonymize --anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"` |
| E2.2 wrapper 以 password 與 seed 同時建構（與前作一致） | `scripts/e2_legacy_wrapper.py:73` | `password=LEGACY_PASSWORD,` |

- [事實] 論文本身的宣稱：§4.3.1 寫「The key space of this scheme is 2^d」，§4.3.2 寫「a key space of up to 2^d × d!」（`past/Privacy-Preserving and Reversible Diffusion Models for Lesion Localization in Chest X-Rays.pdf`，PDF 第 25、26 頁，由 pdftotext 取得）。

## 3. 宣稱 vs 實際（對應 `paper_assets/tables/table_keyspace.csv`）

| 方案 | 模式 | 宣稱 | claim_bits | actual_bits（上界） | 已知明文下搜尋 bits | gap_bits |
|---|---|---|---:|---:|---:|---:|
| S0 Rademacher | password (legacy configuration) | 2^d | 65,536.0 | 32 | 32 | 65,504.0 |
| S0 Rademacher | integer seed | 2^d | 65,536.0 | 32 | 32 | 65,504.0 |
| S1 Signed Permutation | integer seed (sign and permutation coupled) | 2^d * d! | 1,019,572.9 | 32 | 32 | 1,019,540.9 |
| S1 Signed Permutation | password (legacy configuration; seed argument ignored) | 2^d * d! | 1,019,572.9 | 64 | 33 | 1,019,508.9 |

- [事實] d = 65,536；claim_bits 的 S0 = d，S1 = d + log2(d!) = 65,536 + 954,036.9（以 `math.lgamma` 計算）。
- [事實] `actual_bits` 欄是「金鑰可由多少個獨立 bits 決定」的上界（由程式路徑得出），不是攻擊成本；`known_plaintext_search_bits` 是在攻擊者持有一對明文/密文 latent 時、對 seed 做窮舉所需的 bits（第 5 節）。

## 4. 數值驗證（皆以前作自己的 `RademacherKey`、`PermutationKey`、`SignedPermutationKey` 實測）

- V1 [事實] `sha256("Ki@13579")` = `5500887a4be187e69bd05984446bd6aec3c24684f1e2e7e4d3e16c81164c6826`，只用前 4 bytes → 符號 seed = 1426098298（位於 [0, 2^32) 的 0.3320 處）；其餘 224 bits 不影響金鑰。`sha256("Ki@13579_permutation")` = `238306c2c1bc5b6a8ae25f69844494fa6531e7f2ad3a580eebc2d26163a942c2` → 置換 seed = 595789506。
- V2 [事實] 只用整數 seed 重建金鑰：`RademacherKey(seed=1426098298).key` 與 `RademacherKey(password=...)` 在 65,536/65,536 個座標相同（torch.equal = 是）；`PermutationKey(seed=595789506)` 與 password 版在 65,536/65,536 個位置相同（torch.equal = 是）；以兩個整數 seed 組成的 `SignedPermutationKey` 對同一個 N(0,1) latent（seed 1911）的密文與 password 版逐位元相同（是）。即整把 65,536 維金鑰只由 32-bit 整數決定。
- V3 [事實] password 優先於 seed：`RademacherKey(seed=42, password="Ki@13579")` 與只給 password 的金鑰完全相同（是），與只給 seed=42 的金鑰不同（符號一致率 0.4989）；`SignedPermutationKey` 同理，符號與置換皆與只給 password 者相同，與只給 seed=42 者不同（置換位置一致率 0.000015）。因此啟動命令的 `--anonymization_key_seed 42` 對實際金鑰沒有任何作用。
- V4 [事實] 整數 seed 模式下 S1 的耦合：seed=0 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=42 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=1911 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）；seed=123456789 → 符號 = Rademacher(seed)（是）、置換 = Permutation(seed+1000000)（是）。另 seed = 2^32 − 1000000 = 4293967296 時 perm_seed = 2^32，其置換與 seed 0 的置換相同（是），見 V5。
- V5 [事實] torch 2.14.0+cu130 的 CPU generator 只保留整數 seed 的低 32 bits：seed=0 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=42 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=1234567 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed=1426098298 與 seed+2^32 產生相同的金鑰（是）與置換（是）；seed = 2^64 會丟出 `ValueError`。所以即使呼叫端傳入 64-bit 整數，金鑰也至多 32 bits（上界與 SHA-256 截斷無關，獨立成立）。
- V6 [事實] 兩個不同密碼得到完全相同的金鑰：`collide-14550` 與 `collide-37896` 的 SHA-256 前 4 bytes 相同（seed = 2874078466，共計算 37,897 次雜湊；32-bit 生日界預期約 82,137 次），完整 digest 卻有 52 個十六進位字元不同；兩者的 `RademacherKey` 在 65,536/65,536 個座標相同（是）。置換同理：`collide-65750` 與 `collide-76760`（後綴 `_permutation`，seed = 4168062789，76,761 次雜湊）的 `PermutationKey` 在 65,536/65,536 個位置相同（是）。

## 5. 對「32 bits」說法的限定與細節

1. [事實] 預設 legacy 設定（password 模式）下，S0 的 `actual_bits` 上界是 32；S1 的上界是 64，因為符號與置換各由一個獨立的 32-bit 前綴決定（anonymization.py:100 與 anonymization.py:191，且置換的 password 分支先於 seed 分支，anonymization.py:164，所以 `seed + 1000000` 的耦合在 password 模式不會發生）。
2. [事實] 只有整數 seed 模式才有 `seed + 1000000` 耦合（V4），此時 S1 的上界是 32。本專案 legacy 重現（E2.2，`e2_legacy_wrapper.py:70-74`）與前作一樣同時給 seed 與 password，此時 seed 被忽略（V3），走 password 模式。
3. [推定] 在攻擊者持有一對明文/密文 latent 時，S1 的置換可由 |z| 排序配對、符號可由逐元素比較直接讀出（連續分布下幅值理論上互異），因此符號 seed 與置換 seed 可各自獨立窮舉：候選數約 2·2^32 = 2^33，而非 2^64。此推論的 S0 部分與縮小空間 S1 positive control 見 A3.2，完整 S1 KPA 屬 A4.3，不在本任務內。
   [事實] 但 float32 latent 並非幅值互異：A3.3 的 10 個 d=65536 latent 的相異幅值數皆小於 65536（`results/A3.3.json` 的 `inputs.*.stats.n_distinct_magnitudes`，即有座標共用相同幅值），僅憑幅值排序時這些座標的配對有歧義；其處理屬 A4.3，不影響本檔的 bits 上界。
4. [推定] 在僅有密文、且 oracle 無法分開評估符號與置換的情況下，搜尋空間上界仍是 2^64（S1 password 模式）。是否存在這樣的 oracle 不在本任務檢驗範圍。
5. [假設] 若密碼本身是人類可記憶字串（例如本案的 `Ki@13579`），實際熵可能遠低於 32 bits（字典或規則式猜測）；本檔只量測由 SHA-256 前綴施加的 32-bit 上界，不量測密碼熵。
6. [推定] 32 是上界而非精確值：seed → 金鑰的映射在 2^32 個 seed 上是否全為單射本腳本未窮舉驗證（V6 只顯示不同密碼可共用同一 seed）。
7. [事實] 金鑰由 password 確定性導出（V2），且啟動命令寫死單一 password（`run_cfg_inference_Crypto.sh:7`）；[推定] 因此同一 password 下的所有影像共用同一把金鑰，恢復一把即可還原全部——這屬於金鑰重用問題，與位元數是兩個獨立議題。

## 6. 與後續任務的接口

- [事實] A3.2（`results/A3.2.json`）用同一批前作金鑰類別在縮小空間（2^20）上做窮舉 positive control，並量測吞吐後外推到 2^32；本檔不含任何計時。
- [事實] A3.3 驗證 S0 逐元素 |z_enc| = |z| 與 S1 排序幅值不變，這是不需要金鑰就能比對 gallery 的依據；本檔只處理金鑰空間。
- [推定] 修補方向（D5.1）應使用 256-bit 隨機主金鑰，並由 CSPRNG／KDF 展開到 d 維，使每一位元都來自完整熵；該設計的檢驗不在本任務內。

## 7. 重現

```
scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit
```

- 驗證項共 22 個 file:line 錨點與 V1–V6 六組數值檢查；任何一項失敗腳本即以非零狀態結束，不寫出檔案。
