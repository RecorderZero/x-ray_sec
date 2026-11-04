# 跨院系共享：加密影像上雲後，別的院系要怎麼存取？

> 2026-11-02。起因：教授提醒本系統的初衷是「多院系把加密版上傳雲端，跨院系要共享時怎麼存取」。
> 本檔做兩件事：(1) 整理文獻裡已有的解法；(2) 把它們對到 SOTERIA 現有的金鑰結構，提出可行設計。
> **2026-11-04：定案方案見 §8**（M1 接收方綁定共享；§6–§7 已被取代）。
> 2026-11-03：§6、§7 已依審閱意見 `reports/opinion.md` 修訂（浮點容差、輪替層級、L3 邊界、CPA 引用更正、追查移至 future work、v1/v2 證據分離）；§6.6 新增「M2 是否外包 AEAD／S2 角色」待裁決項。根目錄同名文件 `cross_institution_sharing.md` 是另一輪研究紀錄，依使用者裁決不合併。
> 標記：`[事實]` 已查證／已讀程式碼，`[推論]` 尚未驗證的分析，`[建議]` 設計提案，`[待辦]` 尚未做。

---

## 0. 一句話結論

**「發一把臨時鑰匙」這個直覺大致對，但要改成「發一把窄鑰匙」：只能解指定影像、裝在對方公鑰的信封裡發出去，並留下稽核紀錄。**
「臨時」這件事密碼學做不到：金鑰一旦交出去，對方就能永久解開那幾張影像。時效只能靠「發放時的存取控制」與資料使用協議（DUA）來管。
如果要做到真正可撤銷，就**不能把金鑰交出去**，改成讓運算到資料那邊做（TEE、聯邦學習）。

SOTERIA 現有的 per-image HKDF 結構**本來就支援窄鑰匙**，只需要小幅重構 API（§3）。

---

## 1. 我們現在的金鑰結構（共享的起點）

`[事實]`（`src/crypto/kdf.py`、`prng.py`、`mac.py`、`s2_fixed_norm.py`）

```
master_key (32B，每院一把，OS CSPRNG)
   │  HKDF-SHA256(ikm=master_key, salt=nonce_i, info=label)
   ├─ round/0/sign-pre   → D_pre
   ├─ round/0/perm       → P
   ├─ round/0/sign-post  → D_post
   ├─ mac                → HMAC 金鑰（容器完整性）
   └─ private-metadata-aead → AEAD 金鑰（鎖 ‖z‖、lo/hi）
```

- 每張影像 `i` 有自己的 128-bit nonce，所以每張影像的子金鑰都不同。
- HKDF 是單向的 PRF：拿到第 `i` 張的子金鑰，**推不回 master，也推不出第 `j` 張的子金鑰**。
- 所以「只開放第 `i` 張」在數學上已經成立。**目前的缺口在 API**：`transform()`、`open_authenticated()` 都只收 `master_key`，沒有「只用子金鑰解密」的入口。

**絕對不能做的事：把 master key 交給別的院系。** 交出去等於把該院全部影像（包含未來的）都開放了，而且無法收回。

---

## 2. 文獻裡已有的解法（由實用到前沿）

| # | 方法 | 代表文獻 | 核心概念 | 適合情境 | 限制 |
|---|---|---|---|---|---|
| A | **信封加密 + KMS**（envelope encryption） | 業界標準（AWS/GCP KMS）；包裝用 **HPKE**，RFC 9180（Barnes et al., 2022） | 每份資料有自己的資料金鑰（DEK）；共享時把 DEK 用**對方公鑰**包起來交出去 | 院對院、一次性專案共享 | 交出去就收不回；要自己管誰拿過 |
| B | **代理重加密（PRE）** | Blaze–Bleumer–Strauss, EUROCRYPT 1998；Ateniese, Fu, Green, Hohenberger, NDSS 2005 / ACM TISSEC 2006 | 雲端拿「重加密金鑰」把 A 的密文轉成 B 能解的密文，**過程中雲端看不到明文** | A 院不必在線，雲端代轉 | 雲端若與 B 串通，能取得被分享的那份資料（本來就要給 B）；單向方案才安全 |
| C | **屬性加密（CP-ABE）** | Bethencourt, Sahai, Waters, IEEE S&P 2007；**Li, Yu, Zheng, Ren, Lou, IEEE TPDS 24(1), 2013**（雲端個人健康紀錄共享，正中本題） | 用**政策**加密，例如「放射科 AND 研究員 AND 專案X」；具備屬性的人就能解 | 多院、多角色、政策式授權 | 撤銷困難；計算較重；需要屬性授權中心 |
| D | **ABE + PRE + 延遲重加密** | Yu, Wang, Ren, Lou, IEEE INFOCOM 2010 | 把撤銷時的重加密工作外包給雲端，等資料下次被讀時才重加密 | C 方案加上撤銷需求 | 系統較複雜 |
| E | **金鑰輪替／可更新加密** | Everspaugh, Paterson, Ristenpart, Scott, CRYPTO 2017 | 不解密就把密文換成新金鑰的版本 | 撤銷後讓舊鑰匙失效 | 只能擋住「未來的存取」，已下載的擋不了 |
| F | **區塊鏈存取紀錄** | Azaria, Ekblaw, Vieira, Lippman, **MedRec**, OBD 2016 | 鏈上只記錄存取權限與資料指標，資料本身不上鏈 | 多院互不信任、需要不可竄改的稽核 | 本身不做加密，要配合 A–C |
| G | **不交出資料／金鑰：聯邦學習、計算到資料** | Kaissis et al., *Nature Machine Intelligence* 2, 2020；Personal Health Train（荷蘭 2016–） | 模型或演算法到各院跑，資料與金鑰都不出院 | 目的是「訓練模型或統計」而不是「看影像」 | 不能讓對方親眼看到影像 |
| H | **機密運算（TEE，如 Intel SGX/TDX）** | 業界 | 金鑰只在硬體保護區內使用，對方拿到的是結果 | 想要可撤銷，又要在解密後的影像上運算 | 要信任硬體廠商；有側通道風險 |

`[事實]` 2023–2024 有不少「PRE + 區塊鏈 + 雲端」的電子病歷共享論文（例如 Hyperledger Fabric + PRE + IPFS，*Cluster Computing* 2024；跨網域 PRE「FABRIC」，IEEE TDSC 2023），但**專門針對醫學影像（DICOM/PACS）跨院共享的 PRE 研究很少**。本研究可以補這個缺口，論述上是加分。

### 文獻的共同結論（對我們最重要）

1. **不直接分享密文本體的金鑰權限，而是分享「包起來的資料金鑰」**（A、B、C 都是這個模式）。重加密、政策、撤銷都作用在這把幾十 bytes 的小金鑰上，不去碰動輒幾 MB 的影像。
2. **撤銷是所有方案的弱點**。已交出的金鑰收不回；可做的只有擋住未來的存取（E），或從頭就不交出金鑰（G、H）。
3. **存取控制與稽核要和加密分開設計**（F 的定位）。

---

## 3. 對到 SOTERIA：建議設計 `[建議]`

### 3.1 加一層「影像金鑰」（最小改動、收益最大）

```
master_key ──HKDF(salt=nonce_i, info="image-key/v2")──► k_img_i (32B)
                                                          │ HKDF(info=label)
                                                          ├─ D_pre, P, D_post（各輪）
                                                          ├─ mac
                                                          └─ private-metadata-aead
```

- 共享第 `i` 張時，只交出一把 32 bytes 的 `k_img_i`，不必交出 5 把以上的子金鑰。
- 解密 API 改成可以直接接受 `k_img`。現有的 `derive_subkey(master, nonce, label)` 換成 `derive_subkey(k_img, label)`，加密數學（D·P·H·D）完全不變。
- **代價**：金鑰推導是 v1→v2 的格式變更，舊容器需要標記版本，或重新封裝。v1 路徑與凍結結果保持不變；v2 的品質與安全數字**未重跑前不宣稱與 v1 等同**（§7.5）。

### 3.2 需要大量共享時：分層金鑰樹

```
master → project/<專案> → patient/<病患> → image/<nonce>
```

- 交出 `project` 層的金鑰，等於開放整個專案的影像；交出 `patient` 層，只開放一位病患。
- 依據：**分層金鑰指派**（Akl & Taylor, ACM TOCS 1983）。這是經典做法，BIP32 HD wallet 也是同一個模式。
- 限制：分享的範圍必須在**加密時就決定**（路徑要寫進推導）。事後才臨時決定的組合，就退回 3.1 的逐張發放。

### 3.3 發放流程（A 院 → B 院）

1. B 院提出申請（專案、影像清單、用途、期限）；由 A 院的資料委員會或 IRB 核准。
2. A 院的 KMS/HSM（master key 永遠不離開）推導出對應的 `k_img` 或專案金鑰。
3. 用 **HPKE（RFC 9180）** 包給 B 院的公鑰。HPKE 的 AAD 放入授權編號、到期日、用途，讓這些欄位也受完整性保護。
4. 發放紀錄寫進只能追加的稽核日誌（想要多院互信，可以改用鏈上紀錄，見 F）。
5. B 院從雲端下載容器（匿名 PNG + header），用 `k_img` 驗證標籤後解密，再用**同一份 checkpoint 與 sampler 設定**生成還原影像。header 已記錄這些設定。

`[推論]` 第 5 步表示**對方也需要擴散模型**。依 Kerckhoffs 原則，模型可以公開，安全性只靠金鑰。但是保真度依賴 sampler 設定完全一致，所以模型版本必須跟著資料一起被管理。

### 3.4 兩個容易忽略的安全細節

- **完整性 vs 來源證明**：現在的 HMAC 是對稱式的。B 院拿到第 `i` 張的 mac 子金鑰後，可以**偽造**第 `i` 張的合法標籤。跨院時建議由 A 院再加一個**數位簽章（如 Ed25519）**。這樣 B 院可以驗證來源，但無法冒充 A 院。
- **「臨時」的真實意義**：到期日只能管「KMS 還要不要再發」。已經交出去的 `k_img` 沒有到期這回事。要真正收回權限，只能選擇 G/H（不交出金鑰），或用 E 輪替掉雲端上的版本。後者只擋得住未來的下載。

### 3.5 什麼時候用哪一種

| 需求 | 建議 |
|---|---|
| 一次性的研究合作，對象明確 | **3.1 + HPKE 信封**（最簡單，論文可以直接實作示範） |
| A 院不想每次都在線上處理請求 | 加上 **PRE**（B）：雲端代轉包好的 `k_img`，看不到內容 |
| 多院、多角色、規則複雜 | 用 **CP-ABE**（C；Li et al. TPDS 2013 是直接可以對照的前例）包 `k_img` |
| 只要訓練模型或做統計，不需要看影像 | **聯邦學習或計算到資料**（G），金鑰完全不出院 |
| 需要真正可撤銷 | **TEE**（H）或 G |

### 3.6 為什麼「不在影像層做代理重加密」

`[推論]` 我們的加密是線性的，所以理論上可以給雲端一把 `R = Q_B Q_Aᵀ`，把 A 的密文潛向量轉成 B 的版本。但這個做法不建議：

1. `R` 是稠密的 65536×65536 矩陣（約 2³² 個元素、16 GB），失去 D·P·H·D 的快速結構。
2. 雲端必須先把匿名影像反演回潛向量、套用 `R`，再重新生成影像。多一次反演與生成會**累積誤差，降低還原保真度**。
3. 雲端若與 B 院串通，可以從 `R` 與 `Q_B` 算出 `Q_A`（僅限該張影像）。

在 32 bytes 的 `k_img` 上做 PRE 便宜又標準，所以重加密應該放在金鑰層。

---

## 4. 放進論文／口試的定位

- **研究範圍內（已完成）**：單一院所的「上雲保護」，包括匿名化、可逆、抗連結、完整性。
- **系統延伸（本檔提出，尚未實作）**：跨院共享。SOTERIA 的 per-image HKDF 結構讓「逐張或逐專案發窄鑰匙」**不需要更動加密數學**，只需要加一層影像金鑰，再接上標準的包裝或授權機制（HPKE / PRE / CP-ABE）。
- **口試一句話**：「共享時不交出 master key，而是用對方的公鑰包一把只能解指定影像的子金鑰。因為 HKDF 是單向的，這把子金鑰推不回 master，也解不開其他影像。撤銷是所有密碼系統的共同限制，需要真正撤銷時，就改成聯邦學習或 TEE，讓金鑰不出院。」

## 5. `[待辦]`（若要在論文中示範）

1. `kdf.py` 新增 v2 的影像金鑰層，`transform`、`open_*` 接受 `k_img`（容器 header 帶版本）。
2. 使用 RFC 9180 相容的 HPKE 實作（`cryptography` 內建 `hpke` 模組），以官方測試向量驗證，不自行拼接 X25519／HKDF／AEAD；另加 Ed25519 來源簽章。
3. 測試：B 院用 `k_img_i` 能解第 `i` 張；解第 `j` 張的標籤驗證失敗；B 院無法偽造 A 院的簽章。這些都是確定性結果，N 不影響（與附錄 B 類同）。

---

## 6. 設計細化（2026-11-03 討論；同日依 `reports/opinion.md` 修訂）

> ⚠ **2026-11-04：本章已被 §8 取代**，保留作為設計演進紀錄，不再採用。

### 6.0 範圍邊界（依 opinion §1、R8）

- **要做到**：
  - 限定範圍的共享。
  - 綁定身分的取件。
  - 分層金鑰輪替。
  - 由服務端執行的授權撤銷。
- **不主張**：
  - **洩漏者追查**（列為 future work，見 §6.4 末段）。
  - 遠端收回已經交付的內容。
  - DPHD 具備一般性的 IND-CPA／CCA 機密性。
- **共享層只保護金鑰、完整性與來源**，不會自動補上「公開 DPHD payload 的統計結構」這個缺口：
  - HPKE 保護信封裡的金鑰。
  - metadata AEAD 只保護它涵蓋的欄位。
  - MAC 與簽章提供完整性與來源驗證。

  相關邊界見 §6.6 與 `reports/linearity_security.md` §2。

### 6.1 金鑰樹

```
master_key（A 院；由受信任的金鑰服務以 HSM/KMS 保護）
  └─ patient_key_p,e = HKDF(master, salt=patient_salt_p, info=enc("patient", inst_A, e, "v2"))
       └─ k_img_i     = HKDF(patient_key_p,e, salt=nonce_i, info=enc("image", "v2"))
            ├─ round/r/{sign-pre, perm, sign-post} → D_pre, P, D_post
            ├─ mac
            └─ private-metadata-aead
  └─ grant_key_g,n   = HKDF(master, salt=–, info=enc("grant", inst_A, grant_id, generation n, "v2"))   ← Q1(a)
```

- **`enc(...)`** 是無歧義的編碼（每個欄位加長度前綴）。domain 涵蓋院所、用途、版本與世代，與 v1 的 `cxr-transform/v1/...` 完全分開。
- **`patient_salt_p`** 是 A 院**私有索引**中的 128-bit 隨機值，不使用病歷號。
- **patient key 只發給院內受信對象**（Q2）。跨院一律使用 grant key（§6.4 L2）。
- **epoch 規則**（11-03 修訂）：
  - 已經存在的影像不重新加密，epoch 也不變；院內受信對象先後索取同一批舊影像時，拿到同一把 `patient_key(e)`。
  - `patient_key(e)` 一旦發出，epoch e 就**封存**，這位患者之後的新影像改用 `e+1`。
  - 「發鑰 → 封存 → 新影像入庫」的狀態更新必須**持久化而且原子化**，避免並發或重啟後把不該授權的影像納入舊 epoch。
- **推導的邊界**：
  - 持有 `grant_key_g,n` 的人，可以推導出該 grant 派生域裡任何已知 nonce 的 leaf key。
  - 因此 grant 必須**綁定一組固定的物件與版本**；要擴大範圍時，使用新的 grant 或新的世代 n。
- **刪除授權紀錄不等於在密碼學上抹除金鑰**，也不提供 master 洩漏之後的前向保密。

### 6.2 ⚠ 隱私陷阱：公開 header 不能出現患者節點

- **問題**：如果公開 header 帶有「這張影像屬於 patient 節點 p」這類**由患者衍生、而且固定不變**的識別值，任何人都能把同一位患者的影像連起來。舊的 inner-linkage 實驗本身不會因此作廢，但它的結果**無法涵蓋這個新公開的資訊**。
- **做法**：
  - header 採 **schema allowlist**：只允許 version、algorithm profile ID、per-image 隨機 nonce、shape／dtype、tag、簽章。
  - shape、dtype、profile 在不同影像間相同是合理的，**不要求非常數欄位一定互不相同**。
  - 「nonce → 患者／epoch」的對應只存在 A 院的私有索引。
  - 共享時，A 在授權內容中附上該 grant 的物件清單。
- **限制**：物件長度、檔名、存取模式、批次成員、公開索引仍可能用來連結資料。本原型不證明能抵抗流量分析；刪掉患者 ID 也不能證明整個系統不可連結。

### 6.3 金鑰輪替（E）

**依洩漏層級處理（R2）**：

| 洩漏的是 | 新版本需要的處理 |
|---|---|
| 單張 image key | 換新 nonce，派生出不同的 image key |
| `patient_key(p,e)` | **換到新 epoch**（新的 parent key）；只換 nonce 沒有用 |
| `grant_key(g,n)` | 換新的 grant ID 或世代 n；在同一把 grant key 下只換 nonce 沒有用 |
| master key | 所有下層金鑰都能被重新推導。屬於另一層級的事故處理，不在本原型的證明範圍 |

**M2 的輪替方式**：
- 在受信任閘道內計算 `c' = Q_new Q_oldᵀ c`，金鑰由 HSM/KMS 保護。通用 HSM 不一定能執行 FWHT，也不能忽略閘道記憶體曝露的風險。
- 同時要做：重新封裝私有 metadata，並**保留原始 norm**，不要從反覆還原後的近似 latent 重新估算；重算 MAC 與來源簽章；**原子化更新**物件版本與索引。

**成本（[事實]，2026-11-03 實測，CPU，d=65536，R=1，P on）**：
- 單張加密 71 ms，其中產生置換佔 38 ms（Python 迴圈）；解密 41 ms。一次輪替約 **0.1 s／張**。
- 完整流程的成本（KDF、metadata、簽章、序列化、索引）**還沒量**，由 V12 補上。

**數值誤差（R1）**：
- DPHD 在實數運算下是可逆的；浮點實作**不承諾 latent byte-exact**。
- 實測：float32 單次往返的 max-abs 為 7.2e-7，只有 17% 的座標 byte-exact。
- 改用 float64 運算並以 float64 儲存，這一次測試是 100% byte-exact，但這**不是保證**：數值極接近 0 的座標，float32 的 ulp 也極小，仍可能出錯。
- 驗收時：容器的 bytes 讀寫要求精確；latent 往返用預先登記的容差驗證（§7.4 V2、V9）。

**M1（只存匿名 PNG）**：必須反演、重新加密、重新生成影像，每輪替一次就多一次反演誤差，不建議。

**限制**：
- 目前**還沒有適合本原型的安全、緊湊的更新 token**。直接交出兩組轉換因子會洩漏金鑰，交出稠密矩陣則有 16 GB。這只說明這兩種做法不適合，不代表所有 PRE／updatable encryption 構造都不可能。
- 輪替之後，服務**不再提供舊版本**。但這無法證明惡意雲端或接收端沒有留存副本。
- 外層 AEAD 如果只是用同一把 DEK 重新包裝，也撤銷不了已經拿到那把 DEK 的人。

### 6.4 B 院散布金鑰怎麼辦？

**系統邊界**：
- 接收端一旦拿到可以離線解密的材料（金鑰＋密文，或已解密的 latent／影像），系統就無法阻止它轉交。
- 線上服務可以拒絕**之後的**請求，但收不回已經交付的內容。
- 「曾授權給誰」不等於「證明誰洩漏」：A 自己也推導得出 grant key，B 的金鑰可能被竊，接收者也可能外流明文、衍生的子金鑰或解碼程式。**洩漏者識別列為 future work。**

**「用 B 的私鑰簽章讓 A 稽核」要放在 C 繞不過去的關卡**，也就是取件與發鑰。離線解密時，對稱金鑰不會檢查持有者是誰。

| 層級 | 做法 | 能做到 | 做不到 |
|---|---|---|---|
| **L1 簽章取件** | 本機 Ed25519 挑戰-回應（原型機制；未實作完整的 OAuth DPoP／mTLS，因此**不宣稱符合 RFC 9449／8705**）。簽章內容綁定：協定 domain、服務 audience、操作、grant ID、物件與版本、接收者、伺服器挑戰、有效時間。挑戰由服務端**原子化消耗** | C 只有外流的資料金鑰、沒有 B 的私鑰 → 取不到容器 | B 同時外流金鑰與密文；B 的私鑰外流或被代簽（前提是對手**沒有**對應私鑰或簽章代理能力） |
| **L2 grant 專屬的取件時轉加密** | 雲端只存一份原始密文。B 取件並通過 L1 後，受信任閘道把物件轉加密成 grant 派生域下的 `c_B`（新 nonce），交給 B 保存。**雲端不存副本** | 交付的版本與金鑰被限制在指定 grant 之內，**院內儲存用的金鑰與 B 隔離**；不同 grant 之間無法互開 | 不做洩漏者識別；不能防止 B 外流 |
| **L3 金鑰不出院（閘道）** | B 每次存取都要簽章；閘道確認授權後才解密並回傳。閘道的規則見下方 | **授權狀態更新後，服務拒絕之後的請求**。撤銷在「交付前的最後一次授權確認」生效 | 已交付的 latent／明文；B 可以離線保存並生成影像 |
| **L4 B 端硬體綁定** | 把金鑰包給 B 的 HSM/TEE 裡不可匯出的金鑰，並做遠端證明 | 在**硬體、attestation、受信程式與設定都成立**的前提下，取不出金鑰 | 本期不做 |
| **制度** | DUA、基本稽核紀錄 | 管理與故障調查 | — |

**L3 閘道的規則（R4）**：
1. 只接受**服務端的 object ID**，不接受客戶端送來的 latent 或完整容器。
2. 從受信任的儲存命名空間取出**確定的版本**，驗證 **A 院簽章**、完整性與授權範圍。只通過 MAC 不夠，因為持有共享對稱金鑰的人可以產生合法的 MAC。
3. 拒絕未知、被替換、已過期或不屬於該 grant 的物件。
4. 上傳介面不能繞過上述來源驗證。客戶端放進儲存區的任意向量，不會因此被允許解密。
5. 限制長度、shape、dtype、演算法版本，並要求數值為有限值，避免惡意 header 造成不受控的資源配置。

**為什麼不能開放任意向量解密（R5 更正）**：任意向量解密介面會擴大攻擊者的能力。
- 「**一次查詢**即可重建」是**前作 S0/S1** 的結果（PROGRESS A4.3／A4.4）。
- S2 的證據是 d=1024、同一 nonce 下的**完整基底 CPA／滿秩 KPA 對照實驗**（D7.4）。兩者條件不同。
- **不主張一次查詢即可重建 DPHD**。選擇明文與解密 oracle 也分開命名。

**口試一句話**：「我們處理的是跨院授權與金鑰隔離，不處理洩漏者識別。接收端拿到可以離線解密的材料後，系統無法阻止它轉交；線上閘道可以拒絕之後的存取，但收不回已經交付的 latent 或明文。更強的外洩追蹤留給後續研究。」

**Future work**：traitor tracing、鑑識浮水印、抗串謀、衍生解碼器追蹤、證據保全與責任歸屬、外部不可竄改稽核、真實雲端／KMS/HSM／TEE、PRE／ABE／安全的更新 token。

### 6.5 建議組合

- **院內**：直接發 `patient_key`，並依 §6.1 推進 epoch。
- **跨院**：L1 + L2，加上 A 院 Ed25519 簽章、基本稽核紀錄與 DUA。
- **需要服務端撤銷**：使用 L3（O2，已裁決要做）。

### 6.6 M2 要不要再包一層 AEAD？S2 在共享情境的角色（Q5，待裁決）

**判斷**：如果共享的目標只是「把檔案鎖在雲端，讓授權的院所讀到原始內容」，**直接用標準 AEAD 加密原始影像就夠了，而且更好**：
- bytes 完全精確。
- 有標準的 IND-CCA 證明。
- 接收方不需要擴散模型。
- 容量是原圖加上數十 bytes，而 M2 每張約 263 KB。

在這種情況下，替 M2 再包一層 AEAD 會讓 DPHD 對機密性**沒有額外貢獻**，等同於繞一圈做 AEAD。§6.1–§6.4 的金鑰樹、HPKE、L1/L2/L3、輪替設計，**全部可以原封不動地套在 AEAD 的 DEK 上**，而且更簡單：輪替就是 bytes 的重新加密，結果精確。

**S2／SOTERIA 只在「沒有金鑰的人也必須拿到一張影像」時才有角色**：
- 匿名影像是一張外觀自然的胸部 X 光，可以在需要影像格式的系統裡流通。
- 持有金鑰的人可以從它還原。

需要誠實面對的事實：
- 我們自己的結果顯示，匿名影像的診斷效用接近隨機（classification ≈ chance）。所以它的價值在**格式相容與作為不洩漏身分的替代影像**，不在臨床使用。
- 「匿名影像＋另附 AEAD 原圖」也能做到可逆。SOTERIA 相對於這種組合的差異，是「單一物件」與「還原資訊藏在影像本身」。這需要在論文中明確說明使用情境。

**候選方案（待使用者裁決 Q5）**：

| 方案 | 雲端存放 | 機密性來源 | SOTERIA 的角色 |
|---|---|---|---|
| (A) 純 AEAD | AEAD(原圖) | 標準 AEAD | 無；共享原型與 SOTERIA 脫鉤 |
| (B) 雙層 | M1 匿名 PNG（可流通）＋ AEAD(原圖)（授權還原） | 標準 AEAD | 只負責產生匿名影像；不使用 S2 的可逆性 |
| (C) 現行計畫 | M2 DPHD payload（不再外包 AEAD） | DPHD + nonce + 簽章（R8 邊界：雲端可以觀察 payload 的統計結構） | 完整的 SOTERIA，包括可逆性 |

論文定位依 `PROPOSAL.md`／`WORKFLOW.md`，維持「authenticated reversible obfuscation」，**不宣稱取代標準加密**。本研究的核心貢獻是：分析前作線性潛空間加密的攻擊面、提出修補，並在指定資料分布與攻擊下做評估。

---

## 7. 執行計畫（2026-11-03 版；⚠ 已被 §8 取代）

> ⚠ **2026-11-04：本章已被 §8 取代**，Q1–Q3 的裁決隨之失效；Q5 定為 (D)。保留作為紀錄。

### 7.1 目標與範圍

> 在 M2 部署下，建立支援限定範圍共享、身分綁定取件、分層金鑰輪替及服務端授權撤銷的研究原型。保留 v1 行為與既有結果；v2 另行驗證協定正確性與數值誤差。不主張洩漏者追蹤、已交付內容的遠端收回，或一般 IND-CPA／CCA 機密性。

- **做**：
  - v2 金鑰樹、M2 二進位容器、RFC 9180 相容的 HPKE 信封、A 院 Ed25519 簽章。
  - L1 簽章取件、L2 grant 專屬的取件時轉加密、依洩漏層級的輪替、基本稽核紀錄、O2（L3 最小閘道）。
  - 全部在單機程序內模擬 A、B、C 三方與雲端。
- **不做**：
  - 真實網路、雲端、KMS/HSM、TEE（L4）。
  - PRE/CP-ABE、多授權中心。
  - 洩漏者識別與鑑識浮水印、阻止明文外流、外部見證或 WORM 稽核。
- **資源**：全程 CPU。
- ⚠ **Q5 若選 (A) 或 (B)**，payload 改為 AEAD，S3／S4 與 V2／V9 要相應調整。

### 7.2 設計決定 `[決定]`

| # | 項目 | 內容 |
|---|---|---|
| D1 | 部署模式 | **M2**（儲存 DPHD latent payload）。匿名 PNG 是選配，不列入原型。外層 AEAD 待 Q5 裁決 |
| D2 | 金鑰樹與 domain | §6.1：master → patient(e) → image → 子金鑰；grant 由 HKDF(master, enc(grant_id, generation)) 推導（Q1）；欄位以無歧義編碼，與 v1 domain 分離 |
| D3 | header | schema allowlist（§6.2）。**algorithm profile ID** 涵蓋 rounds、permutation、fixed-norm、dtype、byte order 與版本。模型 checkpoint、sampler、生成設定放在**已簽署的 manifest**，容器綁定其 digest。序列化格式固定，並定義簽章與 MAC 涵蓋的範圍；拒絕未知欄位、不合法欄位與降級設定 |
| D4 | epoch（11-03 修訂） | 已有影像不重新加密、epoch 不變。院內受信對象拿到同一把 `patient_key(e)`；發出後 epoch e 即封存，新影像改用 e+1。狀態更新持久化並原子化 |
| D5 | 跨院（11-03 修訂） | L2 grant 專屬的取件時轉加密；雲端只存一份原始密文。每個 grant 綁定固定的物件與版本集合，擴大範圍時使用新 grant 或新世代 |
| D6 | 金鑰傳遞 | HPKE base mode，**RFC 9180 相容實作**：使用 `cryptography` 50.0.2 的 `hazmat.primitives.hpke`（Suite／KEM／KDF／AEAD；本機已確認模組存在），**以官方測試向量驗證**，不自行拼接。base mode 不驗證發件者，所以由 A 另外簽署授權與信封的綁定內容，並維護可信的公鑰註冊。Ed25519 與 X25519 金鑰分開使用。expiry 放進 AAD 只能防竄改，期限由服務端執行 |
| D7 | 來源證明 | A 院以 Ed25519 簽署容器與授權。HMAC 保留作為對稱式完整性檢查，**不視為來源證明** |
| D8 | L1 | §6.4 的挑戰-回應：挑戰原子化消耗，防並發重放、跨物件重放與跨操作重放 |
| D9 | 輪替 | 依 §6.3 的洩漏層級處理。重新封裝 metadata 時保留原始 norm；重算 MAC 與簽章；原子化更新版本與索引；服務不再提供舊版本 |
| D10 | 稽核 | 基本紀錄：發放、取件、拒絕、輪替、撤銷。**不記錄祕密金鑰或不必要的患者資訊**。hash-chain 只在保有可信鏈頭與筆數時才能偵測竄改；無法抵抗能同時改寫日誌與檢查點的管理者 |
| D11 | 相容性 | v1 的 `kdf.py`／`sot.py` 路徑行為完全不變；v2 只新增檔案與參數 |
| D12 | fixed-norm 契約 | 加密時先固定 norm 再做 DPHD；解密時做 inverse，再用私有 metadata 中的原始 norm 恢復。M2 不需要 PNG 的 lo/hi 欄位 |

### 7.3 實作項目（新檔為主，不動 v1 行為）

| 步驟 | 檔案 | 內容 |
|---|---|---|
| S1 | `src/crypto/prng.py`（小重構） | 把「推導子金鑰」與「由子金鑰產生 sign／perm」拆開，新增 `sign_vector_from_key()` 與 `permutation_from_key()`。v1 函式改成呼叫它們，**輸出必須與重構前固定的 v1 test vectors bit-identical** |
| S2 | `src/crypto/kdf_v2.py` | 無歧義編碼的 `patient_key`、`image_key`、`grant_key`、`image_subkey` |
| S3 | `src/crypto/sot_v2.py` | 依 D12：fixed-norm → DPHD（以 image key 推導）→ inverse → 恢復 norm |
| S4 | `src/crypto/container_v2.py` | M2 二進位容器：allowlist header、profile ID、AEAD 私有 metadata（原始 norm）、payload、HMAC、Ed25519；固定序列化格式 |
| S5 | `src/sharing/` | `envelope.py`（HPKE）、`grants.py`（發放、轉加密、撤銷、世代）、`store.py`（模擬雲端、L1、版本管理）、`rotation.py`、`audit.py`、`index.py`（私有索引、epoch 封存） |
| S6 | `src/sharing/gateway.py` | **O2**：L3 最小閘道（§6.4 規則 1–5） |
| S7 | `tests/security/test_sharing_v2.py` | §7.4 的全部測試 |
| S8 | `scripts/sharing_demo.py` | 以 fixture key 跑完整情境 → `paper_assets/tables/table_sharing_v2.{csv,json}`（測試結果、容量、耗時）；使用 output_guard |

### 7.4 驗收測試（具體條件下的功能／正確性驗證，不等於通用的密碼學安全證明）

**預先登記的數值容差**（沿用 skill §5 的加密不變量門檻）：
- latent 往返 **max-abs < 1e-5**，適用於單次往返與連續 10 次輪替。
- 連續 100 次輪替只回報 max-abs、相對 L2、有限值檢查，不設門檻。
- 這些門檻不在跑完之後調整。

| 測試 | 驗收重點 |
|---|---|
| V1 | v1 已知 test vectors，sign／perm 輸出 bit-identical（要和重構前的 baseline 比對，不是只看測試有沒有過） |
| V2 | 容器 bytes 讀寫精確；latent 往返符合上述容差（float32） |
| V3 | 用錯的 image key 無法通過完整性驗證；失敗時不輸出部分明文 |
| V4 | 院內 patient 範圍、epoch 隔離、封存後新增的影像進入新 epoch、並發與重啟後狀態正確 |
| V5 | HPKE 官方測試向量；接收者錯誤；信封或 AAD 被竄改；發件者來源綁定 |
| V6 | A 的簽章涵蓋所有語意欄位；非法容器 fail-closed；無法用合法欄位替換來偷換物件 |
| V7 | L1：身分、範圍、期限、撤銷（包括撤銷 B 的公鑰）、重放、物件與操作綁定；另外保留對照組：已取得「金鑰＋密文」的人可以離線解密（限制的示範） |
| V8 | L2：grant 隔離、合法還原（符合容差）、不同 grant 之間無法互開、來源物件庫的容量不隨授權增加（授權紀錄、nonce 狀態、索引、日誌的成長另外報告）；**不測洩漏者識別** |
| V9 | 各洩漏層級的輪替都正確；**負控制**：持有 parent key 的人，在 parent 不變、只換 nonce 時仍能推導新的 leaf（這是預期行為，不是 bug）；多次輪替的誤差；版本原子更新 |
| V10 | 公開 schema 不含由患者衍生的固定 ID；私有索引不外洩；不據此推論整個系統不可連結 |
| V11 | 日誌與授權操作一致；在保有可信鏈頭與筆數的前提下，修改、刪除、截尾、重排都能被偵測 |
| V12 | 實測完整流程的 CPU 耗時與各類儲存成本；效能數字不是安全證明 |
| O2 | 閘道只處理可信來源、指定版本、已授權的物件。偽造 MAC 但沒有 A 簽章的容器、客戶端上傳的物件、撤銷後的請求都會被拒；明列「已交付的 latent 可以被保留」這個限制 |

### 7.5 對既有結果的影響（依 R7）

> v1 路徑及凍結結果保持不變，仍是 v1 的實驗證據。v2 先完成協定與數值正確性測試；未重跑的影像品質、re-id、inner-linkage 或攻擊實驗，不宣稱已證實與 v1 等同。O3 仍待裁決，若日後要提出 v2 的量化安全或品質結論，再依固定協定補驗。

### 7.6 可選項

| 選項 | 內容 | 狀態 |
|---|---|---|
| O1 | float16 payload 的還原 PSNR（需要 GPU） | ⏸ 待定（Q4） |
| O2 | L3 最小閘道（§6.4 規則 1–5，CPU） | ✅ 要做（Q3） |
| O3 | 用 v2 金鑰重跑 inner-linkage（N=200） | ⏸ 待定（Q4）。不做的話，就不提出「v2 已實證與 v1 等價」 |

### 7.7 流程與治理（11-03 使用者裁決）

1. 依 S1 → S8 實作。**每個子任務通過後立即原子 commit**（CLAUDE.md §4；commit 前跑 `scripts/check_staged_files.sh`，不使用 `git add .`），並追加 `PROGRESS.md`。
2. 稽核方在 commit 之後獨立複驗：決定性、v1 迴歸、header schema、負控制。
3. 不寫入任何真實金鑰材料：只用 fixture key 或程序內臨時產生的金鑰，不落地。
4. 全部通過後產出 `table_sharing_v2`，並在本檔補上「§8 結果」。

### 7.8 問題清單

| 問題 | 選項 | 建議 |
|---|---|---|
| Q1 grant_key 的來源 | (a) HKDF 推導 (b) 隨機產生並登記 | (a) |
| Q2 院內發放 patient key | 允許／不允許 | 允許，僅限院內 |
| Q3 L3 閘道 | 做／不做 | 做 O2 |
| Q4 O1、O3 | 做／不做 | 先不做 |
| Q5 M2 payload 的保護方式 | (A) 純 AEAD (B) 匿名 PNG ＋ AEAD 原圖 (C) 現行 DPHD payload | 見 §6.6；影響論文定位，由使用者決定 |

### 7.9 使用者裁決紀錄

| 日期 | 問題 | 裁決 |
|---|---|---|
| 11-03 | D4 | ✅ 已有影像不重新加密；發鑰後新影像才進入新 epoch |
| 11-03 | D5 | ✅ 取件時轉加密，雲端不存副本 |
| 11-03 | Q1 | ✅ (a) HKDF 推導 |
| 11-03 | Q2 | ✅ 允許院內發放 patient key；跨院走 L2 |
| 11-03 | Q3 | ✅ 做 O2 |
| 11-03 | Q4 | ⏸ 待定 |
| 11-03 | 範圍 | ✅ 洩漏者追查移到 future work（依 opinion §1） |
| 11-03 | 治理 | ✅ commit 維持「每個子任務通過即 commit」，稽核方在 commit 之後複驗 |
| 11-03 | 文件 | ✅ 根目錄 `cross_institution_sharing.md` 與本檔**不合併** |
| 11-03 | Q5 | ⏸ 待定（§6.6） |

**狀態**：已依 `reports/opinion.md` 修訂 §6、§7。Q4、Q5 待定，**尚未實作**。（2026-11-04：已被 §8 取代）

## 8. 定案方案：SOTERIA 接收方綁定共享（2026-11-04；Q5 = (D)，Q6–Q8 已裁決；⏸ 尚未實作）

> 本章**取代** §6、§7。§6–§7 保留為設計演進紀錄，其中的金鑰樹、epoch、grant、L2、輪替、L3 等設計**不再採用**；Q1–Q3 的裁決隨之失效。
> 審閱意見 `reports/opinion.md` 的原則（浮點容差、來源簽章、header schema、v1/v2 證據分離、不過度宣稱）**全部沿用**。
> **2026-11-04 第二版**：依 `reports/opinion.md` §8（N1–N10）修訂：
> - 拆開 V2 的三種往返。
> - 改用 private-range PNG。
> - 補上 sender 綁定，固定唯一一種外層封裝。
> - 註冊表加上時效與防回退。
> - 縮窄 v2 的主張範圍。
> - 補上 E2 的步驟與配對協定、E1 的逐樣本配對。
> - 限縮「不落地」的意義，並讓安全測試分層到達要驗證的那一層。
> - GPU 規則恢復為 30 分鐘門檻。

### 8.1 前提與目標

**前提（使用者 11-03／11-04 確認）**：
- A 院保有原圖 x0，或另有一份可以完美還原的備份（例如以標準 AEAD 加密的備份，不在本章範圍）。
- **SOTERIA 的用途是共享**，不是 A 院自己存檔。

**目標**：
- A 院為**指定的接收方 B** 產生一張 B 專屬的匿名影像 anon_B，經雲端或直接傳給 B。
- 在正常協定下，**只有 B 院解密服務**能解封還原金鑰並還原影像。
- 應用程式**不以明文形式持久保存** k_img_B（§8.5）。

**研究輸入的範圍（N7）**：
- x0 是研究用的輸入：原圖經 equalize → INTER_AREA 縮到 256×256 → JPEG q=100 往返 → normalization（`scripts/e1_ddim_runner.py` 的 `preprocess`），**不是原始 DICOM bytes**。
- 還原影像是**研究原型的輸出**，不宣稱能直接取代臨床原始影像。完整 DICOM metadata、患者對應、診斷等效性與 PACS 互通，都屬於部署階段的驗證。

**與標準加密的關係**：HPKE 與接收方綁定**不是 DPHD 獨有的貢獻**。本方案是在既有的可逆影像表示上，加入標準的交付控制；**不宣稱比直接用 AEAD 共享更安全或更適合臨床**。

### 8.2 威脅模型與主張範圍

**比較基準是現行流程**：病患向原醫院申請光碟（通常是未加密的 DICOM），自己帶到新醫院。

| 威脅 | 處理 |
|---|---|
| 傳輸或儲存途中被第三方（C、雲端、遺失的媒體）取得 anon_B | ✅ 範圍內：在正常協定下，沒有 B 的私鑰就解不開 wrap_B。**這不是「所有只拿到影像的攻擊都無法還原」的證明** |
| 竄改 anon_B 或容器、冒充 A 院發送 | ✅ 範圍內：A 院 Ed25519 簽章，加上 MAC 與 AEAD |
| 重放舊版的金鑰註冊表，讓 A 繼續用已撤銷的金鑰發送 | ✅ 範圍內（有限度）：§8.6 的版本與時效檢查 |
| k_img_B 經由日誌、檔案、例外訊息或 API 意外外洩 | ✅ 範圍內：§8.5（只保證受監控的應用程式路徑） |
| 持有 checkpoint 的攻擊者反演匿名圖（T2-WB）、只拿到影像的攻擊者（T2-BB） | ⚠ **v1 的結果只是 v1 的證據**。v2 本輪只主張 E1（合法接收方的保真度）與 E2（指定 scorer 下的跨接收方連結）的結果；**其他 WB／BB 攻擊沒有在 v2 複驗**，不因為 E1／E2 通過就視為已驗證（N5） |
| A 或 B 院的內部人員**蓄意**外流 k_img_B 或還原後的影像 | ❌ 範圍外：與現行光碟流程相同 |
| B 的長期私鑰被竊、B 的解密服務被攻陷 | ❌ 範圍外。**另外要明列一項限制**：B 的私鑰日後如果外洩，**過去送出的 wrap_B 仍然解得開**。本模式不具接收方被攻陷後的前向保密（RFC 9180 §9.7） |
| 授權到期後停止離線還原 | ❌ 不支援：已交付的物件不會失效（N4） |
| 同一個容器重複還原、重複匯入 PACS | ❌ 不處理：原型允許重複還原，沒有去重或防重放功能；HPKE 本身也不處理應用層重放（N10） |
| 流量分析、一般性的 IND-CPA | ❌ 不主張（`linearity_security.md` §2.1）。定位維持 authenticated reversible obfuscation |

### 8.3 流程

```
【A 院加密服務】每張影像、每個接收方各做一次；整段包在 try/finally 中
 1. 從簽署的金鑰註冊表（§8.6）取得 B 的記錄：inst_id_B、key_id_B、X25519 公鑰 pk_B、用途、狀態
    → 必須通過根金鑰簽章、版本、時效、狀態檢查，否則 fail-closed，不發送
 2. k_img_B ← CSPRNG 32 bytes；nonce_B ← CSPRNG 16 bytes
    （研究實驗改用標示為測試用途的 fixture stream，見 §8.8 的「金鑰策略」）
 3. info   = LP(profile_id, nonce_B, inst_id_B, key_id_B, inst_id_A, signing_key_id_A)      ← profile_id 固定放第一個欄位（§8.4 規則）
    wrap_B = hpke.Suite(X25519, HKDF_SHA256, CHACHA20_POLY1305).encrypt(k_img_B, pk_B, info)
 4. 子金鑰 = HKDF(k_img_B, salt=nonce_B, info=LP(profile_id) ‖ LP(label))      ← 精確格式見 §8.4.1
           label ∈ {round/0/sign-pre, round/0/perm, round/0/sign-post, mac, private-metadata-aead}
 5. x0 → 反演 → z → fixed-norm（C = 248.261，記下原始 norm）→ DPHD → 生成 → 匿名 float 影像
    → private range：記下 lo/hi，量化成 8-bit uint8 → PNG 無損編碼（只含 IHDR/IDAT/IEND，不寫任何 lo/hi chunk）
    ＊用 encode_private_range_png()，不可用 encode_range_preserving_png()（後者會把 lo/hi 寫進公開的 text chunk）
 6. meta_pt  = 固定格式（原始 norm 的 float64 bits ‖ lo 的 float32 bits ‖ hi 的 float32 bits）
    meta_ct  = ChaCha20-Poly1305(meta_key, nonce = 12 個 0 bytes, aad = H, meta_pt)
               ← meta_key 由這張影像隨機的 k_img_B 推導，只用一次，因此固定 nonce 是安全的；不支援更新 metadata
    H        = LP(header 欄位，不含 tag 與 sig；meta_ct 放在 H 之後的獨立欄位)
 7. tag = HMAC-SHA256(mac_key, LP("soteria-share/v2/mac", H, meta_ct, PNG))
    sig = Ed25519_A(LP("soteria-share/v2/sig", H, meta_ct, PNG, tag))
    容器 = preamble ‖ LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag ‖ sig      ← tag、sig 都不涵蓋自己；preamble 與 MAC／簽章的精確範圍見 §8.4.1
 8. finally：把 k_img_B 與所有子金鑰的 bytearray 歸零

【傳送】經雲端（TLS）或直接交給 B。傳輸層與 PNG／DICOM 嵌入的 adapter 屬於落地部署（§8.4）。

【B 院解密服務】整段包在 try/finally 中；第 1–5 步沒有全部通過之前，不解碼 PNG、不載入模型、不使用 GPU
 1. 有大小上限的 framing 與 schema 解析（§8.4）；超過上限或格式錯誤就拒絕
 2. 依 signing_key_id_A，從**釘選信任根的註冊表**取得 A 的 Ed25519 公鑰 → 驗證 sig
    ＊不信任容器自己附帶的任何公鑰
 3. 確認 inst_id_B、key_id_B 是自己的
 4. 取得完整的 manifest → 驗證 digest 與 A 的簽章 → 解析成允許的 profile；不使用任何本機預設值
 5. k_img_B = Suite.decrypt(wrap_B, sk_B, info) → 推導子金鑰 → 驗證 tag
 6. meta_pt = AEAD 解密 → 取出 norm、lo、hi
 7. 解碼 PNG 得到 q → 匿名 float 影像 = q/255·(hi−lo)+lo → 反演 → DPHD⁻¹ → 乘上 norm/C 恢復 → 生成 → 還原影像
 8. 只輸出還原影像（研究原型的輸出）；finally：歸零 k_img_B 與子金鑰
```

`LP(...)` 表示每個欄位都加長度前綴的無歧義編碼，禁止用字串拼接。

> 每個參數的意義、大小、是否公開，以及為什麼需要，整理在 `reports/soteria_share_parameters.md`。

**API 依據**：本機 `cryptography` 50.0.2 的 `Suite.encrypt(plaintext, public_key, info=None)` 與 `Suite.decrypt(ciphertext, private_key, info=None)`，與官方文件一致（opinion §8.1 已核對）。
- 沒有 AAD 參數，所以用 `info` 綁定封裝前就確定的情境欄位；其餘欄位由 MAC 與簽章涵蓋。
- HPKE base mode 不驗證發件者，所以必須有 A 的簽章。
- A 不需要 master key。

### 8.4 容器、profile 與 manifest

**唯一的外層二進位容器**：`LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag(32 B) ‖ sig(64 B)`。
- 每個欄位都有大小上限，例如 H ≤ 4 KiB、PNG ≤ 1 MiB。解析器先檢查長度，再配置記憶體（N3）。
- 原型**不允許**「把含簽章的 header 塞進 PNG，再對整個 PNG 簽章」。這種做法會自我引用；而且 PACS 改寫 chunk 或壓縮方式時，即使像素相同，bytes 也會改變，簽章就失效。
- 要嵌進 PNG chunk 或 DICOM private tag，列為部署 adapter，原始的已簽署容器保持不變。

**公開欄位 H 的 allowlist**：

| 欄位 | 說明 |
|---|---|
| `version`、`profile_id`、`manifest_digest` | **允許跨影像重複**（屬於設定資訊） |
| `inst_id_A`、`signing_key_id_A` | **允許重複**（屬於發送院所資訊，N3、N8） |
| `inst_id_B`、`key_id_B` | **允許重複**（屬於接收路由資訊） |
| `nonce_B`、`wrap_B` | 每次都新產生 |

**禁止出現的欄位**：任何由患者衍生、固定不變的 ID（病歷號、patient 節點、檔名中的識別碼），以及明文的 norm、lo、hi。PNG 只允許 IHDR、IDAT、IEND，**拒絕所有附屬 chunk**（tEXt、iTXt、zTXt 等）。

**profile_id 的規則（11-04 新增）**：
- profile_id 必須是帶命名空間、全域唯一的字串，格式為 `soteria-share/<profile 名稱>/<版本>`，**不得使用不透明的編號**。
- 它同時承擔協定的網域分離（`soteria-share`）與版本資訊；info 的欄位結構由 profile_id 決定，而且 profile_id 一律放在 info 的第一個欄位。
- 因此 info 不再另外放 `"soteria-share"`、`"v2"`。
- H 的 `version` 欄位**保留**：它是容器 framing 的版本，B 要先靠它讀懂外層結構，才拿得到 profile_id。
- MAC／簽章的網域分離字串（`soteria-share/v2/mac`、`/sig`）也**保留**：它們區分的是用途。

**profile `soteria-share/m1-private-range/v2` 固定以下全部設定（N3）**：

| 類別 | 固定值 |
|---|---|
| HPKE | DHKEM(X25519, HKDF-SHA256)、HKDF-SHA256、ChaCha20-Poly1305 |
| 金鑰推導 | HKDF-SHA256 子金鑰標籤（§8.3 第 4 步）、round index 0 |
| 轉換 | rounds=1、permutation=on、fixed-norm C=248.261 |
| 數值格式 | latent 為 float32、little-endian；norm 以 float64 bits 儲存，lo/hi 以 float32 bits 儲存，**不使用十進位** |
| PNG | 8-bit 灰階、private range |
| MAC 與 metadata AEAD | MAC 為 HMAC-SHA256；metadata AEAD 為 ChaCha20-Poly1305，金鑰只用一次、nonce 固定，AAD = H |
| 編碼 | LP 長度前綴編碼與各欄位大小上限 |

**manifest**：
- 記錄 checkpoint 的 SHA-256、sampler（DPM-Solver++(2M)、100 步、backward=null、精確的 timestep schedule）與生成設定，由 A 簽署。
- 容器只放 manifest 的 digest。B **必須取得完整的 manifest**，驗證 digest 與簽章後，才能啟動推論。

### 8.4.1 位元組層級契約（opinion F4；2026-11-04 於 S2 開始時固定）

本節是 encoder／decoder 的**唯一依據**。和 §8.3、§8.4 有衝突時，以本節為準。所有整數都是 unsigned；「LE」= little-endian、「BE」= big-endian。

**(1) 長度前綴 LP**
- `LP(x) = u32_BE(len(x)) ‖ x`。這個格式本身就是 canonical。
- 解碼時，如果長度超過剩餘的 bytes（截斷或溢位）就拒絕。
- 一個結構解完之後如果還有**多餘的尾端 bytes**，也拒絕。

**(2) 最外層容器**

```
preamble = b"SOTERIA\x00"（8 B）‖ u16_BE(container_version = 2)            固定 10 B，永遠位於檔案開頭
container = preamble ‖ LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag(32 B) ‖ sig(64 B)
```

- 容器版本放在**位置固定的前導區**，不放在 H 裡，所以在還不知道版本時，也能先讀出它。不認得的版本一律拒絕。
- 大小上限：|H| ≤ 4096 B、|meta_ct| = 32 B、|PNG| ≤ 1 048 576 B，tag 與 sig 的長度固定。
- 解析器先讀 preamble 與各個 LP 長度，確認都不超過上限，**之後**才配置記憶體。

**(3) H 的欄位（固定順序、固定數量；`H = LP(f1) ‖ … ‖ LP(f8)`，不允許其他欄位或尾端 bytes）**

| # | 欄位 | 格式 |
|---|---|---|
| f1 | profile_id | ASCII，符合 `^soteria-share/[a-z0-9-]+/v[0-9]+$`，≤ 64 B；本原型只允許 `soteria-share/m1-private-range/v2` |
| f2 | manifest_digest | 32 B 原始 SHA-256 |
| f3 | inst_id_A | ASCII ID，符合 `^[A-Za-z0-9._:-]{1,64}$` |
| f4 | signing_key_id_A | 同上 |
| f5 | inst_id_B | 同上 |
| f6 | key_id_B | 同上 |
| f7 | nonce_B | 16 B |
| f8 | wrap_B | 80 B（HPKE X25519 的 enc 32 B ＋ 密文 32 B ＋ tag 16 B） |

**(4) HPKE info**

`info = LP(profile_id) ‖ LP(nonce_B) ‖ LP(inst_id_B) ‖ LP(key_id_B) ‖ LP(inst_id_A) ‖ LP(signing_key_id_A)`

**(5) 子金鑰**

- 公式：`HKDF-SHA256(ikm = k_img_B, salt = nonce_B, info = LP(profile_id) ‖ LP(label), length = 32)`。由 profile_id 承擔版本，取代 §8.3 原本寫的 `LP(label, "v2")`。
- label 為 ASCII，只允許 `round/0/sign-pre`、`round/0/perm`、`round/0/sign-post`、`mac`、`private-metadata-aead` 這五個值。

**(6) metadata**

- `meta_pt = f64_LE(norm) ‖ f32_LE(lo) ‖ f32_LE(hi)`，共 16 B。
- `meta_ct = ChaCha20-Poly1305(meta_key).encrypt(nonce = 12 個 0x00, data = meta_pt, aad = preamble ‖ LP(H))`，共 32 B。
- 合法性：norm 必須是有限值且 > 0；lo、hi 必須是有限值且 hi ≥ lo；拒絕 NaN 與 Inf。
- hi = lo 表示常數影像：encoder 把 q 全部設為 0，decoder 得到全部等於 lo 的影像（與 `m1_storage.encode_private_range_png` 現有的行為相同）。

**(7) MAC 與簽章**

```
tag = HMAC-SHA256(mac_key, LP("soteria-share/v2/mac") ‖ preamble ‖ LP(H) ‖ LP(meta_ct) ‖ LP(PNG))
sig = Ed25519_A(LP("soteria-share/v2/sig") ‖ preamble ‖ LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag)
```

**(8) 量化與 PNG**

- 量化：沿用 `m1_storage.encode_private_range_png`，全部以 float32 運算：`q = clip(np.rint((x−lo)/(hi−lo)·255), 0, 255)`，捨入方式是 round-half-to-even。
- 反量化：`x = q/255·(hi−lo)+lo`，同樣以 float32 運算。
- **解壓縮之前先檢查 PNG 結構**：
  - 開頭是 PNG 簽名，接著依序是 IHDR、一個以上**連續**的 IDAT、IEND，每個 chunk 的 CRC 都正確，最後沒有多餘 bytes。
  - IHDR 必須是 width = height = 256、bit depth = 8、color type = 0（灰階）、compression = 0、filter = 0、interlace = 0。
  - 解壓縮後的資料必須正好是 256 × (1 + 256) = 65 792 B，超過就中止，避免 decompression bomb。
  - 出現任何其他 chunk 都拒絕。

**(9) latent**

- 形狀 (1, 1, 256, 256)，float32，d = 65 536。
- fixed-norm 常數 C = `np.float64("248.261")`（`s2_fixed_norm.FIXED_NORM`）。

**(10) 固定測試向量**

在 S4 加入，涵蓋：正常編碼、截斷、長度溢位、未知版本、尺寸錯誤、非法 metadata、尾端多餘 bytes。

### 8.5 「不落地」的定義與規則（N9）

**定義**：應用程式**不以明文形式持久保存** k_img_B 或子金鑰。

- wrap_B 本來就會隨容器保存。它是能夠還原金鑰的「封裝形式」，不在禁止之列。
- 內部輔助函式（例如 `keys.py` 的 unwrap）把金鑰回傳給服務流程是必要的。**禁止的是對外公開的服務 API 暴露金鑰**。

| 端點 | 規則 |
|---|---|
| A 加密服務 | k_img_B 與子金鑰只存在 `bytearray`。不寫入 log、檔案、例外訊息、除錯輸出。正常結束、例外、取消等**所有結束路徑**都用 `try/finally` 歸零 |
| B 解密服務 | 規則同 A；對外介面只輸出還原影像 |

**限制**：
- Python 無法保證記憶體真的被抹除，因為 `bytes` 是不可變物件，會被複製。
- 測試只能涵蓋受監控的路徑；native 函式庫內部的副本、swap、core dump、APM 記憶體快照都在範圍外，列為部署限制。
- **HSM 只保護 B 的長期私鑰**。解封出來的 k_img_B 和子金鑰仍然會進到 Python。
- 目前 `Suite.decrypt` 只接受軟體私鑰物件。與真實 HSM 的整合需要另外驗證，**不宣稱「只要換掉金鑰存放位置，就完成 HSM 保護」**。

### 8.6 金鑰註冊表（Q6 = (a)；N4）

**信任根**：根金鑰由設定**釘選**，不能跟著待驗證的 JSON 一起被替換。

**整份註冊表由根金鑰簽署一次**（opinion F1）：簽章涵蓋 canonical 編碼的完整 payload，包括 `registry_version`、`issued_at`、`expires_at` 以及**完整的記錄集合**。每筆記錄的內容為院所 ID、key ID、公鑰、用途（X25519 用於接收、Ed25519 用於簽章，**兩者分開**）、狀態（有效／撤銷）。單筆記錄可以另外簽章，但**不能取代**外層的整體簽章。

**驗證契約**（A 查詢收件金鑰、B 查詢發件人驗簽金鑰，共用同一套）：
1. 先驗證整份註冊表的簽章，以及所有合法性條件：用可信時鐘確認在有效期限內；版本 ≥ 本機最高已見版本；同一版本的 digest 必須和本機記錄的一致。
2. **全部通過之後**，才以原子方式把最高已見版本與它的 digest 持久化（重啟後仍然保留）。驗證失敗時不改動本機狀態，避免未驗證的「超高版本」污染可信狀態。
3. 查詢 key ID：必須已知、狀態有效、用途正確。
4. 任何一項不成立，或查不到可信狀態，就 **fail-closed**，不發送（A）或不接受（B）。

**撤銷的意義**：只代表 A 不再用這個 key ID 發送**新的**物件，不會讓已經交付的物件失效。換新金鑰也無法讓攻擊者手上的舊副本失效（見 §8.2 的前向保密限制）。

這是有限度的防回退機制，**不宣稱已經完成真實的聯盟即時 PKI**。

### 8.7 不做的事（Non-goals）

- 真實的雲端、傳輸層、HSM/TEE 整合、聯盟 PKI、PNG／DICOM 嵌入 adapter。
- 洩漏者追查、鑑識浮水印、阻止明文外流、防重放與去重、授權到期後停止離線還原、接收方被攻陷後的前向保密。
- 在 v2 重跑 WB／BB 攻擊（只縮窄主張，不新增必做實驗）；臨床診斷等效性驗證；自行重寫 HPKE。
- A 院自己的加密存檔；金鑰樹、epoch、grant、L2、L3、輪替（已被本方案取代）。

### 8.8 驗證計畫

**預先登記的判準（跑之前就固定，跑完不調整）**：

| 編號 | 範圍 | 判準 |
|---|---|---|
| P1 | 純轉換往返：fixed-norm → DPHD → inverse → 恢復 norm，**不經過 PNG** | 固定 fixtures：d=65536、float32；輸入為 `t2bb_latent_cache_dpm` 中的真實 latent 前 16 個，加上 seed 1911 的 N(0, I) 16 個。latent max-abs **< 1e-5**。**不宣稱這個絕對界適用於所有輸入幅度** |
| P2 | 容器序列化往返 | bytes 完全一致 |
| P3 | 真實 PNG + 恆等 stub + 完整服務介面 | 與一份**獨立撰寫的參考流程**（量化 → 反量化 → inverse → 恢復 norm）逐元素比對，max-abs < 1e-5。⚠ 2026-11-04 S8 更正：兩套獨立的浮點實作會讓極少數值落在 rint 的 .5 邊界兩側，量化結果**無法保證逐像素相同**（8 個 fixture 中最多 1/65536 個像素差 1 個灰階）。所以 P3 拆成兩半：**解碼端**由參考流程直接使用服務產出的 q，並用獨立推導的金鑰解開 metadata，維持原登記的 max-abs < 1e-5；**編碼端**改檢查 max |Δq| ≤ 1，且不同的像素比例 ≤ 1e-3。後者是**看到實測後才加的門檻**，照實記錄。
  2026-11-04 AF-054 補強：參考流程改成**從 x0 自己算出** norm、lo、hi，並和服務寫入的 metadata 比對；解碼端改用參考流程自己的 metadata，不再採用服務寫入的值。這樣 sender 的 metadata 錯誤（例如 norm ×1.01、lo/hi 偏移）都能抓到，`paper_assets/tables/table_share_p3.json` 中 3 種突變全部偵測到。編碼端門檻在事先沒看過的 32 個 held-out seed 上全部通過（每張最多 4 個像素不同）。這條**事後門檻**若要作為正式的驗收依據，**需要使用者確認**。另外**報告**相對原始輸入的有損誤差（不設門檻）。理論參考值：理想四捨五入下，單一像素的量化誤差界是 (hi−lo)/510；逆正交轉換保持的是 L2 而非 L∞，恢復 norm 又會乘上 norm/C，所以這個像素界不能直接當成 latent 的界。**不可以跳過 PNG 讓測試通過** |
| P4 | E1 保真度 | 用 `paper_assets/rq4_formal_dphd/per_sample_long.csv` 的 S2fn 列，以 `sample_id + patient_id` 一對一配對。對每位患者的 ΔPSNR 做病患重抽樣，求平均差的 95% CI。CI 完全落在 **±0.5 dB** 內，才寫「**平均重構 PSNR 在預設界限內**」；**不延伸**到 SSIM／LPIPS、每一位患者、病灶或安全性 |

**金鑰策略（N7）**：
- 研究實驗中，k_img_B、nonce_B 與接收方金鑰對都由**標示為公開測試用途的 fixture stream** 產生，例如由 `hashlib.sha256(b"soteria-share/test-fixture/v2/...")` 推導，讓 anon_B 可以重現。
- HPKE 內部的臨時金鑰仍然是隨機的，所以 wrap_B 的 bytes 每次執行都不同，但不影響 anon_B 的內容。
- 正式部署才使用 CSPRNG。實驗結果**不宣稱能抵抗知道 fixture 的攻擊者**。E2 的攻擊者本來就不使用任何金鑰。

**確定性測試（CPU，pass/fail；只代表具體條件下的功能與正確性，不是一般性的安全證明；每項測試都要真的打到它要驗證的那一層，N10）**

| # | 測試 | 通過條件 |
|---|---|---|
| V1 | v1 迴歸 | `prng.py` 重構後，v1 的 sign／perm 輸出與**重構前固定的 test vectors** bit-identical；既有的 `tests/security/*` 全部通過 |
| V2 | 往返（N1） | P1、P2、P3 三項分開測試，各自依登記的判準判定 |
| V3 | HPKE 層（直接測函式庫） | 錯誤的私鑰、錯誤的 info、截斷或格式錯誤的 wrap → 全部失敗 |
| V3' | 接收方綁定（整合測試） | 送到 C → C 的服務因 key_id 不符而拒絕；即使把 key_id 改成 C 的並由測試用 signer 重新簽章，HPKE 也無法解封。兩種情況都**不呼叫 inverter／generator，也不輸出任何部分結果**（用 spy 驗證） |
| V4 | 竄改，分層觸發 | (i) 簽章層：簽章範圍內任一 byte 被改 → sig 失敗。(ii) 受控的 fault injection，個別觸發各層，並**斷言觸發的是預期那一層的錯誤類型**，不只是籠統的 raises：改 PNG 後用測試 signer 重新簽章 → MAC 失敗；改 meta_ct 後**用 fixture 的 mac_key 重算 tag**，再用測試 signer 重新簽章 → 通過簽章與 MAC，觸發 AEAD 失敗（opinion F2）；兩個物件的 wrap 對調後重算 tag 與 sig → HPKE info 不符。另外用直接呼叫 metadata opener 的單元測試補強。重算 MAC 或簽章只是受控測試，不代表真實攻擊者擁有這些金鑰。(iii) 整合測試：任何一層失敗時，都不呼叫推論、不輸出部分結果（用 spy 驗證） |
| V5 | 來源 | 未知的 signing_key_id、不在註冊表內的 signer、註冊表狀態無效 → 拒絕。容器內夾帶的公鑰一律被忽略 |
| V6 | 每次都產生新金鑰（N8） | 每次呼叫的 k_img_B、nonce_B 都不同；不同接收方互相隔離；錯誤的私鑰 fail-closed。「不同接收方的 payload 不同」只在非退化的 fixtures 上當 smoke 檢查，**不能據此推論無法連結** |
| V7 | 不落地（受監控路徑的洩漏檢查，N9） | ① 對外公開的服務 API 沒有回傳金鑰的函式。② 在正常流程，以及**故意在 HPKE、PNG、簽章、推論各步驟注入失敗**的情況下執行，檔案、logging、stdout／stderr、例外訊息中都找不到 k_img_B 或子金鑰的 raw、hex、base64 形式。③ finally 執行後，受控的 buffer 已經歸零 |
| V8 | 註冊表（N4） | 撤銷、過期、未知 key ID、簽章錯誤、查不到狀態 → 拒絕發送。**載入新版撤銷記錄後，重放舊版的「有效」註冊表，仍然不能恢復發送** |
| V9 | 公開視角（N2、N8） | 檢查**整個公開容器與 PNG chunks**，不只是 Python dict。欄位符合 allowlist；允許重複的欄位如 §8.4 所列；不含由患者衍生的固定 ID；不含明文 norm、lo、hi；含附屬 chunk 的 PNG 被拒絕 |
| V10 | 成本 | HPKE 與 DPHD 的 CPU 耗時、A 端與 B 端擴散模型的 GPU 耗時**分開報告**；每張容器的 bytes；效能數字不是安全證明 |
| V11 | 資源限制（N3） | 超過大小上限或 framing 格式錯誤的輸入，在配置大型記憶體、解碼 PNG、載入模型之前就被拒絕 |

**實驗（GPU；先 smoke；正式執行前依 §8.10 回報估時並取得裁決）**

**E1：B 的還原保真度（N7）**

- **協定**：沿用 `scripts/rq4_formal_redo.py` 的設定：security_v1、N=200、DPM-Solver++(2M) 100 步、backward null、private-range PNG、病患 cluster bootstrap 10000 次、seed 1911。
- **沿用舊結果前要先核對**：
  - `analysis_manifest.json` 的 split／checkpoint hashes、前處理、精確的 timestep schedule、rounds=1、permutation=on、C=248.261、精度、metrics 與 LPIPS clipping。
  - 保留它的 `git_dirty_provenance.md`。46.40 dB 這個基準值來自當時的 dirty-tree 紀錄，**不描述為乾淨 commit 的全新重跑**。
- **與 v1 的配對**：依 P4，以 `sample_id + patient_id` 一對一配對。
- **要報告的內容**：PSNR、SSIM、LPIPS、ΔPSNR 的 CI；差值的分布尾端與最差案例。
- **不做**：不為了讓結果好看而調參、換 seed 或重新抽樣。
- **產出**：`paper_assets/tables/table_share_fidelity_n200.{csv,json}`。
- **估時（[推論]，11-04 更正）**：
  - **規劃用的估計值**：`paper_assets/quality_speed/comparison.csv` 記錄完整 4-pass 流程（反演 → 加密 → 生成 → 反演 → 解密 → 生成，steps=100）約 **6.3 s／張**。A 端只做前 2 pass，約 3.2 s；B 端做後 2 pass，約 3.2 s。
  - E1 合計約 200 × 6.3 s ≈ **21 分鐘**，加上 metrics 與模型載入，估計 25 分鐘以內。v1 基準直接取既有的逐樣本結果，不重跑。
  - `rq4_formal_dphd/table_runtime.csv` 的正式紀錄平均約 12.6 s（逐樣本：前 181 張約 13.2 s、最後 19 張約 6.2 s）。這是**不同測量情境下的觀測，可能存在負載或其他條件差異，原因待核對**；不能據此推定哪一個才是「正確」的數字（opinion F6）。
  - 啟動前依當時 smoke 實測的完整 pipeline、metrics、I/O 與 GPU 負載重新估時，超過 30 分鐘就先裁決。
  - 論文引用時間時，要標明樣本數、設定、計時區間、硬體與量測條件；**不因數字較慢就排除正式資料**。如果無法建立可比性，就分開列出，或暫不做公平的效能比較。
  - smoke 會實測每張的耗時。如果推算的正式版超過 30 分鐘（例如 GPU 正被共用），**先回報並取得裁決**。

**E2：跨接收方連結（Q7；N6）**

- **資料**：security_v1 中 200 位不同患者，每人一張 x0，每張產生 B、C 各一個版本 → **200 張原圖、400 張匿名輸出**。如果 E1 的 B 版本在 profile、split、公開視角與 provenance 上都完全相同，就直接重用。
- **配對**：B 固定為 gallery、C 固定為 query。正對是同一張 x0 的 B/C，負對是不同患者的 B/C。全配對下共 200 個正對、39800 個負對。隨機基準：Top-1 = 1/200、Top-5 = 5/200。
- **攻擊者能看到的資訊**：只有公開 PNG 的 q/255，以及允許公開的欄位。不得讀取 sample ID、患者 ID、原圖路徑、private metadata、金鑰，也不得使用發送端暫存的 float 影像或接收端反量化後的影像。對照表只有評估器能使用。正對與負對的 B/C 路由資訊相同，避免接收方 ID 和標籤混在一起。
- **scorer**：cos、norm、DenseNet121。preprocessing、weights hash、分數方向、tie 處理、bootstrap 的 seed 與次數全部**預先凍結**。不看測試標籤事後翻轉方向，也不挑最好的 scorer。如果需要校準，另外使用病患不重疊的 dev 集。
- **對照組**：
  - 正控制：用原圖 x0 的 B/C 複本，確認 scorer 與配對流程本身能正常運作。
  - 負控制：用固定種子打散配對。
- **CI（F3 = (a)，使用者 11-04 裁決）**：gallery **固定**為原本 200 人的 B 版本；以 query 病患為單位，有放回地重抽 query 的結果區塊（每位 query 病患的正對分數與 199 個負對分數作為一個區塊），做 2000 次，seed 1911，取 2.5／97.5 百分位。結果一律標為「**以固定 gallery 為條件的 CI**」，**不宣稱**涵蓋 gallery 母體的不確定性。這與既有的 `t2bb_inner_linkage._boot_auc_ci()` 做法一致。**所有 image pairs 不能當成獨立樣本**。
- **結果的解讀**：AUC 與 Top-k 是科學結果，**不以「接近隨機」當成工程上的通過門檻**。CI 含 0.5，只代表在這些指定 scorer 下沒有偵測到差異，不是不可連結的證明。
- **產出**：`paper_assets/tables/table_share_cross_recipient_linkage.{csv,json}`。
- **估時（[推論]，11-04 更正）**：重用 E1 的 B 版本時，只需要在 A 端再生成 C 版本（2 pass），約 200 × 3.2 s ≈ **11 分鐘**。scorer 只處理公開 PNG（cos、norm、DenseNet121 embedding），不需要反演，再加幾分鐘，估計 15 分鐘左右。smoke 實測後，如果推算超過 30 分鐘，**先回報並取得裁決**。

### 8.9 實作步驟（新檔為主，v1 行為不變；每個子任務通過就原子 commit）

| 步驟 | 檔案 | 內容 | 對應驗證 |
|---|---|---|---|
| S1 | `src/crypto/prng.py`（小重構） | 拆出 `sign_vector_from_key()`、`permutation_from_key()`；v1 函式改成呼叫它們。**先固定重構前的 test vectors** | V1 |
| S2 | `src/share/keys.py` | HPKE 封裝與解封、`LP()` 編碼、v2 子金鑰推導、`bytearray` 歸零工具、fixture stream（標示為測試用） | V3、V6、V7 |
| S3 | `src/share/transform.py` | fixed-norm → DPHD（用 k_img_B 推導的 D/P/D）→ inverse → 恢復 norm；內部重用 v1 的 FWHT | P1 |
| S4 | `src/share/container.py` | 外層二進位容器、framing 大小上限、schema allowlist、profile 驗證、private-range PNG 編解碼（重用 `encode_private_range_png`／`decode_private_range_png`）、PNG chunk 白名單、metadata AEAD、MAC、Ed25519 | P2、V4、V9、V11 |
| S5 | `src/share/registry.py` | 釘選的信任根、簽署記錄、版本、時效、最高已見版本、fail-closed | V5、V8 |
| S6 | `src/share/manifest.py` | 簽署 manifest、digest 驗證、解析成允許的 profile | V4、V5 |
| S7 | `src/share/sender.py`、`receiver.py` | A 加密服務、B 解密服務（嚴格依 §8.3 的步驟順序）。反演與生成以注入函式的方式取得：正式流程接 DPM++ pipeline，測試時接 stub | P3、V3'、V4、V7 |
| S8 | `tests/security/test_soteria_share.py` | V1–V11（CPU，使用 stub pipeline 加上獨立撰寫的參考流程） | V1–V11 |
| S9 | `scripts/soteria_share_demo.py` | CPU 示範：A→B、C 攔截、分層竄改、註冊表回退 → `paper_assets/tables/table_share_protocol.{csv,json}`（使用 output_guard） | V10 |
| S10 | `scripts/soteria_share_experiments.py` | GPU runner，分 `--mode e1` 與 `--mode e2`：E1 逐樣本配對、E2 預先登記的配對協定、正負控制、病患重抽 CI；先 smoke；使用 output_guard | E1、E2 |

### 8.10 流程與治理

1. 依 S1 → S10 實作。每個子任務**通過驗證就原子 commit**：commit 前跑 `scripts/check_staged_files.sh`，不用 `git add .`，並追加 `PROGRESS.md`。稽核方在 commit 後獨立複驗。
2. **GPU 規則（使用者 2026-11-04 重申）**：
   - 預估佔用 GPU **超過 30 分鐘**的任務，啟動前必須**先告知使用者並取得裁決**。回報內容包括目的、預估時間、GPU 記憶體、要寫出的 artifact、較小的替代方案。
   - smoke（幾分鐘）可以直接跑。
   - 10-27／10-28 的「預先同意」只適用於當時使用者不在電腦前的那段期間，**已經不再適用**。
   - 依 §8.8 的規劃估計，E1 約 21–25 分鐘、E2 約 15 分鐘。這只是規劃用的估計值；實際依 smoke 實測重新估時，推算超過 30 分鐘就先問。
3. **不寫入任何真實金鑰材料**：只用標示為測試用途的 fixture，或程序內臨時產生的金鑰。
4. **對既有結果的影響**：v1 的路徑與凍結結果都不變。v2 只主張 E1、E2 的結果（§8.2）。

### 8.11 問題與已採選項（皆已裁決，不需要重複詢問）

| 問題 | 已採選項 |
|---|---|
| Q5 部署方案 | (D) M1 接收方綁定 |
| Q6 金鑰註冊表的信任來源 | (a) 由聯盟或主管機關簽署（原型以 JSON + 簽章模擬，§8.6） |
| Q7 E2 跨接收方連結 | 要做（§8.8） |
| Q8 HPKE 官方向量 | (a) 依賴套件本身的測試，本專案做整合往返與負面測試；跨實作互通留到部署階段 |

### 8.12 裁決紀錄

| 日期 | 項目 | 裁決 |
|---|---|---|
| 11-04 | Q5 | ✅ 選 **(D)**：M1、接收方綁定（HPKE 封裝 k_img_B）、k_img_B 不以明文持久保存 |
| 11-04 | 前提 | ✅ A 院保有原圖，或有可以完美還原的備份；SOTERIA 用於共享 |
| 11-04 | 威脅範圍 | ✅ 內部人員蓄意外流與私鑰被竊列為範圍外；以現行光碟流程作為比較基準 |
| 11-04 | 規則 B | ✅ anon_B 的傳送使用一般的安全網路方法（TLS），屬於落地部署 |
| 11-04 | §6–§7 | ✅ 被本章取代；Q1–Q3 失效 |
| — | Q4（舊） | 已經不適用（O1、O3 屬於舊方案）；新方案對應的是 E1、E2 |
| 11-04 | Q6 | ✅ **(a)** 由聯盟或主管機關簽署的金鑰註冊表（原型以 JSON + 簽章模擬） |
| 11-04 | Q7 | ✅ **要做** E2 跨接收方連結實驗（N=200） |
| 11-04 | Q8 | ✅ **(a)** HPKE 正確性由 `cryptography` 套件本身的測試涵蓋，本專案只做往返與負面測試。與 opinion §4.2「以官方向量驗證」的要求不同，**由使用者明確接受**。理由：套件 API 無法注入臨時金鑰，也沒有 AAD 參數，因此無法直接重跑 RFC 9180 向量。落地前可以補 (c) 跨實作互通測試（例如 `pyhpke`，裝在獨立的測試環境） |
| 11-04 | GPU 規則 | ✅ 恢復「預估超過 30 分鐘先告知並取得裁決」；10-27／10-28 的預先同意只是暫時性的 |
| 11-04 | 規劃修訂 | ✅ 依 `reports/opinion.md` §8（N1–N10）修補本章 |
| 11-04 | opinion §9 | F1（整份註冊表簽章與持久化防回退，§8.6）、F2（V4 真正觸發 AEAD 層，§8.8）、F5（參數文件主張改保守）、F6（runtime 改為「不同情境的觀測」，§8.8）✅ 已修入。**F4**（byte-level 編碼契約與輸入合法性）於 S2／S4 開始時固定在本章並附 test vectors。**F3**（E2 雙側 bootstrap 的重複患者處理）⏸ 待決定，必須在 S10／E2 正式執行前定案 |
| 11-04 | AUD-20261104-01 | AF-052（`b728dc2`）、AF-053（`7ae6e33`）、AF-055（`311f384`）、AF-054（`41a7514`）已修，等稽核方複驗 |
| 11-04 | P3 編碼端門檻 | ✅ 使用者同意**先採用**事後門檻：max|Δq| ≤ 1，且不同的像素比例 ≤ 1e-3。依據：8 個訂門檻用的 seed 加 32 個 held-out seed 全部通過（`table_share_p3.json`）。之後引用時要註明這是事後訂的門檻 |
| 11-04 | F3 | ✅ **(a)**：gallery 固定，只重抽 query 區塊，結果標為以固定 gallery 為條件的 CI（§8.8 E2） |
| 11-04 | 最終比較 | ✅ 使用者要求最後比較「現版本（v2 接收方綁定）」與「原始 master key DPHD（v1 S2fn）」的差異，由 E1 逐樣本配對 ΔPSNR 回答（基準 46.40 dB [46.16, 46.62]） |
| 11-04 | 進度 | ⏸ 使用者指示：先請稽核方查核，**暫停後續實作（S10）** |

**狀態（11-04）**：S1–S10 完成；AF-052～055 已由稽核方複驗 CLOSED（AUD-20261104-02）；E1、E2 正式結果見 §8.13，待稽核方複驗。

---

### 8.13 S10 結果（2026-11-04；E1、E2 正式版；runner commit `e250142`）

**E1：B 的還原保真度（v2 接收方綁定 vs 原始 master key DPHD v1 S2fn）**
`paper_assets/tables/table_share_fidelity_n200.{csv,json}`、`..._per_sample.csv`；security_v1 N=200，DPM-Solver++(2M) 100 步、backward null、private-range PNG。

| 指標 | v2（本方案） | v1 S2fn（凍結基準） | 逐樣本配對差（v2 − v1） |
|---|---|---|---|
| PSNR（dB） | 46.389 [46.149, 46.614] | 46.399 [46.160, 46.621] | **−0.010 [−0.246, +0.216]** |
| SSIM | 0.99471 [0.99438, 0.99503] | 0.99473 [0.99440, 0.99505] | −0.00001 [−0.00013, +0.00010] |
| LPIPS | 0.0193 [0.0178, 0.0208] | 0.0195 [0.0180, 0.0210] | −0.0002 [−0.0006, +0.0002] |
| 匿名圖 vs x0 PSNR | 5.25 dB | 5.27 dB | — |

- [事實] 預先登記的判準成立：ΔPSNR 的 95% 病患 bootstrap CI 完全落在 ±0.5 dB 之內。可以寫「**平均重構 PSNR 在預設界限內**」，不延伸到 SSIM／LPIPS、每一位患者、病灶或安全性。
- [事實] **逐樣本波動很大**：ΔPSNR 的標準差 1.69 dB，35 張差距超過 1 dB、15 張超過 3 dB，最大 −9.5／+7.1 dB；v1 與 v2 的逐樣本相關只有 0.49。但兩版的分布幾乎相同（最低 37.5／37.1、p5 43.4／43.7、中位數 46.75／46.73 dB；低於 42 dB 的 v1 有 10 張、v2 有 6 張），最差的 5 張也大多不是同一批病患。
- [推論] 單張影像的還原品質會隨「這次抽到的隨機金鑰（即匿名圖）」改變，這是 DPHD 本身的性質，v1 也有，**不是 v2 造成的退步**。論文報告逐張品質時應註明：同一張影像換一把金鑰，PSNR 可能相差數 dB。
- [事實] 每張耗時：A 端 3.11 s、B 端 3.12 s（RTX 5090，本次量測條件）；真實匿名圖的容器平均 56.0 KB（53.7–57.8 KB）。
- [事實] runtime 綁定檢查通過：checkpoint SHA-256、steps、101 個 timestep 都和已簽署的 manifest 一致。

**E2：跨接收方連結（同一張 x0 的 B 版本與 C 版本）**
`paper_assets/tables/table_share_cross_recipient_linkage.{csv,json}`；200 張原圖、400 張匿名輸出；gallery 是 B、query 是 C；攻擊者只看得到公開 PNG（q/255）；CI 以固定 gallery 為條件（F3 = (a)）。

| 條件 | scorer | AUC [95% CI] | Top-1（隨機 0.005） |
|---|---|---|---|
| v2 公開 PNG | cos | 0.504 [0.473, 0.539] | 0.010 |
| v2 公開 PNG | norm | 0.504 [0.469, 0.543] | 0.005 |
| v2 公開 PNG | DenseNet121 | 0.476 [0.439, 0.508] | 0.000 |
| 正控制：原圖 x0 | cos／norm／DenseNet121 | 1.000 | 1.000 |

- [事實] 三個指定 scorer 的 CI 都含 0.5，Top-1 都在隨機基準附近：**在這些 scorer 下，未偵測到同一張影像的 B、C 版本可以被連起來**。這不是不可連結的證明，也不涵蓋學習式 re-id 或 WB 反演攻擊（§8.2）。
- [限制] 正控制是兩份完全相同的 x0，只能證明「配對與評估流程本身正常」，**無法證明 scorer 對較弱訊號的偵測力**。較強的正控制（例如 v1 的 S0／S1 共用金鑰版本）不在本次預先登記的範圍內。
- DenseNet121 的 AUC 0.476 略低於 0.5，CI 上界 0.508 仍含 0.5。依預先登記，**不事後翻轉分數方向**。

**一句話**：在同樣的 x0、模型與 sampler 下，接收方綁定的 v2 與原始 master key DPHD 的平均還原品質沒有可偵測的差異（ΔPSNR −0.01 dB，CI ±0.25 dB 內）；同一張圖分給不同院所的兩個版本，用這三種 scorer 連不起來。


### 參考文獻

- M. Blaze, G. Bleumer, M. Strauss. Divertible protocols and atomic proxy cryptography. EUROCRYPT 1998.
- G. Ateniese, K. Fu, M. Green, S. Hohenberger. Improved proxy re-encryption schemes with applications to secure distributed storage. NDSS 2005; ACM TISSEC 9(1), 2006. <https://eprint.iacr.org/2005/028>
- J. Bethencourt, A. Sahai, B. Waters. Ciphertext-policy attribute-based encryption. IEEE S&P 2007.
- M. Li, S. Yu, Y. Zheng, K. Ren, W. Lou. Scalable and secure sharing of personal health records in cloud computing using attribute-based encryption. IEEE TPDS 24(1):131–143, 2013. doi:10.1109/TPDS.2012.97
- S. Yu, C. Wang, K. Ren, W. Lou. Achieving secure, scalable, and fine-grained data access control in cloud computing. IEEE INFOCOM 2010. doi:10.1109/INFCOM.2010.5462174
- A. Everspaugh, K. Paterson, T. Ristenpart, S. Scott. Key rotation for authenticated encryption. CRYPTO 2017. <https://eprint.iacr.org/2017/527>（作者有勘誤，引用時請用修訂版）
- A. Azaria, A. Ekblaw, T. Vieira, A. Lippman. MedRec: Using blockchain for medical data access and permission management. OBD 2016. doi:10.1109/OBD.2016.11
- G. Kaissis, M. Makowski, D. Rückert, R. Braren. Secure, privacy-preserving and federated machine learning in medical imaging. *Nature Machine Intelligence* 2:305–311, 2020. doi:10.1038/s42256-020-0186-1
- R. Barnes, K. Bhargavan, B. Lipp, C. Wood. Hybrid Public Key Encryption. RFC 9180, 2022. <https://www.rfc-editor.org/rfc/rfc9180>
- S. Akl, P. Taylor. Cryptographic solution to a problem of access control in a hierarchy. ACM TOCS 1(3), 1983.
- B. Chor, A. Fiat, M. Naor. Tracing traitors. CRYPTO 1994.
- B. Campbell et al. OAuth 2.0 Mutual-TLS Client Authentication and Certificate-Bound Access Tokens. RFC 8705, 2020.
- D. Fett et al. OAuth 2.0 Demonstrating Proof of Possession (DPoP). RFC 9449, 2023.
- Personal Health Train / FAIR data stations. <https://www.hl7.org/fhir/uv/fhir-for-fair/STU1/PHT.html>
