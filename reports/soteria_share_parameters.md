# SOTERIA 接收方綁定共享：§8.3 流程參數說明

> 2026-11-04。對應 `reports/cross_institution_sharing.md` §8.3（定案第二版）。逐一說明流程中每個參數**是什麼、多大、誰產生、是否公開、為什麼需要**，以及**少了它或錯了會怎樣**。
> 標記：`[設計]` 為 §8 的規劃（尚未實作）；`[事實]` 為已核對的程式碼或實測。
> 11-04 修訂：依 opinion F5 把安全與誤差的說法改得更保守（§0、§5、§7、§10、§11）；info 刪除重複的 `"soteria-share"`、`"v2"`，改由 profile_id 承擔網域分離與版本（§3、§8）。

---

## 0. 總覽：一張匿名影像容器裡有什麼

```
容器 = LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag ‖ sig

H（公開 header）  : version, profile_id, manifest_digest,
                    inst_id_A, signing_key_id_A, inst_id_B, key_id_B,
                    nonce_B, wrap_B
meta_ct（加密）   : 原始 norm、lo、hi
PNG（公開）       : 匿名影像（8-bit，private range）
tag（公開）       : HMAC-SHA256
sig（公開）       : A 院 Ed25519 簽章
```

| 類別 | 參數 |
|---|---|
| **祕密，絕不公開** | k_img_B、所有子金鑰（D_pre、P、D_post、mac_key、meta_key）、B 的私鑰 sk_B、A 的簽章私鑰、原圖 x0、latent z、原始 norm、lo/hi（明文） |
| **公開；正常協定下解封需要指定接收方的私鑰**（不宣稱所有影像推論攻擊都會失效） | H 的所有欄位（包含 wrap_B）、meta_ct、PNG、tag、sig |
| **公開的設定** | profile 內容、manifest（模型與 sampler）、FIXED_NORM C、註冊表 |

---

## 1. 身分與金鑰註冊表（A 院加密服務第 1 步、B 院解密服務第 2 步）

| 參數 | 是什麼 | 為什麼需要 | 少了或錯了會怎樣 |
|---|---|---|---|
| **註冊表根公鑰** | 聯盟或主管機關的簽章公鑰，**寫死在設定檔裡**（釘選） | 註冊表裡每筆記錄的可信度都來自它，是整個信任鏈的起點 | 如果允許跟著註冊表 JSON 一起更換，攻擊者可以塞入自己的根金鑰，偽造整份註冊表 |
| **inst_id_B** | B 院的院所 ID | 指定「這份容器要給誰」，也會綁進 info | 缺少的話無法路由；B 院也無法確認這份是給自己的 |
| **key_id_B** | B 院**接收用金鑰**的 ID | 一個院所可能先後換過幾把金鑰，靠它選對公鑰與私鑰；撤銷時也是以它為單位 | 選錯金鑰就解不開；如果沒有 ID，就無法分辨新舊金鑰 |
| **pk_B** | B 院的 X25519 公鑰（32 B） | HPKE 用它把 k_img_B 加密給 B | 用了假的 pk_B，等於把金鑰交給攻擊者，所以 pk_B 一定要來自已驗證的註冊表 |
| **用途欄位** | 標示這把金鑰是「接收用（X25519）」還是「簽章用（Ed25519）」 | **一把金鑰只做一件事**，避免跨用途的攻擊 | 混用會讓一種用途的弱點牽連到另一種用途 |
| **狀態（有效／撤銷）** | 金鑰目前是否可以使用 | B 的私鑰可能外洩時，A 必須停止用它發送 | 沒有狀態欄位，就無法撤銷 |
| **registry_version** | 註冊表的版本號 | A 記住看過的最高版本，拒絕更舊的版本 | 缺少的話，攻擊者可以重放「撤銷之前」的舊註冊表，讓 A 繼續用已外洩的金鑰（回退攻擊） |
| **issued_at／expires_at** | 註冊表的發行時間與到期時間 | 讓過期的註冊表自動失效，迫使 A 定期取得最新狀態 | 一份舊的「有效」註冊表可以被無限期使用 |
| **inst_id_A、signing_key_id_A** | A 院的院所 ID 與**簽章用金鑰**的 ID | B 院靠它到註冊表找到 A 的 Ed25519 公鑰來驗證簽章；同時綁進 info | 缺少的話，B 不知道該用哪把公鑰驗證，也無法重建 info（N3）。**容器裡不能夾帶公鑰本身**，否則攻擊者可以附上自己的公鑰 |

---

## 2. 每一次共享新產生的值（第 2 步）

| 參數 | 大小 | 誰產生 | 是什麼、為什麼需要 |
|---|---|---|---|
| **k_img_B** | 32 B | A，用 CSPRNG（研究實驗用標示為測試用途的 fixture） | **這張影像給 B 的那一份**的唯一主金鑰。所有子金鑰都從它推導。每張影像、每個接收方都不同，用完即丟，A 事後也無法重新推導。這就是「接收方綁定」與「A 不需要 master key」的來源 |
| **nonce_B** | 16 B | A，用 CSPRNG | ① 推導子金鑰時當作 HKDF 的 salt；② 當作這個物件的唯一識別碼；③ 綁進 info |

**既然 k_img_B 已經是隨機的，為什麼還需要 nonce_B？**

理論上，k_img_B 每次都是新的隨機值，子金鑰已經不會重複，所以 nonce_B 不是必要條件。保留它是為了**多一層保險**：
- 如果 CSPRNG 出錯，或研究用的 fixture 不小心在兩張影像重複使用了同一把 k_img_B，不同的 nonce_B 仍然能讓子金鑰不同，避免 DPHD 在兩張影像上使用同一個 Q。同一個 Q 被用兩次，正是線性轉換最怕的金鑰重用。
- 它也延續 v1「每張影像一個 nonce」的設計，讓兩版容器的結構一致。

---

## 3. HPKE 封裝（第 3 步）

| 參數 | 是什麼 | 為什麼需要 |
|---|---|---|
| **Suite = (X25519, HKDF_SHA256, CHACHA20_POLY1305)** | HPKE 的演算法組合：KEM 用 X25519 做金鑰協商、KDF 用 HKDF-SHA256、AEAD 用 ChaCha20-Poly1305 | 這是 RFC 9180 的標準組合，`cryptography` 50.0.2 直接支援。固定寫在 profile 中，防止降級到較弱的組合 |
| **info** | 「情境資訊」：用 LP 編碼的 `(profile_id, nonce_B, inst_id_B, key_id_B, inst_id_A, signing_key_id_A)` | HPKE 會把 info 混進金鑰推導。B 解封時必須提供**完全相同**的 info 才解得開，所以 wrap_B 被**綁定**在這個情境上：不能把它搬到別的容器、別的 nonce、別的接收方或別的發送方。函式庫沒有 AAD 參數，因此用 info 承擔這個綁定 |
| **profile_id**（info 的第一個欄位） | `soteria-share/m1-private-range/v2` | 一個欄位同時負責三件事：① **網域分離**：前綴 `soteria-share` 確保這個 wrap 只在本協定有效，不會和其他也用 HPKE 的系統混淆；② **版本**：決定 info 的欄位結構；③ **演算法組合**：見 §8。原本另外放的 `"soteria-share"`、`"v2"` 與 profile_id 重複，11-04 已刪除。前提是遵守 §8 的命名規則：profile_id **不得**改成不透明的編號，否則網域分離就會消失 |
| **wrap_B** | 80 B = 臨時公鑰 enc（32 B）＋加密後的 k_img_B（32 B）＋AEAD tag（16 B）`[事實：11-03 實測 80 B]` | 這就是「只有 B 打得開的 k_img_B」。它會隨容器公開保存，但沒有 sk_B 就解不開。**HPKE 內部的臨時金鑰每次都是隨機的**，所以同樣的輸入，每次產生的 wrap_B 都不同 |

**HPKE 內部的臨時金鑰對**由函式庫產生並用完即丟，不會出現在我們的程式中（也因此無法注入，見 Q8）。它的公鑰部分就是 wrap_B 裡的 enc。

---

## 4. 子金鑰推導（第 4 步）

公式：`子金鑰 = HKDF-SHA256(ikm = k_img_B, salt = nonce_B, info = LP(label, "v2"))`，每把 32 B。

| label | 推導出的子金鑰 | 用途 | 為什麼要獨立一把 |
|---|---|---|---|
| `round/0/sign-pre` | D_pre 的 ChaCha20 金鑰 → 65536 個 ±1 | DPHD 第一層：隨機翻轉正負號 | **打散輸入的大小分布**，消除 |Hz| 這類可用來連結的固定特徵（先前 DH 分析的結論） |
| `round/0/perm` | P 的 ChaCha20 金鑰 → Fisher-Yates 置換 | DPHD：打亂 65536 個座標的位置 | 部署設定 permutation=on。N=200 的消融顯示 P 對學習式 re-id 沒有觀察到顯著效果，但它是部署 profile 的一部分，所以保留 |
| `round/0/sign-post` | D_post 的 ChaCha20 金鑰 → 65536 個 ±1 | DPHD 最後一層：再隨機翻轉正負號 | **防止 H-strip**：沒有它，攻擊者可以直接用 Hadamard 矩陣把 H 剝掉 |
| `mac` | mac_key | 計算 tag（第 7 步） | 不同用途使用不同金鑰，一把金鑰外洩不會牽連其他用途 |
| `private-metadata-aead` | meta_key | 加密 meta_pt（第 6 步） | 同上 |

- **`round/0`**：第 0 輪。profile 固定 rounds=1，所以只有第 0 輪。標籤保留輪次編號，是為了日後支援多輪時格式不必改。
- **為什麼 label 後面加 "v2"**：和 v1 的 `cxr-transform/v1/...` 做網域分離。即使 k_img 的位元相同，v1 和 v2 推導出的子金鑰也不會一樣。
- **為什麼用 HKDF，不直接用 k_img_B**：一把主金鑰要安全地分成多把彼此獨立的金鑰，標準做法就是 HKDF（RFC 5869），這也延續了 v1 的 `kdf.py`。

---

## 5. 影像轉換參數（第 5 步）

| 參數 | 是什麼 | 為什麼需要 |
|---|---|---|
| **x0** | 研究輸入影像：原圖經 equalize → 縮到 256×256 → JPEG q=100 往返 → 正規化 `[事實：e1_ddim_runner.py:130-133]` | 系統的輸入。**不是原始的 DICOM bytes**，所以保真度是相對於這份 x0 計算的 |
| **z** | x0 經 DPM-Solver++(2M)、100 步反演得到的 latent（d = 65536 = 256×256） | DPHD 作用在 latent 上，不直接作用在像素上。這樣生成出來的匿名影像才是一張自然的 CXR |
| **FIXED_NORM C = 248.261** | 公開常數 `[事實：s2_fixed_norm.py:46]` | 正交轉換保持長度：‖Qz‖ = ‖z‖。如果直接加密，**latent 的長度本身就是指紋**，可以用來連結與識別（AF-025）。所以先把每個 z 縮放到同樣的長度 C，再加密 |
| **原始 norm（‖z‖）** | 這張影像 latent 原本的長度，存成 float64 | 還原時要乘回去，否則還原影像的亮度尺度會錯。它是指紋，所以**必須加密**（放在 meta_pt） |
| **u = z·C/‖z‖** | 長度固定後的 latent | 實際送進 DPHD 的向量 |
| **c = D_post·P·(H/√d)·D_pre·u** | 加密後的 latent | 設計上讓它的分布接近高斯雜訊，擴散模型拿它生成匿名影像。匿名影像的外觀、與原圖的關聯性、保留了多少資訊，都**由指定的實驗評估**，不能從 DPHD 或高斯外觀直接推論兩者在統計上獨立 |
| **匿名 float 影像** | c 經 100 步生成得到的影像（浮點數） | 這就是匿名影像本體，但還要量化才能存成 PNG |
| **lo／hi** | 匿名 float 影像的最小值與最大值，存成 float32 | 量化時用 `q = round((x−lo)/(hi−lo)·255)`，把全部 256 個灰階都用上，減少量化誤差。反量化需要原值，而它們也帶有資訊，所以**必須加密**（N2：不可寫進公開的 PNG chunk） |
| **q** | 8-bit（uint8）的匿名影像 | PNG 能無損儲存的格式。**PNG 的「無損」只是對 q 無損，float → uint8 的量化本身是有損的**（N1） |
| **PNG** | q 的 PNG 無損編碼，只包含 IHDR、IDAT、IEND chunk | 一般影像流程都能處理的格式。拒絕 tEXt 等附屬 chunk，避免 lo/hi 或其他資訊被不小心寫進公開欄位 |

---

## 6. 私有 metadata 加密（第 6 步）

| 參數 | 大小 | 為什麼需要 |
|---|---|---|
| **meta_pt** | 16 B = norm（float64，8 B）‖ lo（float32，4 B）‖ hi（float32，4 B） | 還原時必需、但不能公開的三個數值。**用位元格式儲存，不轉成十進位文字**，避免四捨五入改變還原結果 |
| **meta_key** | 32 B | 從 k_img_B 推導，**只用這一次** |
| **AEAD nonce = 12 個 0 bytes** | 12 B | ChaCha20-Poly1305 要求「同一把金鑰下 nonce 不可重複」。meta_key 只加密這一筆資料就丟棄，所以固定的 nonce 是安全的。代價是**不支援更新 metadata**：要更新就要換新的 k_img_B |
| **aad = H** | — | 把整個公開 header 綁進 AEAD 驗證。header 被改，metadata 就解不開；meta_ct 也不能被搬到別的容器 |
| **meta_ct** | 32 B = 密文 16 B ＋ tag 16 B | 加密後的 metadata，可以公開 |

---

## 7. 完整性與來源（第 7 步）

| 參數 | 大小 | 是什麼 | 為什麼需要 |
|---|---|---|---|
| **H** | ≤ 4 KiB | 公開 header 的 LP 編碼，**不含 tag 與 sig** | tag 和 sig 要簽的是「除了自己以外的所有內容」，不能自我引用（N3） |
| **tag** | 32 B | `HMAC-SHA256(mac_key, LP("soteria-share/v2/mac", H, meta_ct, PNG))` | ① **金鑰確認**：B 解封後先驗證 tag，確認自己拿到的 k_img_B 和這份內容是配套的，才開始昂貴的反演；② 延續 v1「先驗證、再開啟」的設計。⚠ **MAC 不能取代來源簽章**：任何人都可以自己選一把 image key、用 B 的公鑰做 HPKE 封裝，再算出一個合法的 MAC，只有可信的 A 院簽章才能阻止這種冒充（opinion F5） |
| **sig** | 64 B | `Ed25519_A(LP("soteria-share/v2/sig", H, meta_ct, PNG, tag))` | **來源證明**：證明這份容器確實是 A 院發出的，而且沒被改過。HPKE base mode 本身不驗證發件者，只靠 HPKE 的話，任何人都能用 pk_B 包一份假影像給 B；MAC 是對稱金鑰，也無法證明是 A 發的 |
| **"soteria-share/v2/mac"、"…/sig"** | — | 網域分離字串 | 確保 MAC 和簽章只在本協定、本用途下有效，防止把其他系統的簽章搬過來用 |

**誠實說明：tag 和 sig 有部分重疊。** sig 已經涵蓋了全部內容，HPKE 也會偵測錯誤的金鑰。tag 的用途是金鑰確認與完整性檢查，**不是來源驗證的備援**。實作後如果評估多餘，可以在之後的版本拿掉。目前保留是為了和 v1 的結構一致。

---

## 8. 封裝與編碼（第 7 步）

| 參數 | 是什麼 | 為什麼需要 |
|---|---|---|
| **LP(...)** | 長度前綴編碼：每個欄位前面先寫它的長度，再寫內容 | **避免歧義**：如果直接把字串接在一起，`"ab"+"c"` 和 `"a"+"bc"` 會產生同樣的 bytes，攻擊者可能利用這點讓兩組不同的欄位得到相同的簽章 |
| **容器格式 `LP(H) ‖ LP(meta_ct) ‖ LP(PNG) ‖ tag ‖ sig`** | 唯一的外層二進位格式 | 簽章是針對確切的 bytes 計算的，所以格式必須唯一而且固定。PNG 嵌入與 DICOM tag 嵌入都屬於部署時的 adapter |
| **大小上限**（例如 H ≤ 4 KiB、PNG ≤ 1 MiB） | 解析前先檢查 | 防止惡意的超大輸入在驗證之前就耗盡記憶體或觸發 GPU 工作（V11） |
| **version** | 容器 framing 的版本（不是 profile） | B 院要先靠它讀懂容器的外層結構，才拿得到 profile_id，所以**不能用 profile_id 取代**；也讓 B 院能拒絕不認得的格式 |
| **profile_id** | `soteria-share/m1-private-range/v2`。**規則**：必須是 `soteria-share/<profile 名稱>/<版本>` 格式、全域唯一、帶命名空間的字串，不得使用不透明的編號；一律放在 info 的第一個欄位 | 用一個 ID 固定**所有**演算法選擇：HPKE 組合、KDF 標籤、rounds、permutation、C、數值格式、PNG 格式、MAC、metadata AEAD 的 nonce 規則。B 只接受允許清單內的 profile，**防止降級攻擊**（例如被誘導去用不含 D_post 的設定） |

---

## 9. 模型與 sampler 的 manifest（B 院第 4 步）

| 參數 | 是什麼 | 為什麼需要 |
|---|---|---|
| **manifest** | 由 A 簽署的設定檔，內容為 checkpoint 的 SHA-256 `[事實：3f61b80a…]`、sampler DPM-Solver++(2M)、100 步、backward=null、精確的 timestep schedule、生成設定 | B 要從匿名影像還原，**必須使用和 A 完全相同的模型與 sampler**，否則反演出來的 latent 不對，還原品質會大幅下降 |
| **manifest_digest** | manifest 的雜湊值，放在 H 裡 | 容器不放整份 manifest，只放雜湊值來綁定。B 必須取得完整的 manifest，驗證雜湊值與 A 的簽章，**不能使用任何本機預設值** |

---

## 10. B 院解密服務

| 參數或步驟 | 是什麼 | 為什麼需要 |
|---|---|---|
| **sk_B** | B 院的 X25519 私鑰 | 唯一能解開 wrap_B 的東西。理想上放在 HSM。但 HSM 只保護 sk_B 本身，解封出來的 k_img_B 仍然會進入 Python（§8.5） |
| **第 1–5 步通過之前，不解碼 PNG、不載入模型** | 驗證的順序 | 驗證很便宜，反演很昂貴。未驗證的輸入不能觸發 GPU 工作（N3） |
| **反量化 `x = q/255·(hi−lo)+lo`** | 把 uint8 還原成 float 匿名影像 | 量化的逆運算，需要解密後的 lo/hi |
| **DPHD⁻¹ = D_preᵀ·(H/√d)ᵀ·Pᵀ·D_postᵀ** | 正交矩陣的逆就是它的轉置 | 從 ĉ 解回 û。理想的逆正交步驟在 **L2 範數**下不會放大誤差；這不涵蓋 L∞、之後乘上 norm/C 的縮放、浮點誤差，以及非線性的生成器 |
| **乘上 norm/C** | 把長度恢復成原始 norm | fixed-norm 的逆運算 |
| **try/finally 歸零** | 所有結束路徑都清除 k_img_B 與子金鑰 | 「不落地」：正常結束、例外、取消時都不留下明文金鑰（§8.5） |

---

## 11. 一張圖看哪個參數保護哪件事

| 要保護的事 | 靠哪些參數 |
|---|---|
| 只有 B 能還原 | k_img_B、wrap_B、pk_B／sk_B、info |
| 不是別人冒充 A | sig、signing_key_id_A、註冊表根公鑰 |
| 內容沒被改 | sig、tag、meta_ct 的 AEAD tag、aad = H |
| 匿名影像不洩漏身分（**設計目標**，由指定攻擊評估，不是已證明的性質） | D_pre、P、D_post、fixed-norm C、加密後的 norm 與 lo/hi、private-range PNG、header 不放患者 ID |
| 不同接收方的版本彼此獨立 | 每次都新產生的 k_img_B 與 nonce_B（是否真的連不起來，由 E2 實測） |
| 已撤銷的金鑰不會再被用來發送 | 註冊表的狀態、registry_version、issued_at／expires_at |
| 能正確還原 | manifest（模型與 sampler）、profile、norm、lo/hi |
| 不會被惡意輸入拖垮 | 大小上限、驗證通過之前不使用 GPU |
