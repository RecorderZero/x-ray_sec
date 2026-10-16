# Q10.3 M1（image-only）與 M2（latent-payload）部署模式對照

> 數字引自 formal N=200 凍結表（`paper_assets/q8_formal/table_fidelity.csv`、`table_storage.csv`）與 Q10.4 AEAD-ref（`paper_assets/tables/table_aead_reference.csv`）、Q10.1 誤差歸因（`table_error_attribution.csv`）。標記：`[事實]` 已量測並附來源；`[限制]` 適用邊界；`[決定]` 部署建議。
> 大目標（Q10）：正確解讀可逆性——**M2 可做到原始 latent bit-exact；M1 影像本質上受 sampler + 8-bit 量化限制，不預設 pixel bit-exact**（見 `reports/metric_semantics.md`）。

---

## 1. 兩種部署模式定義

| 模式 | 儲存內容 | 解密後得到 | 可逆性性質 |
|---|---|---|---|
| **M1 image-only** | 僅匿名影像（PNG）＋ authenticated container header | 影像 → 目標模型 inversion → latent → 逆變換 → 重建 x0 | **非 bit-exact**：經 sampler 反轉/重生成 + 8-bit PNG，受量化與 sampler 限制 |
| **M2 latent-payload** | 加密的 latent payload（ciphertext）＋ container | 直接解密 → **精確 latent** → 可再解碼成影像 | **latent bit-exact**：原始 bytes 100% 還原（AEAD 下限，Q10.4） |

---

## 2. Fidelity 對照（recon vs x0，formal N=200）

| 指標 | M1（S2 部署，保值域私有 v2） | M2 天花板（精確 latent → 影像） | 來源 |
|---|---|---|---|
| 影像 PSNR | **38.82 dB** | **41.87 dB**（所有方法相同） | `table_fidelity`：`m1_image_inversion` / `t1_exact_latent_m2` |
| 影像 SSIM | 0.970 | 0.984 | 同上 |
| 影像 LPIPS | 0.067 | 0.031 | 同上 |
| latent cosine vs z | 0.999794 | 1.0（bit-exact rate 1.0） | 同上 |
| 原始 bytes 還原 | — | **100% byte-exact**（Q10.4） | `table_aead_reference` |

* `[事實]` M2 的精確 latent 影像天花板 41.87 dB 全方法一致；其殘差來自 VAE 解碼 + 8-bit，**與加密無關**。M1 的 38.82 dB 與天花板差約 3.05 dB，來源是 DDIM invert/regen 的 latent 誤差（Q10.1 誤差歸因 stage 2→3）。
* `[限制]` M1 無法 pixel bit-exact：影像一旦經 sampler 反轉與 uint8 PNG 存檔，資訊不可逆損失。前作「cosine 0.9989 / MaxAbsError 0.714」即屬此類影像空間殘差，**不是 structured transform 的性質**（transform-only round-trip 到浮點精度無損，Q10.2）。

---

## 3. Security 對照

| 面向 | M1 image-only | M2 latent-payload |
|---|---|---|
| 攻擊者在 PACS/檔案系統拿到 | 匿名影像 | 加密 latent payload（ciphertext） |
| 對應主威脅情境 | **T2-WB**：需持目標 checkpoint 自行 inversion 得近似 latent | **T1**：ciphertext 洩漏，需 master key 才能解 |
| 加密層安全（norm/linkage/geometry/cross-nonce） | 與 M2 **相同**：皆為 latent 層、與存檔無關；見 defense_package（D7） | 同左 |
| 完整性 / 竄改 | container HMAC：竄改 tag/nonce/ciphertext/version 全拒（`table_tamper`，D7-C8） | 同左；另 AEAD 原語下限：ciphertext/tag/nonce/key/aad 任一竄改 fail-closed（Q10.4，`table_aead_reference`） |
| 額外暴露 | 僅影像（不含精確 latent） | ciphertext 的大小/結構，但以 AEAD 封裝、per-image nonce |

* `[事實]` 加密層再識別/旁道結果（T1 norm、T2-WB linkage、幾何、cross-nonce、repeat）皆為 latent 層與容器層性質，**M1/M2 共用、與影像存檔無關**；AF-025 固定 norm v2 後 T1/T2-WB norm Top-1=0.005。來源 `paper_assets/defense_package/claims.csv`（D7-C1…C10）。
* `[事實]` M2 的 bit-exact 可逆性與竄改拒絕是 AEAD 的保證（Q10.4：6 種 payload 全 100% byte-exact、5 類竄改全拒），與 sampler 無關；這是「若要宣稱原始 bytes bit-exact，只能引用 AEAD-ref」的依據（WORKFLOW §Q10 注意）。

---

## 4. Storage 對照（formal N=200 平均 bytes）

| 方法 | M1 total | M1 vs P | M2 total | M2 額外 metadata（S2） |
|---|---|---|---|---|
| P（明文） | 33,042 | 1.00× | 262,144 | — |
| S0 | 53,706 | 1.63× | 262,144 | — |
| S1 | 54,141 | 1.64× | 262,144 | — |
| **S2** | **74,526**（2.26× P） | 2.26× | **350,565** | private_metadata 65 B＋tag 32 B＋nonce 16 B（其餘為 base64 4/3 膨脹） |

* `[事實]` M2 精確 latent payload＝262,144 B（65,536×4 float32）；S2 M2 達 350,565 B（base64 膨脹 + 容器/AEAD metadata）。M1（影像）明顯小於 M2（latent），S2 M1 約為 M2 的 1/4.7。來源 `table_storage`。
* `[限制]` M2 以約 4.7× 的儲存成本換取精確 latent 可逆；M1 以較小儲存換取「僅影像、sampler 限制」的可逆。

---

## 5. 部署建議

* `[決定]` **M1 image-only**：適用於只需保留去識別影像、可接受 sampler 限制保真度（~38.8 dB）、且要最小儲存的情境；主安全情境為 T2-WB。
* `[決定]` **M2 latent-payload**：適用於需要精確 latent 還原 / 原始 bytes bit-exact（例如後續重算、審計、無損再解碼）的情境；以較大儲存換取 AEAD 保證的精確可逆；主安全情境為 T1。
* `[事實]` 兩模式的**加密層安全相同**；差別在**可逆性層級**（影像 vs 原始 bytes）與**儲存成本**，而非密碼強度。
* `[限制]` 本報告不加入新 sampler 或新資料集（WORKFLOW §Q10 注意）；conditioning（y=0 條件 vs 無條件 null）對 M1 保真度的影響列為 GPU BDIA 附錄的 deferred 項（Q10.1 stage 5），本輪未納入正式數字。
