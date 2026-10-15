# F9.3 Adaptive norm limitation：T1 vs T2-WB 與固定 norm remediation

> 本註記彙整 norm-only adaptive linkage 的正式結果與機制分析，供論文 §5.4 的誠實限縮使用。所有數字來自凍結的 N=200 formal 表格，不手抄；標記依專案慣例。
> 來源：`paper_assets/tables/table_norm_leakage.csv`（D7.3，S2 v1）、`paper_assets/tables/table_AF025_fixed_norm_norm_leakage.csv`（AF-025，S2 v2）、`paper_assets/tables/table_D7_T2WB_inversion.csv`（inversion 保真度）。

## 1. 結果（security_v1 N=200，norm-only scorer，random=0.005）

| 情境 | 存檔 | 校正 | S2 v1（正交，未固定 norm）| S2 v2（固定 norm C=248.261）|
|---|---|---|---|---|
| T1（精確 latent） | exact_latent | — | **Top-1=1.000**（CI [0.981, 1.000]）| **0.005** |
| T2-WB（重新反演） | legacy_png | raw／dev_median | 0.005／0.005 | 0.005 |
| T2-WB（重新反演） | range_preserving | raw | 0.005 | 0.005 |
| T2-WB（重新反演） | range_preserving | dev_median | **0.250**（CI [0.195, 0.314]、Top-5=0.745、median rank 3）| 0.005 |

inversion 保真度（S2 v1，T2-WB positive control cosine=1.0，結果可解釋）：legacy latent cosine **0.990**、norm_ratio **1.0132**；range cosine **0.9998**、norm_ratio **0.9967**（`table_D7_T2WB_inversion.csv`）。

## 2. 機制

- `[事實]` 正交變換必然滿足 ‖Qz‖₂=‖z‖₂，所以 **norm 是任何正交 S2 不可避免的不變量**；只要攻擊者能取得 latent 的 norm，norm-only linkage 就可行。這是構造性質，不是實作 bug。
- `[事實]` **T1（精確 latent）是最強洩漏**：攻擊者拿到精確 norm，Top-1=1.0。這直接來自正交性（§5.4：不得宣稱「只有 norm 一個不變量」，但 norm 確實是已驗證的其中一個通道）。
- `[事實]` **T2-WB 成功率取決於 inversion error 與存檔**：重新反演使 norm 乘上一個近常數比值（legacy≈1.013、range≈0.997）。
  - legacy_png 丟棄原值域，使反演回來的 norm 尺度不穩定、偏差不一致 → 即使用 dev 校正，Top-1 仍為 0.005（≈random）。
  - range_preserving 保住值域，norm 比值穩定 → 以**只在 dev 凍結**的 median bias 校正後，T2-WB Top-1 升到 **0.25**（random 的 50 倍）。
- `[結論，重要]` **legacy 的低 norm 攻擊率不是密碼強度，而是隱私／可逆性的存檔 trade-off**：legacy 同時把 M1 重構品質壓到約 31 dB（AF-024）。不能把存檔造成的尺度破壞當成 S2 的安全性；range_preserving 保住品質，也同時讓 dev-calibrated norm 通道可用（AF-025 的核心張力）。

## 3. Remediation（AF-025 固定 norm S2 v2，已 CLOSED）

- `[決定]`（使用者 AUD-20261014-04）採選項 2：加密前把每個 latent 縮放到**固定公開常數 C=248.261**（float64 bits 凍結，族群平均 norm），真實 norm 與 M1 lo/hi 以獨立 HKDF 子鑰的 ChaCha20-Poly1305 加密為 private metadata，外層 HMAC 先驗證，解密後恢復真實 norm。
- `[事實]` 固定 norm 後，**所有 protected latent 的 norm 皆等於 C**（formal max dev from C=1.49e-5，float32），norm 不再攜帶任何 linkage 資訊：T1 由 1.0→**0.005**、T2-WB（range，dev_median_to_C 校正）由 0.25→**0.005**，皆回到 random。合法還原品質不變（M2 往返 cosine 1.0、M1 private-range legit cosine 0.9998；AUD-20261015-01）。
- `[事實]` 固定 norm 後的 repeat／structured probe 無殘留（`table_AF025_fixed_norm_repeat.csv`、`table_AF025_fixed_norm_probe.csv`，AUD-20261015-01 已獨立複驗）。

## 4. 論文限縮（§5.4）

- `[限制]` norm 是**已驗證的其中一個**結構通道，固定 norm 針對它；不得據此宣稱 S2 「只有 norm 一個不變量」或等同標準加密（§5.1）。其他 basis／constant／sparse probe 僅作結構診斷（D7.6），不作通用安全證明。
- `[限制]` 必須分開報 original／sorted／norm-only 與 T1／T2-WB；T2-WB 的 storage-dependent 行為要與 range-attribution 並列，說明 legacy 低攻擊率來自值域破壞而非密碼強度。
- `[注意]` T2-WB 的 25%（v1）用的是**只在 dev 凍結**的 median bias，formal 未用 formal 資料擬合校正；這是保守做法，避免「看了正式資料才挑校正」。
