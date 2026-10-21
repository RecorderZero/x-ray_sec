# Results index (Z12.2)

> One row per research question / track → key table(s), run id, conclusion, limitation.
> Numbers trace to frozen artifacts (never hand-copied); claim wording + Section-5 limits
> in `paper_assets/claim_matrix/index.md`; full cell list in `paper_assets/final_experiment_matrix.{csv,md}`.
> Reproduction: `paper_assets/reproduction/` (P11.2, bit-exact). Checksums: `paper_assets/SHA256SUMS`.

## Core (RQ1–RQ4, N=200 formal — Q8_FORMAL_20261015…7e4c3368 + packages)

| RQ | 一句話結論 | 關鍵表 | 限制 |
|---|---|---|---|
| **RQ1** 前作攻擊 | S0 逐元素幅值 / S1 sorted-幅值 gallery linkage Top-1=1.0（精確 latent 上界）；KPA 1–2 對還原金鑰；CPA 固定金鑰可重建解密映射；T4 竄改**全被接受**（無完整性） | `attack_package`（C4/C5/C9/C10/C11/C12-14）、`tables/table_{T1_magnitude,KPA,CPA,tamper}` | T2-WB 成功率隨存檔/反演誤差變；前作實際部署=legacy min-max 存檔（range 作參考） |
| **RQ2** S2 防禦 | S2 把 original/sorted 幅值攻擊壓到隨機（Top-1 0.010，random 0.005）；per-image nonce 下跨 nonce 資訊不可轉移（cosine 0.0011、per-image-Q Top-1 0.005）；HMAC 竄改 **7/7 拒絕**（無認證對照 0/1） | `defense_package`（D7-C1/C2/C5/C6/C8）、`tables/table_{structured_probe,cross_nonce,tamper}` | 不宣稱等同 AES/AEAD/一般 IND-CPA（§5.1）；nonce 不隱藏單一密文 L2 norm（§5.2） |
| **RQ3** norm 旁道 | T1 精確 latent 的 norm 相關=1.0（可 linkage）；AF-025 **fixed-norm** 把 T1 與 T2-WB 的 norm-only Top-1 降到隨機（0.005） | `defense_package`（D7-C3/C4/C9/C10）、`tables/table_{norm_leakage,AF025_fixed_norm_*}` | norm linkage 的顯著性本就依存檔（range_preserving 才顯著）；fixed-norm 為處置 |
| **RQ4** 保真/可逆 | S2 M1 影像 PSNR **38.82 dB**（SSIM 0.970、LPIPS 0.067；私有保值域部署）；**加密本身成本**（同存檔）S2−S0 中位 **+0.30 dB**；部署對照 +7.68 dB 幾乎全來自 legacy 存檔丟值域。latent round-trip cosine 0.999794，殘差為 sampler+8-bit（非 transform） | `q8_formal/table_fidelity`、`tables/table_statistics_formal`、`table_error_attribution` | M1 非 pixel bit-exact（sampler+PNG）；bit-exact 走 M2 latent-payload+AEAD；N=200 |

## P2 / 附錄 tracks

| Track | 一句話結論 | 關鍵表 / run | 限制 |
|---|---|---|---|
| **T2-BB inner**（匿名↔匿名 unlinkability） | direct-latent：S0/S1 透明（cos AUC 0.657=plaintext），S2 回隨機（0.433）；實際公開 PNG + frozen-encoder（DenseNet/RAD-DINO）：S0/S1/S2 皆近隨機、正控制 P 0.976/0.810 | `table_t2bb_inner_linkage`、`table_t2bb_m1_frozen`（AUD-20261022-01 PASS） | **AF-041**：§6 的 S0/S1 近隨機成立於「image-encoder 介面 + range-discarding 存檔」雙前提，與 §5 minmax 一致、非推翻透明性；range_preserving 下 S0/S1 cos 仍 0.657 |
| **方法比較**（外部 re-id） | 外部發表級攻擊者（Packhäuser-arch）shadow-val 0.911；同 latent-view 表徵下 S2 0.445 < S0 0.616（per-image-nonce 效果）；原始 checkpoint plaintext 0.982、identity 控制 plaintext_latent 0.550、S2 0.477 | `table_reid_external`、`method_comparison.md §3`、樹外 sanity | **AF-042**：plaintext 真實-CXR 天花板是不同表徵（latent-view 對 CXR 網路 OOD，未加密即 0.550），S2 安全錨在同表徵 S0 基線，不宣稱「擊退 0.98 攻擊者」 |
| **BDIA 2×2 消融** | sampler（BDIA vs DDIM）對 m1 PSNR 影響 <0.4 dB、latent cosine 不變、NFE 相同（1996）→ DDIM 已足夠；conditioning（asym vs symnull）差 ~1 dB（匿名條件的品質代價） | `table_bdia_2x2`、BDIA2X2_…efbf6f85（N=200） | negative/robustness 結果；symmetric 列改匿名語意，只報保真、不宣稱安全 |
| **reinvert 消融** | 前作 deanonymize 把 x_rec 再反演是多餘 round-trip：直接用 x_rec_T 的 healthy 圖與之幾乎逐像素相同（cos 0.9999）、且快 1.50× | `reinvert_ablation/`、`predecessor_decrypt_flow.md §3.1` | 單次 GPU 浮點、N=10；此 pipeline gs=2 健康化幅度本就小 |

## Freeze / 品質保證

| 項目 | 結果 | 出處 |
|---|---|---|
| 關鍵結果重跑（P11.2） | E2.3 latent cache（主表確定性輸入）**10/10 bit-exact** 重現 | `paper_assets/reproduction/` |
| claim audit（P11.6） | 0 無證據 over-claim（perfect/IND-CPA/pipeline-lossless 皆無） | `paper_assets/claim_audit_p11.md` |
| checksum/manifest（P11.7） | 335 檔 SHA-256 + freeze fingerprint | `paper_assets/SHA256SUMS`、`freeze_manifest.json` |
| 全測試（Z12.1） | 見 `reports/final_test_report.md` | — |
