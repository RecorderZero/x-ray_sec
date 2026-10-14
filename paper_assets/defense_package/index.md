# P7.1 Defense Package

本素材包由 `scripts/build_defense_package.py` 從 D7 表格重建；headline 不手抄。
S2a 與 S2 的 confidentiality transform 相同，差別是 S2 加入 authentication。

## 主素材

- 方法／整合圖：`image/D6.7_s2_dev.png`
- 攻擊與 norm limitation：`image/D7_attack_summary.png`、`paper_assets/tables/table_attack_original.csv`、`table_norm_leakage.csv`
- cross-nonce／geometry／probe／repeat／tamper：同目錄各 D7 表格。

## Claim index

| ID | Claim | Headline | Source |
|---|---|---|---|
| D7-C1 | T1 original-|z| linkage | Top-1=0.010 (random=0.005) | `paper_assets/tables/table_attack_original.csv@2d615915e8c6` |
| D7-C2 | T2-WB original-|z| linkage across storage | legacy=0.010; range=0.010 | `paper_assets/tables/table_attack_original.csv@2d615915e8c6` |
| D7-C3 | Exact-latent norm limitation | T1 norm Top-1=1.000 | `paper_assets/tables/table_norm_leakage.csv@cdaf47fc0a45` |
| D7-C4 | Storage-dependent adaptive norm linkage | T2-WB dev-calibrated Top-1 legacy=0.005; range=0.250 | `paper_assets/tables/table_norm_leakage.csv@cdaf47fc0a45` |
| D7-C5 | Full-map transfer across nonce | same nonce MaxAbs=2.41e-15; cross nonce cosine=0.0011 | `paper_assets/tables/table_cross_nonce.csv@afb8d3cdac76` |
| D7-C6 | Pairwise geometry | fixed-Q Top-1=1.000; per-image-Q=0.005 | `paper_assets/tables/table_geometry.csv@b481067e2cf5` |
| D7-C7 | Same-plaintext repeat | unique protected payloads=100/100 | `paper_assets/tables/table_repeat.csv@78285b58d5d1` |
| D7-C8 | Authentication | S2 rejected=7/7; S2a rejected=0/1 | `paper_assets/tables/table_tamper.csv@af52e845d625` |
| D7-C9 | T2-WB inversion positive-control quality | latent cosine legacy=0.990134; range=0.999798 | `paper_assets/tables/table_D7_T2WB_inversion.csv@f90d555fc1f7` |

## 限制

- magnitude／norm 與 geometry 結果是指定攻擊的實測，不是通用安全證明。
- D7.4 的顯式 full-map KPA/CPA 為 d=1,024 synthetic diagnostic；同 nonce positive control 通過。
- range-preserving T2-WB norm linkage 仍明顯高於 random，屬 AF-025，需使用者裁決保護強度。
