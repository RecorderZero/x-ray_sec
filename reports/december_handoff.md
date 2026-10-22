# December handoff (Z12.3) — thesis outline + figure placement

> For writing the thesis in December. Each section lists its claim source, the
> table/figure to place, and the run id. Numbers come from frozen artifacts only
> (see `reports/results_index.md`, `paper_assets/claim_matrix/index.md`,
> `paper_assets/final_experiment_matrix.{csv,md}`). Nothing new is to be run.

## Section outline

| § | 章節 | 主張來源 | 插圖 / 表 | 備註 |
|---|---|---|---|---|
| 1 | Introduction | — | — | 問題：醫療 CXR 匿名化的可逆/金鑰控制 vs 不可逆去識別 |
| 2 | Background | `reports/metric_semantics.md`、`deployment_modes.md` | — | 雙空間報告規範；為何 diffusion 而非 GAN（見 `predecessor_decrypt_flow.md §4`） |
| 3 | Threat model | `paper_assets/threat_model_attack_map.md`、`claim_matrix §5` | Fig: 威脅模型圖 | T1/T2-WB/T2-BB/T3/T4 與 Kerckhoffs |
| 4 | Method S2 | `contribution.md` | Fig: pipeline（invert→fixed-norm→SOT→AEAD→generate） | per-image nonce + fixed-norm + SOT + AEAD |
| 5 | Break（前作攻擊 RQ1） | `attack_package`、`table_{T1_magnitude,KPA,CPA,tamper}` | Table 1: linkage Top-1；Table: KPA/CPA/tamper | C4/C5/C9/C10/C11/C12-14 |
| 6 | Fix（防禦 RQ2/RQ3） | `defense_package`、`table_{structured_probe,cross_nonce,norm_leakage,AF025_fixed_norm_*}` | Table 2: S2 攻擊壓制；Table: fixed-norm 處置 | D7-C1…C10 |
| 7 | Fidelity（RQ4） | `q8_formal/table_fidelity`、`table_statistics_formal`、`table_error_attribution` | Fig: `q8_formal/fig_P8_visual_grid.png`；Table 3: PSNR/SSIM/LPIPS + paired | 同存檔 +0.3 dB vs 部署 +7.7 dB（存檔歸因） |
| 8 | Unlinkability（T2-BB） | `table_t2bb_inner_linkage`、`table_t2bb_m1_frozen` | Table 4: anon↔anon linkage AUC | **AF-041 限縮**：雙前提、range_preserving cos 仍 0.657 |
| 9 | Method comparison | `method_comparison.md`、`table_reid_external` | Table 5: 定位表；Table: 外部 re-id | **AF-042 限縮**：同表徵 S0 基線、identity 控制 0.550 |
| 10 | Ablations | `table_bdia_2x2`、`reinvert_ablation/grid_gs2.png` | Fig: reinvert 並排圖；Table: BDIA 2×2 | sampler 不是瓶頸；reinvert 可省 1 pass |
| 11 | Conclusion | `claim_matrix` | — | 10 條 supported 宣稱 + Section-5 限縮 |

## 插圖清單（現有）

| 圖 | 檔案 | 用於 § | 向量檔（300 dpi） | 需補 |
|---|---|---|---|---|
| S2 視覺 grid | `paper_assets/q8_formal/fig_P8_visual_grid.png` | §7 | `fig_P8_visual_grid.pdf` ✅ | caption + take-away（P11.4） |
| reinvert 並排 | `paper_assets/reinvert_ablation/grid_gs2.png` | §10 | `grid_gs2.pdf` ✅ | caption |
| localization parity | `paper_assets/localization_parity/`（每病人零件 PNG） | §7 附錄 | — | 先合成單張 grid（12 月，選圖/版面屬寫作決定），再加 caption |
| 威脅模型 / pipeline | 待繪 | §3/§4 | — | 新繪（示意圖，非數據圖） |

> PDF 由 `scripts/p11_figure_export.py` 無損嵌入原始像素（`FlateDecode`，非 PIL 的
> JPEG/DCTDecode），bit-exact、可重跑。兩張 grid 是真實 CXR 像素的點陣圖，**不做 SVG**
> （SVG 只會 base64 包同一份點陣,無向量化效益）；真正該向量化的是 §3/§4 的示意圖。

## 尚待（P11.4 caption / P11.5，非阻擋，12 月寫作期）
- P11.4：**向量匯出已完成**（上表兩張 grid 的 PDF）。剩 caption 文字（take-away + 限制，跟 §7/§10 正文口徑一致）、localization 合成圖、§3/§4 示意圖。
- P11.5：§5–§10 的可引用草稿已散落於各 `reports/*.md` 與 package `index.md`，需彙整成連續 method/result notes（跟論文正文同步寫最省力）。
- 威脅模型圖與 pipeline 圖為示意圖，12 月繪製。

## Freeze 狀態
- git：`main`（使用者手動 push）；最新 freeze fingerprint 見 `paper_assets/freeze_manifest.json`（定稿前重跑 `scripts/p11_checksums.py`）。
- 測試：193 passed（`reports/final_test_report.md`）。
- 重現：E2.3 cache 10/10 bit-exact（`paper_assets/reproduction/`）。
- 禁止：新增方法/資料集、換 seed/sample、覆蓋舊 run、手改表格數字（WORKFLOW Week 8 規定）。
