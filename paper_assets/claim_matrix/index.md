# Claim matrix v1（F9.5）

> 由 `scripts/build_claim_matrix.py` 從已凍結、已稽核的 `attack_package/claims.csv`（A3／A4／A4.6）與 `defense_package/claims.csv`（D5／D6／D7／AF-025）彙整；headline 數字不手抄。
> 來源：`attack_package/claims.csv`@01141d1fb72d、`defense_package/claims.csv`@48aa3b491169。
> 狀態：10 項已由 N=200 formal 證據支持，0 項待補。品質列（H3c／RQ4）由 `q8_formal/table_fidelity.csv`@b9fcf38adb4e 與 `tables/table_statistics_formal.csv`@05a5496f0472 讀入，數字不手抄。

| 假設 | RQ | 一句話 | 狀態 | headline | 證據 |
|---|---|---|---|---|---|
| H1a | RQ1 | S0 magnitude gallery attack 顯著高於 random baseline | supported | C4-t1-s0-abs: 1.0 | C6-t2wb-s0: legacy 1.000 / range 1.000 | attack_package:C4-t1-s0-abs;C6-t2wb-s0 |
| H1b | RQ1 | S1 sorted-magnitude attack 顯著高於 random baseline | supported | C5-t1-s1-sorted: 1.0 | C7-t2wb-s1: legacy 0.465 / range 0.995 | attack_package:C5-t1-s1-sorted;C7-t2wb-s1 |
| H1c | RQ1 | 重用 transform 時，一組已知明密文（或少數影像對）可危及其他影像 | supported | C9-kpa: S0 3/3(1對) / S1 3/3(2對) | C10-cpa: 0.0 | C12-imgonly-s0: sign K=1 legacy 0.956 / range 0.992 ; K=8 legacy bit-exact 50/50 | C13-imgonly-s1: perm K=8 range 1.000 / legacy 0.645 ; range K=4 0.905 | C14-recon-e2e: legit~attacker PSNR/SSIM: S0-legacy 100/1.00 ; S1-range 100/1.00 ; S1-legacy 14.3/0.35 | attack_package:C9-kpa;C10-cpa;C12-imgonly-s0;C13-imgonly-s1;C14-recon-e2e |
| H2a | RQ2 | S2 顯著降低 original／sorted magnitude attack 至接近 random | supported | D7-C1: Top-1=0.010 (random=0.005) | D7-C2: legacy=0.010; range=0.010 | defense_package:D7-C1;D7-C2 |
| H2b | RQ2 | 不同 nonce 下，某張影像取得的 KPA／CPA／geometry 資訊不能直接解開其他影像 | supported | D7-C5: same nonce MaxAbs=2.41e-15; cross nonce cosine=0.0011 | D7-C6: fixed-Q Top-1=1.000; per-image-Q=0.005 | defense_package:D7-C5;D7-C6 |
| H2c | RQ2 | HMAC 對所有預定 bit／header／metadata tamper case 均拒絕（S2a 無認證作對照） | supported | D7-C8: S2 rejected=7/7; S2a rejected=0/1 | defense_package:D7-C8 |
| H3a | RQ3 | T1（精確 latent）norm correlation 保持 1；AF-025 固定 norm 將其降到 random | supported_remediated | D7-C3: T1 norm Top-1=1.000 | D7-C10: norm Top-1 T1=0.005; T2-WB=0.005 | defense_package:D7-C3;D7-C10 |
| H3b | RQ3 | norm-only linkage 在 T1 成功，T2-WB 成功率取決於 inversion error／存檔；固定 norm 後兩者降到 random | supported_remediated | D7-C4: T2-WB dev-calibrated Top-1 legacy=0.005; range=0.250 | D7-C9: latent cosine legacy=0.990134; range=0.999798 | D7-C10: norm Top-1 T1=0.005; T2-WB=0.005 | defense_package:D7-C4;D7-C9;D7-C10 |
| H3c | RQ3 | 真實 inversion latent 並非理想 N(0,I)，structured transform 可能改變生成品質 | supported | structured transform 本身無損（transform-only latent cosine 1.000000、max_abs 8.3e-07）；S2 M1 latent round-trip cosine 0.999794，殘差來自 sampler+8-bit 量化而非 transform；S2 M1 影像品質與明文 P 同級（見 RQ4） | q8_formal:table_fidelity |
| RQ4 | RQ4 | P／S0／S1／S2 的 latent round-trip、PSNR、SSIM、LPIPS、cosine、MaxAbsError 與 runtime 比較（含 paired test） | supported | S2 M1 PSNR 38.82 dB（SSIM 0.970、LPIPS 0.067；保值域私有部署）；前作加密方案 S0 31.15／S1 31.42 dB（legacy PNG 部署），明文 P 37.46 dB；配對 S2−S0 中位 +7.68／S2−S1 +7.73 dB（rank-biserial r=1.00，Holm p=5.0e-33，N=200） | q8_formal:table_fidelity;tables/table_statistics_formal |

## Section-5 宣稱限縮（必須在論文遵守）

| 條款 | 內容 | 出處 |
|---|---|---|
| §5.1 | 不宣稱 structured transform 等同 AES／AEAD／一般 IND-CPA；只針對已驗證的 norm 旁道與逐元素／排序幅值攻擊強化 | PROPOSAL.md §5.1；AUD-20261015-01 |
| §5.2 | nonce 只讓每張影像用不同 subkey／Q，不隱藏單一 ciphertext 的 L2 norm，也不提供 integrity | PROPOSAL.md §5.2；D7-C5/C6 |
| §5.3 | 不以 bit-exact=100% 作為 structured transform 的通過條件；DICOM byte 完全一致須走 AEAD-ref（Q10.4） | PROPOSAL.md §5.3 |
| §5.4 | original／sorted／norm-only 與 T1／T2-WB 分開報；固定 norm S2 v2 以相同 N=200 補驗，不因假說篩選 | PROPOSAL.md §5.4；AUD-20261014-04/15-01 |

## 品質列證據與限縮（H3c／RQ4，已由 Q8_FORMAL + legacy supplement 落地）
- H3c、RQ4 headline 由 `q8_formal`（N=200 formal，Q8_FORMAL parent + Q8_LEGACY_SUPPLEMENT）的 `table_fidelity.csv` 與 F9.4 `table_statistics_formal.csv`（paired Wilcoxon + rank-biserial + Holm）讀入，數字不手抄。
- 品質主欄為 mixed deployment（P／S0／S1 legacy PNG、S2 private-range v2）；全 range-preserving 為 attribution 另列。**不得把 S2−S0 的差異歸因為加密本身**：同存檔下 S2−S0 僅約 +0.5 dB，部署差異的 ~+7.7 dB 幾乎全來自保值域存檔。

