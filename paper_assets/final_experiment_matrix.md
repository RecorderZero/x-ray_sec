# P11.1 Final experiment matrix

> Built by `scripts/p11_experiment_matrix.py` from the frozen claim packages (attack/defense, run_ids inside), the Q8_FORMAL quality parent, and the P2/appendix tracks added after F9. `n` read from the live table files; numbers never hand-copied.
> RQ1-4 core evidence + Section-5 claim limits live in `claim_matrix/index.md` (10 claims).

Total cells: 34 (RQ1-4 frozen main + P2/appendix).

| track | cell | experiment | run id / provenance | n | status |
|---|---|---|---|---|---|
| T1/T3 | C1-keyspace | 前作有效金鑰上界 S0=32／S1=64 bits；已知一對明密文=0；唯密文 seed 搜尋 S0≈2^32、S1≈2^33 |  |  | frozen |
| T1 COA | C2-coa-s0 | S0 單一密文 latent 即可辨識金鑰（全 200 張皆高於 2^32 門檻；外推屬 [推定]） |  |  | frozen |
| T1 COA | C3-coa-s1 | S1 置換階段單一密文辨識 199/200（最弱 security_v1_0018 低於門檻） |  |  | frozen |
| T1 gallery | C4-t1-s0-abs | S0 精確 latent magnitude linkage Top-1（上限） |  |  | frozen |
| T1 gallery | C5-t1-s1-sorted | S1 精確 latent sorted-magnitude linkage Top-1（上限） |  |  | frozen |
| T2-WB gallery | C6-t2wb-s0 | S0 abs linkage Top-1（前作部署＝legacy；range 作參考） | A4.1_20261012T083425685874Z_3132fa49 |  | frozen |
| T2-WB gallery | C7-t2wb-s1 | S1 sorted-magnitude（尺度不變，AF-033）linkage Top-1；前作部署＝legacy，range 作參考 | A4.1_20261012T083425685874Z_3132fa49 |  | frozen |
| T2-WB sidechannel | C8-t2wb-norm | 校正後 norm-only 僅在 range-preserving 顯著（S0/S1）；AF-025 處置依 S2a/S2 於 D7 決定 | A4.1_DEV_CAL_20261012T133002259066Z_0fa671c3; A4.1_20261012T |  | frozen |
| T3 KPA | C9-kpa | S0 一對即還原金鑰；S1 真實 latent 需 2 對（float32 量值重複） |  |  | frozen |
| T3 CPA | C10-cpa | 固定金鑰下重建解密映射（d=4096 全基底；前作實際僅需 1 次查詢），fresh max-abs 0 |  |  | frozen |
| T4 tamper | C11-tamper | S0/S1 無完整性檢查，全部竄改皆被接受（無拒絕） |  |  | frozen |
| T2-WB→金鑰還原 | C12-imgonly-s0 | 只憑（原圖,匿名圖）影像：S0 單對即還原多數符號（legacy／range），約 8 對對 50 位受害者位元相同解密（A4.6） | A4.1_20261012T083425685874Z_3132fa49 |  | frozen |
| T2-WB→金鑰還原 | C13-imgonly-s1 | 只憑影像：S1 單對無用、多對可破；保存值域顯著加速（range K=4 置換 0.90、K=8 位元相同；legacy K=8 僅 0.645）（A4.6） | A4.1_20261012T083425685874Z_3132fa49 |  | frozen |
| T2-WB 端到端 | C14-recon-e2e | 端到端：以 8 對還原金鑰重構受害者原圖，S0（兩存檔）與 S1（range）與合法持金鑰者位元相同；S1 legacy 仍部分（A4.6c） | A4.6_RECON_20261013T052324279030Z_165666a1; A4.1_20261012T08 |  | frozen |
| RQ2/RQ3 | D7-C1 | T1 original-|z| linkage | 2d615915e8c6 |  | frozen |
| RQ2/RQ3 | D7-C2 | T2-WB original-|z| linkage across storage | 2d615915e8c6 |  | frozen |
| RQ2/RQ3 | D7-C3 | Exact-latent norm limitation | cdaf47fc0a45 |  | frozen |
| RQ2/RQ3 | D7-C4 | Storage-dependent adaptive norm linkage | cdaf47fc0a45 |  | frozen |
| RQ2/RQ3 | D7-C5 | Full-map transfer across nonce | afb8d3cdac76 |  | frozen |
| RQ2/RQ3 | D7-C6 | Pairwise geometry | b481067e2cf5 |  | frozen |
| RQ2/RQ3 | D7-C7 | Same-plaintext repeat | 78285b58d5d1 |  | frozen |
| RQ2/RQ3 | D7-C8 | Authentication | af52e845d625 |  | frozen |
| RQ2/RQ3 | D7-C9 | T2-WB inversion positive-control quality | f90d555fc1f7 |  | frozen |
| RQ2/RQ3 | D7-C10 | AF-025 fixed-norm remediation | a8eae222e1b5 |  | frozen |
| RQ4 | Q8-FORMAL | fidelity/stat/anon/wrongkey/runtime/storage (N=200) | Q8_FORMAL_20261015T173907560215Z_7e4c3368 (+ Q8_LEGACY_SUPPL | 200 | frozen |
| T2-BB inner | T2BB-1 | anon<->anon linkage, direct latent (cos/abs/norm/DenseNet) | t2bb_inner_v1 (seed 1911), cached inversion | 34 | done |
| T2-BB inner | T2BB-2 | learned SupCon attacker (shadow 200) | t2bb_shadow_v1 pilot | 6 | done_negative |
| T2-BB inner | T2BB-3 | actual public-PNG M1 frozen-encoder linkage (N=20) | t2bb_m1 public export + private eval map | 8 | done |
| T2-BB inner | T2BB-4 | image-space control (DenseNet121) | t2bb_inner_v1 | 3 | done |
| Method comp | MC-1 | positioning table vs published medical/reversible anon | refpaper 111/112/113 + web (cited) | - | done |
| Method comp | MC-2 | external re-id attacker (Packhauser-arch reimpl, CheXpert) | REID-external (shadow 1000, seed 1911) | 4 | done |
| Method comp | MC-3 | original-weights cross-set sanity + identity control | Packhauser GPL checkpoint (out-of-tree) | - | done |
| Ablation | ABL-1 | BDIA/DDIM x asym/symnull 2x2 fidelity ablation (N=200) | BDIA2X2_20261020T182731Z_efbf6f85 | 4 | done |
| Ablation | ABL-2 | redundant re-inversion in predecessor healthy-gen (N=10) | reinvert-ablation (seed 1911) | 10 | done |

## Missing-value reasons
- `n` blank for attack/defense package rows: those are per-claim C-codes (multiple sub-runs), not single tables; N is inside each package table.
- No cell is absent for a bad-result reason; P2 rows are optional/appendix and clearly scoped (negative results labelled `done_negative`).
