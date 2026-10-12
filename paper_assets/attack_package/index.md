# Track A 攻擊素材包（P4.1）

> 由 `scripts/build_attack_package.py` 從已追蹤的結果檔產生，數字不手抄。每個 claim 對應表、圖、result JSON 與 run ID，見 `claims.csv`。
> 威脅模型見 `PROPOSAL.md` §3；攻擊協定見 `reports/t2wb_protocol.md`；逐輪紀錄見 `PROGRESS.md`。

| claim | 威脅 | 一句話 | headline |
|---|---|---|---|
| C1-keyspace | T1/T3 | 前作有效金鑰上界 S0=32／S1=64 bits；已知一對明密文=0、唯密文 S0≈2^32 | 33 |
| C2-coa-s0 | T1 COA | S0 單一密文 latent 即可辨識金鑰（全 200 張） | 0 |
| C3-coa-s1 | T1 COA | S1 置換階段單一密文辨識（199/200） | 1 |
| C4-t1-s0-abs | T1 gallery | S0 精確 latent magnitude linkage Top-1 | 1.0 |
| C5-t1-s1-sorted | T1 gallery | S1 精確 latent sorted-magnitude linkage Top-1 | 1.0 |
| C6-t2wb-s0 | T2-WB gallery | S0 abs linkage Top-1（legacy／range-preserving） | 1.000 (attack_success) |
| C7-t2wb-s1 | T2-WB gallery | S1 sorted-abs linkage Top-1（range-preserving 0.975 vs legacy 0.19） | 0.975 (attack_success) |
| C8-t2wb-norm | T2-WB sidechannel | 校正後 norm-only 僅在 range-preserving 達顯著（AF-025） | 0.285 (attack_success) |
| C9-kpa | T3 KPA | S0 一對即還原金鑰；S1 真實 latent 需 2 對（float32 量值重複） | True |
| C10-cpa | T3 CPA | 固定金鑰 d 次查詢重建解密映射（fresh max-abs 0） | 0.0 |
| C11-tamper | T4 tamper | S0/S1 無完整性檢查，竄改皆被接受 | True |

## 注意
- S0/S1 的「保存值域 vs 不保存」對比**不用來裁決 AF-025**：前作部署本就不保存值域。AF-025 的處置依據是 S2a/S2 的跨存檔攻擊（D7，尚未執行）。
- lo/hi 旁道在每病人單筆 record 下標 `not_evaluable`（需不同影像的具標籤 enrollment）。
