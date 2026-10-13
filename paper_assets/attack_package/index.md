# Track A 攻擊素材包（P4.1）

> 由 `scripts/build_attack_package.py` 從已追蹤的結果檔產生，數字不手抄。每個 claim 對應表、圖、result JSON 與 run ID，見 `claims.csv`。
> 威脅模型見 `PROPOSAL.md` §3；攻擊協定見 `reports/t2wb_protocol.md`；逐輪紀錄見 `PROGRESS.md`。

| claim | 威脅 | 一句話 | headline |
|---|---|---|---|
| C1-keyspace | T1/T3 | 前作有效金鑰上界 S0=32／S1=64 bits；已知一對明密文=0；唯密文 seed 搜尋 S0≈2^32、S1≈2^33 | S0 32 / S1 33 bits |
| C2-coa-s0 | T1 COA | S0 單一密文 latent 即可辨識金鑰（全 200 張皆高於 2^32 門檻；外推屬 [推定]） | 200/200 above-threshold (0 below) |
| C3-coa-s1 | T1 COA | S1 置換階段單一密文辨識 199/200（最弱 security_v1_0018 低於門檻） | 199/200 above-threshold |
| C4-t1-s0-abs | T1 gallery | S0 精確 latent magnitude linkage Top-1（上限） | 1.0 |
| C5-t1-s1-sorted | T1 gallery | S1 精確 latent sorted-magnitude linkage Top-1（上限） | 1.0 |
| C6-t2wb-s0 | T2-WB gallery | S0 abs linkage Top-1（前作部署＝legacy；range 作參考） | legacy 1.000 / range 1.000 |
| C7-t2wb-s1 | T2-WB gallery | S1 sorted-magnitude（尺度不變，AF-033）linkage Top-1；前作部署＝legacy，range 作參考 | legacy 0.465 / range 0.995 |
| C8-t2wb-norm | T2-WB sidechannel | 校正後 norm-only 僅在 range-preserving 顯著（S0/S1）；AF-025 處置依 S2a/S2 於 D7 決定 | S0 0.255 (attack_success) / S1 0.285 (attack_success) [range-preserving] |
| C9-kpa | T3 KPA | S0 一對即還原金鑰；S1 真實 latent 需 2 對（float32 量值重複） | S0 3/3(1對) / S1 3/3(2對) |
| C10-cpa | T3 CPA | 固定金鑰下重建解密映射（d=4096 全基底；前作實際僅需 1 次查詢），fresh max-abs 0 | 0.0 |
| C11-tamper | T4 tamper | S0/S1 無完整性檢查，全部竄改皆被接受（無拒絕） | 30/30 |
| C12-imgonly-s0 | T2-WB→金鑰還原 | 只憑（原圖,匿名圖）影像：S0 單對即還原多數符號（legacy／range），約 8 對對 50 位受害者位元相同解密（A4.6） | sign K=1 legacy 0.956 / range 0.992 ; K=8 legacy bit-exact 50/50 |
| C13-imgonly-s1 | T2-WB→金鑰還原 | 只憑影像：S1 單對無用、多對可破；保存值域顯著加速（range K=4 置換 0.90、K=8 位元相同；legacy K=8 僅 0.645）（A4.6） | perm K=8 range 1.000 / legacy 0.645 ; range K=4 0.905 |
| C14-recon-e2e | T2-WB 端到端 | 端到端：以 8 對還原金鑰重構受害者原圖，S0（兩存檔）與 S1（range）與合法持金鑰者位元相同；S1 legacy 仍部分（A4.6c） | legit~attacker PSNR/SSIM: S0-legacy 100/1.00 ; S1-range 100/1.00 ; S1-legacy 14.3/0.35 |

## 注意
- 前作部署欄為 **legacy PNG**（不保存值域）；range-preserving 欄只作「存檔方式影響」的參考（AUD-20261012-05 使用者裁決）。
- S0/S1 的「保存值域 vs 不保存」對比**不用來裁決 AF-025**：前作部署本就不保存值域。AF-025 的處置依據是 S2a/S2 的跨存檔攻擊（D7，尚未執行、尚未裁決）。
- C7 採尺度不變 scorer（AF-033）：前作部署（legacy）下 S1 為 0.465，非 0.19。
- lo/hi 旁道在每病人單筆 record 下標 `not_evaluable`（需不同影像的具標籤 enrollment）。
- C12–C14（A4.6）：攻擊者只需少數幾組（原圖,匿名圖）對即可竊得整把全域金鑰，之後同金鑰下任一影像都可還原；S0（兩存檔）與保存值域的 S1 可達與合法持金鑰者**位元相同**的重構，前作部署（legacy）的 S1 需更多配對。根因為金鑰與影像無關且幅值為不變量，直接支持 D5 的 per-image nonce、CSPRNG 金鑰與 authenticated container。
