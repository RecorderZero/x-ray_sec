# 每週進度摘要（Weekly Summary）

> 本檔是給論文撰寫與指導討論用的**每週彙整**。逐輪的細節紀錄在 `PROGRESS.md`，稽核紀錄在 `AUDIT.md`，研究設計在 `PROPOSAL.md`／`WORKFLOW.md`。
> 本檔引用的圖表都放在 `appendix/week<N>/`，以 `W<N>-F<k>`（圖）與 `W<N>-T<k>`（表）編號，詳見 `appendix/README.md`。

## 撰寫規範（每週沿用）

### 何時寫、誰寫
1. 每週最後一個工作日，在當週稽核輪次結束後撰寫；新的一週**加在檔案最下面**，舊週不改寫。發現舊週有誤時，在新週的「更正」小節說明。
2. 先執行 `scripts/run_cfg_ddim.sh python -m scripts.build_weekly_appendix --week <N>` 產生 `appendix/week<N>/`，確認重建兩次結果相同，再寫摘要。
3. 摘要與附錄同一個 commit 提交（`docs: [W<N>] weekly summary and appendix`）。

### 數字與宣稱
4. 摘要中的每個數字，都必須來自 `appendix/week<N>/` 的素材或其 MANIFEST `sources`；在數字後標出素材編號，例如「31.47 dB（W1-T7）」。不得從記憶或對話手抄數字。
5. 沿用專案標記：`[事實]`（已量測或已讀碼，附來源）、`[推定]`、`[假設]`、`[決定]`（使用者裁決或實作方決定，附日期）、`[結論]`。不得把 `[推定]` 寫成 `[事實]`。
6. 科學結果只報告，不以「是否符合假說」判斷成敗（WORKFLOW §3.4）；負結果與失敗案例照實列出。
7. 依 PROPOSAL §5 限縮宣稱：不寫 perfect security、IND-CPA、lossless、零失真等無證據字樣。

### 每週固定章節
| 小節 | 內容 |
|---|---|
| `<N>.1 範圍與狀態` | 子任務表：ID、內容、稽核判定、主要 commit |
| `<N>.2 本週重點` | 3–6 條最重要的結論，每條附素材編號 |
| `<N>.3 實驗結果與分析` | 依主題分段；每段先放圖表，再寫觀察、分析與限制 |
| `<N>.4 論文可用素材` | 附錄索引表：編號、內容、建議用途（正文主表／正文／附錄／僅內部）、建議章節 |
| `<N>.5 決策紀錄` | 本週的使用者裁決與實作方決定，附日期與出處 |
| `<N>.6 稽核狀態` | 稽核輪次、開立／關閉的 finding、Gate 判定 |
| `<N>.7 限制與未解問題` | 樣本數、未驗證的假設、待裁決事項 |
| `<N>.8 下週計畫` | 對應 WORKFLOW 子任務 ID |

---

## Week 1（2026-10-05～10-11）｜環境、前作重現與正確的評估基準（E0、E1、E2）

### 1.1 範圍與狀態

| 子任務 | 內容 | 稽核判定 | 主要 commit |
|---|---|---|---|
| E0.1 | evidence reset：舊 preflight 數字重建或撤回 | Achieved（AUD-20261008-02） | `9eb4ccc` |
| E1.1 | 環境、checkpoint、套件來源盤點（fail-closed） | Achieved | `44a05a8`、`625f16d` |
| E1.2 | batch 1/4/8 速度與 VRAM | Achieved | `625f16d`、`4fcb7ca` |
| E1.3 | dev split `dev_v1.1`（20 張、20 位病人） | Achieved | `44a05a8` |
| E1.4 | DDIM smoke（推論流程） | Achieved | `1579c96`、`4fcb7ca` |
| E1.5 | run artifact validator 與 managed run | Achieved | `625f16d`、`cf1879f` |
| E2.1 | formal split `security_v1`（200 位病人） | Achieved（AUD-20261008-02） | `44a05a8` |
| E2.2 | P/S0/S1 legacy wrapper 逐位元重現前作 | Achieved（AUD-20261010-01） | `2177c2a`、`15e5a62`、`faf2fa9` |
| E2.3 | latent cache（security_v1、dev_v1.1） | Achieved（AUD-20261010-03） | `a3a9a60`、`b12ca19` |
| E2.4 | 指標語意：前作 cosine 0.9989 的真正含義 | Achieved（AUD-20261010-03） | `905580a`、`6db5347` |
| E2.5 | P/S0/S1 dev round-trip（T1、T2-WB／M1） | Achieved（AUD-20261010-03） | `4156be7`、`5590937` |
| （延伸）AF-024 | 保存值域的 M1 存檔協定與三欄對照 | 第 1、2、4、5 點完成（AUD-20261010-06） | `f5c967e`、`a4b0db9`、`fd4f799` |

* **G0：通過**（AUD-20261010-03）。
* 本週 commit 範圍：`f180569..caa82dd`。

### 1.2 本週重點

1. `[事實]` **前作流程可由其原始程式逐位元重現。** P/S0/S1 的 wrapper 與直接呼叫學長函式、guidance −1 與 0、解密路徑、transform round-trip 的 MaxAbs 全為 0（W1-T4）。以前作的比較方式類比，S1 得到 30.20 dB／影像 cosine 0.9985，與前作論文的 30.23 dB／0.998978 同等級（W1-T7）。
2. `[事實]` **前作宣稱的「latent cosine 0.9989」其實是輸出影像的像素 cosine。** 在本專案 dev 上，兩張不同病人的 X 光 cosine 中位數就有 0.884；同一張圖加雜訊到 30.6 dB 即為 0.9987（W1-F2、W1-T6）。這個數字只代表約 30 dB，不代表 latent 一致或零失真。
3. `[事實]` **latent 距離無法預測影像可逆性。** P 的解密 latent cosine 僅 0.299，影像卻有 37.71 dB；S0/S1 在 legacy PNG 下 latent cosine 0.990，影像只有約 31 dB（W1-F3、W1-T7）。可逆性一律以影像端指標判斷。
4. `[事實]` **前作約 31 dB 的損失來自存檔方式，不是加密。** 前作存檔時做 per-image min-max 卻不保存值域；改用保存值域的 range-preserving PNG 後，S0/S1 回到 38.44／39.17 dB，接近 float 交接的 38.70／39.40 dB（W1-F1、W1-T8）。
5. `[事實]` **保存值域也讓攻擊者看得更清楚。** S0/S1 的 T2-WB 攻擊者 |z| Pearson 從 0.976 升到 0.999（W1-T8）。稽核方據此開立 AF-025：A4／D7 的安全評估必須兩欄並列並加測 norm 旁道。
6. `[事實]` **可重現性建立在 batch 1 上。** E2.3 cache 以 batch 1 重算全部逐位元相同，但 batch 4 建 cache、batch 1 重算的 MaxAbs 達 6.93e-4（W1-T5；`reports/compute_budget.md`）。正式實驗因此固定 batch 1。

### 1.3 實驗結果與分析

#### (a) 實驗設定與資料（E1.1、E1.3、E2.1）

* `[事實]` 環境為 Python 3.10.19、PyTorch 2.14.0+cu130、RTX 5090；checkpoint strict load 的 missing／unexpected keys 皆為 0，參數 113,998,722（W1-T1）。
* `[事實]` `dev_v1.1` 為 20 張、20 位病人，健康／積水各 10；`security_v1` 為 200 張、200 位病人，各 100，並排除了重現前作訓練抽樣的 22,253 位病人（W1-T2）。
* `[假設]` 前作訓練抽樣的重建，依賴 CSV 列順序與 pandas `sample(random_state=1911)` 跨版本行為一致；前作的 256×256 訓練影像目錄沒有保留，無法逐檔比對（AUD-20261008-02 §8-Q1）。

#### (b) 前作重現（E2.2）

* `[事實]` 前作實際的加密與解密生成都寫死 `guidance_scale=-1`：以 y=0 健康類別做條件、單次模型呼叫、沒有 CFG 混合。實測與 guidance 0 逐位元相同（W1-T4），但每張影像一個完整來回只要 15.55 s，guidance 0 需 23.13 s（batch 1，W1-T9）。
* `[事實]` 本專案的 P（不加密）改用恆等金鑰，走與 S0/S1 完全相同的加密流程，三者只差在金鑰。P 等同於不加金鑰的 forward→backward（W1-T4）。

#### (c) 指標語意（E2.4）

![W1-F2](appendix/week1/figures/W1-F2_pixel_cosine_vs_psnr.png)

* `[事實]` 前作程式計算的是 base 流程輸出與加解密流程輸出的影像 cosine，沒有用到任何 latent；解密端的 `samples` 是第 3 步的偽健康影像（`reports/metric_semantics.md` §1）。
* `[事實]` 影像只會正向縮放時，cosine 就能等於 1：y=0.5x 時 cosine=1，PSNR 卻只有 10.79 dB（W1-T6）。
* `[推定]` 在加性、近正交誤差下，像素 cosine 只是 PSNR 的單調函數（W1-F2）。因此論文不得把前作的 0.9989 稱為 latent 一致性或零失真。

#### (d) 可逆性基線與存檔歸因（E2.5、AF-024）

![W1-F1](appendix/week1/figures/W1-F1_storage_reversibility.png)

| 還原對 x0 PSNR（dB，20 張平均 [95% CI]） | legacy PNG（前作） | range-preserving PNG（本專案） | float（上限） |
|---|---|---|---|
| P（恆等金鑰） | 37.71 [37.25, 38.07] | 37.81 [37.32, 38.19] | 37.62 |
| S0 Rademacher | 31.47 [30.60, 32.30] | 38.44 [37.98, 38.85] | 38.70 |
| S1 Signed Perm. | 31.30 [30.75, 31.89] | 39.17 [38.98, 39.34] | 39.40 |

（數值出自 W1-T8；legacy PNG 與 float 欄併自 E2.5 run，見 W1-T7。）

* `[事實]` 保存精確 latent（T1）時，S0/S1 的 transform 逐位元可逆，生成器輸入與 P 相同，還原即為 42.23 dB；這是依構造成立，不是另外量測（W1-T7）。T1 的損失全部來自 DDIM 反演與生成本身。
* `[事實]` 加密匿名影像的值域約為 [−0.53, 1.0]，不在 [0,1]（`results/AF024_m1_storage_per_sample.csv` 的 `storage_lo`／`storage_hi`）。legacy PNG 讀回時等於對整張圖做一次未保存的亮度／對比仿射，再經反演與金鑰反運算放大（`reports/metric_semantics.md` §7）。
* `[事實]` S0/S1 的匿名影像對 x0 只有約 5.3–5.4 dB，視覺上已打亂（W1-T7、W1-F4）。
* `[結論]` 前作的可逆性瓶頸是存檔流程造成的工程失真，不是翻號／置換加密的限制（使用者 2026-10-10 裁決，AUD-20261010-05）。

![W1-F4](appendix/week1/figures/W1-F4_roundtrip_grid.png)

#### (e) latent 與影像指標的脫鉤（E1.4、E2.5）

![W1-F3](appendix/week1/figures/W1-F3_latent_vs_image_fidelity.png)

* `[事實]` 推論流程的 re-inversion latent cosine：第二次 inversion 共用同一份 t=0 雜訊時為 0.906，屬樂觀上限；雜訊未知時只有 0.371（W1-T10）。
* `[事實]` 稽核方的方向敏感度診斷顯示：P 的 re-inversion 誤差（RMSE 約 1.13）與大幅白雜訊同級，影像卻幾乎不變；S0 的解密誤差小很多（RMSE 0.132），影像影響卻與同大小白雜訊相當（AUD-20261010-02 §2）。
* `[推定]` 生成器是多對一的；誤差落在不敏感方向時，latent 差很多、影像幾乎一樣。各方向等權的 latent 距離因此無法預測影像可逆性。

#### (f) 可重現性與成本（E0.1、E1.2、E2.3）

* `[事實]` E0 canonical preflight 中，SOT-WHT 在 d=4,096 的 float32 bit-exact 為 100/100，d=65,536 為 97/100，float64 MaxAbs 2.44e-14；WHT 保留 norm，norm-only gallery Top-1=1.0（W1-T3）。舊的 31/32、98/100 與 Householder 表已標為 historical。
* `[事實]` 學長加密流程每張影像一個完整來回：batch 1 為 15.55 s，batch 8 為 11.07 s（W1-T9）。正式實驗固定 batch 1，N=200 的 E2–F9 預算上限為 22.29 GPU 小時（`reports/compute_budget.md`）。
* `[事實]` E2.3 cache 以 batch 1 抽樣重算，security_v1 20/20、dev 5/5 逐位元相同（W1-T5）。

### 1.4 論文可用素材（`appendix/week1/`）

| 編號 | 內容 | 建議用途 | 建議章節 |
|---|---|---|---|
| W1-F1 | 三種存檔協定下的 M1 還原 PSNR | 正文 | Results／可逆性歸因 |
| W1-F2 | 像素 cosine 對 PSNR、不同病人基準與前作 0.9989 | 正文 | 對前作指標的評析 |
| W1-F3 | 解密 latent cosine 對影像 PSNR（兩種存檔） | 正文 | Results／指標選擇 |
| W1-F4 | P/S0/S1 定性對照（原圖、匿名、T1、M1、差異） | 正文 | Results／定性比較 |
| W1-F5 | 恆等金鑰 P smoke（含局部幻覺觀察） | 附錄 | P 基線 |
| W1-F6 | 推論流程 smoke | 僅內部 | — |
| W1-T1 | 環境、checkpoint、diffusion 與前處理設定 | 附錄 | Experimental setup |
| W1-T2 | dev_v1.1／security_v1 組成與 split hash | 正文 | 資料 |
| W1-T3 | E0 canonical preflight | 附錄 | 前置可行性 |
| W1-T4 | E2.2 前作逐位元重現 | 附錄 | 前作重現 |
| W1-T5 | E2.3 latent cache 與重算 | 附錄 | 可重現性 |
| W1-T6 | 像素 cosine 基準與反例 | 正文 | Metric semantics |
| W1-T7 | E2.5 前作可逆性基線主表 | 正文主表 | Results |
| W1-T8 | AF-024 三種存檔對照（W1-F1 的數值表） | 正文 | Results／可逆性歸因 |
| W1-T9 | 加密流程計算成本 | 附錄 | 計算成本 |
| W1-T10 | 推論流程 re-inversion（共用／未知雜訊） | 僅內部 | — |

完整圖說草稿、來源 SHA-256 與 run ID 見 `appendix/week1/MANIFEST.csv`。

### 1.5 決策紀錄

| 日期 | 決定 | 出處 |
|---|---|---|
| 10-08 | `[決定]` Pillow 回復為學長 requirements 的 8.4.0；所有指令經 `scripts/run_cfg_ddim.sh`（停用 user-site） | AF-004 |
| 10-08 | `[決定]` formal split 採方案 (a)：重現學長抽樣並排除其病人 | AF-012 |
| 10-08 | `[決定]` formal N 維持 200；擴大時另建 `security_v2` | AF-016 |
| 10-08 | `[決定]` 實作方可代寫 WORKFLOW／PROPOSAL，內容須與使用者裁決一致 | AUD-20261008-03 |
| 10-09 | `[決定]` 主表沿用學長生成設定 guidance −1（y=0 條件、無 CFG 混合），不改成無條件 | AF-018、AUD-20261009-02 |
| 10-09 | `[決定]` 由 Claude 擔任實作與測試方，依 loop spec 逐子任務 commit | PROGRESS 2026-10-09 |
| 10-10 | `[決定]` T2-WB `Inconclusive` 規則：攻擊成功照實報告；未成功才看各方案專屬正控制 | AUD-20261010-02 |
| 10-10 | `[決定]` 影像指標維持不 clip，只在報告中註明 | AF-023 第 6 點 |
| 10-10 | `[決定]` M1 存檔須保存影像實際值域；前作 ~31 dB 歸因於存檔流程 | AUD-20261010-05、AF-024 |
| 10-10 | `[決定]` P/S0/S1/S2a/S2 兩種存檔欄都跑；對外以 range-preserving 欄為主結果 | PROGRESS 2026-10-10 |
| 10-10 | `[決定]`（實作方）正式 pass 固定 batch 1 | `reports/compute_budget.md` |

### 1.6 稽核狀態

* 稽核輪次：AUD-20261008-01 至 AUD-20261010-07，共 13 輪。
* Finding：AF-001～AF-025 共 25 筆，已關閉 23 筆。
* 仍開立：
  * AF-024（MAJOR）：只剩 HMAC 涵蓋 lo/hi，排在 D5。
  * AF-025（MAJOR）：保存值域也會幫到攻擊者，A4／D7 須兩欄並列並測 norm 與 lo/hi 旁道。稽核方 dev 實測，norm-only T2-WB 從 legacy PNG 的 1–2/20 升到 range-preserving 的 14–18/20（AUD-20261010-07），處置方向待使用者裁決。

### 1.7 限制與未解問題

* dev 只有 20 張；E2.5 與 AF-024 的結論在 dev 上成立，正式數字以 security_v1（N=200）為準。
* T2-WB 攻擊尚未在 security_v1 實測；本週只建立了它的前提，即攻擊者能取得的 latent 保真度。
* `[觀察]` M1 還原影像在少數病例出現原圖沒有的細小字樣狀紋理（W1-F4、W1-F5），屬生成模型的局部幻覺，尚未量化。
* runtime 數字在 GPU 獨占時量測；與其他程序共用 GPU 時速度約慢 2.4 倍（`results/perf_probe_batch1.json`）。
* AF-025 的處置方向待使用者裁決，候選包括把 lo/hi 放進加密 payload、加入以金鑰導出的 norm 擾動，或接受為已知限制。

### 1.8 下週計畫（Week 2，10-12～10-18）

* A3.1–A3.4：前作金鑰空間、小空間暴力搜尋、S0/S1 不變量、T1 gallery linkage。
* A4.1–A4.5：T2-WB inversion 與 magnitude gallery（兩種存檔欄，加測 norm-only 與 lo/hi 旁道）、KPA、CPA、tamper baseline。
* P3.1、P4.1：攻擊示意圖與 M1 攻擊素材包。
