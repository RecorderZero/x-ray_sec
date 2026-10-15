# X-ray Security Research File Index

本檔是專案的「檔案地圖」。未來新增、重新命名或淘汰腳本、CSV/JSON、報告、圖片時，必須在同一個 commit 更新這裡；大型逐樣本輸出、dataset、checkpoint、cache 與 secret 不列入 Git，只記錄其用途與預期位置。

## 目前狀態

- E0.1、E1.1–E1.5 已由 AUD-20261008-02 複驗通過；E2.2 已完成 S0/S1 legacy regression，並補上恆等金鑰 P（AF-017）。
- 2026-10-09 起 P/S0/S1 一律走學長**加密流程**（`ddim_sample_loop_anonymization`）；E1 的推論流程結果只作病灶定位流程的歷史基線。AF-017、AF-020、AF-021 已由 AUD-20261010-01 關閉；AF-019 的 E2.5 與 A4 T2-WB 均已完成並關閉。
- A3／A4／P3.1／P4.1 已完成且 G1 PASS；D7/G4 PASS。AF-025 的 fixed-norm S2 v2 與 N=200 補驗已由 AUD-20261015-01 獨立複驗通過並 CLOSED；目前無未關閉的 MAJOR／MINOR finding。
- Q8.1–Q8.5／P8.1 pilot 基礎設施已完成 CPU 驗證：P/S0/S1/S2-fixed-norm v2 的品質、匿名性、wrong-key/nonce、runtime、storage 與固定選圖共用同一套可追溯 runner/analyzer；真實 GPU smoke／pilot 結果另由 managed run 產生，不把 synthetic self-test 當科學證據。
- Q8 真實 N=2 GPU smoke `Q8_PILOT_SMOKE_20261015T030330332601Z_955298c1` 與經使用者裁決的 N=20 dev pilot `Q8_PILOT_20261015T044834956998Z_a03d6951` 均成功且 artifacts valid；N=20 aggregate 五表、analysis manifest、P8 選樣與視覺格位於 `paper_assets/q8_pilot/`，逐樣本 long table 與原始 run 只留本機、不入 Git。兩者皆為 `formal=false`，不可取代 Week 7 N=200 formal。
- LPIPS 已使用官方 torchvision AlexNet backbone（SHA-256 `7be5be79…cdee02`）在上述 smoke arrays 上以 `--lpips require` 計算成功；不需安裝額外 `lpips` 套件。
- E2.3 latent cache 已凍結（`security_v1` 200 張、`dev_v1.1` 20 張，batch 1，抽樣重算全部 bit-exact）；E2.4 指標語意報告與 E2.5 dev round-trip 已完成，G0 的子任務全部有交付物。
- T2-WB `Inconclusive` 規則依使用者 2026-10-10 裁決：攻擊成功照實報告；未成功時才看各方案專屬的正控制（`reports/t2wb_protocol.md` §4）。
- 目標 checkpoint 可 strict load，missing/unexpected keys 均為 0。
- E2.1 formal split 已建立且通過獨立驗證；formal N 依使用者裁決先維持 200，僅在小實驗跑通後另建新版 split 擴大。輸入必須使用已凍結的 legacy CheXpert preprocessing：grayscale → histogram equalization → OpenCV `INTER_AREA` 256×256 → JPEG quality 100 round-trip → per-image min-max `[0,1]`。
- 固定 dev split：`dev_v1.1`，20 張、20 位不同病人，健康／積水各 10 張；hash 排除本機路徑。
- 正式 split：`security_v1`，200 位病人；重現學長抽樣後排除 22,253 位曾入選訓練病人，健康／積水各 100 位。
- 主線 P/S0/S1/S2a/S2 的生成一律依學長程式寫死的 `guidance_scale=-1`（y=0 健康類條件生成、單次模型呼叫、無 CFG 混合；與 `guidance_scale=0` 逐位元等價，見 AF-021；使用者 2026-10-09 裁決），inversion 為 `null=True` 無條件；論文用語不得寫成「guidance 0」或「無條件生成」。guidance=4 僅可作病灶健康化附表。
- D5.1–D5.7 已完成實作、clean-HEAD CPU 證據與稽核：100,000 nonce 無碰撞且 reuse fail closed；14-case HMAC tamper matrix 全過；AF-024、AF-030、AF-037、AF-038 均已關閉。
- 主要環境：`CFG_DDIM`、Python 3.10.19、PyTorch 2.14.0+cu130、RTX 5090；D5 primitive provider `cryptography==50.0.2`。

## 核心規劃與紀錄

| 檔案 | 用途 |
|---|---|
| `README.md` | 本索引；回答「檔案在哪裡、做什麼、是否仍使用」。 |
| `AGENTS.md` | 專案協作硬規則：防禦性研究範圍、GPU 裁決、Git 排除，以及背景程序／平台安全提示後的強制狀態回報。 |
| `PROPOSAL.md` | 研究方法、威脅模型、S0/S1/S2 與論文主張邊界。 |
| `WORKFLOW.md` | 八週任務、子任務 ID、pass criteria 與交付項目。 |
| `PROGRESS_summary.md` | 每週摘要（論文撰寫與討論用）：檔首為每週撰寫規範；每週列範圍與狀態、重點、結果分析、論文可用素材、決策、稽核狀態、限制與下週計畫。數字一律引用 `appendix/week<N>/` 的素材編號。 |
| `PROGRESS.md` | 只增不改的實驗日誌；所有新條目使用最新七段式格式。 |
| `CLAUDE.md` | 每個 session 自動載入的執行規則：角色與語言、subagent 自主、GPU >30 分鐘需使用者裁決、既有常規摘要。 |
| `loop_engineering_spec.md` | loop、安全閘門、長任務狀態機、重試與 Git 規則。 |
| `.gitignore` | 排除 dataset、checkpoint、cache、run-level/per-sample 大型輸出與 secrets。 |
| `AUDIT.md` | 稽核方的 finding 與複驗紀錄；實作方唯讀，回覆追加於 `PROGRESS.md`。 |
| `appendix/README.md` | 論文素材附錄規範：`week<N>/` 結構、`W<N>-F<k>`／`W<N>-T<k>` 編號、MANIFEST 欄位、產生方式與使用原則。 |
| `paper_assets/attack_package/` | P4.1 攻擊素材包：`claims.csv`（11 claim × 表／圖／run ID）與 `index.md`。 |
| `appendix/week1/` | Week 1 論文素材：6 張圖、10 張表與 `MANIFEST.csv`（含圖說草稿、建議用途、來源 SHA-256 與 run ID）；由 builder 產生，不手動編輯。 |
| `appendix/week2/` | Week 2 Track A 攻擊素材：3 張圖、10 張表與 `MANIFEST.csv`，涵蓋金鑰空間、不變量、T1／T2-WB、KPA／CPA 與 tamper baseline。 |
| `appendix/week3/` | Week 3 D5 防禦素材：key pipeline、nonce registry 與 authenticated-container tamper matrix 共 3 張表及 `MANIFEST.csv`。 |
| `appendix/week4/` | Week 4 D6 防禦素材：SOT 正確性／分布／R-P 消融／結構診斷、S2 dev integration 與五方法比較，共 3 張圖、8 張表及 `MANIFEST.csv`。 |
| `appendix/week5/` | Week 5 D7／P7／AF-025 素材：T1／T2-WB linkage、cross-nonce、geometry、probe、repeat、tamper、defense package 與 fixed-norm v2 補驗，共 2 張圖、13 張表及 `MANIFEST.csv`。 |
| `audit/README.md` | 稽核程式與輸出索引；由稽核方維護。 |

## E0/E1 可重現程式

| 檔案 | 主要功能 | 何時使用 |
|---|---|---|
| `scripts/build_e0_evidence_reset.py` | 彙整 d=4,096/d=65,536 與 seed sweep，產生 CSV、舊數字對照及去計時 stable hash。 | preflight 程式或設定改變後。 |
| `scripts/run_cfg_ddim.sh` | 固定 `PYTHONNOUSERSITE=1` 後在 CFG_DDIM 執行命令，阻止 `~/.local` 套件混入。 | 所有文件化的 CFG_DDIM 指令。 |
| `scripts/env_guard.py` | 共用 fail-closed 檢查；套件載自 active `sys.prefix` 外或 user-site 啟用時立即中止。 | E1/E2 runner 與 split builder 啟動時。 |
| `scripts/managed_run.py` | 受控執行命令，保存真實 stdout/stderr/exit code、atomic status、PID、heartbeat 與 task/GPU locks；自身設定寫入 `runner_config.json`，以 `EXPERIMENT_RUN_DIR` 把 run 目錄交給子程序；`--validate-artifacts` 時子程序成功後先以 `validate_run` 驗證 manifest/per-sample/summary，未通過即標為 failed（exit 3）。 | E2.3 起所有長任務的外層 runner。 |
| `scripts/e1_environment_inventory.py` | strict load checkpoint，記錄 Python、PyTorch、CUDA、GPU、GPU 驅動版本（AF-031）、Pillow、pytest、OpenCV、模型參數及 hash。 | 環境、套件、驅動或 checkpoint 改變後。 |
| `scripts/create_dev_split.py` | 從 CheXpert validation CSV 建立 20 位病人互斥 split；stable hash 不含 local path。 | split 版本升級時；不要為改善結果任意重抽。 |
| `scripts/create_security_split.py` | 重現學長每類 16,000 張抽樣、排除其病人，再建立固定隨機 formal split。 | E2.1 重建或擴大 formal N 時。 |
| `scripts/e1_ddim_runner.py` | headless CFG-DDIM benchmark/smoke，走學長**推論流程**（`ddim_sample_loop_known_progressive`，manifest `pipeline` 註明）；smoke 的 re-inversion 分報共用雜訊（樂觀上限）與未知雜訊（seed+1,000,000）兩版（AF-019）；可在 `managed_run.py` 下寫入受控 run 目錄。 | E1 重驗；`preprocess`／`create_runtime` 供 E2 共用。P/S0/S1 比較改用 `e2_anonymization_runner.py`。 |
| `scripts/e2_legacy_wrapper.py` | 直接呼叫學長 anonymization sampler 的 anonymize／deanonymize；`P` 為恆等金鑰（全 +1 Rademacher），與 S0/S1 只差在金鑰（AF-017）；`legacy_invert`／`apply_legacy_key`／`legacy_generate` 讓 cached x_T 可重用，並以 test 證明與完整 legacy sampler bit-exact（生成固定 guidance=-1，與學長原始碼字面一致）；CLI 檢查 wrapper/direct、guidance 0 等價、deanonymize 與 transform round-trip 皆 bit-exact。 | E2.2 重現與後續 S0/S1 pipeline 共用。 |
| `scripts/run_artifacts.py` | 共用 run 目錄寫入（manifest 另記 `imported_project_modules`：被 import 的 repo 內模組 SHA-256；`git_dirty` 只看已追蹤檔）：standalone 時自建目錄、寫 log/exit code 並立即驗證；在 `managed_run.py` 下改寫入 `EXPERIMENT_RUN_DIR`，由 managed_run 擁有 log 並在結束後驗證。 | E1/E2 runner 寫出 manifest、per-sample、summary。 |
| `scripts/e2_anonymization_runner.py` | 學長**加密流程**（`ddim_sample_loop_anonymization`）上的 P/S0/S1 實驗：`benchmark` 量 inversion（null=True）與生成（guidance −1／0）每步成本；`p-smoke` 以恆等金鑰 P 跑 4 張 dev 圖的輸出、重算、re-inversion 與 M1（float／學長 PNG 交接）端到端還原（AF-017、AF-019）；`cache` 為 E2.3 latent cache（batch 1，`latents.npy` 存於 run 目錄，seed 1911 抽樣以 batch 1 重算，MaxAbs ≤ 1e-5）。 | E2.3 起 P/S0/S1 共用；長任務經 `managed_run.py --validate-artifacts`。 |
| `scripts/e2_roundtrip_runner.py` | E2.5：從 E2.3 cache 出發，對 P/S0/S1 分報 transform-only、T1（精確 latent）與 T2-WB／M1（學長 PNG 交接為主、float 交接為上限）；影像指標對 x0 與對 P 輸出兩種基準，latent 另報攻擊者視角（ẑ_ano vs z_ano）與解密後保真度（各方案專屬正控制量）；bootstrap 95% CI。 | E2.5 dev round-trip；之後 S2a/S2 沿用同一路徑定義。 |
| `scripts/e2_metric_semantics.py` | E2.4（AF-022）：CPU、seed 1911，量測像素空間 cosine 基準（不同病人、加雜訊至指定 PSNR、去均值、cosine 對 PSNR 表）與「cosine=1 不等於相等」反例，並與稽核方數字比對。 | 指標語意報告的數字來源。 |
| `scripts/perf_probe.py` | 診斷用 batch 1 吞吐探針：eager／CUDA Graphs（以快取同一 CPU 運算結果的 monkey-patch 讓 legacy `timestep_embedding` 可被 capture）與多行程並行，並比對 latent SHA-256。 | 決定正式 run 的加速方式前重新量測。 |
| `scripts/m1_storage.py` | AF-024：M1 存檔協定 `range_preserving_png/v1`，以真實 8-bit 灰階 PNG 儲存，float32 lo/hi 以 IEEE-754 位元保存在 PNG text chunk 與紀錄中，`np.rint` 量化，讀回 `q/255·(hi−lo)+lo`；缺值域時拒絕解碼。前作 `legacy_png_handoff` 不變。 | 本專案 M1 存檔；S2 的 HMAC 須涵蓋 lo/hi（D5）。 |
| `scripts/m1_storage_compare_runner.py` | AF-024：從 E2.3 dev cache 對 P/S0/S1 以 range-preserving PNG 重跑 M1，逐張驗證匿名影像與 E2.5 相同，再唯讀併入 E2.5 的 legacy PNG／float 結果，輸出三欄並列表。 | P/S0/S1 存檔協定對照；S2a/S2 沿用同一協定。 |
| `scripts/p3_attack_overview.py` | P3.1：固定病例（dev_v1.1_000／_010）的原圖、latent、sorted-|·| profile（z／S0／S1 重合）與 S0／S1 匿名圖；GPU 生成 4 張匿名圖。 | P3.1 攻擊示意圖。 |
| `scripts/a3_invariants.py` | A3.3：量化 S0 逐座標 |z|、S1 sorted-|z| 與 norm 不變量（合成＋真實 latent、password＋seed 金鑰），含 negative control 與解析式。 | A3.3 不變量。 |
| `scripts/a3_bruteforce.py` | A3.2：縮小空間（2^20）暴力搜尋正控制，對 S0（合成 d=4096、真實 d=65536 預篩）與 S1（符號/置換拆開）各還原祕密 seed，含負控制與 2^32 外推。 | A3.2 brute-force 正控制。 |
| `scripts/a3_keyspace_audit.py` | A3.1：從程式路徑證明 legacy 金鑰有效位元上界（S0 password 32、S1 password 64、整數 seed 模式 32），含 sha256[:4] 碰撞與整數 seed 低 32 位截斷的數值驗證。 | A3.1 金鑰空間稽核。 |
| `scripts/a3_t1_gallery.py` | A3.4：T1（精確 latent）magnitude／sorted-magnitude／norm gallery linkage；P 正控制、shuffle 負控制；Top-1/5、mAP、CMC 與 bootstrap 95% CI。CPU，讀 E2.3 cache。 | A3.4 攻擊上限；A4 為其 T2-WB 版本。 |
| `scripts/a4_3_kpa.py` | A4.3 KPA：S0 一對精確還原金鑰；S1 真實 latent 量值重複需 2 對（合成 d=4096 一對即可）；held-out 解密驗證。 | A4.3。 |
| `scripts/a4_4_cpa.py` | A4.4 CPA：固定金鑰下 d 次基底查詢重建 S0/S1 解密映射，全新明文 max-abs 0；明列 fixed-Q 前提。 | A4.4。 |
| `scripts/a4_5_deployment_tamper.py` | A4.5 部署路徑竄改（GPU）：竄改存檔的匿名影像後跑完整解密管線（inversion→逆金鑰→生成），確認無任何拒絕。 | A4.5 部署路徑 baseline。 |
| `src/crypto/kdf.py` | D5.1/D5.2：256-bit random master key、明確的 scrypt password mode、RFC 5869 HKDF-SHA256 與 round/purpose domain separation。 | S2 key pipeline；不得把輸出截成短 seed。 |
| `src/crypto/prng.py` | D5.3：完整 HKDF subkey 直接驅動 ChaCha20；image nonce 僅作 HKDF salt，stream IV 固定為 counter 0，避免 AF-037 counter overflow；產生 sign bits 與 rejection-sampling Fisher–Yates permutation。 | D6 structured transform 的 key material。 |
| `src/crypto/nonce.py` | D5.4：SQLite 持久化、跨 process UNIQUE(scope,nonce) registry；同 master-key scope 顯式 reuse fail closed。 | deployment nonce 發放／登記。 |
| `src/crypto/container.py`、`src/crypto/mac.py` | D5.5/D5.6：strict canonical container、M1 lo/hi float32 bits、payload bytes 與 AAD 的 HMAC-SHA256；驗證成功前不呼叫 opener。 | S2 authenticated container。 |
| `src/crypto/key_io.py` | D5.7：僅接受 32-byte raw master key、0600、新檔不覆寫；不使用 pickle-capable loader。 | 本機 key file。 |
| `scripts/d5_security_evidence.py` | 產生 D5.4 100,000 nonce 統計與 D5.6 tamper matrix，不保存 nonce、master/subkey。 | D5 正式 CPU 證據。 |
| `src/crypto/sot.py` | D6.1/D6.2：torch 正規化 FWHT、D5 CSPRNG 直接導出的逐輪 sign/permutation、`Q_R…Q_1` forward 與反向輪序 inverse；支援 CPU/CUDA。 | S2a/S2 transform 核心。 |
| `src/crypto/s2.py` | D6.7：M1/M2 authenticated S2 封裝；seal 強制經持久化 nonce registry，open 強制 tag 驗證成功後才 decode/inversion/inverse。 | S2 部署路徑，不可繞過此層自行解密。 |
| `src/crypto/s2_fixed_norm.py` | AF-025 S2 v2：latent 先縮放至固定 `C=248.261`；原始 norm 與 M1 lo/hi 以 ChaCha20-Poly1305 private metadata 保存，外層 HMAC 驗證後才解密與恢復。 | AF-025 裁決後的正式 S2 路徑；`s2.py` 保留為 v1 稽核基準。 |
| `scripts/d6_sot_evidence.py` | D6.1–D6.6 CPU 證據：數值可逆、顯式 Q、Gaussian、真實 latent 動差、R/P 攻擊／runtime／precision 消融及結構診斷。 | SOT 改動後重建 D6 CPU 結果。 |
| `scripts/d6_s2_dev_runner.py` | D6.4/D6.7 GPU runner：20 張 × 8 組 generation-quality 消融；4 張 authenticated M1/M2 dev 整合。 | 一律由 `managed_run.py --validate-artifacts` 執行。 |
| `scripts/d7_metrics.py` | D7 共用 linkage 指標：float64 feature、悲觀 tie-break、Wilson CI 與 bootstrap mAP。 | 固定沿用 A3/A4 經 AF-028/033 修正後的評估口徑。 |
| `scripts/d7_cpu_security.py` | D7.2–D7.8 CPU formal evidence：T1 magnitude/norm、cross-nonce KPA/CPA、geometry、structured probe、100 次 repeat 與 tamper。 | 讀取凍結的 security_v1 latent cache，不載入 diffusion model。 |
| `scripts/d7_t2wb_inversion.py` | D7.1 GPU runner：S2 R=1/P-off 匿名生成後，分別經 legacy／range-preserving 存檔再做 target-model inversion。 | 支援 dev smoke/calibration 與 formal N=200 managed run。 |
| `scripts/d7_t2wb_analyze.py` | 合併 D7 T1 與 T2-WB query cache，執行相同 magnitude scorer、dev-calibrated norm、正負控制。 | 正式產生 `table_attack_original.csv` 與 `table_norm_leakage.csv`。 |
| `scripts/build_defense_package.py` | P7.1 deterministic builder：由 D7 正式表格建立 claim→source hash 索引、素材導覽與 attack summary 圖。 | D7 表格更新後以 `--force` 重建；逐位元重建驗證已通過。 |
| `scripts/af025_fixed_norm_cpu.py` | 固定 norm v2 的 formal N=200 T1 norm、structured probes 與 100-nonce repeat 證據。 | AF-025 補驗的 CPU 部分。 |
| `scripts/af025_fixed_norm_t2wb.py` | 固定 norm v2 的 target-model T2-WB runner；分開保存攻擊者 public-PNG inversion 與合法 private-range inversion。 | 先 smoke/dev，再跑 formal N=200 managed run。 |
| `scripts/af025_fixed_norm_analyze.py` | 合併固定 norm T1、dev calibration 與 formal T2-WB，輸出 norm Top-k、controls 與 provenance。 | formal run 完成後執行。 |
| `scripts/method_comparison_runner.py` | 固定第一張積水病例，重跑 P/S0/S1/S2a/S2 的原圖、stored anonymous、重構圖；P/S0/S1 用 legacy 丟值域 PNG，S2a/S2 用 range-preserving PNG，另區分 x0 cosine 與前作 estimand。 | 產生五方法直觀比較圖與單張 metrics。 |
| `tests/security/test_environment_lock.py` | AF-038 regression：確認 D5 的三個精確套件版本全部存在於 Week 3 environment freeze。 | D5／後續環境變更時。 |
| `scripts/a4_5_tamper.py` | A4.5 tamper baseline：S0/S1 無完整性檢查，竄改 latent／儲存紀錄皆被接受並量化影響。 | A4.5。 |
| `scripts/a4_t2wb_gallery.py` | A4.2：T2-WB magnitude／sorted-magnitude／norm gallery linkage（兩種存檔分欄、方案專屬解密正控制、shuffle 負控制、Wilson CI、Inconclusive 判定、AF-025 norm 校正與 lo/hi 旁道）。CPU，讀 E2.3 gallery 與 A4.1 query cache。 | A4.2 攻擊。 |
| `scripts/a4_t2wb_inversion.py` | A4.1：從 security/dev E2.3 cache 生成 S0/S1 anonymous image，分別經 legacy PNG 與 range-preserving PNG 讀回後用目標模型 inversion；逐筆記錄 z_ano 對 zhat_ano 指標，並把 query latent cache 留在受控 run 目錄。 | 先以 `--limit` smoke；正式 N=200 必須由 `managed_run.py --validate-artifacts` 執行。 |
| `scripts/build_attack_package.py` | P4.1：從已追蹤結果檔組 Track A 攻擊素材包（每 claim 對應表／圖／JSON／run ID；數字不手抄）。 | 攻擊章節 claim-evidence 對照。 |
| `scripts/build_weekly_appendix.py` | 從已提交的結果檔產生 `appendix/week<N>/` 的圖、表與 MANIFEST；每週素材清單登記在 `WEEKS[<N>]`；重建須逐位元相同。 | 每週撰寫 `PROGRESS_summary.md` 前執行。 |
| `scripts/output_guard.py` | AF-029：寫入 results/、paper_assets/、reports/、image/、appendix/ 的腳本共用；輸出檔已存在時預設拒絕，需 `--force` 才覆寫。 | 所有會寫追蹤檔的腳本啟動時。 |
| `scripts/artifact_schema.py` | 建立／驗證 manifest、per-sample CSV、summary；檢查 ID、finite 與摘要一致性。 | 新實驗 runner 寫出結果後。 |
| `scripts/check_staged_files.sh` | commit 前拒絕禁傳路徑、secret 名稱及超過 90 MiB 的 staged file。 | 每次 commit 前必跑。 |
| `scripts/q8_pilot_runner.py` | Q8/P8 GPU runner：從已驗證的 E2.3 latent cache 比較 P/S0/S1/S2-fixed-norm v2，保存寬表、hash-verified arrays、wrong-key/nonce 控制、分段時間／記憶體與 storage accounting；pilot/formal 使用每 run 隨機 key/nonce。 | 先做 ≤4 張 smoke，再做 dev N=20 pilot；formal N=200 留到 Week 7。 |
| `scripts/q8_analyze.py` | CPU analyzer：驗證 Q8 run 後產生五張 Q8 表、CI-ready long table、P8 固定／中位／最差視覺格與 analysis manifest；LPIPS 缺權重時明確 deferred。 | GPU run 完成後執行，不需重跑 diffusion。 |
| `scripts/q8_metrics.py` | Q8/P8 共用統計、NN rank、LPIPS 可用性、選圖、array evidence 與 S2 container storage 拆解定義。 | runner、analyzer 與單元測試共用；變更會改 estimand，需先凍結規格。 |
| `reports/q8_pilot_infrastructure.md` | Q8/P8 pilot 的方法、公開視角、LPIPS clipping、key/nonce policy、cache provenance、失敗規則與執行指令。 | 執行或稽核 Q8 前先閱讀；不含科學結果。 |
| `paper_assets/q8_pilot/analysis_manifest.json`、`P8_selection.json` | N=20 dev pilot 的來源 run/cache/script/output hashes、LPIPS 權重與 clipping 揭露，以及 fixed-label／median／worst 預先規則選樣。 | Q8/P8 pilot provenance；`formal=false`，不可當正式主表。 |
| `paper_assets/q8_pilot/table_fidelity.csv`、`table_anonymity.csv`、`table_wrong_key.csv`、`table_runtime.csv`、`table_storage.csv` | Q8.1–Q8.5 N=20 aggregate 結果：bootstrap CI、公開視角 NN、authenticated controls、分段成本與 M1/M2 storage accounting。 | pilot 稽核與 Week 7 formal 規格凍結；不含逐樣本資料。 |
| `paper_assets/q8_pilot/fig_P8_visual_grid.png` | P8.1 依固定規則呈現 label 0、label 1、S2 M1 中位與最差案例的 original／各方法 public anonymous／reconstruction／S2 error。 | pilot 視覺檢查；不得改成手挑最佳案例。 |

## E0/E1 結果檔

| 檔案 | 說明／如何解讀 |
|---|---|
| `canonical_preflight.csv` | E0 唯一 canonical synthetic preflight 摘要；舊 31/32、98/100 不得混用。 |
| `reports/evidence_reset.md` | 說明哪些舊 claim 被撤回或限制，以及現版 preflight 能支持什麼。 |
| `reports/preprocessing_audit.md` | 學長實際 CheXpert 前處理的程式證據、與本 runner 的一致性及剩餘限制。 |
| `reports/metric_semantics.md` | E2.4（AF-022）：前作「Cosine Sim 0.9989」在論文（latent）與程式（輸出影像、base 對解密輸出）的差異與行號；像素 cosine 基準；本專案 latent／影像指標定義與「對 x0／對 P 輸出」並列；論文用語規範。 |
| `reports/keyspace_method_note.md` | A3.1 方法說明：claim vs actual 位元數、21 個 file:line 錨點、V1–V6 數值檢查，以及「32 bits」在 password／整數 seed 模式的限定。 |
| `reports/t2wb_protocol.md` | AF-019：可逆性以影像端端到端為主指標、latent 只作診斷；加密流程 P smoke 事實；T2-WB 攻擊的正控制／負控制定義，以及使用者 2026-10-10 裁決的 `Inconclusive` 規則（攻擊成功照實報告；未成功時看各方案專屬正控制）。 |
| `reports/compute_budget.md` | E2–F9 各 scheme/attack 在加密流程上的 forward／generation 半週期矩陣（實測 batch 1/4/8 成本、guidance −1）與 N=200/500/1,000 GPU 時數；記錄 N=200 裁決，以及 batch 組成會改變結果（batch 4 vs 1 MaxAbs 6.9e-4）而正式 pass 固定 batch 1 的決定。 |
| `artifacts/environment_baseline.txt` | 當次環境、套件實際來源、GPU、checkpoint hash、freeze hash 與 strict-load 結果。 |
| `artifacts/environment_freeze.txt` | Week 1／E1 基準，由 CFG_DDIM 的 `pip freeze --all` 產生；保留不覆寫。 |
| `artifacts/environment_freeze_w3.txt` | Week 3／D5 環境快照，納入 `cryptography`、`cffi`、`pycparser` 的實際版本，用於 AF-038 重現。 |
| `requirements-d5.txt` | D5 新增密碼套件的精確版本鎖定：`cryptography` 及相關相依 `cffi`、`pycparser`。 |
| `model_inventory.csv` | 一列式 checkpoint/model inventory，適合程式與試算表讀取。 |
| `splits/dev_v1.csv` | 歷史 dev split；hash 含 local path，不再作新 run 預設。 |
| `splits/dev_v1.1.csv` | 現行 20 張固定 dev images；成員同 v1，stable split hash 排除 local path。 |
| `splits/security_v1.csv` | E2 formal 200 人 split，健康／積水各 100；病人與重建的前作訓練抽樣互斥。 |
| `splits/security_v1_manifest.json` | formal split seed、抽樣規則、排除人數、hash 與 label counts。 |
| `results/E1.2_benchmark.json` | 推論流程 batch 1/4/8 的 cycle 時間、估計 noise=500 時間與 peak VRAM（managed run `E1.2_20261009T153114394206Z_8339c2f5`）；加密流程成本改見 `AF021_anonymization_benchmark.json`。 |
| `results/E1.4_ddim_smoke.csv` | **推論流程**（病灶定位用，`ddim_sample_loop_known_progressive`）四張 x0→z→xrec 的分段 runtime、VRAM、finite、image/latent metrics；re-inversion 分共用雜訊（cos 0.906，樂觀上限）與未知雜訊（cos 0.371）兩版（AF-019）；不是 P/S0/S1 比較用的 P 基線（AF-017）。由 managed run `E1.4_20261009T153121667310Z_aa04e79c` 產生。 |
| `results/AF021_anonymization_benchmark.json` | 加密流程 batch 1/4/8 的 inversion／生成（guidance −1 與 0）每步每張成本與半週期估計；`compute_budget.md` 的依據。 |
| `results/AF017_P_anonymization_smoke.csv` | 恆等金鑰 P 在加密流程上的 4 張 smoke：P 輸出、inversion 重算、re-inversion（float／PNG）latent 指標與低／高頻、\|z\| 診斷，以及 M1 端到端影像指標。 |
| `image/AF017_P_anonymization_smoke.png` | 上述 4 張的 original、P 輸出、M1 float、M1 PNG 與固定色階 0–0.1 差異圖。 |
| `results/E2.4_metric_semantics.json` | E2.4 像素 cosine 基準與反例的數值（dev_v1.1 20 張、190 對），含與稽核方的 `cross_check`。 |
| `results/E2.3_DONE.json` | E2.3 正式 latent cache（`security_v1` 200 張，batch 1）的 metadata：run ID、`latents.npy` 路徑與 SHA-256、shape／finite 檢查、seed 1911 抽 20 張以 batch 1 重算的 MaxAbs（全為 0）。cache 本體在 `artifacts/runs/E2.3/<run_id>/latents.npy`，不入 Git。 |
| `results/E2.3_dev_v1.1_cache.json` | 同上，dev_v1.1 20 張（E2.5 的輸入），抽 5 張重算。 |
| `results/E2.5_roundtrip_per_sample.csv` | E2.5 dev 20 張 × P/S0/S1 的逐筆結果（managed run `E2.5_20261009T191319376995Z_f990c7d7`）：transform-only、T1、M1 PNG／float 的影像（對 x0、對 P 輸出）與 latent（攻擊者視角、解密後保真度）指標。 |
| `results/perf_probe_batch1.json` | 診斷用（非正式實驗）：batch 1 下多行程並行與 CUDA Graphs 的吞吐與逐位元檢查，含各行程每張影像的 latent SHA-256；量測時 GPU 與 ollama、稽核腳本共用，比例僅供參考。 |
| `results/AF024_m1_storage_per_sample.csv` | AF-024 dev 20 張 × P/S0/S1：range-preserving PNG 的 lo/hi、PNG SHA-256、攻擊者／解密 latent 與還原影像指標，並唯讀併入 E2.5 的 legacy PNG／float 欄（managed run `AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8`）。 |
| `paper_assets/tables/table_keyspace.csv` | A3.1 主表：各 scheme／模式的 claim_bits、key_upper_bound_bits、seed_search_bits_separable、kpa_full_pair_search_bits（=0）、ciphertext_only_seed_search_bits 與 file:line 依據（AF-026）。 |
| `results/A3.3.json` | A3.3 不變量數值：S0/S1 的幅值與 norm 最大差、S1 被置換座標比例、float32/64 norm 殘差與 negative control。 |
| `results/A3.2.json` | A3.2 暴力搜尋結果：還原的 seed、候選數、負控制、拆開搜尋與吞吐外推（吞吐/牆鐘在 timing_nondeterministic）。 |
| `results/A4.3.json`、`paper_assets/tables/table_KPA.csv` | A4.3 KPA 結果與控制。 |
| `results/A4.4.json`、`paper_assets/tables/table_CPA.csv` | A4.4 CPA 結果與控制（每查詢換金鑰的 nonce 模擬）。 |
| `paper_assets/tables/table_tamper_deployment.csv` | A4.5 部署路徑竄改結果（2 張，accepted=True）。 |
| `results/A4.5.json`、`paper_assets/tables/table_tamper_baseline.csv` | A4.5 tamper baseline 結果。 |
| `results/D5.4_nonce_registry.json` | D5.4：100,000 個 128-bit random nonce collision/reuse registry 結果，不含 raw nonce/key。 |
| `results/D5.6_tamper_matrix.json`、`paper_assets/tables/table_D5_tamper_matrix.csv` | D5.6：未竄改接受；wrong key/tag、payload、nonce、version、params、AAD、storage/quantization、record/PNG lo-hi 等竄改全部 fail closed，並記 opener 呼叫次數。 |
| `results/D6.1.json`、`results/D6.2.json` | D6 數值正確性：d=65,536、R=1–4、P on/off 的 float64/float32 inverse；d=1,024 顯式 Q 的正交性、norm 與 `Q_2Q_1` 公式。 |
| `results/D6.3.json`、`image/D6.3_gaussian_distribution.png` | D6 Gaussian synthetic 的 KS、前四階動差、histogram 與 Q-Q；圖與 p-value 都只作診斷，不是安全證明。 |
| `results/D6.4.csv`、`results/D6.4_generation_per_sample.csv`、`paper_assets/tables/table_D6_generation_quality.csv` | 20 張 dev 真實 latent 在 8 組 R/P 下的動差位移、norm/cosine，以及 target model 匿名影像品質與 runtime。 |
| `paper_assets/tables/table_rounds_permutation.csv` | D6.5 R=1/2/3/4 × P on/off 的 original-/sorted-magnitude、norm-only、CPU runtime 與 float32 precision；D7 候選選擇依據。 |
| `paper_assets/tables/table_structure.csv` | D6.6 每組設定的 trace、same-vector cosine expectation、basis probe、矩陣 sparsity 與正交誤差；明列不是安全證明。 |
| `results/D6.7_dev_integration.csv`、`image/D6.7_s2_dev.png` | 選定 R=1/P off 的 4 張 M2/T1 與 authenticated range-preserving M1 端到端結果；圖依序呈現原圖、匿名圖、M2 還原、M1 還原與差異。 |
| `results/D7_cpu_security.json` | D7.2–D7.8 CPU formal evidence provenance：security_v1 N=200、cache/split hash、S2 參數與各表路徑。 |
| `paper_assets/tables/table_D7_T2WB_inversion.csv` | D7.1 formal N=200 的兩種存檔 inversion 品質、runtime 與 bootstrap 95% CI；逐樣本 query 留在 managed run，不進 Git。 |
| `paper_assets/tables/table_attack_original.csv`、`table_norm_leakage.csv` | D7.2/D7.3 的 T1/T2-WB 正式結果；包含兩種存檔、相同 magnitude scorer、dev-only norm calibration 與正負控制。 |
| `paper_assets/tables/table_cross_nonce.csv`、`table_geometry.csv`、`table_structured_probe.csv`、`table_repeat.csv`、`table_tamper.csv` | D7.4–D7.8：cross-nonce KPA/CPA、pairwise geometry、非 Gaussian probes、100 次重複加密與 S2/S2a tamper 對照。 |
| `paper_assets/defense_package/`、`image/D7_attack_summary.png` | P7.1 defense 素材包：10 條 claim→source hash 索引、方法／攻擊／norm limitation／AF-025 修補／tamper 素材導覽與 D7 Top-1 圖。 |
| `results/D7_T1_T2WB_linkage.json` | D7.2/D7.3 完整分析 provenance：formal/dev run ID、dev norm bias、positive/negative controls 與各 storage/scorer metrics。 |
| `paper_assets/tables/table_AF025_fixed_norm_T1.csv`、`table_AF025_fixed_norm_norm_leakage.csv` | AF-025 固定 norm v2 的 formal N=200 T1/T2-WB norm-only 結果；含 raw、dev calibration、負控制與 random baseline。 |
| `paper_assets/tables/table_AF025_fixed_norm_probe.csv`、`table_AF025_fixed_norm_repeat.csv` | AF-025 補驗的 structured probes 與同明文 100-nonce repeat evidence。 |
| `paper_assets/tables/table_AF025_fixed_norm_T2WB_inversion.csv`、`results/AF025_fixed_norm_T1_T2WB.json` | fixed-norm formal/dev run provenance、攻擊者 public-PNG inversion、合法 private-range positive control 與 runtime 摘要。 |
| `reports/method_comparison.md`、`image/method_comparison.png` | 固定 `dev_v1.1_010` 的 P/S0/S1/S2a/S2 三欄比較（原圖／stored anonymous／重構圖），含 storage protocol 與 cosine estimand 解讀。 |
| `paper_assets/tables/table_method_comparison.csv`、`results/method_comparison.json` | 上述單圖的 PSNR、SSIM、L∞、float/uint8 最大 pixel error、MSE/RMSE/MAE、x0 cosine、P-output cosine與完整 provenance。 |
| `results/A3.4_T1_gallery.json` | A3.4 逐 scheme×scorer 的 Top-1/5、mAP、CI、CMC 與正負控制（security_v1 200 張，T1）。 |
| `results/E2.2.json` | 單張 noise=500、guidance=-1 的 P（恆等金鑰）/S0/S1 wrapper 對 legacy direct-call regression，P 另比對無金鑰 forward→backward：anonymize、deanonymize、guidance 0 等價對照與 transform round-trip 的 MaxAbs；不含 raw key。 |
| `paper_assets/tables/table_baseline_correctness.csv` | E2.5 主表：P/S0/S1 × 23 項指標的平均、bootstrap 95% CI（B=10,000、seed 1911）、中位數與範圍。 |
| `image/E2.5_roundtrip_grid.png` | 預先固定的 dev_v1.1_000（健康）與 _010（積水）在 P/S0/S1 下的原圖、匿名圖、T1 還原、M1 PNG 還原與固定色階差異圖。 |
| `paper_assets/tables/table_m1_storage_comparison.csv` | AF-024 三欄並列表：P/S0/S1 × legacy PNG／range-preserving PNG／float × 10 項指標的平均、bootstrap 95% CI、中位數與範圍。 |
| `paper_assets/tables/table_T1_magnitude.csv` | A3.4 主表：P/S0/S1 × abs/sorted_abs/norm 的 Top-1/5、mAP、median rank 與負控制。 |
| `image/P3.1_attack_overview.png` | P3.1 攻擊示意圖：magnitude 不變量如何支撐 A3.4 的 gallery linkage。 |
| `image/A3.4_T1_cmc.png` | A3.4 CMC 曲線（三 scorer 分欄，P/S0/S1）。 |
| `image/E1.4_ddim_smoke_contact_sheet.png` | 四列視覺檢查圖；每列是 original、reconstruction、absolute difference。 |
| `artifacts/preflight/canonical_direction_candidates_d4096_n100.json` | d=4,096、N=100 canonical preflight 原始輸出。 |
| `artifacts/preflight/canonical_direction_candidates_d65536_n100.json` | d=65,536、N=100 canonical preflight 原始輸出。 |
| `artifacts/preflight/canonical_sot_seed_sweep_d65536_n100.json` | seeds 0–99、R=1/2/4 bit-exact sweep 原始輸出。 |
| `artifacts/preflight/direction_candidates_cpu*.json` | 舊抽樣數的 historical preflight，只供追溯，不是正式引用來源。 |

## 測試

| 檔案 | 驗證內容 |
|---|---|
| `tests/unit/test_artifact_schema.py` | valid synthetic artifact 應通過；duplicate ID、NaN/Inf、空 numeric fields、缺欄或 exit code 不一致應失敗。 |
| `tests/unit/test_env_guard.py` | active-prefix 正控制，以及越界套件／啟用 user-site 的 fail-closed 負控制。 |
| `tests/unit/test_managed_run.py` | 真 stdout 成功控制與 traceback/非零 exit/status=failed 負控制；`--validate-artifacts` 的有效 artifacts 正控制，以及 NaN、缺 manifest 兩個負控制。 |
| `tests/unit/test_chexpert_preprocessing.py` | 直接載入學長原始前處理函式作 oracle，逐值守住 equalize/INTER_AREA/JPEG/min-max。 |
| `tests/unit/test_split_hash.py` | 驗證更換 local path 前綴不改變 split hash。 |
| `tests/unit/test_roundtrip_runner.py` | E2.5 runner 的 cache 載入檢查：有效 cache 正控制；split hash、樣本順序、latent hash 被竄改時拒絕；bootstrap 摘要可重現。 |
| `tests/unit/test_run_artifacts.py` | 共用 run 目錄寫入：standalone run 通過 validator 且 git commit 取自 run 開始時；`git_dirty` 不計未追蹤檔；manifest 記錄 import 的專案模組 SHA-256；run 期間原始碼被修改時 finalize 失敗；managed 模式沿用 `EXPERIMENT_RUN_DIR`。 |
| `tests/unit/test_m1_storage.py` | AF-024：任意值域（含負值、>1）`decode(encode(x))` 誤差 ≤ (hi−lo)/510；lo/hi 精確保存；常數影像；缺值域 metadata 與非有限值 fail closed；對超出 [0,1] 的影像優於前作 min-max。 |
| `tests/unit/test_output_guard.py` | AF-029：已存在的輸出無 `--force` 時被拒絕且檔案不變；不存在或加 `--force` 時允許；預設 `force=False`。 |
| `tests/unit/test_staged_guard.py` | 在暫存 repo 驗證 checkpoint 副檔名與 private-key 內容會被拒絕。 |
| `tests/unit/test_q8_pilot.py` | Q8/P8 CPU 合成後端測試：metrics、artifact schema、公開視角、auth fail-closed、cache provenance、key/nonce role policy、storage/runtime、LPIPS deferred 與固定選圖。 |
| `tests/security/test_sot.py` | D6.1/D6.2：CPU/CUDA inverse、d=65,536 float64 門檻、顯式 Q 正交與輪次公式、wrong-key/wrong-nonce controls、FWHT 輸入契約。 |
| `tests/security/test_s2.py` | D6.7：M1/M2 authenticated round-trip、tag 驗證先於 inversion、wrong key/nonce fail closed，以及跨 M1/M2 nonce reuse rejection。 |
| `tests/security/test_s2_fixed_norm.py` | AF-025：固定 norm M2 往返、M1 public PNG 不含 lo/hi、合法值域／原 norm 恢復，以及 private metadata tamper 在 inverter 前拒絕。 |
| `tests/integration/test_legacy_ddim_equivalence.py` | GPU 比對 wrapper 與學長 progressive sampler 的 latent/reconstruction bit-exact。 |
| `tests/test_legacy_repro.py` | E2.2：P/S0/S1 transform 精確可逆；GPU 比對 P/S0/S1 anonymize／deanonymize wrapper 與 legacy sampler bit-exact、guidance −1 與 0 逐位元相同（AF-021），以及恆等金鑰 P 等同無金鑰的 legacy forward→backward（AF-017）；cached x_T 路徑（invert → key → generate）與完整 legacy sampler bit-exact（E2.3）。 |

執行：

```bash
scripts/run_cfg_ddim.sh python -m pytest -q tests/unit
```

## 攻擊與設計驗證程式

| 檔案 | 用途 |
|---|---|
| `attack/README_trackA.md` | Track A 攻擊腳本的入口說明與執行順序。 |
| `attack/attack_keyspace.py` | 審計舊方法 seed/key-space 與 brute-force 正向控制。 |
| `attack/attack_poc.py` | 早期攻擊 proof-of-concept；主要供歷史追溯。 |
| `attack/attack_tier1_real.py` | 對真實 latent 執行 Tier-1 linkage/attack。 |
| `attack/attack_trackA_latent.py` | Track A latent-level invariants、KPA/CPA 相關實驗。 |
| `attack/defense_design_check.py` | 歷史候選防禦檢查；docstring 表格非 canonical，須搭配 evidence reset 解讀。 |
| `attack/validate_sot_bitexact_sweep.py` | 重建指定維度、seed 區間與 R 的 float32 bit-exact 計數。 |
| `attack/validate_direction_candidates.py` | E0 CPU synthetic preflight：SOT-WHT、CDF torus pad、KCI toy chain。 |
| `attack/validate_direction_candidates.py.orig` | 該腳本的歷史備份，不應當作目前執行入口。 |
| `attack/verify_cpa.py` | 驗證固定線性 transform 在 chosen-plaintext queries 下的恢復。 |
| `attack/verify_hh_formula.py` | 驗證 Householder 相關公式與數值實作。 |

## 學長程式：先看哪些文件

| 檔案 | 用途 |
|---|---|
| `past/SourceCode/SHELL_SCRIPTS_GUIDE.md` | 22 個 shell scripts 的逐檔用途、可否直接跑及需要修的路徑。 |
| `past/SourceCode/README/CFG_DDIM_README.md` | CFG-DDIM 資料、preprocessing、模型旗標與訓練／推論使用法。 |
| `past/SourceCode/README/ANONYMIZATION_INTEGRATION_REPORT.md` | 匿名化整合方式、pixel-space latent shape 與參數。 |
| `past/SourceCode/README/SIGNED_PERMUTATION_REPORT.md` | Signed Permutation 實作與過往驗證說明。 |
| `past/SourceCode/安裝環境教學.md` | 學長原環境安裝紀錄；目前以 CFG_DDIM inventory 為準。 |
| `past/SourceCode/requirements.txt` | 舊專案依賴列表；不可直接覆蓋目前 conda 環境。 |

## 學長 shell scripts 快速用途

完整風險與路徑修正見 `past/SourceCode/SHELL_SCRIPTS_GUIDE.md`。

| 檔案 | 主要功能 |
|---|---|
| `start_visdom_cfg_ddim.sh` | 啟動本機 Visdom 8850。 |
| `run_cfg_training.sh` | 訓練原圖／deblur CFG diffusion models。 |
| `run_clf_model_training.sh` | 訓練 classifier-guidance diffusion backbone。 |
| `run_clf_classifier_training.sh` | 訓練 classifier-guidance 的獨立 classifier。 |
| `run_cfg_inference.sh` | CFG-DDIM 一般推論。 |
| `run_cfg_inference_Crypto.sh` | CFG/實際活動命令為 uncond 的匿名化與解匿名化。 |
| `run_clf_inference_Crypto.sh` | CLF-DDIM 匿名化／解匿名化。 |
| `run_uncond_ddim_inference.sh` | guidance=0 的 DDIM baseline。 |
| `run_uncond_inference_Crypto.sh` | uncond 90 張匿名化／解匿名化。 |
| `run_inference_hyp.sh` | 掃描 noise level 與 guidance scale。 |
| `run_evaluation_metrics_hyp.sh` | 評估超參數掃描 NPZ。 |
| `run_evaluation_metrics.sh` | 比較 FPGAN、CLF-DDIM、CFG-DDIM metrics。 |
| `run_figure_maker.sh` | 從 NPZ 產生論文比較圖。 |
| `run_encryption_reversibility_eval_sample3.sh` | 三張快速可逆性評估。 |
| `run_encryption_reversibility_eval_sample90.sh` | 九十張正式可逆性評估。 |
| `run_paper_comparison.sh` | 多模型、多 key scheme 的總比較入口草案。 |
| `run_comparison_experiments.sh` | 早期匿名化比較原型。 |
| `run_training.sh`、`run_training_v2.sh` | 舊版訓練設定，只供追溯。 |
| `run_inference.sh`、`run_inference_v2.sh`、`run_experiment_orig.sh` | 舊版推論設定，只供追溯。 |

## 學長 Python 核心檔案

| 檔案 | 用途 |
|---|---|
| `past/SourceCode/scripts/chexpert_preproc.py` | 原始 CheXpert 做 histogram equalization、256 resize、JPEG 輸出與 label/path 重建。 |
| `past/SourceCode/scripts/cfg_image_train.py` | CFG diffusion 訓練入口。 |
| `past/SourceCode/scripts/cfg_image_sample.py` | CFG DDIM inversion/reconstruction 推論。 |
| `past/SourceCode/scripts/cfg_image_sample_anonymization.py` | 舊 Rademacher/Signed Permutation 匿名化與解匿名化。 |
| `past/SourceCode/scripts/cfg_image_sample_enhanced_anonymization.py` | 後期增強匿名化實驗入口。 |
| `past/SourceCode/scripts/unconditional_ddim_anonymization.py` | guidance=0 匿名化基線。 |
| `past/SourceCode/scripts/image_train.py`、`image_sample.py` | classifier-guidance diffusion backbone 的訓練／採樣。 |
| `past/SourceCode/scripts/classifier_train.py`、`classifier_sample_known*.py` | noisy classifier 訓練與 classifier-guided sampling。 |
| `past/SourceCode/scripts/evaluation_metrics*.py` | 影像品質、定位、可逆性及比較 metrics。 |
| `past/SourceCode/scripts/figure_maker_select.py` | 選案例並排版論文圖片。 |
| `past/SourceCode/scripts/test_enhanced_anonymization.py` | 舊增強匿名化的自測。 |
| `past/SourceCode/guided_diffusion/bratsloader.py` | CheXpert/BRATS dataset、label filtering、min-max loader。 |
| `past/SourceCode/guided_diffusion/gaussian_diffusion.py` | DDPM/DDIM forward、reverse、inversion、anonymization 核心。 |
| `past/SourceCode/guided_diffusion/anonymization.py` | Rademacher、Permutation、Signed Permutation key transforms。 |
| `past/SourceCode/guided_diffusion/script_util.py` | model/diffusion defaults 與 factory。 |
| `past/SourceCode/guided_diffusion/unet_v1.py` | 目前 checkpoint 對應的 v1 U-Net。 |
| `past/SourceCode/guided_diffusion/unet.py`、`unet_v2.py` | 其他歷史 U-Net variants。 |
| `past/SourceCode/guided_diffusion/train_util.py` | training loop、checkpoint 與共用 visualize/min-max。 |
| `past/SourceCode/guided_diffusion/respace.py`、`resample.py` | timestep respacing 與 training schedule sampler。 |
| `past/SourceCode/guided_diffusion/dist_util.py`、`logger.py` | device/distributed/checkpoint I/O 與 logging。 |
| `past/SourceCode/guided_diffusion/nn.py`、`losses.py`、`fp16_util.py` | 網路元件、diffusion losses 與 mixed precision utilities。 |

## 舊結果與大型本機資料

| 路徑 | 用途／規則 |
|---|---|
| `past/SourceCode/results/encryption_reversibility_evaluation*/` | 學長過往可逆性 CSV、報告與比較圖；只作歷史證據，不覆寫。 |
| `past/SourceCode/results/visdom_cfg_inference_*/` | 過往 Visdom inference log/progress。 |
| `past/SourceCode/results/Model/` | checkpoint；Git 排除，禁止上傳。 |
| `dataset/` | 原始資料集；Git 排除且實驗時唯讀。 |
| `artifacts/runs/` | 未來長任務的 run directory；含 log/status/per-sample，本機保存、不提交。 |
| `refpaper/` | 參考論文 PDF；通常不提交大型檔。 |
| `past/Documents/` | 學長論文原始 LaTeX、圖與歷史文件。 |

## 找檔案的建議順序

1. 查任務定義：`WORKFLOW.md`。
2. 查實際做過什麼：`PROGRESS.md`。
3. 查方法與安全主張：`PROPOSAL.md`。
4. 查小型正式結果：`results/`、`reports/`、`splits/`、`image/`。
5. 查學長舊流程：先看 `past/SourceCode/SHELL_SCRIPTS_GUIDE.md`，再讀對應 script。
6. 查長任務：依 `task_id/run_id` 到 `artifacts/runs/`，不要只看 stdout。
