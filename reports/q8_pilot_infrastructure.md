# Q8.1–Q8.5／P8.1 pilot 基礎設施（實作說明）

本檔只說明工具如何運作與其保守宣稱；**不含任何科學結果**。Week 7 formal 數字須由實際 run 重新產生。

## 檔案與分工

| 檔案 | 角色 |
|---|---|
| `scripts/q8_pilot_runner.py` | GPU 端 runner。每張影像、每個方法（P／S0／S1／S2-fixed-norm v2）寫一列寬表 `per_sample.csv`，並寫出影像 arrays（`arrays/*.npy`、`array_index.json` 含逐 slice SHA-256）。**不算 LPIPS**。 |
| `scripts/q8_analyze.py` | CPU 端 analyzer。驗證 run，產生 `table_fidelity/anonymity/wrong_key/runtime/storage.csv`、`fig_P8_visual_grid.png`、`P8_selection.json`、`per_sample_long.csv`、`analysis_manifest.json`。 |
| `scripts/q8_metrics.py` | 共用定義：bootstrap／Wilson CI、pessimistic NN rank、P8 選圖規則、LPIPS 狀態、儲存位元組拆解、arrays 證據。 |
| `tests/unit/test_q8_pilot.py` | 30 個 CPU 測試（合成後端）。 |

## 凍結的語意（沿用，未更動）

- S2 = `S2-fixed-norm/v2`、R=1、P off、公開固定 `C=248.261`、真實 norm 與 M1 lo/hi 在 AEAD private metadata；HMAC 先於 metadata 解密、PNG 解碼、inversion、inverse。
- 主要 storage cell = `range_preserving`（S2 用 authenticated private-range container；P/S0/S1 用 public-range PNG）。`legacy`（前作 min-max PNG）為可選歸因 cell：`--storages range_preserving legacy`；對 S2 這是繞過 container 的反事實。
- 生成 `guidance_scale=-1`、inversion `null=True`、`noise_level=500`、seed 1911、batch 1。

## 各表的估計量

- **Q8.1 fidelity**：路徑 `m1`（影像 inversion）、`t1`（M2 exact latent）、`transform_only`；空間 image（PSNR/SSIM/LPIPS/cosine/L∞/RMSE/uint8 相等率）與 latent（cosine/RMSE/MaxAbs/norm ratio/bit-exact/低高頻 cosine/|z| Pearson）；對 x0、對 P 輸出、對 z。另有 S2−S0、S2−S1、S2−P、S0−P、S1−P 配對差異。S2 平均 PSNR 比 S0 低超過 1 dB 時只在 `review_trigger` 欄標註「先檢查實作」，**不是 pass/fail gate**。
- **Q8.2 anonymity**：主要 view＝`public_available`，即持有儲存物件者實際可重建的影像：P/S0/S1＝以公開 lo/hi 還原值域的 range-preserving PNG；S2＝僅 q/255（lo/hi 已 AEAD 加密）。**公開資訊在協定上刻意不對稱**（表內 `note` 欄註明）。此 view 同時算影像差異（PSNR/SSIM/LPIPS/cosine/L∞）與 NN rank，並保留 `generator_float`（診斷）與 `stored_pixels_q255`（8-bit q，診斷）。NN rank 以 gallery=同批原圖，`cosine_raw` 與 `pearson` 兩種 scorer，pessimistic ties；已驗證的 `anonymous_public_available` array 存於 run 目錄；負控制 = 種子化 cyclic-shift derangement；正控制 = P（未保護）；另附「無關影像」數值下限。
- **Q8.3 wrong key／nonce**：S2 的 authenticated open 以 wrong key、偽造 nonce（沿用原 tag）開啟，記錄拒絕率（Wilson）與 inverter 呼叫次數（必須 0）；另有繞過驗證的反事實（解密者仍被給予真實 range 與 norm，屬保守 oracle）之影像／latent 指標，並與正確條件、無關影像下限並列，且給 `wrong − unrelated` 配對差。S0/S1 用另一個 legacy password；P 無金鑰；P/S0/S1 無 authenticator，拒絕率標 `not_applicable`。
- **Q8.4 runtime／memory**：stage 互不重疊：`fixed_norm_scale`、`transform_forward/inverse`、`container_seal/open_total`（含 KDF、private-metadata AEAD、HMAC、PNG codec）、`storage_encode/decode`、`diffusion_*`；`micro_*` 為同輸入的單一原語重測（KDF、MAC compute/verify、inverse transform，**與 pipeline 階段不可相加**）；`control_*` 為 wrong-key 控制，排除於部署總量；`cached_latent_to_m1`（快取 z → 變換／封裝 → 生成 → M1 編碼）與 `m1_to_reconstruction`（M1 解碼／開啟 → inversion → inverse transform → 生成）為同一樣本已量測互斥階段之和；兩者**明確排除**最初 x0→z 的 inversion（該步驟由 E2.3 cache 一次性、與金鑰無關地完成）。median 與 p95（numpy linear）。GPU 每階段 `max_memory_allocated` 峰值；CPU 行程 RSS 為 process high-water（非每階段）。S0/S1 的 transform 時間含 wrapper 每次重建 legacy 金鑰（password KDF）。
- **Q8.5 storage**：S2 容器拆為 nonce(16)／tag(32)／manifest（canonical JSON，不含密文與 private metadata）／private metadata（AEAD 密文原始位元組）／latent payload／image payload／encoding overhead（hex/base64/JSON 框架的非負殘差），總和恆等於序列化總長；M1、M2、M2+preview 分列。P/S0/S1 無 container，僅 payload。
- **P8.1**：`fixed`＝split 順序第一張 label 0 與第一張 label 1；`median`＝`S2__m1__psnr_db` 升冪的下中位數；`worst`＝最小值（同分以 sample_id）。規則寫在 `q8_metrics.SELECTION_RULE`，結果與完整排序寫入 `P8_selection.json`；同一張被多類選中就重複顯示，不替換。固定顯示範圍 [0,1]，PNG 輸出逐位元可重現。

## LPIPS 輸入裁切揭露

LPIPS（torchmetrics `normalize=True`）要求輸入在 [0,1]，所以被比較影像先 clip，而 PSNR/SSIM/cosine/L∞ 使用**未裁切**浮點（SSIM `data_range=1`）。因此 LPIPS 與 PSNR 在 clip fraction>0 時看到的像素不同。三張表的每個 LPIPS 列都有 `input_view`、`clip_fraction_mean`、`clip_fraction_max`（被比較影像中落在 [0,1] 之外的像素比例，對樣本取平均／最大；配對差取兩輸入較大者），`analysis_manifest.json` 的 `lpips.clip_fraction_per_array` 記錄每個 (method, group) array；LPIPS deferred 時 clip fraction 仍會算出並揭露。

## LPIPS（不得替代）

本環境沒有 `lpips` 套件；`torchmetrics` 內含 LPIPS 線性層權重，但需要 torch hub cache 內已存在的 torchvision AlexNet 權重（`alexnet-owt-7be5be79.pth`），且本工具**絕不下載**。`q8_metrics.lpips_status()` 檢查並記錄權重 SHA-256。

- `q8_analyze --lpips auto`（預設）：不可用時所有 LPIPS 列的數值欄留空，`status` 明寫 `lpips deferred: <原因>`，`analysis_manifest.json` 記錄 `lpips.state=deferred`；
- `--lpips require`：不可用時在寫出任何檔案前中止（Week 7 formal 建議使用）；
- `--lpips off`：標 `not_requested`。

LPIPS 僅在 CPU analyzer 由已雜湊驗證的 arrays 計算，所以 GPU run 不依賴它；日後補齊權重後，對同一個 run 重跑 analyzer 即可，不必重跑 GPU。真實 LPIPS 路徑在本機**未能執行驗證**（缺權重）；測試以注入的假 metric 驗證接線，以及缺權重／錯誤 hash 的 fail-closed 行為。

## 金鑰／nonce 政策（依 evidence role）

- `synthetic_selftest`／`smoke`：預設 `fixture`（公開確定性 key 與 nonce，只用於管線自檢，可逐位元重現）。
- `pilot`／`formal`：**強制** `random`——每個 run 新產生的 CSPRNG 256-bit master key 與隨機且互異的 128-bit nonce；`NonceRegistry` 在封裝時強制拒絕重用（每張 M1、M2 各一個，registry 計數必須 =2N；wrong nonce 不登錄但保證互異）。`--key-mode fixture` 對這兩個 role 直接拒絕，analyzer 亦會拒絕分析 fixture 的 pilot/formal run。
- manifest 只記 `master_key_sha256`、`key_mode`、`nonce_mode`、`raw_key_material_recorded=false`；原始金鑰不寫入任何檔案（測試以位元組搜尋 run 目錄驗證）。
- 可重現性：random 模式下受保護影像每次 run 本來就不同；可重現性依賴 split／checkpoint／E2.3 cache 的 hash、逐樣本 x0／latent hash、經雜湊驗證的 arrays，以及確定性的 analyzer（同一 run 重跑輸出逐位元一致）。
- P/S0/S1 使用前作寫死的 legacy password／seed（它們是被比較的基準定義，不是本研究的 S2 金鑰）。

## E2.3 cache provenance（模型建立前 fail closed）

`validate_latent_cache`：cache run artifact schema 與 `succeeded` 狀態；manifest 的 `steps` = CLI `--noise-level`、`pipeline`、`guidance_scale`、`generation_conditioning`、`inversion_conditioning`、`source_sampler_sha256`、`checkpoint_hash`、`dataset_split_hash`、`sample_ids`（含順序）、`exit_code`／`failure_reason`／`skip_reason`、`dtype`；`config.json` 的 `mode=cache`、`noise_level`、`pipeline`、`guidance_scale`、`limit=0`、split 檔名；`load_latent_cache` 逐 latent 雜湊；逐樣本以 `preprocess(path)[0]`（原始 (1,H,W) 形狀）重算 `x0_sha256` 與 cache 比對。通過後才建立模型。manifest 記錄 cache 的 manifest／config／index／latents 四個 SHA-256 與 `latent_cache_x0_verified`。

## 證據與失敗處理

- 沿用 `open_run`／`finalize_run`／`managed_run.py --validate-artifacts`，manifest 含 git commit／dirty、script／imported-module hash、split／checkpoint／latent-cache hash、config hash。
- fail closed：任何樣本例外或非有限值 → 寫 `failures.json`、exit 4、不產生 `per_sample.csv`／`summary.json`；重試用新 run ID，不更換 seed／樣本／設定。
- S2 authenticated wrong-key／wrong-nonce 若未被拒絕（rejected≠1）或 inverter 被呼叫（calls≠0），runner 直接中止（exit 4）；analyzer 也會對不一致證據 fail closed（另檢查 patient ID 唯一、x0 hash 與 array 一致、nonce registry 計數、key_mode 與 role、cache provenance 完整）。runner 亦要求 patient ID 唯一。
- analyzer 檢查：artifact schema、`failures.json` 不存在、manifest `failures=0`、array 檔與逐 slice hash、數值有限、sample 數與 ID、`evidence_role`；每張表含 `n_expected／n／n_failed`。
- `--evidence-role` 必須明示（`smoke`≤4 張、`pilot`、`formal` 僅限完整 200 張 `security_v1`、`synthetic_selftest`）。`synthetic_selftest` 用假的逐點可逆 generator，只驗證管線，analyzer 預設拒絕分析（需 `--allow-synthetic`）。
- 輸出檔經 `output_guard`，預設拒絕覆寫。`per_sample_long.csv`、`arrays/` 屬大型 per-sample 輸出，不應提交。

## 指令

CPU 測試：

```bash
scripts/run_cfg_ddim.sh python -m pytest -q tests/unit/test_q8_pilot.py
```

CPU 管線自檢（合成後端，無 GPU）：

```bash
scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id Q8_PILOT_SELFTEST --validate-artifacts -- \
  python -m scripts.q8_pilot_runner --evidence-role synthetic_selftest --synthetic-selftest --limit 2 --task-id Q8_PILOT_SELFTEST
```

GPU（尚未執行；超過 30 分鐘需使用者裁決）：

```bash
# smoke，約 5 分鐘
scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id Q8_PILOT_SMOKE --validate-artifacts -- \
  python -m scripts.q8_pilot_runner --evidence-role smoke --limit 2 --task-id Q8_PILOT_SMOKE \
  --split splits/dev_v1.1.csv --latent-cache <E2.3 dev cache run> --checkpoint <ckpt>
# pilot（N=20 dev，主 storage），預估約 45 分鐘
scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id Q8_PILOT --validate-artifacts -- \
  python -m scripts.q8_pilot_runner --evidence-role pilot --task-id Q8_PILOT \
  --split splits/dev_v1.1.csv --latent-cache <E2.3 dev cache run> --checkpoint <ckpt>
# 分析（CPU）
scripts/run_cfg_ddim.sh python -m scripts.q8_analyze --run-dir artifacts/runs/Q8_PILOT/<run_id> \
  --output-dir paper_assets/q8_pilot --lpips auto
```
