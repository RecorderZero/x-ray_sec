# Final test report (Z12.1)

> Command: `scripts/run_cfg_ddim.sh python -m pytest tests/ -q` (2026-10-22).
> Scoped to our suite `tests/`; the predecessor tree `past/SourceCode/scripts/test_*`
> is excluded (read-only, unmaintained — a stale import there is not our deliverable).

## Result: **193 passed, 0 failed** (1 benign warning), 106 s

| 類別 | 檔案數 | 說明 |
|---|---|---|
| `tests/unit/` | 15 | 指標、q8 pilot runner、artifacts、crypto 不變量、aggregation 等 |
| `tests/integration/` | 1 | `test_legacy_ddim_equivalence`（GPU：legacy invert/reconstruct 對照） |
| `tests/security/` | 10 | SOT/Hadamard、cross-nonce、tamper、AEAD、fixed-norm 等安全性質 |
| `tests/` (root) | 1 | `test_legacy_repro`（GPU：確定性重現） |

- **GPU 測試**：3 支以 `skipif(not cuda)` gate，本次 CUDA 在場故全部實跑（非 skip），皆通過。
- **警告**：1 個 `PILLOW_VERSION` 棄用警告（torchmetrics 相依，無害，不影響結果）。
- **P0 failure**：無。無需記錄 limitation。

## 與其他 freeze 證據的關係
- 關鍵結果重跑（P11.2）另在 `paper_assets/reproduction/`：E2.3 latent cache 10/10 bit-exact。
- claim audit（P11.6）`paper_assets/claim_audit_p11.md`：0 無證據 over-claim。
- checksum（P11.7）`paper_assets/SHA256SUMS` + `freeze_manifest.json`。

## 重跑方式
`scripts/run_cfg_ddim.sh python -m pytest tests/ -q`（約 1.8 分；無 GPU 時 3 支測試自動 skip，其餘 CPU 測試仍須全綠）。
