# CLAUDE.md — 本專案的執行規則（每個 session 自動載入）

> 研究設計見 `PROPOSAL.md`／`WORKFLOW.md`，執行治理見 `loop_engineering_spec.md`，稽核見 `AUDIT.md`，進度見 `PROGRESS.md`。本檔只記錄「使用者對 AI 執行方式的常設指示」，不重複上述文件。

## 1. 角色與語言
- 角色依每個 session 的指派（實作／測試方，或稽核方）；不確定時先問。
- 對使用者的結論一律用繁體中文；內部思考可用英文。

## 2. Subagent（使用者 2026-10-12 指示）
- 需要時可**自行開啟 subagent**平行處理，不必每次徵求同意。
- 適合委派：CPU、獨立、規格明確的任務（例如各自的攻擊腳本、文件用語修正、全樣本數值檢查）。核心、稽核會重點複驗的結果（如主攻擊 pipeline）自己做。
- subagent 一律：CPU、唯讀 `past/SourceCode/`、不 commit、不寫金鑰材料；回傳後由實作方審閱並獨立驗證確定性與 provenance 再 commit。

## 3. GPU 長任務（使用者 2026-10-12 指示）
- **預估佔用 GPU 超過 30 分鐘的實驗，啟動前必須先告知使用者並取得裁決**（是否現在可以跑）。預估 ≤ 30 分鐘的可直接跑。
- 告知時附上：預估時間、GPU 記憶體、要寫出的 artifact、對應子任務。
- smoke／單張示範（數分鐘）不受此限；但正式 N=200 之類的長 run 一律先問。
- 已在背景執行的長 run 可繼續監控，不需重問。

## 4. 既有常規（摘要，細節見治理文件）
- Python 一律經 `scripts/run_cfg_ddim.sh`（停用 user-site）。
- 每完成一個子任務做原子 commit；commit 前跑 `scripts/check_staged_files.sh`；不 `git add .`。
- `AUDIT.md`、`audit/` 為稽核方所有，實作方唯讀；回覆追加於 `PROGRESS.md`（只增不改）。
- 會寫入追蹤結果檔的腳本用 `scripts/output_guard.py`（預設拒絕覆寫，需 `--force`）。
- 每週結束做週摘要：`scripts/build_weekly_appendix.py` + `PROGRESS_summary.md`。
