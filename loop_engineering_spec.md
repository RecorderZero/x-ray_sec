# Loop Engineering 自動化執行與防護規範指南

本文件旨在規範 AI Agent 在執行 Loop Engineering（自動化迴圈工程 / 迭代開發）時的執行邏輯、安全防護、資源管理、斷路保護與版本控制規範。

---

## 1. 執行前準備與對齊機制 (Pre-Flight Alignment)

在 AI 啟動任何程式碼修改或進入 Loop 迭代之前，必須嚴格執行「三要素確認」，嚴禁在目標模糊或無客觀標準的情況下盲目動手：

1. **確認任務目標 (Objective)**：
   * 必須對照 `WORKFLOW.md` 當前週次的子任務（例如 `E1.4`, `A4.2`），明確本輪 Loop 具體要完成的功能或解決的問題。
2. **確認三層通過標準 (Pass Criteria)**：
   * **工程正確性硬閘門**：test vector、schema、無 NaN/Inf、inverse error、HMAC fail-closed、nonce uniqueness 等；未通過不得進入下一任務。
   * **證據完整性硬閘門**：positive/negative control、固定 split、per-sample rows、summary、config、run ID、程式與 checkpoint hash 齊全；缺任一項不得宣稱任務完成。
   * **科學結果（只報告，不作成功門檻）**：Top-k、PSNR、SSIM、LPIPS、攻擊成功率及 S2 是否優於 S0。假說不成立或攻擊成功仍可完成任務，前提是 protocol 與 controls 正確。
   * 禁止沿用示例數值（如 PSNR > 25、cos > 0.95）作正式門檻；所有數值門檻只能來自 `WORKFLOW.md` 已預先登記的 correctness criteria。
   * T2-WB inversion quality 是可解釋性條件：若 positive-control inversion 品質不足，結論標為 inconclusive，不得寫成方案成功抵禦攻擊。
3. **確認驗證測試腳本 (Test Script)**：
   * 必須先確認或編寫好對應的單元測試（Unit Test）或驗證腳本（Verification Script）。
   * **推進閘門原則**：AI 每一輪修改必須以「測試腳本是否 Pass」作為客觀依據。**只有確認通過測試，才允許結束當前子任務並往下一個目標前進**。

---

## 2. 安全防護與人工干預機制 (Human-in-the-Loop / HitL)

為避免 AI 自動化執行造成不可逆的系統毀損或資產遺失，**嚴禁讓 AI 擁有 100% 的無限制全自動權限**。凡涉及以下高風險操作，AI 必須強制中斷暫停，主動向使用者說明影響範圍並等待人工審查核可 (Y/N)：

### 2.1 高風險操作清單（強制暫停等待確認）
* **檔案系統破壞性指令**：
  * 任何刪除指令，包含但不限於 `rm`、`rmdir`、PowerShell `Remove-Item`。
  * 刪除或覆蓋既有模型權重、Checkpoints（如 `*.pt`, `*.ckpt`, `*.pth`, `*.safetensors`）。
  * 刪除或覆蓋原始資料集（如 `data/` 下之影像或特徵快取）。
* **版本控制危險操作**：
  * 強制推送 `git push --force` 或 `git push -f`。
  * 重置或清空工作目錄：`git reset --hard`、`git clean -fd`。
  * 刪除本機或遠端分支、Tag。
* **外部資源與付費操作**：
  * 涉及額外扣款的 API 呼叫（如未受限的商業 API Key 呼叫、雲端付費執行個體啟動）。
  * 修改環境全域設定或提升系統管理者權限。

---

## 3. 資源、額度與 Context 管理 (Resource & Context Management)

### 3.1 API 與執行額度管理 (Quota & Rate Limits)
* **5 小時滾動上限 (5-Hour Limit)**：
  * 當觸及額度上限時，系統將進入冷卻鎖定。
  * 必須確保每個迭代都有 **Checkpoint (存檔) 機制**，以便冷卻結束後能從斷點繼續，避免遺失 Context。
* **週用量上限 (Weekly Limit)**：
  * 長時間封鎖機制。為避免單一無限迴圈乾耗週配額，必須嚴格限制單一子任務的最大迭代次數（預設上限 10 次 Loop）。
* **無效耗能防範**：
  * 嚴禁 AI 在沒有 Exit Condition (離開條件) 的情況下無休止試錯。

### 3.2 Token 膨脹管理 (Context Window Management)
* **原則**：隨著 Loop 輪次增加，對話記錄迅速膨脹，導致單次 Request 消耗的 Input Token 呈指數型上升，且容易引發模型注意力發散（Context Confusion）。
* **做法**：
  * 外層控制器應定時進行 **Context Compression（上下文壓縮）**。
  * 核心 Context 只需保留：(1) 最初的 System Prompt / 本規範、(2) 最新檔案狀態與架構、(3) 最近 2 次的關鍵報錯 Log 與最新測試輸出。
  * 清理掉中間無效或已修正的歷史試錯過程。

---

## 4. 長時數任務執行規範 (Long-Running Tasks)

預估超過 15 分鐘的訓練、批次 inversion 或正式評估，一律使用可恢復的背景 run；不得在互動式 CLI 前景等待。

### 4.1 啟動前檢查

每個長任務在啟動前必須完成：

1. 固定 `task_id` 與唯一 `run_id`，建議格式為 `<task_id>_<UTC timestamp>_<config hash 8碼>`。
2. 建立全新 `artifacts/runs/<task_id>/<run_id>/`；若已存在則拒絕啟動，禁止覆寫舊 run。
3. 保存 resolved config、完整 command、Git commit／dirty state、dataset split hash、checkpoint hash、環境與 GPU 資訊。
4. 執行 1–4 samples smoke test，確認輸入、輸出、metric schema 與估計時間。
5. 檢查可用磁碟、GPU VRAM、輸出容量上限與資料路徑；資料集以唯讀方式使用。
6. 若程式支援 resume，必須先確認 checkpoint／partial-output 的恢復語意；不支援時明確標記 `resumable=false`。

### 4.2 Run directory 與狀態機

每個 run 至少包含：

```text
artifacts/runs/<task_id>/<run_id>/
  config.json
  command.txt
  manifest.json
  status.json
  pid
  heartbeat
  stdout.log
  stderr.log
  exit_code
  summary.json          # 成功或失敗後產生
  per_sample.*          # 保留本機，不提交 Git
```

`status.json` 僅允許以下狀態：`queued → running → succeeded|failed|interrupted`。更新時先寫同目錄暫存檔再 atomic rename，避免中斷留下半份 JSON。至少記錄 `task_id`、`run_id`、PID、host、GPU、start/update/end time、progress completed/total、last checkpoint、exit code 與最後錯誤摘要。

### 4.3 背景啟動與監控

- 使用 `tmux` 或受控 runner 背景啟動，啟動後立即回報 run ID、PID、log 與 status 路徑；不得只回報 shell job number。
- runner 必須捕捉 `EXIT`、`SIGINT`、`SIGTERM`，寫入最終狀態與 exit code。
- 每 60 秒更新 heartbeat；監控端若超過 5 分鐘無 heartbeat，標記 stalled 並檢查 PID/GPU，不直接重啟。
- 同一 `task_id + config hash` 同時只允許一個 active run；同一 GPU 使用 lock，避免重複佔用。
- 監控採短輪詢並立即返回，不以前景 blocking wait 佔住連線。
- Email／Webhook 僅在使用者已明確設定目的地與憑證時啟用；未設定時以本地 status/log 為準，不自行對外傳送。

### 4.4 重試、恢復與斷路

- OOM 可依預先規則降低 batch size 後重試一次，且必須建立新 run ID。
- transient I/O／worker failure 最多自動重試一次；同類錯誤連續 3 次觸發斷路器。
- 不得為取得較好的科學結果自動更換 seed、sample、split、metric 或方法參數。
- 恢復時沿用原 config hash 與 split，另記 `resumed_from`；不得覆寫原 stdout、per-sample rows 或 summary。
- 任務完成後先驗證預期樣本數、重複／缺失 sample ID、NaN/Inf、summary 與 per-sample 一致性，再將狀態設為 `succeeded`。

---

## 5. 工作日誌與結構化反思機制 (Loop Journal & Reflection)

為確保每一輪迭代都有清晰的目的性與可追溯性，AI 在執行每一輪 Loop 時，必須在本地專案根目錄維護一份 `PROGRESS.md`，並嚴格遵循以下「日誌樣板」進行記錄與自我反思：

### 5.1 日誌結構樣板 (Journal Structure)

`PROGRESS.md` 檔首所列格式是唯一 canonical template。每輪只在檔尾追加一個條目，不改寫舊紀錄：

```markdown
## YYYY-MM-DD · Wn · Track X · 一句話標題

### 1. 當前目標 (Objective)
* 對應 `WORKFLOW.md` 子任務代號：[例如 A1.4]
* 本輪要解決的核心問題或實現的功能是什麼？

### 2. 執行動作 (Actions Taken)
* 修改或新增了哪些檔案/程式碼？（標記具體檔名與函式）
* 下達了哪些測試指令？

### 3. 執行結果 (Results & Observations)
* 測試是否通過？（附上客觀 Pass/Fail 數據，如 `PSNR = 28.5 dB`）
* 是否出現預期外的錯誤？

### 4. 達標判定 (Assessment)
* [ ] **已達標 (Achieved)**：完全符合預期，測試 Pass。
  - 下一步：執行原子化 `git commit`（記錄變更檔案與 feat/fix），並進入下一個 Sub-task。
  - 圖表保存：若有關鍵實驗結果圖表，保存在 `image/` 或指定目錄中，檔名必須具備可讀性與辨識度。
* [ ] **未達標 (Failed)**：未通過測試或指標未符標準。

### 5. 歸因分析與下一輪修正策略 (Reflection & Next Action)
*(僅在「未達標」時填寫)*
* **失敗原因分析**：為什麼這次修改無效？根本原因為何？
* **下輪修改計畫**：根據本次失敗經驗，下一輪 Loop 應該調整什麼方向？

### 6. ⚠ 與 WORKFLOW.md 不符
- 發現的衝突。沒有就寫「無」。有的話題給使用者裁決，不要自己改 WORKFLOW.md
```

---

## 6. 原子化版本控制策略 (Atomic Commits)

為確保實驗與程式碼變更具備百分之百的可追溯性，**每成功完成一個子任務（Sub-task）且確認測試通過後，必須立即執行一次 `git commit`**。

### 6.1 Commit 執行規範

- **禁止廣域 staging**：不得使用 `git add .`、`git add -A` 或等價操作；只能列出本子任務明確產生的檔案。
- **提交前安全閘門**：每次 commit 前依序執行：

```bash
set -euo pipefail
git add -- <明確檔案清單>
scripts/check_staged_files.sh
git diff --cached --stat
git diff --cached --check
git commit ...
```

只有所有命令皆以 exit code 0 完成才可 commit；任一步失敗時 `set -e` 必須立即中止。`dataset/`、checkpoint、cache、`artifacts/runs/`、大型 per-sample 輸出與 secret 不得以 `git add -f` 繞過。需要版本化的結果只提交小型 summary、manifest、表格與論文圖。

1. **單一任務原子性**：一個 Commit 僅對應一個獨立子任務，禁止將多個未經驗證的任務混合提交。
2. **Comment 格式規範**：Commit message 必須清楚備註本次完成的功能類別（修 bug 或是增加新功能），並具體列出修改或新增的檔案清單：

```bash
git commit -m "[feat/fix]: <子任務代號> <簡述本次完成之功能或修復之問題>

- 類型: 新增功能 (Feature) 或 修復 Bug (Bugfix)
- 任務依據: WORKFLOW.md <子任務代號>
- 變更檔案:
  - [新增/修改]: <檔案路徑 1>
  - [新增/修改]: <檔案路徑 2>
- 驗證結果: <工程正確性與證據完整性閘門；科學 metrics 另列>"
```

3. **Commit 範例**：
```bash
git commit -m "feat: [E1.4] 完成 DDIM smoke 與 artifact 驗證

- 類型: 新增功能
- 任務依據: WORKFLOW.md E1.4
- 變更檔案:
  - 新增: tests/integration/test_ddim_smoke.py
  - 修改: src/pipeline/ddim_smoke.py
- 驗證結果: smoke 無 NaN/Inf、schema validator 通過；PSNR 僅記錄"
```

---

## 7. 斷路器與異常中斷機制 (Circuit Breaker & Alerts)

外層 Controller 或監控機制必須即時監控 AI 的執行狀態，當滿足以下任一條件時，必須**強制觸發斷路器**：

1. **最大迭代限制**：達到預設的最大 Loop 次數（預設單一任務 10 次）。
2. **重複錯誤門檻**：連續捕獲 3 次相同 Exception 或相同指令失敗。
3. **邏輯停滯 (Stagnation)**：AI 連續 3 次輸出的 Code 或 Log 相似度過高，無實質推進。
4. **設計衝突停損**：發現底層架構矛盾，或同一個 Milestone 判準失敗兩次以上（依 `WORKFLOW.md` 規則應立刻回頭檢討）。

### 斷路器執行流程：
1. **保存狀態**：將當前進度、變數與 Log 自動寫入 `error.log`。
2. **發送警報**：透過 Email / Webhook / 終端警示發送通知給使用者，內容包含：
   * 中斷原因與最後報錯 Log。
   * 已嘗試次數、嘗試過的修改方向與後續修復建議。
3. **安全中止**：中斷 Session 或執行 `sys.exit()`，防止 Token 與配額持續無效消耗，等待人工介入檢視。
