# appendix/：論文可用素材（每週彙整）

本資料夾收錄每週可直接放進論文的圖、表與其來源證據。內容一律由 `scripts/build_weekly_appendix.py` 從**已提交到 Git 的結果檔**產生，不手動編輯任何數字或圖。每週的文字摘要寫在根目錄的 `PROGRESS_summary.md`，並以本資料夾的編號引用素材。

## 1. 結構

```text
appendix/
  README.md                 本檔：規範
  week<N>/
    MANIFEST.csv            本週素材索引（唯一真實來源）
    figures/W<N>-F<k>_<slug>.png
    tables/W<N>-T<k>_<slug>.csv
```

## 2. 命名與編號

* 圖為 `W<N>-F<k>`，表為 `W<N>-T<k>`；`<N>` 是週次，`<k>` 從 1 起連號。
* 編號一經發布就**不重排、不重用**。素材被取代時保留原檔，在下一週的 MANIFEST 以新編號收錄，並在 `PROGRESS_summary.md` 註明「取代 W1-F3」。
* `<slug>` 用英文小寫與底線，描述內容（如 `storage_reversibility`）。
* 圖內文字用英文（便於直接放進英文論文）；圖說（caption）草稿用中文，寫在 MANIFEST。

## 3. MANIFEST.csv 欄位

| 欄位 | 內容 |
|---|---|
| `id` | `W<N>-F<k>` 或 `W<N>-T<k>` |
| `kind` | `figure` 或 `table` |
| `file` | 素材相對路徑 |
| `paper_use` | `正文主表`、`正文`、`附錄`、`僅內部`（不放論文，只供追溯） |
| `section` | 建議放置的論文章節 |
| `caption_zh` | 圖說草稿（含資料集、樣本數、CI 等必要條件） |
| `supports_claim` | 這份素材支持的論述，與對應的 finding／裁決 |
| `sources` | 來源檔 `path@sha256 前 12 碼`，以 `; ` 分隔 |
| `run_ids` | 產生來源數據的 managed run ID |
| `sha256` | 本素材檔的 SHA-256 |

## 4. 產生方式

```bash
scripts/run_cfg_ddim.sh python -m scripts.build_weekly_appendix --week <N>
```

* 每週的素材清單登記在 `scripts/build_weekly_appendix.py` 的 `WEEKS[<N>]`；新增一週只需加一個清單，不改舊週。
* 圖可直接複製既有 `image/` 圖檔，或由函式從 CSV／JSON 重新繪製；表一律由來源檔衍生，不手抄數值。文獻值（例如前作論文的 30.23 dB）須在表內註明出處。
* 重建兩次結果須逐位元相同（MANIFEST 的 `sha256` 不變），才可提交。
* 圖表樣式依 dataviz 規範：類別色固定為 `#2a78d6`、`#eb6834`、`#1baf7a`（已驗證三色 all-pairs 色盲可分）；第三色對比偏低，因此每張圖都必須有對應的數值表或直接標籤。網格與座標軸為淺色實線，長條圖從 0 起，雙 y 軸禁止使用。

## 5. 使用原則

* 論文中的每個數字都要能回溯到某個 `W<N>-T<k>` 或其 `sources`。
* `僅內部` 的素材不得放進論文，只作為紀錄；例如推論流程（病灶定位用）的結果不是 P/S0/S1 的比較基線。
* 素材的結論等級沿用專案標記：`[事實]`（已量測，附來源）、`[推定]`、`[假設]`；caption 不得寫出超過證據的宣稱（PROPOSAL §5）。
