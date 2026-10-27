# appendix_audit/ · 論文附錄素材（稽核方整理）

本資料夾依週次存放**可放進論文或附錄**的圖、表、原始資料與文件快照，對應根目錄 `AUDIT_summary.md` 的各週摘要。撰寫規則見 `AUDIT_summary.md` 檔頭「撰寫方式」。

## 結構

```
appendix_audit/
  README.md                ← 本檔：資料夾規範與各週索引
  weekNN_<任務群>/
    MANIFEST.csv           ← 每個檔案的來源路徑、SHA-256、大小、產生者、複製時 git HEAD、說明、論文用途
    figures/               ← wNN_figXX_<slug>.png（本週產生，英文標籤，300 dpi）
                              wNN_src_*.png（實作方影像的複本）
    tables/                ← wNN_tabXX_<slug>.csv（每張圖的資料表）
                              wNN_src_*.csv（實作方表格的複本）
    data/                  ← 原始數據：wNN_src_*（實作方）、wNN_aud_*（稽核方獨立量測）
    docs/                  ← 實作方報告的快照（wNN_src_*.md）
    scripts/               ← make_figures.py（重新產生圖表）、collect_assets.py（複製素材＋產生 MANIFEST）
```

## 規範

1. **只複製、不搬移**：原檔留在原位，這裡的檔案是快照；SHA-256 記在 `MANIFEST.csv`，可隨時比對是否與原檔一致。
2. **命名前綴**：`wNN_fig`／`wNN_tab`＝本週稽核方產生；`wNN_src_`＝實作方原檔複本；`wNN_aud_`＝稽核方獨立量測原始資料。
3. **只放小檔**：不放 latent cache、dataset、checkpoint、run 目錄。
4. **可重現**：從 repo 根目錄執行
   ```bash
   scripts/run_cfg_ddim.sh python appendix_audit/weekNN_<任務群>/scripts/make_figures.py
   ```
   ```bash
   scripts/run_cfg_ddim.sh python appendix_audit/weekNN_<任務群>/scripts/collect_assets.py
   ```
5. **以 `AUDIT.md` 為準**：本資料夾與 `AUDIT_summary.md` 只是整理；有出入時以 `AUDIT.md` 的稽核紀錄為準，並在下一週摘要中更正。

## 各週索引

| 週次 | 資料夾 | 範圍 | 摘要位置 |
|---|---|---|---|
| Week 1（10/05–10/10） | `week01_E0-E2/` | E0.1、E1.1–E1.5、E2.1–E2.5（G0） | `AUDIT_summary.md` § Week 1 |
| Week 2（10/11–10/13） | `week02_A3-A4/` | A3.1–A3.4、P3.1、A4.1–A4.5、P4.1（G1 待 10/18） | `AUDIT_summary.md` § Week 2 |
| Week 3（10/13–10/19） | `week03_D5/` | D5.1–D5.7、A4.6（G1 通過；G2 待 10/25） | `AUDIT_summary.md` § Week 3 |
| Week 4（10/14–） | `week04_D6/` | D6.1–D6.7、S2 dev 整合（G3 通過） | `AUDIT_summary.md` § Week 4 |
| Week 5（10/14–10/15） | `week05_D7-P7/` | D7.1–D7.8、P7.1、AF-025（G4 通過；AF-025 CLOSED） | `AUDIT_summary.md` § Week 5 |
| Week 6（10/15） | `week06_Q8-P8/` | Q8.1–Q8.5、P8.1 N=20 dev pilot（mixed deployment） | `AUDIT_summary.md` § Week 6 |
| Week 7（10/16–10/19） | `week07_F9-G5-Q10/` | F9.1–F9.5＋G5 主表凍結、Q10.1–Q10.5（誤差歸因/精度/AEAD/雪崩） | `AUDIT_summary.md` § Week 7 |
| Week 8（10/22–10/24） | `week08_P2-P11/` | P2 附錄（T2-BB-M1-Frozen／外部 re-id／BDIA 2×2）＋P11／Z12 凍結交付＋Q11 定位 parity（AF-041/042/043 全關閉） | `AUDIT_summary.md` § Week 8 |
