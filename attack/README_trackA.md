# Track A — 對 Labarbarie et al. 2025 的密碼分析

## 一、環境

```bash
conda create -n trackA python=3.10 -y && conda activate trackA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
pip install pillow numpy matplotlib
pip install facenet-pytorch          # 選配，exp_E 用
```

## 二、資料檢查（下載鏡像後第一件事）

```bash
python3 -c "
from PIL import Image; import glob
fs = sorted(glob.glob('celeba_hq/**/*.*', recursive=True))[:5]
for p in fs:
    im = Image.open(p); print(p, im.format, im.size, im.mode)
"
```

- `PNG (1024,1024)` → 與官方一致
- `JPEG (256,256)`  → 可用，但 limitation 要註明「非無損來源」

## 三、執行順序

```bash
# 1) 先小規模確認 pipeline 正確（~10 分鐘）
python attack_trackA_latent.py --data_dir ./celeba_hq --n 20 --mode A,B

# 2) 看 fig1_magnitude_leak.png 的 (d) 是否看得出五官 → 這是 Figure 1
# 3) 看 exp A 的 L1/L2 是否為 0，T2 的 cos 落在哪裡

# 4) 正式跑再識別（N=300 先試，再放大到 3000/30000）
python attack_trackA_latent.py --data_dir ./celeba_hq --n 300 --mode C --threat T2
python attack_trackA_latent.py --data_dir ./celeba_hq --n 300 --mode C --threat T1

# 5) KPA 與獨立度量
python attack_trackA_latent.py --data_dir ./celeba_hq --mode D,E

# 6) 對照組：關掉遮罩，等同學長的純 Rademacher 版本
python attack_trackA_latent.py --data_dir ./celeba_hq --n 300 --mode C --no_mask
```

## 四、三個必須確認的接口

| 位置 | 待辦 |
|---|---|
| `LatentDM.__init__` | 確認 `--model_id` 可下載。候選：`CompVis/ldm-celebahq-256`、`runwayml/stable-diffusion-v1-5`。論文用「SD 論文的無條件模型、訓練於 FFHQ」，若無完全對應權重，用同族模型並在論文說明即可 —— 幅值不變量不依賴特定權重。 |
| `face_mask()` | 目前是中央橢圓近似（`raise ImportError` 那行）。論文正式數據建議接真正的 BiSeNet face parser。近似版足以呈現三個洩漏通道的相對關係。 |
| `exp_C` 的 gallery | N=300 是暖身。論文數據要對齊其協定，用 CelebA-HQ 全部 30,000 張。記憶體吃緊時把 latent 存成 float16 或分批算相似度。 |

## 五、預期產出與對應的論文位置

| 產出 | 論文位置 | 判讀 |
|---|---|---|
| `exp A` 的 L1 / L2 = 0 | Table 1（不變量驗證） | 必然為 0，是解析結果的數值確認 |
| `exp A` 的 T2 cos | §5.x 威脅模型可行性 | 對回 Tier-0 E2 表格。Tier-0 顯示 cos=0.50 時 top-1 仍為 1.00 |
| `fig1_magnitude_leak.png` | **Figure 1** | (d) 若看得出五官，一張圖講完整個攻擊 |
| `fig2_cmc_T1/T2.png` | **Figure 2** | 三條曲線分別對應 L1／L2／L1+L2 三個洩漏通道 |
| `fig3_kpa.png` | Figure 3 | T1 應接近 100%，T2 視反演誤差 |
| `exp E` | 度量對照表 | 說明其 FaceNet 指標為何對潛空間攻擊失明 |

## 六、三個洩漏通道

程式把它們分開量，因為它們在論文裡是三個不同的論點：

- **L1 遮罩內**：`|M ⊙ z_ano| = |M ⊙ z|` — 符號翻轉保幅值。這是核心漏洞。
- **L2 遮罩外**：`(1−M) ⊙ z_ano = (1−M) ⊙ z` — **完全未加密，純明文**。論文把它當賣點宣傳（保留背景、髮型、姿勢），但背景與髮型是準識別符。
- **L3 Eq.(7)**：反擴散每一步都重注入原始 `z_t` 到非遮罩區，使匿名影像的背景由原始 latent 生成。

`exp_C` 的 `unmasked` 那條曲線量的就是 L2 —— 如果它單獨就能 top-1 = 1.00，代表**光靠未加密的背景就足以再識別**，這比 L1 更難辯解。

## 七、寫作時的框架

引用其 §5 Conclusion：

> "Future work could ... conduct a systematic evaluation of its resilience against de-anonymization attacks."

然後寫：

> We undertake this evaluation and find that the sign-flipping construction preserves an exact
> per-element magnitude invariant, enabling ciphertext-only re-identification at N% top-1 accuracy
> on their own evaluation protocol.

原作者已明確把這件事列為未完成工作 —— 這是最理想的切入點，既有禮貌又站得住。
