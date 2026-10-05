#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Track A : 對 Labarbarie et al. 2025 的密碼分析
   "Secure and Reversible Face Anonymization with Diffusion Models"
=============================================================================
 目標：證明其加密構造洩漏幅值不變量，並在其自身的評估協定下完成再識別。

 復現對象（論文原式）
 --------------------
   Eq.(3)  k = 2b − 1 ∈ {−1,+1}^d ,  b ~ Bernoulli(0.5)
   Eq.(5)  DDIM forward (inversion)  : x0 -> zT
   Eq.(6)  z_T^ano = M ⊙ (k ⊙ z_T) + (1 − M) ⊙ z_T
   Eq.(4)  DDIM backward             : zT -> x0
   Eq.(7)  每步重注入非遮罩區          : z_t^ano = M ⊙ z_t^ano + (1 − M) ⊙ z_t
   設定    T = 50 steps, 無條件模型, BiSeNet 臉部遮罩, CelebA-HQ 256x256

 三個洩漏通道（本檔要量化的東西）
 --------------------------------
   L1  遮罩內：符號翻轉保幅值      -> |M ⊙ z_ano| = |M ⊙ z|
   L2  遮罩外：完全未加密           -> (1−M) ⊙ z_ano = (1−M) ⊙ z   （純明文）
   L3  Eq.(7) 每步重注入原始 z_t    -> 匿名影像的背景由原始 latent 生成

 威脅模型
 --------
   Kerckhoffs：攻擊者知道演算法與公開模型權重，不知金鑰。
   T1  直接取得 z_ano（雲端外洩 / 傳輸攔截）        -> 反演誤差 = 0
   T2  只取得匿名影像，需自行 DDIM inversion 還原   -> 有 round-trip 誤差
   攻擊者另持候選影像庫（CelebA-HQ / FFHQ 皆為公開資料）。

 用法
 ----
   python attack_trackA_latent.py --data_dir /path/to/celeba_hq --n 300 --mode all
=============================================================================
"""
import argparse, os, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
#  模型封裝：把 encode / decode / eps 預測抽象出來，換模型只改這個類別
# ═══════════════════════════════════════════════════════════════════════════
class LatentDM:
    """
    包裝一個 latent diffusion model。候選（擇一，先確認可下載）：
      - "CompVis/ldm-celebahq-256"      LDM 訓練於 CelebA-HQ，最貼近論文的人臉 LDM
      - "runwayml/stable-diffusion-v1-5" SD 1.5，f=8 / 4ch latent
    論文用的是「SD 論文的無條件模型、訓練於 FFHQ」。若找不到完全對應的權重，
    改用同族公開模型即可 —— 幅值不變量不依賴特定權重，只需在論文說明。
    """

    def __init__(self, model_id: str, device="cuda", dtype=torch.float32):
        self.device, self.dtype, self.model_id = device, dtype, model_id
        from diffusers import DDIMScheduler

        if "stable-diffusion" in model_id:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype,
                                                           safety_checker=None)
            pipe.to(device)
            self.vae, self.unet = pipe.vae, pipe.unet
            self.sched = DDIMScheduler.from_config(pipe.scheduler.config)
            # 無條件：用空字串的 text embedding 固定住，等效於 unconditional
            with torch.no_grad():
                ids = pipe.tokenizer([""], padding="max_length", truncation=True,
                                     max_length=pipe.tokenizer.model_max_length,
                                     return_tensors="pt").input_ids.to(device)
                self.ctx = pipe.text_encoder(ids)[0]
            self.scale = pipe.vae.config.scaling_factor
            self.cond = lambda b: self.ctx.expand(b, -1, -1)
        else:
            from diffusers import LDMPipeline, VQModel, UNet2DModel
            self.vae = VQModel.from_pretrained(model_id, subfolder="vqvae").to(device)
            self.unet = UNet2DModel.from_pretrained(model_id, subfolder="unet").to(device)
            self.sched = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.scale, self.ctx, self.cond = 1.0, None, lambda b: None

        self.vae.eval(); self.unet.eval()
        self.ac = self.sched.alphas_cumprod.to(device=device, dtype=dtype)

    # ---- VAE ---------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x):                       # x ∈ [-1,1], [B,3,H,W]
        out = self.vae.encode(x.to(self.device, self.dtype))
        z = out.latent_dist.sample() if hasattr(out, "latent_dist") else out.latents
        return z * self.scale

    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / self.scale).sample.clamp(-1, 1)

    # ---- eps 預測 -----------------------------------------------------------
    @torch.no_grad()
    def eps(self, z, t):
        tt = torch.full((z.shape[0],), int(t), device=self.device, dtype=torch.long)
        c = self.cond(z.shape[0])
        out = self.unet(z, tt, encoder_hidden_states=c).sample if c is not None \
              else self.unet(z, tt).sample
        return out[:, :z.shape[1]]             # learn_sigma 時只取前半


# ═══════════════════════════════════════════════════════════════════════════
#  DDIM 確定性前向 / 後向  —— 嚴格照論文 Eq.(4)(5)
# ═══════════════════════════════════════════════════════════════════════════
def _timesteps(n_train, T):
    return torch.linspace(0, n_train - 1, T).long().tolist()


@torch.no_grad()
def ddim_invert(m: LatentDM, z0, T=50, keep_trace=False):
    """Eq.(5)：z0 -> zT。keep_trace=True 時回傳每步的 z_t（Eq.(7) 需要）"""
    ts = _timesteps(len(m.ac), T)
    z, trace = z0.clone(), {}
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        if keep_trace:
            trace[t] = z.clone()
        e = m.eps(z, t)
        z0h = (z - (1 - m.ac[t]).sqrt() * e) / m.ac[t].sqrt()
        z = m.ac[tn].sqrt() * z0h + (1 - m.ac[tn]).sqrt() * e
    if keep_trace:
        trace[ts[-1]] = z.clone()
        return z, trace
    return z


@torch.no_grad()
def ddim_sample(m: LatentDM, zT, T=50, mask=None, trace=None):
    """Eq.(4)。給 mask+trace 時同時執行 Eq.(7) 的非遮罩區重注入。"""
    ts = _timesteps(len(m.ac), T)[::-1]
    z = zT.clone()
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        e = m.eps(z, t)
        z0h = (z - (1 - m.ac[t]).sqrt() * e) / m.ac[t].sqrt()
        z = m.ac[tn].sqrt() * z0h + (1 - m.ac[tn]).sqrt() * e
        if mask is not None and trace is not None and tn in trace:
            z = mask * z + (1 - mask) * trace[tn]      # Eq.(7)
    return z


# ═══════════════════════════════════════════════════════════════════════════
#  加密原語  —— Eq.(3) + Eq.(6)
# ═══════════════════════════════════════════════════════════════════════════
def rademacher(shape, gen, device):
    return (torch.randint(0, 2, shape, generator=gen).float() * 2 - 1).to(device)


def encrypt(z, k, mask=None):
    """Eq.(6)：z_ano = M ⊙ (k ⊙ z) + (1 − M) ⊙ z"""
    return z * k if mask is None else mask * (k * z) + (1 - mask) * z


def face_mask(x, latent_hw, device):
    """
    BiSeNet 臉部遮罩。若環境沒有 face parser，退回中央橢圓近似 ——
    近似版足以重現三個洩漏通道的相對關係，但論文正式數據建議裝 BiSeNet。
    """
    try:
        raise ImportError                       # <<<TODO>>> 接上你的 BiSeNet
    except ImportError:
        h = w = latent_hw
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
                                indexing="ij")
        m = (((xx / 0.62) ** 2 + ((yy + 0.05) / 0.78) ** 2) <= 1).float()
        return m[None, None].to(device)


def cos_sim(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


# ═══════════════════════════════════════════════════════════════════════════
#  資料
# ═══════════════════════════════════════════════════════════════════════════
def load_images(d, n, size=256):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    fs = sorted(p for e in exts for p in Path(d).rglob(e))[:n]
    if not fs:
        raise FileNotFoundError(f"{d} 下找不到影像")
    out = []
    for p in fs:
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        out.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1)
    print(f"[data] {len(out)} 張，來源 {fs[0].suffix}，"
          f"原始尺寸 {Image.open(fs[0]).size}")
    return torch.stack(out), [str(p) for p in fs]


# ═══════════════════════════════════════════════════════════════════════════
#  A. 不變量驗證 + round-trip 誤差   —— 決定攻擊落在哪一欄
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def exp_A(m, imgs, mask, args, out):
    print("\n" + "=" * 74 + "\n A. 不變量驗證與 round-trip 誤差\n" + "=" * 74)
    g = torch.Generator().manual_seed(args.seed)
    rows = []
    for i in range(min(args.n_probe, len(imgs))):
        z0 = m.encode(imgs[i:i + 1])
        zT, tr = ddim_invert(m, z0, args.T, keep_trace=True)
        k = rademacher(zT.shape, g, m.device)
        za = encrypt(zT, k, mask)

        # L1/L2：不變量是否成立（理論上必然，這裡做數值確認）
        l1 = (mask * (za.abs() - zT.abs())).abs().max().item()
        l2 = ((1 - mask) * (za - zT)).abs().max().item()

        # T2：攻擊者路徑
        anon = m.decode(ddim_sample(m, za, args.T, mask, tr))
        za_hat = ddim_invert(m, m.encode(anon), args.T)
        rows.append(dict(idx=i, l1_max=l1, l2_max=l2,
                         cos_T2=float(cos_sim(za_hat, za)),
                         cos_vae=float(cos_sim(m.encode(m.decode(z0)), z0))))
        if i == 0:
            torch.save(dict(z0=z0.cpu(), zT=zT.cpu(), za=za.cpu(), k=k.cpu(),
                            anon=anon.cpu(), za_hat=za_hat.cpu()), out / "sample0.pt")

    a = {kk: np.array([r[kk] for r in rows]) for kk in rows[0] if kk != "idx"}
    print(f"  L1  遮罩內 max||z_ano|−|z||  = {a['l1_max'].max():.3e}   (應為 0 → 幅值完全洩漏)")
    print(f"  L2  遮罩外 max|z_ano−z|      = {a['l2_max'].max():.3e}   (應為 0 → 該區為純明文)")
    print(f"  T2  round-trip cos           = {a['cos_T2'].mean():.5f} "
          f"[min {a['cos_T2'].min():.5f}, p5 {np.percentile(a['cos_T2'],5):.5f}]")
    print(f"  參考 VAE 往返 cos            = {a['cos_vae'].mean():.5f}")
    print("\n  → 把 T2 的 cos 對回 Tier-0 的 E2 表格，即可判定 T2 攻擊可行性。")
    print("    （T1 情境下誤差為 0，攻擊必然成立，不需此數字。）")
    json.dump({k: v.tolist() for k, v in a.items()}, open(out / "expA.json", "w"))
    return a


# ═══════════════════════════════════════════════════════════════════════════
#  B. 幅值洩漏視覺化  ->  論文 Figure 1
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def exp_B(m, imgs, mask, args, out, idx=0):
    print("\n" + "=" * 74 + "\n B. 幅值洩漏視覺化\n" + "=" * 74)
    g = torch.Generator().manual_seed(args.seed)
    z0 = m.encode(imgs[idx:idx + 1])
    zT, tr = ddim_invert(m, z0, args.T, keep_trace=True)
    k = rademacher(zT.shape, g, m.device)
    za = encrypt(zT, k, mask)
    anon = m.decode(ddim_sample(m, za, args.T, mask, tr))

    def gray(t):                                # latent 多通道 -> 可視化單圖
        v = t[0].mean(0).float().cpu()
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    def rgb(t):
        return ((t[0].permute(1, 2, 0).float().cpu() + 1) / 2).clamp(0, 1)

    panels = [(rgb(imgs[idx:idx + 1]), "(a) Original"),
              (rgb(anon),              "(b) Anonymized — what the attacker sees"),
              (gray(za),               "(c) Encrypted latent  z_ano"),
              (gray(za.abs()),         "(d) |z_ano| = |z|   ← THE LEAK"),
              (gray(mask.expand_as(za)), "(e) Face mask M")]
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
    for ax, (im, t) in zip(axs, panels):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t, fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "fig1_magnitude_leak.png", dpi=200); plt.close()
    print("  已存 fig1_magnitude_leak.png")
    print("  → 檢查 (d) 是否看得出五官輪廓。若是，這張圖就是論文 Figure 1。")


# ═══════════════════════════════════════════════════════════════════════════
#  C. 再識別攻擊（CMC）—— 三個洩漏通道分開量
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def exp_C(m, imgs, mask, args, out):
    print("\n" + "=" * 74 + "\n C. 再識別攻擊 (CMC)\n" + "=" * 74)
    g = torch.Generator().manual_seed(args.seed)
    N = len(imgs)
    mflat = mask.flatten().bool()
    gal, qry = {c: [] for c in ("full", "masked", "unmasked")}, \
               {c: [] for c in ("full", "masked", "unmasked")}

    for i in range(N):
        z0 = m.encode(imgs[i:i + 1])
        zT, tr = ddim_invert(m, z0, args.T, keep_trace=True)
        k = rademacher(zT.shape, g, m.device)
        za = encrypt(zT, k, mask)
        if args.threat == "T1":
            zq = za                                       # 直接取得密文
        else:
            anon = m.decode(ddim_sample(m, za, args.T, mask, tr))
            zq = ddim_invert(m, m.encode(anon), args.T)   # 攻擊者自行反演

        gf, qf = zT.flatten().cpu(), zq.flatten().cpu()
        mk = mflat.repeat(zT.shape[1]) if mflat.numel() != gf.numel() else mflat
        mk = mk.cpu()
        gal["full"].append(gf.abs());          qry["full"].append(qf.abs())
        gal["masked"].append(gf[mk].abs());    qry["masked"].append(qf[mk].abs())
        gal["unmasked"].append(gf[~mk]);       qry["unmasked"].append(qf[~mk])   # 明文，不取 abs
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N}")

    res = {}
    for ch in gal:
        G = torch.stack(gal[ch]).float(); Q = torch.stack(qry[ch]).float()
        G = G - G.mean(1, keepdim=True); G = G / (G.norm(dim=1, keepdim=True) + 1e-9)
        Q = Q - Q.mean(1, keepdim=True); Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
        S = Q @ G.T
        rank = (S > S.gather(1, torch.arange(N).unsqueeze(1))).sum(1)
        cmc = [(rank < r).float().mean().item() for r in range(1, 21)]
        res[ch] = cmc
        print(f"  {ch:9s}  top-1 = {cmc[0]:.3f}   top-5 = {cmc[4]:.3f}   "
              f"(隨機 = {1/N:.2e})")

    plt.figure(figsize=(5.4, 4.2))
    lbl = {"full": "全 latent 幅值 (L1+L2)", "masked": "僅遮罩內幅值 (L1)",
           "unmasked": "僅遮罩外明文 (L2)"}
    for ch, c in res.items():
        plt.plot(range(1, 21), c, "o-", ms=3, label=lbl[ch])
    plt.axhline(1 / N, ls="--", c="gray", lw=1, label="random")
    plt.xlabel("rank k"); plt.ylabel("CMC top-k"); plt.ylim(-.03, 1.03)
    plt.title(f"Re-identification, N={N}, threat={args.threat}")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(out / f"fig2_cmc_{args.threat}.png", dpi=200); plt.close()
    json.dump(res, open(out / f"expC_{args.threat}.json", "w"))
    print(f"  已存 fig2_cmc_{args.threat}.png")
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  D. 已知明文攻擊
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def exp_D(m, imgs, mask, args, out):
    print("\n" + "=" * 74 + "\n D. 已知明文攻擊 (KPA)\n" + "=" * 74)
    g = torch.Generator().manual_seed(args.seed)
    acc = {"T1": [], "T2": []}
    mb = mask.bool()
    for i in range(min(args.n_probe, len(imgs))):
        z0 = m.encode(imgs[i:i + 1])
        zT, tr = ddim_invert(m, z0, args.T, keep_trace=True)
        k = rademacher(zT.shape, g, m.device)
        za = encrypt(zT, k, mask)
        sel = mb.expand_as(zT)                                  # 只有遮罩內有金鑰可談
        acc["T1"].append(((torch.sign(za) * torch.sign(zT) == k)[sel]).float().mean().item())
        anon = m.decode(ddim_sample(m, za, args.T, mask, tr))
        zh = ddim_invert(m, m.encode(anon), args.T)
        acc["T2"].append(((torch.sign(zh) * torch.sign(zT) == k)[sel]).float().mean().item())

    for t in acc:
        a = np.array(acc[t])
        print(f"  {t}  金鑰位元正確率 = {a.mean():.4%}  ±{a.std():.4%}   (隨機 = 50%)")
    plt.figure(figsize=(4.6, 4))
    plt.boxplot([acc["T1"], acc["T2"]], labels=["T1: z_ano leaked", "T2: via inversion"])
    plt.axhline(.5, ls="--", c="gray", label="random"); plt.ylabel("key-bit recovery")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(out / "fig3_kpa.png", dpi=200); plt.close()
    json.dump(acc, open(out / "expD.json", "w"))
    print("  已存 fig3_kpa.png")
    return acc


# ═══════════════════════════════════════════════════════════════════════════
#  E. 獨立隱私度量：對照他們自己的 Table 1
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def exp_E(m, imgs, mask, args, out):
    print("\n" + "=" * 74 + "\n E. 與其匿名化度量的對照\n" + "=" * 74)
    try:
        from facenet_pytorch import InceptionResnetV1
        net = InceptionResnetV1(pretrained="vggface2").eval().to(m.device)
    except ImportError:
        print("  未安裝 facenet-pytorch，略過。 pip install facenet-pytorch")
        return None
    g = torch.Generator().manual_seed(args.seed)
    sims = []
    for i in range(min(args.n_probe, len(imgs))):
        x = imgs[i:i + 1].to(m.device)
        z0 = m.encode(x); zT, tr = ddim_invert(m, z0, args.T, keep_trace=True)
        k = rademacher(zT.shape, g, m.device)
        anon = m.decode(ddim_sample(m, encrypt(zT, k, mask), args.T, mask, tr))
        e = net(F.interpolate(torch.cat([x, anon]), 160, mode="bilinear"))
        sims.append(float(F.cosine_similarity(e[0:1], e[1:2])))
    s = np.array(sims)
    print(f"  FaceNet cos(原圖, 匿名影像) = {s.mean():.4f} ± {s.std():.4f}")
    print(f"  論文 Table 1 自報          = 0.0755 ± 0.1610")
    print("\n  → 重點不是我們是否重現這個數字，而是：這個度量衡量的是「渲染出來的臉」，")
    print("    而 exp_C 的攻擊完全不碰渲染影像。該度量在設計上就對潛空間攻擊失明。")
    json.dump(sims, open(out / "expE.json", "w"))
    return s


# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_id", default="CompVis/ldm-celebahq-256")
    ap.add_argument("--out", default="attack_out_trackA")
    ap.add_argument("--n", type=int, default=300, help="gallery 大小")
    ap.add_argument("--n_probe", type=int, default=20, help="A/D/E 的抽樣數")
    ap.add_argument("--T", type=int, default=50, help="論文設定 T=50")
    ap.add_argument("--threat", choices=["T1", "T2"], default="T2")
    ap.add_argument("--no_mask", action="store_true", help="關閉遮罩（純 Rademacher 對照）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1911)
    ap.add_argument("--mode", default="all",
                    help="all 或以逗號指定，如 A,B,C")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[init] model = {args.model_id}   device = {dev}   T = {args.T}")
    m = LatentDM(args.model_id, device=dev)
    imgs, paths = load_images(args.data_dir, args.n, args.size)

    z_probe = m.encode(imgs[:1])
    print(f"[init] latent shape = {tuple(z_probe.shape[1:])}   d = {z_probe[0].numel():,}")
    mask = None if args.no_mask else face_mask(imgs[:1], z_probe.shape[-1], dev)
    if mask is not None:
        print(f"[init] 遮罩覆蓋率 = {mask.mean().item():.1%} "
              f"（其餘 {1-mask.mean().item():.1%} 為純明文）")

    todo = ["A", "B", "C", "D", "E"] if args.mode == "all" else args.mode.split(",")
    if "A" in todo: exp_A(m, imgs, mask if mask is not None else torch.ones_like(z_probe[:, :1]), args, out)
    if "B" in todo: exp_B(m, imgs, mask if mask is not None else torch.ones_like(z_probe[:, :1]), args, out)
    if "C" in todo: exp_C(m, imgs, mask if mask is not None else torch.ones_like(z_probe[:, :1]), args, out)
    if "D" in todo: exp_D(m, imgs, mask if mask is not None else torch.ones_like(z_probe[:, :1]), args, out)
    if "E" in todo: exp_E(m, imgs, mask if mask is not None else torch.ones_like(z_probe[:, :1]), args, out)

    print("\n" + "=" * 74)
    print(f" 完成。輸出在 {out}/")
    print(" 建議：先跑 --mode A,B --n 20 確認 pipeline 正確，再開 --mode C --n 300。")
    print("=" * 74)


if __name__ == "__main__":
    main()
