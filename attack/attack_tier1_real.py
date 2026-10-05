#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Tier-1 : 真實攻擊流程 (需要前作的 CFG-DDIM U-Net checkpoint)
=============================================================================
 Tier-0 已證明：加密洩漏幅值不變量，且在低反演誤差下可完成再識別。
 Tier-1 要回答的唯一問題：
     「真實 DDIM inversion 的 round-trip 誤差，是否落在攻擊可行的區間？」

 你只需要把 <<<TODO>>> 的四個接口接上學長的程式碼，其他都寫好了。
 產出三張論文用圖：
     fig1_magnitude_leak.png   -- |Z| 幅值圖 vs 原圖 vs 匿名影像 (視覺衝擊，Figure 1)
     fig2_reid_roc.png         -- 再識別 ROC / CMC 曲線
     fig3_kpa_keyrecovery.png  -- 已知明文攻擊的金鑰復原率
=============================================================================
"""
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE   = "cuda"
OUT      = Path("attack_out"); OUT.mkdir(exist_ok=True)
N_IMAGES = 200          # gallery 大小，先用 50 跑通再放大
L_STEPS  = 500          # 與原論文一致的 inversion 步數
SEED     = 1911
torch.manual_seed(SEED); np.random.seed(SEED)


# ===========================================================================
# <<<TODO 1>>>  接上前作的模型與擴散排程
# ===========================================================================
def load_model():
    """回傳 (unet, alphas_cumprod)。unet(x_t, t, y=None) -> eps 預測。"""
    raise NotImplementedError("接學長的 build_model() / load_state_dict()")


# ===========================================================================
# <<<TODO 2>>>  接上資料載入 (CheXpert 篩選後的 split，seed=1911)
# ===========================================================================
def load_images(n):
    """回傳 tensor [n,1,256,256]，值域 [-1,1]，已做 CLAHE + area resize。"""
    raise NotImplementedError("接學長的 dataset / dataloader")


# ===========================================================================
# DDIM inversion / sampling —— 照論文 4.2.2 與 4.4.3，全部用 unconditional
# ===========================================================================
@torch.no_grad()
def ddim_invert(unet, x0, ac, steps=L_STEPS):
    """x0 -> Z_T (deterministic, unconditional eps_∅)"""
    ts = torch.linspace(0, len(ac) - 1, steps).long()
    x = x0
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        eps = unet(x, t.expand(x.size(0)).to(DEVICE), None)
        x0h = (x - (1 - ac[t]).sqrt() * eps) / ac[t].sqrt()
        x = ac[tn].sqrt() * x0h + (1 - ac[tn]).sqrt() * eps
    return x


@torch.no_grad()
def ddim_sample(unet, xT, ac, steps=L_STEPS, y=None, w=0.0):
    """Z_T -> x0。w=0 即論文的匿名化路徑 (unconditional)。"""
    ts = torch.linspace(0, len(ac) - 1, steps).long().flip(0)
    x = xT
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        tt = t.expand(x.size(0)).to(DEVICE)
        if y is None or w == 0.0:
            eps = unet(x, tt, None)
        else:
            e_c, e_u = unet(x, tt, y), unet(x, tt, None)
            eps = (1 + w) * e_c - w * e_u
        x0h = (x - (1 - ac[t]).sqrt() * eps) / ac[t].sqrt()
        x = ac[tn].sqrt() * x0h + (1 - ac[tn]).sqrt() * eps
    return x


# ===========================================================================
# 加密原語 (與 Tier-0 相同，照論文 4.3)
# ===========================================================================
def rademacher_key(shape, g):  return (torch.randint(0, 2, shape, generator=g).float() * 2 - 1)
def enc_rad(Z, k):             return k * Z
def signed_perm_key(numel, g): return (torch.randint(0,2,(numel,),generator=g).float()*2-1), torch.randperm(numel, generator=g)
def enc_sp(Z, k, pi):          return (k * Z.flatten())[pi].view_as(Z)


def cos_sim(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


# ===========================================================================
# 實驗 A：round-trip 誤差量測  (決定攻擊是否可行的關鍵數字)
# ===========================================================================
@torch.no_grad()
def exp_A_roundtrip(unet, ac, imgs):
    """
    量測威脅模型 T2 下攻擊者實際能拿到的反演品質：
        x0 -> Z -> 加密 -> 匿名影像 -> (攻擊者) 反演 -> Ẑ_enc
    然後比對 Ẑ_enc 與真 Z_enc 的 cosine similarity。
    把這個數字對回 Tier-0 的 E2/E3 表格，就知道攻擊落在哪一欄。
    """
    g = torch.Generator().manual_seed(SEED)
    rows = []
    for i in range(len(imgs)):
        x0 = imgs[i:i+1].to(DEVICE)
        Z  = ddim_invert(unet, x0, ac)
        k  = rademacher_key(Z.shape, g).to(DEVICE)
        Ze = enc_rad(Z, k)
        anon = ddim_sample(unet, Ze, ac, w=0.0)          # 匿名影像 (公開傳輸的東西)
        Ze_hat = ddim_invert(unet, anon, ac)             # 攻擊者自行反演
        rows.append(float(cos_sim(Ze_hat, Ze)))
    rows = np.array(rows)
    print(f"[A] round-trip cos(Ẑ_enc, Z_enc): mean={rows.mean():.5f} "
          f"min={rows.min():.5f} p5={np.percentile(rows,5):.5f}")
    np.save(OUT / "roundtrip_cos.npy", rows)
    return rows


# ===========================================================================
# 實驗 B：幅值洩漏視覺化  -> Figure 1
# ===========================================================================
@torch.no_grad()
def exp_B_visual(unet, ac, imgs, idx=0):
    g = torch.Generator().manual_seed(SEED)
    x0 = imgs[idx:idx+1].to(DEVICE)
    Z  = ddim_invert(unet, x0, ac)
    k  = rademacher_key(Z.shape, g).to(DEVICE)
    Ze = enc_rad(Z, k)
    anon = ddim_sample(unet, Ze, ac, w=0.0)

    panels = [
        (x0[0,0].cpu(),              "(a) Original CXR"),
        (anon[0,0].cpu(),            "(b) Anonymized (what the attacker sees)"),
        (Ze[0,0].cpu(),              "(c) Encrypted latent  Z_enc"),
        (Ze[0,0].abs().cpu(),        "(d) |Z_enc|  == |Z|  ← THE LEAK"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (im, t) in zip(axs, panels):
        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "fig1_magnitude_leak.png", dpi=200)
    print("[B] saved fig1_magnitude_leak.png  <- 檢查 (d) 是否看得出解剖結構")


# ===========================================================================
# 實驗 C：再識別攻擊 (CMC / top-k)  -> Figure 2
# ===========================================================================
@torch.no_grad()
def exp_C_reid(unet, ac, imgs, scheme="rademacher"):
    g = torch.Generator().manual_seed(SEED)
    gallery, queries = [], []
    for i in range(len(imgs)):
        x0 = imgs[i:i+1].to(DEVICE)
        Z  = ddim_invert(unet, x0, ac)
        gallery.append(Z.flatten().abs().cpu())
        if scheme == "rademacher":
            k = rademacher_key(Z.shape, g).to(DEVICE); Ze = enc_rad(Z, k)
        else:
            k, pi = signed_perm_key(Z.numel(), g)
            Ze = enc_sp(Z, k.to(DEVICE), pi.to(DEVICE))
        anon   = ddim_sample(unet, Ze, ac, w=0.0)
        Ze_hat = ddim_invert(unet, anon, ac)
        q = Ze_hat.flatten().abs().cpu()
        queries.append(torch.sort(q).values if scheme != "rademacher" else q)

    G = torch.stack([torch.sort(x).values if scheme != "rademacher" else x for x in gallery])
    Q = torch.stack(queries)
    G = (G - G.mean(1, keepdim=True)); G /= G.norm(dim=1, keepdim=True)
    Q = (Q - Q.mean(1, keepdim=True)); Q /= Q.norm(dim=1, keepdim=True)
    S = Q @ G.T                                             # [n_query, n_gallery]
    rank = (S > S.gather(1, torch.arange(len(Q)).unsqueeze(1))).sum(1)
    cmc = [(rank < k).float().mean().item() for k in range(1, 21)]
    print(f"[C:{scheme}] top-1={cmc[0]:.3f}  top-5={cmc[4]:.3f}  "
          f"(random top-1 = {1/len(G):.4f})")

    plt.figure(figsize=(5, 4))
    plt.plot(range(1, 21), cmc, "o-", label=scheme)
    plt.axhline(1/len(G), ls="--", c="gray", label="random")
    plt.xlabel("rank k"); plt.ylabel("CMC top-k accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / f"fig2_reid_{scheme}.png", dpi=200)
    return cmc


# ===========================================================================
# 實驗 D：已知明文攻擊 (真實反演誤差下)  -> Figure 3
# ===========================================================================
@torch.no_grad()
def exp_D_kpa(unet, ac, imgs, n_pairs=20):
    g = torch.Generator().manual_seed(SEED)
    accs_T1, accs_T2 = [], []
    for i in range(n_pairs):
        x0 = imgs[i:i+1].to(DEVICE)
        Z  = ddim_invert(unet, x0, ac)
        k  = rademacher_key(Z.shape, g).to(DEVICE)
        Ze = enc_rad(Z, k)
        # T1: 攻擊者直接拿到 Z_enc (= 論文 6.2 節自己建議的部署方式)
        accs_T1.append(((torch.sign(Ze) * torch.sign(Z)) == k).float().mean().item())
        # T2: 攻擊者只有匿名影像，需自行反演
        anon = ddim_sample(unet, Ze, ac, w=0.0)
        Ze_h = ddim_invert(unet, anon, ac)
        accs_T2.append(((torch.sign(Ze_h) * torch.sign(Z)) == k).float().mean().item())
    print(f"[D] KPA 金鑰位元正確率  T1(直接拿到Z_enc)={np.mean(accs_T1):.4%}  "
          f"T2(需反演)={np.mean(accs_T2):.4%}")

    plt.figure(figsize=(5, 4))
    plt.boxplot([accs_T1, accs_T2], labels=["T1: Z_enc leaked", "T2: via inversion"])
    plt.axhline(0.5, ls="--", c="gray", label="random guess")
    plt.ylabel("key-bit recovery rate"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig3_kpa.png", dpi=200)


if __name__ == "__main__":
    unet, ac = load_model()
    unet.eval().to(DEVICE); ac = ac.to(DEVICE)
    imgs = load_images(N_IMAGES)

    exp_A_roundtrip(unet, ac, imgs[:50])
    exp_B_visual(unet, ac, imgs)
    exp_C_reid(unet, ac, imgs, "rademacher")
    exp_C_reid(unet, ac, imgs, "signed_perm")
    exp_D_kpa(unet, ac, imgs)
    print("\n完成。把 [A] 的 cos 數字對回 Tier-0 的 E2/E3 表格，即可判定攻擊可行性。")
