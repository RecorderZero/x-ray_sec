#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Tier-0 PoC : Cryptanalysis of self-inverse latent-space encryption
              (Rademacher sign-flip / Signed Permutation)
=============================================================================
 目標：在「不需要訓練好的 diffusion model」的前提下，先把三件事釘死
   E1  數學不變量驗證     -- 加密到底洩漏了什麼？(可解析證明，這裡做數值確認)
   E2  Gallery 比對攻擊   -- 洩漏的資訊夠不夠做再識別？需要多小的反演誤差？
   E3  已知明文攻擊 (KPA) -- 拿到一組明密文對，能不能直接把金鑰挖出來？
   E4  排列後的直方圖攻擊 -- Signed Permutation 打亂了位置，還剩什麼？

 威脅模型 (threat model)
 -----------------------
   Kerckhoffs 原則：攻擊者知道演算法、知道公開的 diffusion model 權重，
   但不知道金鑰 k / 置換 pi。攻擊者能取得下列之一：
     (T1) 加密噪聲 Z_enc 本身
          -> 原論文 6.2 節的部署建議就是「只永久儲存/傳輸加密噪聲」，
             所以這是作者自己描述的部署情境，誤差 = 0。
     (T2) 匿名影像 (由 Z_enc 無條件反擴散生成)
          -> 攻擊者需自行 DDIM inversion 還原 Ẑ_enc，會引入 round-trip 誤差。
             本腳本用 cosine similarity 參數化這個誤差。
   攻擊者另持有一個候選影像庫 (gallery)，例如某院的公開資料集，
   目標：判斷某張加密影像是否來自庫中某人 (membership / re-identification)。
=============================================================================
"""
import numpy as np

rng = np.random.default_rng(1911)          # 沿用原論文的 random seed

D          = 256 * 256                      # latent 維度 (pixel-space 256x256, 1 channel)
N_GALLERY  = 200                            # 候選影像庫大小
COS_GRID   = [1.0, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.97, 0.95, 0.90]
RHO_GRID   = [0.0, 0.5, 0.90, 0.97]         # gallery 內的共同結構比例 (見下)

# ---------------------------------------------------------------------------
# latent 模型
#   真實的 DDIM-inverted latent 不是 iid N(0,I)：同院同機器的 CXR 反演後會
#   共享大量解剖/成像結構。用 rho 參數化「所有影像共享的結構比例」：
#       Z_i = sqrt(rho)*S + sqrt(1-rho)*U_i ,  S,U_i ~ N(0,I)
#   rho=0    -> 完全獨立 (對攻擊者最有利，樂觀上界)
#   rho=0.97 -> 影像之間高度相似 (對攻擊者最不利，悲觀下界)
#   corr(Z_i, Z_j) = rho
# ---------------------------------------------------------------------------
def make_gallery(n, d, rho):
    S = rng.standard_normal(d)
    U = rng.standard_normal((n, d))
    return np.sqrt(rho) * S[None, :] + np.sqrt(1.0 - rho) * U


# ---------------------------------------------------------------------------
# 加密原語 (完全照論文 4.3 節)
# ---------------------------------------------------------------------------
def rademacher_key(d):
    return rng.choice([-1.0, 1.0], size=d)

def enc_rademacher(Z, k):
    return k * Z                                    # Z_enc = k ⊙ Z

def signed_perm_key(d):
    return rng.choice([-1.0, 1.0], size=d), rng.permutation(d)

def enc_signed_perm(Z, k, pi):
    return (k * Z)[pi]                              # Z_enc = pi · (k ⊙ Z)


# ---------------------------------------------------------------------------
# round-trip 誤差模型
#   攻擊者從匿名影像做 DDIM inversion 拿回 Ẑ_enc，與真 Z_enc 的 cosine
#   similarity 為 target_cos。原論文報告的 latent cosine sim ≈ 0.9990~0.9987。
# ---------------------------------------------------------------------------
def add_roundtrip_error(Z, target_cos):
    if target_cos >= 1.0:
        return Z.copy()
    n = rng.standard_normal(Z.shape)
    n -= (n @ Z) / (Z @ Z) * Z                      # 取與 Z 正交的分量
    n /= np.linalg.norm(n)
    tan = np.sqrt(1 - target_cos**2) / target_cos
    return Z + tan * np.linalg.norm(Z) * n


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ===========================================================================
# E1  數學不變量驗證
# ===========================================================================
def E1():
    print("=" * 78)
    print("E1  數學不變量驗證 (What does the ciphertext leak?)")
    print("=" * 78)
    Z = rng.standard_normal(D)

    k = rademacher_key(D)
    Ze = enc_rademacher(Z, k)
    same_elem = np.array_equal(np.abs(Ze), np.abs(Z))
    print(f"  Rademacher      : |Z_enc| == |Z|  逐元素完全相同 ?  -> {same_elem}")
    print(f"                    max |  |Z_enc| - |Z|  | = {np.abs(np.abs(Ze)-np.abs(Z)).max():.3e}")
    print("                    => 幅值圖 |Z| 原封不動外洩。加密只動了「符號」這 1 bit/pixel，")
    print("                       其餘所有資訊 (float32 的 31 bits/pixel) 完全沒有被保護。")

    k2, pi = signed_perm_key(D)
    Ze2 = enc_signed_perm(Z, k2, pi)
    same_sorted = np.allclose(np.sort(np.abs(Ze2)), np.sort(np.abs(Z)))
    print(f"\n  SignedPerm      : sort(|Z_enc|) == sort(|Z|) ?     -> {same_sorted}")
    print(f"                    逐元素相同 ? -> {np.array_equal(np.abs(Ze2), np.abs(Z))}")
    print("                    => 位置被打散，但幅值的 multiset (直方圖) 完全保留。")

    print("\n  [結論] 兩種方案都不是保密的『置換-代換』密碼，而是 norm-preserving 的")
    print("         符號/位置重排。任何對 |Z| 或其分布的統計，都是明文可見的。")


# ===========================================================================
# E2  Gallery 比對攻擊 (Rademacher)  —— 主攻擊
# ===========================================================================
def E2():
    print("\n" + "=" * 78)
    print("E2  Gallery 比對再識別攻擊 (Rademacher)  score = corr(|Ẑ_enc|, |Z_j|)")
    print("=" * 78)
    print(f"    gallery = {N_GALLERY} 張,  d = {D},  指標 = top-1 命中率 (隨機 = {1/N_GALLERY:.3f})")
    print()
    header = "  rho \\ cos  " + "".join(f"{c:>9.4g}" for c in COS_GRID)
    print(header); print("  " + "-" * (len(header) - 2))

    for rho in RHO_GRID:
        G = make_gallery(N_GALLERY, D, rho)
        absG = np.abs(G)
        absG = absG - absG.mean(axis=1, keepdims=True)
        absG /= np.linalg.norm(absG, axis=1, keepdims=True)
        row = []
        for c in COS_GRID:
            hit = 0
            for t in range(0, N_GALLERY, 4):        # 抽 50 個 target 測試
                k  = rademacher_key(D)
                Ze = enc_rademacher(G[t], k)
                Zh = add_roundtrip_error(Ze, c)
                q  = np.abs(Zh); q = q - q.mean(); q /= np.linalg.norm(q)
                if int(np.argmax(absG @ q)) == t:
                    hit += 1
            row.append(hit / len(range(0, N_GALLERY, 4)))
        print(f"  {rho:>6.2f}    " + "".join(f"{v:>9.2f}" for v in row))

    print("\n  rho = gallery 內影像的相似度 (0=完全獨立, 0.97=高度相似)")
    print("  cos = 攻擊者反演 Ẑ_enc 與真 Z_enc 的 cosine similarity")
    print("  註：原論文自己報告的 latent cosine similarity ≈ 0.9990，")
    print("      即攻擊者只要能複製作者的反演品質，就落在最左邊那幾欄。")


# ===========================================================================
# E3  已知明文攻擊 (KPA) — 直接把金鑰挖出來
# ===========================================================================
def E3():
    print("\n" + "=" * 78)
    print("E3  已知明文攻擊 (Known-Plaintext Attack)：一組 (Z, Z_enc) 即可回推金鑰")
    print("=" * 78)

    print("\n  [E3-a] Rademacher :  k̂ = sign(Z_enc) * sign(Z)")
    print(f"  {'cos(Ẑ_enc, Z_enc)':>20} | {'金鑰位元正確率':>16} | 說明")
    print("  " + "-" * 62)
    for c in COS_GRID:
        Z  = rng.standard_normal(D)
        k  = rademacher_key(D)
        Ze = enc_rademacher(Z, k)
        Zh = add_roundtrip_error(Ze, c)
        kh = np.sign(Zh) * np.sign(Z)
        acc = float((kh == k).mean())
        note = "完全破解" if acc > 0.999 else ("實質破解" if acc > 0.95 else "部分洩漏")
        print(f"  {c:>20.4g} | {acc:>15.4%}  | {note}")

    print("\n  [E3-b] Signed Permutation :  以幅值排序匹配回推 pi，再回推 k")
    print(f"  {'cos':>20} | {'置換 pi 復原率':>16} | {'金鑰 k 復原率':>15}")
    print("  " + "-" * 62)
    d_small = 4096                                   # 排序匹配 O(d log d)，用小維度示範即可
    for c in COS_GRID:
        Z = rng.standard_normal(d_small)
        k = rng.choice([-1.0, 1.0], size=d_small)
        pi = rng.permutation(d_small)
        Ze = (k * Z)[pi]
        Zh = add_roundtrip_error(Ze, c)
        # 幾乎必然所有 |Z_i| 互異 -> 以幅值大小排序即可唯一配對
        order_src = np.argsort(np.abs(Z))            # 明文中幅值第 r 小的位置
        order_dst = np.argsort(np.abs(Zh))           # 密文中幅值第 r 小的位置
        pi_hat = np.empty(d_small, dtype=int)
        pi_hat[order_dst] = order_src                # pi_hat[j] = 密文位置 j 的來源索引
        pi_acc = float((pi_hat == pi).mean())
        k_hat = np.sign(Zh) * np.sign(Z[pi_hat])     # 逐密文位置回推符號
        k_est = np.empty(d_small); k_est[pi_hat] = k_hat
        k_acc = float((k_est == k).mean())
        print(f"  {c:>20.4g} | {pi_acc:>15.4%}  | {k_acc:>14.4%}")

    print("\n  [結論] 兩種方案在 KPA 下都不安全。金鑰一旦重用於多張影像，")
    print("         單一組明密文對即導致該金鑰下所有影像全面失守。")
    print("         論文宣稱的金鑰空間 2^d 與 2^d·d! 只是暴力破解上界，與實際安全性無關。")


# ===========================================================================
# E4  幅值直方圖攻擊 (Signed Permutation)
# ===========================================================================
def E4():
    print("\n" + "=" * 78)
    print("E4  幅值直方圖攻擊 (Signed Permutation)  score = corr(sort|Ẑ_enc|, sort|Z_j|)")
    print("=" * 78)
    print(f"    位置被打散，僅剩排序後的幅值曲線可用。top-1 命中率 (隨機 = {1/N_GALLERY:.3f})")
    print()
    header = "  rho \\ cos  " + "".join(f"{c:>9.4g}" for c in COS_GRID)
    print(header); print("  " + "-" * (len(header) - 2))

    for rho in RHO_GRID:
        G = make_gallery(N_GALLERY, D, rho)
        sG = np.sort(np.abs(G), axis=1)
        sG = sG - sG.mean(axis=1, keepdims=True)
        sG /= np.linalg.norm(sG, axis=1, keepdims=True)
        row = []
        for c in COS_GRID:
            hit = 0
            for t in range(0, N_GALLERY, 4):
                k, pi = signed_perm_key(D)
                Ze = enc_signed_perm(G[t], k, pi)
                Zh = add_roundtrip_error(Ze, c)
                q = np.sort(np.abs(Zh)); q = q - q.mean(); q /= np.linalg.norm(q)
                if int(np.argmax(sG @ q)) == t:
                    hit += 1
            row.append(hit / len(range(0, N_GALLERY, 4)))
        print(f"  {rho:>6.2f}    " + "".join(f"{v:>9.2f}" for v in row))

    print("\n  對照 E2：Signed Permutation 在『無明文』情境下確實比 Rademacher 安全得多，")
    print("  但論文選的最佳組合 (最高 SSIM 0.912) 正是 Rademacher —— ")
    print("  也就是說，論文推薦的高保真設定，剛好是安全性最差的那一個。")


if __name__ == "__main__":
    E1(); E2(); E3(); E4()
    print("\n" + "=" * 78)
    print("Tier-0 完成。下一步 -> attack_tier1_real.py (需要前作的 U-Net checkpoint)")
    print("=" * 78)
