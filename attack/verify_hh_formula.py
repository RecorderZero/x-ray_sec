#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Householder 洩漏的閉式公式 + 通用設計準則
========================================
主張 1：m 個隨機 Householder 反射的乘積滿足
            E[cos(Hz,z)] = E[tr(H)]/d ≈ exp(-2m/d)
        -> m = d 時 cos ≈ 0.135，仍不安全；需 m ≈ (d ln d)/2 ≈ 4d
主張 2：金鑰導出正交族 {Q_k} 抗相關性攻擊 <=> E_k[tr(Q_k)^2]/d^2 -> 0

執行：python verify_hh_formula.py       （只需 numpy，約 2 分鐘）
"""
import numpy as np
rng = np.random.default_rng(1911)
D = 4096

def hh_apply(z, V):
    out = z.copy()
    for v in V:
        out = out - 2 * v * (v @ out)
    return out

def make_V(m, d, rg):
    V = rg.standard_normal((m, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def fwht(a):
    a = a.copy(); n = len(a); h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i+h].copy(); y = a[i+h:i+2*h].copy()
            a[i:i+h] = x + y; a[i+h:i+2*h] = x - y
        h *= 2
    return a

def sot_keys(R, d, rg):
    return [(rg.choice([-1., 1.], d), rg.permutation(d), rg.choice([-1., 1.], d))
            for _ in range(R)]

def sot(z, keys, d):
    out = z.astype(np.float64)
    for k1, pi, k2 in keys:
        out = fwht(out * k1) / np.sqrt(d)
        out = out[pi] * k2
    return out

def trace_over_d(apply_fn, d, n_probe=20):
    """Hutchinson estimator: tr(Q)/d = E_g[g^T Q g]/d, g ~ Rademacher"""
    est = [rng.choice([-1., 1.], d) @ apply_fn(rng.choice([-1., 1.], d)) for _ in range(0)]
    vals = []
    for _ in range(n_probe):
        g = rng.choice([-1., 1.], d)
        vals.append(g @ apply_fn(g) / d)
    return float(np.mean(vals))

print("=" * 74)
print(" 1. Householder: cos(Hz,z) vs 預測 exp(-2m/d)      d = %d" % D)
print("=" * 74)
print(f"{'m':>8} {'m/d':>7} {'實測cos':>10} {'exp(-2m/d)':>12} {'tr(H)/d':>10}")
for m in [64, 256, 1024, 2048, 4096, 8192, 16384]:
    cs, trs = [], []
    for _ in range(5):
        V = make_V(m, D, rng)
        z = rng.standard_normal(D)
        cs.append(hh_apply(z, V) @ z / (np.linalg.norm(z) ** 2))
        trs.append(trace_over_d(lambda x: hh_apply(x, V), D))
    tag = "  <-- m = d" if m == D else ""
    print(f"{m:>8} {m/D:>7.2f} {np.mean(cs):>10.4f} {np.exp(-2*m/D):>12.4f} "
          f"{np.mean(trs):>10.4f}{tag}")

print("\n" + "=" * 74)
print(" 2. 設計準則 |tr(Q)/d|：SOT-WHT vs Householder vs Haar")
print("=" * 74)
for R in [1, 2, 3]:
    cs, trs = [], []
    for _ in range(5):
        keys = sot_keys(R, D, rng)
        z = rng.standard_normal(D)
        cs.append(sot(z, keys, D) @ z / (np.linalg.norm(z) ** 2))
        trs.append(trace_over_d(lambda x: sot(x, keys, D), D))
    print(f"  SOT-WHT R={R}   cos={np.mean(cs):>9.5f}   |tr/d|={abs(np.mean(trs)):>9.5f}")
print(f"  Haar 理論值                        |tr/d| ~ 1/d = {1/D:.5f}")
print("\n  判定：|tr/d| 是「密文洩漏多少明文方向」的直接量度。")
print("  Householder 的 |tr/d| = exp(-2m/d)，要 -> 1/d 需 m ≈ (d ln d)/2 ≈ 4d，")
print("  此時儲存 %.0f MB，比完整 Q 的 %.0f MB 還糟。" % (4*D*D*4/1e6, D*D*4/1e6))
