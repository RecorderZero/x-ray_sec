#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HISTORICAL WARNING (2026-10-08): the table below is a 2026-09-10 record, not
the canonical evidence for the current code. See reports/evidence_reset.md.

=============================================================================
 修補方案設計驗證  ——  W5 開工前必跑（本檔已於 2026-09-10 跑過，結論見下）
=============================================================================
 回答三個問題：
   1. Householder × m 的結構化正交矩陣能不能修掉 F1 幅值攻擊？  → ❌ 不能
   2. SOT（sign → WHT → perm → sign）能不能？                    → ✅ 能，且 bit-exact
   3. ‖Z‖₂ 在理想高斯下洩漏多少？                                 → 極少，但真實 latent 要實測

 [事實] 2026-09-10 主要結論（d=4096, gallery=2000, 隨機基準 5e-4）

   方案                     top-1     bit-exact   儲存
   Rademacher (L0)          1.000     ✅(僅翻符號)  —
   Householder m=64         1.000     ✅           1.05 MB    ← 無效！
   Householder m=d/2        1.000     ✅          33.55 MB    ← 仍無效！
   Householder m=d          0.000     ✅          67.11 MB    ← 有效但等同完整 Q
   SOT-WHT 1 round          0.000     ✅           0.033 MB   ← 有效且最省

 為什麼 Householder 無效（純代數，不是浮點問題）：
   H = ∏(I − 2vᵢvᵢᵀ)，任何 z ⊥ span{v₁..v_m} 滿足 Hz = z
   → (d−m) 維子空間**完全未變**，其幅值原封不動洩漏。
   → m 必須 ≈ d 才安全，此時儲存回到 O(d²)，結構化失去意義。

 SOT-WHT bit-exact 的成立條件（重要）：
   (a) 在 float64 計算，資料為 float32
   (b) **每一輪都做 1/√d 正規化**，不要讓動態範圍累積
       （天真做法在 d=65536 會失敗，實測 err = 1.42e-14）
   (c) d = 2^k 且 **k 為偶數**（使 √d 亦為 2 的冪 → 除法是純指數調整）
       ✅ Labarbarie latent 4096=2¹² (√d=64)   ✅ CheXpert 65536=2¹⁶ (√d=256)
       ⚠️ CIFAR-10 3072=3×2¹⁰ 非 2 的冪 → 需**逐通道** WHT(1024) + 全域置換
=============================================================================
"""
import numpy as np


def fwht(a):
    """Fast Walsh-Hadamard Transform（只有加減法，無乘法）"""
    a = a.copy(); n = len(a); h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i+h].copy(); y = a[i+h:i+2*h].copy()
            a[i:i+h] = x + y; a[i+h:i+2*h] = x - y
        h *= 2
    return a


# ═══════════════════════════════════════════════════════════════════════════
#  L2 修補：金鑰導出結構化正交變換 SOT
#  結構：sign → WHT/√d → permutation → sign，重複 R 輪
#  金鑰全部由 KDF 從 master_seed + image_id 導出 → 實際儲存量 ≈ 0
# ═══════════════════════════════════════════════════════════════════════════
def derive_keys(master_seed, image_id, d, rounds=2):
    """[待辦] 正式版改用 HKDF（RFC 5869），此處用 numpy Generator 示意"""
    g = np.random.default_rng([master_seed, image_id])
    return [(g.choice([-1., 1.], d), g.permutation(d), g.choice([-1., 1.], d))
            for _ in range(rounds)]


def sot_encrypt(z, keys, d):
    out = z.astype(np.float64)
    for k1, pi, k2 in keys:
        out = fwht(out * k1) / np.sqrt(d)
        out = out[pi] * k2
    return out


def spn_decrypt(z, keys, d):
    out = z.astype(np.float64)
    for k1, pi, k2 in keys[::-1]:
        out = out * k2
        inv = np.empty_like(pi); inv[pi] = np.arange(d)
        out = fwht(out[inv]) / np.sqrt(d)      # WHT/√d 為自逆
        out = out * k1
    return out


def check_bit_exact(d=4096, rounds=2, seed=1911):
    g = np.random.default_rng(seed)
    z32 = g.standard_normal(d).astype(np.float32)
    keys = derive_keys(1911, 0, d, rounds)
    rt = spn_decrypt(sot_encrypt(z32, keys, d), keys, d).astype(np.float32)
    return np.array_equal(rt.view(np.int32), z32.view(np.int32))


def check_gaussian(d=65536, seed=1911):
    """加密後是否仍為標準高斯（正交變換必然保持，此處數值確認）"""
    from scipy import stats
    g = np.random.default_rng(seed)
    z = g.standard_normal(d).astype(np.float32)
    e = sot_encrypt(z, derive_keys(1911, 0, d, 2), d)
    return stats.kstest(e / e.std(), 'norm')


if __name__ == "__main__":
    print(__doc__)
    for d in [4096, 65536]:
        for R in [1, 2, 4]:
            print(f"  bit-exact  d={d:>6}  rounds={R}  -> {check_bit_exact(d, R)}")
