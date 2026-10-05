#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 / CPA 分析：線性加密在選擇明文攻擊下的表現
=============================================
結論：固定金鑰時，d 次選擇明文查詢即可逐欄重建任意線性 Q。
      此性質對「完整 Haar Q / Householder / SOT-WHT」一律成立 ——
      不是本方案的缺陷，是線性映射的定義。
      擋在 CPA 與全面破解之間的唯一機制是 L3 的 per-image nonce。

執行：python verify_cpa.py      （只需 numpy，數秒）
"""
import numpy as np
rng = np.random.default_rng(1911)
D = 1024

def fwht_rows(A):
    A = A.astype(np.float64).copy(); n = A.shape[1]; h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = A[:, i:i+h].copy(); y = A[:, i+h:i+2*h].copy()
            A[:, i:i+h] = x + y; A[:, i+h:i+2*h] = x - y
        h *= 2
    return A

def sot_rows(A, keys, d):
    out = A.astype(np.float64)
    for k1, pi, k2 in keys:
        out = fwht_rows(out * k1) / np.sqrt(d)
        out = out[:, pi] * k2
    return out

keys = [(rng.choice([-1., 1.], D), rng.permutation(D), rng.choice([-1., 1.], D))
        for _ in range(2)]
E = np.eye(D)

print("=" * 70)
print(" A. d 次查詢重建整個 Q（SOT-WHT, R=2, d=%d）" % D)
print("=" * 70)
Q_hat = sot_rows(E, keys, D).T          # 第 i 次查詢 E(e_i) = Q 的第 i 欄
z = rng.standard_normal(D)
err = np.max(np.abs(Q_hat @ z - sot_rows(z[None], keys, D)[0]))
print(f"  查詢數 = {D}")
print(f"  用重建的 Q 預測「未查詢過的新明文」之密文，最大絕對誤差 = {err:.3e}")
print("  -> 完全破解。攻擊者不需要金鑰，直接持有等價的解密映射 Q^T。")

print("\n" + "=" * 70)
print(" B. 單輪情形下，單次查詢 e_0 洩漏什麼")
print("=" * 70)
k1, pi, k2 = keys[0]
y = sot_rows(E[0][None], [keys[0]], D)[0] * np.sqrt(D)
k2_hat = np.sign(y)
acc = max((k2_hat == k2).mean(), (-k2_hat == k2).mean())
print(f"  WHT 第一列全為 1 -> fwht(e_0 * k1) = k1[0] * 全一向量")
print(f"  故 sign(E(e_0)) 直接等於 k2（差一個全域符號）")
print(f"  單次查詢還原 k2 的位元一致率 = {acc:.4f}")

print("\n" + "=" * 70)
print(" C. 論文該怎麼寫")
print("=" * 70)
print("""  1. 這不是 SOT-WHT 獨有：Q·e_i 就是 Q 的第 i 欄，任何線性映射皆然。
  2. 因此 L2 單獨無法宣稱 CPA 安全，必須寫明。
  3. L3 per-image nonce 的真正價值在此 —— 查詢影像 j 得到 Q_{k_j}，
     與目標 i 的 Q_{k_i} 因 KDF 單向而無關。
  4. 前提：image_id 不可由攻擊者控制或預測（L3 的第 4 個前提）。
  5. 呼應保範數 => 不可能 IND-CPA 的定理：要真正解決必須離開線性。""")
