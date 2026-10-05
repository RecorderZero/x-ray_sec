#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Finding #3 : 金鑰空間崩塌 (Key-Space Collapse)
=============================================================================
 這不是設計缺陷，是實作缺陷 —— 而且它單獨就足以否定論文全部的安全性宣稱。

 位置：guided_diffusion/anonymization.py
   RademacherKey._derive_key_from_password()          L95-101
   PermutationKey._derive_permutation_from_password() L186-192
   SignedPermutationKey.__init__()                    L355-363

 程式碼原文：
     hash_digest = hashlib.sha256(password_bytes).digest()
     seed = int.from_bytes(hash_digest[:4], byteorder='big')   # <-- 只取 4 bytes
     return RademacherKey._generate_key(shape, seed)

 SHA-256 產生 256 bits，但只取前 32 bits 當 torch.Generator 的 seed。
 金鑰 k 因此是一個 32-bit 整數的確定性函數 —— 不是 2^65536 個等機率選擇。

 更糟的是 SignedPermutationKey：
     perm_seed = seed + 1000000        # L359，固定偏移
 置換 pi 也由同一個 seed 決定。所以「2^d x d!」的宣稱在實作中完全不成立：
 sign 與 permutation 並非獨立，兩者一起由同一個 32-bit 值決定。

 執行環境：本檔的算術部分不需 torch；驗證部分需要 torch（在你的 conda env 跑）。
=============================================================================
"""
import hashlib, math, time

D = 256 * 256
PASSWORD  = "Ki@13579"          # run_cfg_inference_Crypto.sh 內硬編碼的密碼
SEED_ARG  = 42                  # --anonymization_key_seed 42（password 未設時的後備）


def bar(t): print("\n" + "=" * 76 + f"\n {t}\n" + "=" * 76)


# ---------------------------------------------------------------------------
bar("A. 重現論文程式碼的金鑰導出，看實際 seed 是什麼")
for pw in (PASSWORD, PASSWORD + "_permutation"):
    h = hashlib.sha256(pw.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], byteorder="big")
    print(f"  password = {pw!r}")
    print(f"    sha256 full  = {h.hex()}")
    print(f"    使用的部分   = {h[:4].hex()}   (256 bits 中只用了 32 bits)")
    print(f"    -> seed      = {seed}")
print(f"\n  也就是說：整個 65,536 維的金鑰，完全由 {seed} 這個 32-bit 整數決定。")


# ---------------------------------------------------------------------------
bar("B. 宣稱 vs 實際的金鑰空間")
claim_rad = D
claim_sp  = D + math.lgamma(D + 1) / math.log(2)      # log2(2^d * d!)
real_bits = 32

rows = [
    ("Rademacher（論文 §4.3.1 宣稱）",       claim_rad),
    ("Signed Permutation（論文 §4.3.2 宣稱）", claim_sp),
    ("實作實際值（password 或 seed 導出）",     real_bits),
]
print(f"  {'方案':<40} {'log2(金鑰空間)':>16}")
print("  " + "-" * 58)
for name, bits in rows:
    print(f"  {name:<40} {bits:>16,.0f} bits")

print(f"\n  落差：")
print(f"    Rademacher       宣稱 2^{claim_rad:,}  ->  實際 2^32   （少了 {claim_rad-32:,} bits）")
print(f"    SignedPermutation 宣稱 2^{claim_sp:,.0f}  ->  實際 2^32   （少了 {claim_sp-32:,.0f} bits）")
print(f"\n  對照：AES-128 是 2^128，AES-256 是 2^256。這個方案是 2^32 —— ")
print(f"  比 1998 年就被公開硬體破解的 DES（2^56）還小 2^24 倍。")


# ---------------------------------------------------------------------------
bar("C. 窮舉成本估計")
KEYSPACE = 2 ** 32
print(f"  金鑰空間 = 2^32 = {KEYSPACE:,}")
print()
print(f"  {'攻擊者的判別器 (oracle)':<44} {'單次成本':>12} {'總時間':>16}")
print("  " + "-" * 76)
scenarios = [
    ("已知明文（有任一張原圖）：直接算 k=sign(Z_enc)sign(Z)", 0.0,     None),
    ("部分解碼 5 步 DDIM + 結構性檢查（GPU 批次 1024）",       2.0e-4,  None),
    ("完整解碼 500 步 DDIM（論文的 24 s/張）",                24.0,     None),
]
for name, c, _ in scenarios:
    if c == 0.0:
        print(f"  {name:<44} {'不需窮舉':>12} {'< 1 秒':>16}")
    else:
        tot = KEYSPACE * c
        if tot < 86400 * 365:
            s = f"{tot/86400:,.1f} 天"
        else:
            s = f"{tot/86400/365:,.0f} 年"
        print(f"  {name:<44} {c:>10.1e}s {s:>16}")

print("\n  結論：即使不用已知明文，只要能造出一個『便宜的判別器』（例如只跑 5 步")
print("  部分去噪就足以分辨『像 X 光』還是『純雜訊』），2^32 就落在數天之內。")
print("  而在有已知明文的情況下，根本不需要窮舉。")


# ---------------------------------------------------------------------------
bar("D. 完整攻擊鏈：三個漏洞如何複合成 ciphertext-only 全面破解")
chain = [
 ("1", "取得密文",
      "攻擊者拿到加密噪聲 Z_enc（論文 §6.2 自己建議只儲存/傳輸這個），"
      "或匿名影像後自行 DDIM 反演取得 Ẑ_enc。"),
 ("2", "利用幅值不變量鎖定明文  ← Finding #1",
      "|Z_enc| = |Z| 逐元素相同。拿公開 CheXpert 做 gallery，"
      "比對 corr(|Ẑ_enc|, |Z_j|) 找出這張密文對應哪一張原圖。Tier-0 E2 顯示 top-1 = 1.00。"),
 ("3", "把步驟 2 的結果當已知明文  ← Finding #2",
      "k̂ = sign(Z_enc) ⊙ sign(Z)。Tier-0 E3 顯示 cos=0.998 時金鑰位元正確率 97.99%。"),
 ("4", "金鑰重用讓單點失守擴散為全面失守  ← Finding #3",
      "金鑰由 password 確定性導出且全資料集共用，"
      "所以步驟 3 恢復的 k 可以解開該 password 下的每一張影像。"),
 ("5", "結果",
      "攻擊者從頭到尾沒有金鑰、沒有窮舉、只用了公開資料集與公開模型權重，"
      "即可解密整批醫療影像。"),
]
for n, t, d in chain:
    print(f"\n  [{n}] {t}")
    for line in [d[i:i+66] for i in range(0, len(d), 66)]:
        print(f"      {line}")

print("\n\n  這條鏈的關鍵在於：三個漏洞單獨看都還有辯解空間，")
print("  但複合起來就是完整的 ciphertext-only 破解，且不需要任何非公開資訊。")


# ---------------------------------------------------------------------------
bar("E. 需要 torch 才能跑的驗證（在你的 conda env 執行）")
print("""
  import torch, hashlib
  from guided_diffusion.anonymization import RademacherKey, SignedPermutationKey

  # E-1  證明 password 與 32-bit seed 產生完全相同的金鑰
  h    = hashlib.sha256(b"Ki@13579").digest()
  seed = int.from_bytes(h[:4], "big")
  k_pw   = RademacherKey(shape=(1,256,256), password="Ki@13579").key
  k_seed = RademacherKey(shape=(1,256,256), seed=seed).key
  assert torch.equal(k_pw, k_seed)          # <-- 應為 True，金鑰 = f(32-bit)

  # E-2  證明 SignedPermutation 的 sign 與 permutation 不獨立
  sp = SignedPermutationKey(shape=(1,256,256), seed=42)
  assert torch.equal(sp.rademacher_key.key, RademacherKey((1,256,256), seed=42).key)
  #   置換的 seed 是 42 + 1000000，完全可預測

  # E-3  窮舉可行性計時：批次產生 N 把候選金鑰要多久
  t0 = time.time()
  for s in range(10000):
      torch.Generator().manual_seed(s)
      torch.bernoulli(torch.full((1,256,256), .5),
                      generator=torch.Generator().manual_seed(s))
  print("10k keys:", time.time()-t0, "s  ->  2^32 需",
        (time.time()-t0)*2**32/10000/86400, "天（僅產金鑰，未含判別）")
""")

print("=" * 76)
print(" 建議：把 A/B/C 三張表直接做成論文的 Table 1（宣稱 vs 實際），")
print(" D 的攻擊鏈做成 Figure 2。這是不需要任何實驗就能寫的貢獻。")
print("=" * 76)
