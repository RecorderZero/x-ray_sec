# Privacy–Utility (DPM++ N=100, S2fn, recon 46.727 dB)

| Encoder | Pathology real→real | Pathology real→**recovered** | Pathology real→anon | Identity P (link AUC) | Identity **S2** (link AUC) |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 (ImageNet) | 0.916 | **0.909** | 0.501 | 0.821 | **0.381** |
| rad-dino (CXR) | 0.937 | **0.937** | 0.497 | 0.975 | **0.486** |

External attacker (Siamese ResNet-50 (ImageNet), emb128, |ea-eb| head, BCE (Packhauser 2022 arch, reimpl)): plaintext top-1 0.15 (AUC 0.764) → **S2 top-1 0.05** (AUC 0.445; random 0.05).

**Per-column n (AF-047; sources differ, stated explicitly):**
- Pathology (utility): per_class 200 (test 120/split x 50 patient-splits)
- Identity (linkage): 20 patient pairs (gallery+query), scheme S2
- External (re-id): Siamese ResNet-50 (ImageNet), emb128, |ea-eb| head, BCE (Packhauser 2022 arch, reimpl) Siamese, latent ciphertext

Same DPM++ N=100 pipeline (null backward) and same frozen encoders throughout. Pathology (utility) uses the S2fn deployment recovery; identity linkage is scored on the S2 scheme (same per-image-nonce orthogonal family; S2fn is its fixed-norm fidelity variant, both yield noise-like anon). Pathology AUROC near real after recovery; identity linkage AUC near random (0.5) for S2; external latent-ciphertext attacker S2 at random top-1. P (no encryption) is the identity-leak control. Per-column n differs by source and is stated above (AF-047): utility n is per-class x splits, identity n is patient pairs.
