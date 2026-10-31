# External method comparison (same-data head-to-head)

PriCheXy-Net (Packhäuser et al., MICCAI 2023) released flow-field generator
(mu=0.01) run **out-of-tree** at `/data2/external_gpl/` (GPL-3.0; code/weights
NOT vendored here). Only its anonymized CheXpert PNGs are read by our own
attacker/probe. Numbers below are produced by OUR code.

- `prichexy_pathology.json` — our densenet121 frozen probe on PriCheXy-anon vs real.
- re-id on the same victim set: `../tables/table_reid_external_vs_prichexy.csv`
  (`external_prichexy_anon_victim` row), scored by our Packhäuser re-id attacker.

Result (same CheXpert set, same attacker, same probe):
| method | anon re-id AUC | anon pathology AUC | reversible |
|---|---|---|---|
| plaintext | 0.764 | 0.923 | — |
| PriCheXy (mu=0.01, cross-dataset) | 0.738 | 0.895 | no |
| our S2 (DPHD) | 0.445 | ~chance (recovered 0.908) | yes |

Caveat: cross-dataset (PriCheXy weights trained on ChestX-ray14) + different
attacker than PriCheXy's own; their paper reports 81.8->57.7% on their setup.
Honest reading: PriCheXy's adversarial deformation does not transfer to a
different attacker/dataset; our S2 is attacker-agnostic (cryptographic).

## AF-050 resolution (2026-10-31)
Earlier `table_reid_external_vs_prichexy` had S2 == S2perm bit-identical because
the config default had been flipped to P-on, so the "S2" condition was already
P-on. Re-run with `S2_PERM=0`: S2(P off)=0.569 != S2perm(P on=deployment)=0.445
in the SAME table/net -> proves the permutation is applied and changes the
attacker score. Deployment = S2perm (DPHD, 0.445).
