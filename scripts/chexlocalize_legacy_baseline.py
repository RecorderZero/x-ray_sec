#!/usr/bin/env python3
"""Predecessor-original localization baseline on CheXlocalize val GT.

Answers: is our parallel-latent pipeline's low IoU a regression, or is pixel-mask
effusion localization just low-IoU for this counterfactual-diffusion family?  We
run the predecessor's ORIGINAL anomaly-detection recipe (thesis setting) on the
SAME CheXlocalize validation images, scored with the SAME metrics as
scripts.chexlocalize_pilot, so the numbers sit on one ruler.

Predecessor original (faithful to ddim_sample_loop_known / q11_localization_parity):
  * legacy DDIM, dense T=500 (NOISE_LEVEL=500), the legacy decode convention;
  * inversion null=True, generation y=0 conditional (guidance_scale sentinel);
  * pseudo-healthy = CFG toward y=0 at guidance_scale=4.0 (thesis GUIDANCE);
  * anomaly map = |x0 - healthy|  (original vs pseudo-healthy, no recovery diff).
Preprocessing is e1 (same as the pilot and the frozen legacy results -- the correct
[-1,1] input range), so the only differences from the pilot are the METHOD knobs
(legacy sampler, |x0-healthy|, gs=4), not the loader.  No encryption: this is the
localization method itself, not the privacy layer (layer-A proved the re-inversion
deployment wrapper is near-equivalent anyway).

NOT a formal result: fixed uncalibrated threshold 0.5, validation split, no method
selection.  Read IoU/Dice/pxAP jointly with the negative footprint (a broad map can
inflate overlap without being lesion-specific).

  PYTHONPATH=.cache/localization_deps scripts/run_cfg_ddim.sh \
      python -m scripts.chexlocalize_legacy_baseline --per-class 2 --gs 2 4   # smoke
  PYTHONPATH=.cache/localization_deps scripts/run_cfg_ddim.sh \
      python -m scripts.chexlocalize_legacy_baseline --per-class 10 --gs 2 4  # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.chexlocalize_manifest import PATHOLOGY, sha256
from scripts.chexlocalize_pilot import decode_mask, evaluate_map, select_balanced
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.e2_legacy_wrapper import legacy_generate, legacy_invert, legacy_model_kwargs
from scripts.exact_inv_sweep import CKPT
from scripts.respaced_ddim_sweep import SEED

NOISE_LEVEL = 500  # thesis dense DDIM


@torch.inference_mode()
def legacy_cfg_generate(diffusion, model_fn, x_T, guidance_scale):
    """legacy_generate, but at an arbitrary CFG guidance_scale (healthy branch)."""
    return diffusion.ddim_anonymization_backward(
        model_fn, x_T, NOISE_LEVEL, clip_denoised=True,
        model_kwargs=legacy_model_kwargs(x_T.shape[0], x_T.device),
        guidance_scale=float(guidance_scale))


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument('--manifest', type=Path, default=Path('.cache/localization/val_manifest.json'))
    p.add_argument('--per-class', type=int, default=2)
    p.add_argument('--gs', type=float, nargs='+', default=[4.0],
                   help='pseudo-healthy CFG guidance scales (thesis setting is 4.0)')
    p.add_argument('--checkpoint', type=Path, default=CKPT)
    p.add_argument('--out-dir', type=Path, default=Path('paper_assets/localization_legacy_baseline'))
    args = p.parse_args()
    if not 1 <= args.per_class <= 10:
        p.error('baseline pilot only: per-class 1..10')
    if any(g < 0 for g in args.gs):
        p.error('guidance scales must be >= 0')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    selected = select_balanced(manifest, args.per_class)
    if sha256(Path(manifest['annotations_path'])) != manifest['annotations_sha256']:
        raise ValueError('GT changed since manifest creation')
    for r in selected:
        if sha256(Path(r['image_path'])) != r['image_sha256']:
            raise ValueError('Image changed since manifest creation')
    annotations = json.loads(Path(manifest['annotations_path']).read_text())

    _, diffusion, model_fn = create_runtime(Path('past/SourceCode'), args.checkpoint, device)

    records, start = [], time.perf_counter()
    for item in selected:
        x0 = preprocess(Path(item['image_path']))[0].unsqueeze(0).to(device)
        gt = decode_mask(annotations[item['gt_key']][PATHOLOGY])
        if int(gt.sum()) != item['mask_area_pixels']:
            raise ValueError('Decoded mask area changed')
        x_T = legacy_invert(diffusion, model_fn, x0, NOISE_LEVEL)
        recon = legacy_generate(diffusion, model_fn, x_T, NOISE_LEVEL)   # y=0 recovery (sanity)
        recon_psnr = image_pair_metrics('recovery', x0, recon)['recovery_psnr_db']
        for gs in args.gs:
            healthy = legacy_cfg_generate(diffusion, model_fn, x_T, gs)
            if not bool(torch.isfinite(healthy).all()):
                raise ValueError('non-finite healthy')
            heat = (x0 - healthy).abs()     # predecessor recipe: original vs pseudo-healthy
            metrics = evaluate_map(heat, gt)
            row = {'sample_id': f'{item["sample_id"]}/legacy-DDIM/gs{gs:g}',
                   'source_sample_id': item['sample_id'], 'patient_id': item['patient_id'],
                   'gt_positive': bool(gt.any()), 'guidance': gs, 'recon_psnr_db': recon_psnr,
                   **metrics}
            records.append(row)
            print(json.dumps({k: row[k] for k in
                              ('sample_id', 'gt_positive', 'iou', 'dice', 'pixel_ap',
                               'pointing_hit', 'predicted_positive_fraction', 'raw_fraction_gt_001')},
                             allow_nan=False), flush=True)

    groups = []
    for gs in args.gs:
        rr = [r for r in records if r['guidance'] == gs]
        pos = [r for r in rr if r['gt_positive']]
        neg = [r for r in rr if not r['gt_positive']]
        groups.append({'sampler': 'legacy-DDIM', 'noise_level': NOISE_LEVEL, 'guidance': gs,
                       'heatmap': 'abs(x0 - healthy)', 'positive_n': len(pos), 'negative_n': len(neg),
                       **{k: float(np.mean([r[k] for r in pos])) for k in ('iou', 'dice', 'pixel_ap', 'pointing_hit')},
                       'negative_predicted_fraction': float(np.mean([r['predicted_positive_fraction'] for r in neg])),
                       'negative_raw_fraction_gt_001': float(np.mean([r['raw_fraction_gt_001'] for r in neg]))})

    payload = {'seed': SEED, 'formal': False, 'split': 'val', 'sampler': 'legacy-DDIM',
               'noise_level': NOISE_LEVEL, 'guidance': args.gs, 'heatmap': 'abs(x0 - healthy)',
               'recovery_condition': 'y=0 conditional (legacy); inversion null=True',
               'preprocessing': 'e1 (same as pilot and frozen legacy results)',
               'threshold': 0.5, 'threshold_policy': 'fixed exploratory, NOT validation-tuned',
               'manifest_sha256': sha256(args.manifest), 'checkpoint_sha256': sha256(args.checkpoint),
               'selected_ids': [r['sample_id'] for r in selected], 'source_images': len(selected),
               'groups': groups, 'per_sample': records, 'elapsed_seconds': time.perf_counter() - start}
    (args.out_dir / 'baseline.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

    print('\n=== predecessor-original legacy-DDIM baseline (|x0-healthy|); mean over positives ===')
    for g in groups:
        print(f"  gs={g['guidance']:<4g} IoU={g['iou']:.4f} Dice={g['dice']:.4f} pxAP={g['pixel_ap']:.4f} "
              f"point={g['pointing_hit']:.2f} | neg_pred%={100*g['negative_predicted_fraction']:.3f} "
              f"neg>.01%={100*g['negative_raw_fraction_gt_001']:.2f}")
    print(f"wrote {args.out_dir}/baseline.json ({len(selected)} images, {time.perf_counter()-start:.1f}s)")


if __name__ == '__main__':
    main()
