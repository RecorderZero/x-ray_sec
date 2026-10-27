"""Validation-only annotated pilot of the parallel-latent localization pipeline.

No test selection, no threshold search, no clinical/safety claims. Run with
managed_run --task-id LOC-VAL --validate-artifacts. Full-resolution GT masks
are unchanged; 256-square raw maps are bilinearly resized before scoring.
S2 is the explicit numerical fixture (NOT authenticated deployment).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.chexlocalize_manifest import PATHOLOGY, sha256
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.localization_cfgpp import VARIANT, ddim_cfgpp_backward, dpm_cfgpp_backward
from scripts.localization_metrics import mask_metrics
from scripts.localization_sampler_smoke import canonical_cfg_backward, finite, gs0_sanity, recovered_latent
from scripts.respaced_ddim_sweep import Respaced, SEED, schedule
from scripts.run_artifacts import open_run, finalize_run


def select_balanced(manifest, per_class):
    if manifest['split'] != 'val' or manifest['mask_validation'] != 'decoded_binary':
        raise ValueError('Only decoded validation manifest allowed; test is held out')
    selected = []
    used_patients = set()
    for positive in (True, False):
        candidates = sorted(manifest['records'], key=lambda r: r['sample_id'])
        count = 0
        for r in candidates:
            if r['mask_positive'] != positive or r['patient_id'] in used_patients:
                continue
            selected.append(r)
            used_patients.add(r['patient_id'])
            count += 1
            if count == per_class:
                break
        if count != per_class:
            raise ValueError('Insufficient independent patients in requested class')
    return selected


def decode_mask(rle):
    from pycocotools import mask as coco
    obj = dict(rle)
    if isinstance(obj['counts'], str):
        obj['counts'] = obj['counts'].encode('ascii')
    elif isinstance(obj['counts'], list):
        obj = coco.frPyObjects(obj, *obj['size'])
    result = coco.decode(obj)
    if result.shape != tuple(rle['size']) or not np.isin(result, [0, 1]).all():
        raise ValueError('Invalid decoded mask')
    return result


def normalize_map(raw):
    lo, hi = float(raw.min()), float(raw.max())
    # Exact constant guard; raw amplitude separately reported, not hidden.
    return np.zeros_like(raw) if hi == lo else (raw - lo) / (hi - lo)


def evaluate_map(raw_256, gt, threshold=0.5):
    raw = F.interpolate(raw_256.detach().float().cpu(), size=gt.shape,
                        mode='bilinear', align_corners=False)[0, 0].numpy()
    finite('upsampled map', torch.from_numpy(raw))
    normalized = normalize_map(raw)
    metrics = mask_metrics(normalized, gt, threshold)
    metrics.update({'raw_mean': float(raw.mean()), 'raw_max': float(raw.max()),
                    'raw_fraction_gt_001': float((raw > 0.01).mean())})
    return metrics


def save_panel(path, original, recovery, healthy, heatmap, gt, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, arr, label in zip(axes,
            [original[0, 0].cpu(), recovery[0, 0].cpu(), healthy[0, 0].cpu(),
             heatmap[0, 0].cpu(), gt],
            ['Original', 'Recovery', 'Pseudo-healthy', 'Raw difference', 'GT effusion']):
        ax.imshow(arr, cmap='magma' if label == 'Raw difference' else 'gray',
                  vmin=0, vmax=0.1 if label == 'Raw difference' else 1)
        ax.set_title(label)
        ax.axis('off')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('.cache/localization/val_manifest.json'))
    p.add_argument('--per-class', type=int, default=1)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--include-s2', action='store_true')
    p.add_argument('--checkpoint', type=Path, default=CKPT)
    args = p.parse_args()
    if not os.environ.get('EXPERIMENT_RUN_DIR'):
        p.error('Use managed_run --task-id LOC-VAL --validate-artifacts')
    if not 1 <= args.per_class <= 10 or not 1 <= args.steps <= 499:
        p.error('Pilot only: per-class 1..10, steps 1..499; formal needs separate protocol')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    manifest = json.loads(args.manifest.read_text())
    selected = select_balanced(manifest, args.per_class)
    if sha256(Path(manifest['annotations_path'])) != manifest['annotations_sha256']:
        raise ValueError('GT changed since manifest creation')
    for r in selected:
        if sha256(Path(r['image_path'])) != r['image_sha256']:
            raise ValueError('Image changed since manifest creation')
    annotations = json.loads(Path(manifest['annotations_path']).read_text())
    schemes = ['original'] + (['S2fn-transform-only'] if args.include_s2 else [])
    b = schedule(args.steps)
    config = {'manifest_sha256': sha256(args.manifest), 'annotations_sha256': manifest['annotations_sha256'],
              'checkpoint_sha256': sha256(args.checkpoint), 'selected_ids': [r['sample_id'] for r in selected],
              'steps': args.steps, 'boundaries': b.tolist(), 'seed': SEED, 'formal': False,
              'split': 'val', 'schemes': schemes, 'samplers': ['aligned-DDIM', 'DPM++2M'],
              'cfg_gs': [0.0, 2.0], 'cfgpp_lambda': [0.0, 0.5, 1.0], 'cfgpp_variant': VARIANT,
              'threshold': 0.5, 'threshold_policy': 'fixed exploratory, NOT validation-tuned',
              'heatmap': 'abs(recovery-healthy), original GT geometry, bilinear align_corners=False',
              'normalization': 'per-map minmax; constants zero; raw amplitude also reported',
              'cohort': 'first sorted independent patients per GT-mask class; positive/negative balanced',
              'metrics': 'all GT-positive incl empty predictions; GT-negatives overlap/AP null; no probability gating',
              'preprocessing': 'e1 grayscale/equalizeHist/INTER_AREA256/JPEG100/minmax; no crop',
              'flow': 'shared decrypted latent -> recovery + healthy; no recovery reinversion',
              'recovery_condition': 'y=0 conditional, NOT unconditional',
              'security_scope': 'S2 numerical fixture, rounds2/permutation/fixednorm/rangePNG; no AEAD',
              'runtime_scope': 'prototype wall time includes finite checks/CPU metrics; not speed ranking',
              'resumable': False}
    ctx = open_run('LOC-VAL', config, Path('artifacts/runs'))
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    _, diffusion, model = create_runtime(Path('past/SourceCode'), args.checkpoint, device)
    rs = Respaced(diffusion, model, device)
    dpm = DPMSolverpp(rs)
    records = []
    start = time.perf_counter()
    for sample_no, item in enumerate(selected):
        x = preprocess(Path(item['image_path']))[0].unsqueeze(0).to(device)
        gt = decode_mask(annotations[item['gt_key']][PATHOLOGY])
        if int(gt.sum()) != item['mask_area_pixels']:
            raise ValueError('Decoded mask area changed')
        for name, sampler in [('aligned-DDIM', rs), ('DPM++2M', dpm)]:
            for scheme in schemes:
                z = recovered_latent(sampler, x, b, scheme, item['sample_id'])
                rec = finite('recovery', sampler.backward(z.clone(), b))
                rec_metrics = image_pair_metrics('recovery', x, rec)
                variants = [('CFG', s) for s in (0.0, 2.0)] + [('CFG++', s) for s in (0.0, 0.5, 1.0)]
                for kind, scale in variants:
                    tick = time.perf_counter()
                    if kind == 'CFG':
                        healthy = canonical_cfg_backward(sampler, z.clone(), b, scale)
                    else:
                        fn = dpm_cfgpp_backward if name == 'DPM++2M' else ddim_cfgpp_backward
                        healthy = fn(sampler, z.clone(), b, scale)
                    finite('healthy', healthy)
                    if kind == 'CFG' and scale == 0:
                        gs0_sanity(rec, healthy)
                    heat = (rec - healthy).abs()
                    metrics = evaluate_map(heat, gt)
                    ident = f'{item["sample_id"]}/{name}/{scheme}/{kind}/{scale}'
                    row = {'sample_id': ident, 'source_sample_id': item['sample_id'],
                           'patient_id': item['patient_id'], 'gt_positive': bool(gt.any()),
                           'sampler': name, 'scheme': scheme, 'guidance': kind, 'scale': scale,
                           'recovery_psnr_db': rec_metrics['recovery_psnr_db'],
                           **metrics, 'branch_and_metrics_seconds': time.perf_counter()-tick}
                    records.append(row)
                    stem = f'{sample_no}_{name}_{scheme}_{kind}_{scale}'
                    np.savez_compressed(ctx.run_dir / (stem+'.npz'), recovery=rec.cpu().numpy(),
                                        healthy=healthy.cpu().numpy(), heatmap=heat.cpu().numpy())
                    if scale != 0:
                        save_panel(ctx.run_dir / (stem+'.png'), x, rec, healthy, heat, gt, ident)
                    print(json.dumps(row, allow_nan=False), flush=True)
    expected = len(selected)*2*len(schemes)*5
    if len(records) != expected or len({r['sample_id'] for r in records}) != expected:
        raise ValueError('Missing or duplicate result rows')
    groups = []
    for name in ('aligned-DDIM', 'DPM++2M'):
        for scheme in schemes:
            for kind, scale in [('CFG', 0.0), ('CFG', 2.0), ('CFG++', 0.0), ('CFG++', 0.5), ('CFG++', 1.0)]:
                rr = [r for r in records if (r['sampler'],r['scheme'],r['guidance'],r['scale']) == (name,scheme,kind,scale)]
                pos, neg = [r for r in rr if r['gt_positive']], [r for r in rr if not r['gt_positive']]
                groups.append({'sampler': name, 'scheme': scheme, 'guidance': kind, 'scale': scale,
                               'positive_n': len(pos), 'negative_n': len(neg),
                               **{k: float(np.mean([r[k] for r in pos])) for k in ('iou','dice','pixel_ap','pointing_hit')},
                               'negative_predicted_fraction': float(np.mean([r['predicted_positive_fraction'] for r in neg])),
                               'negative_raw_fraction_gt_001': float(np.mean([r['raw_fraction_gt_001'] for r in neg]))})
    (ctx.run_dir/'localization_summary.json').write_text(json.dumps(
        {'formal': False, 'source_images': len(selected), 'groups': groups,
         'limitations': ['exploratory validation only', 'no external baselines yet', 'S2 numerical fixture only'],
         'elapsed_seconds': time.perf_counter()-start}, indent=2, allow_nan=False)+'\n')
    finalize_run(ctx, Path(__file__), records,
                 ['recovery_psnr_db','raw_mean','raw_max','predicted_positive_fraction'],
                 {'dataset_split_hash': sha256(args.manifest), 'checkpoint_hash': config['checkpoint_sha256'],
                  'formal': False, 'source_sample_count': len(selected), 'seed_list': [SEED],
                  'steps': args.steps, 'scheme': schemes, 'rounds': 2 if args.include_s2 else 0,
                  'nonce_mode': 'public deterministic fixture; nonproduction' if args.include_s2 else 'none',
                  'container_version': 'none; numerical fixture only',
                  'gt_available': True, 'elapsed_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
