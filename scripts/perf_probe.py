#!/usr/bin/env python3
"""Throughput probe for batch-1 legacy inversion: eager vs CUDA Graphs.

Each invocation inverts the first ``--images`` dev images at batch 1 with the
legacy anonymization forward pass and writes per-image latent SHA-256 and
timing to ``--output``.  Running several invocations concurrently measures
multi-process throughput; comparing hashes with an eager single-process run
checks bit-exactness.  This is a diagnostic, not a formal experiment.

  scripts/run_cfg_ddim.sh python -m scripts.perf_probe --checkpoint <ckpt> \\
      --mode eager --output /tmp/probe_eager.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, load_split, preprocess
from scripts.e2_legacy_wrapper import legacy_invert


def patch_timestep_embedding() -> None:
    """Make the legacy timestep embedding capturable without changing its values.

    The predecessor builds ``freqs`` on the CPU on every call and copies it to
    the GPU, which CUDA-graph capture forbids.  The patch computes ``freqs``
    once with exactly the same CPU expression, keeps the device copy, and
    leaves the rest of the function unchanged (monkey-patch; past/ untouched).
    """
    import math

    import guided_diffusion.unet_v1 as unet_v1

    cache: dict[tuple, torch.Tensor] = {}

    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        key = (half, max_period, timesteps.device)
        if key not in cache:
            cache[key] = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=timesteps.device)
        freqs = cache[key]
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    unet_v1.timestep_embedding = timestep_embedding


class GraphedModelFn:
    """Replay one captured CUDA graph per (null flag, shape) instead of eager calls."""

    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.graphs: dict[tuple, tuple] = {}

    def _capture(self, x, t, y, p_uncond, null, clf_free):
        static_x, static_t, static_y = x.clone(), t.clone(), y.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model_fn(static_x, static_t, static_y, p_uncond, null, clf_free)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.model_fn(static_x, static_t, static_y, p_uncond, null, clf_free)
        return graph, static_x, static_t, static_y, static_out

    def __call__(self, x, t, y=None, p_uncond=-1, null=False, clf_free=True):
        key = (bool(null), p_uncond, bool(clf_free), tuple(x.shape), x.dtype, t.dtype)
        if key not in self.graphs:
            self.graphs[key] = self._capture(x, t, y, p_uncond, null, clf_free)
        graph, static_x, static_t, static_y, static_out = self.graphs[key]
        static_x.copy_(x)
        static_t.copy_(t)
        static_y.copy_(y)
        graph.replay()
        return static_out.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--noise-level", type=int, default=200)
    parser.add_argument("--mode", choices=("eager", "eager-patched", "graph"), default="eager")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_split(args.split)[: args.images]
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    if args.mode in {"eager-patched", "graph"}:
        patch_timestep_embedding()
    fn = GraphedModelFn(model_fn) if args.mode == "graph" else model_fn
    images = [preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device) for row in rows]
    with torch.inference_mode():
        legacy_invert(diffusion, fn, images[0], 3)  # warm-up / graph capture
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        results = []
        for row, x0 in zip(rows, images):
            began = time.perf_counter()
            latent = legacy_invert(diffusion, fn, x0, args.noise_level)
            torch.cuda.synchronize(device)
            array = latent.detach().float().cpu().numpy()
            results.append({
                "sample_id": row["sample_id"],
                "seconds": time.perf_counter() - began,
                "latent_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            })
        wall = time.perf_counter() - started
    payload = {
        "mode": args.mode,
        "noise_level": args.noise_level,
        "images": len(rows),
        "wall_seconds": wall,
        "seconds_per_step_per_image": wall / len(rows) / (args.noise_level - 1),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}))


if __name__ == "__main__":
    main()
