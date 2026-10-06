# CheXpert Preprocessing Audit for E2

## Verdict

The original E1 direct bicubic preprocessing did **not** match the senior project's training pipeline. The current `scripts/e1_ddim_runner.py` has been corrected and now reproduces the documented training path closely enough to begin E2:

1. OpenCV grayscale read.
2. Global histogram equalization (`cv2.equalizeHist`).
3. Resize to 256×256 with `cv2.INTER_AREA`.
4. JPEG quality 100 encode/decode round-trip, matching the default output of `chexpert_preproc.py` and later Pillow loading.
5. Per-image min-max normalization to `[0, 1]`, matching `ChexpertDataset` via `visualize()`.

## Evidence

- `past/SourceCode/scripts/chexpert_preproc.py::_preprocess_image()` performs grayscale loading, histogram equalization by default, and `INTER_AREA` resize.
- `ChexpertResNormPipeline.execute()` defaults to `.jpg` and writes JPEG quality 100.
- `past/SourceCode/guided_diffusion/bratsloader.py::ChexpertDataset.__getitem__()` loads the preprocessed file as grayscale and applies per-image min-max normalization.
- `past/SourceCode/scripts/cfg_image_train.py` uses that `ChexpertDataset` directly with `frontal_only` and `sample_n=16000`.
- `past/SourceCode/README/CFG_DDIM_README.md` explicitly says `chexpert_preproc.py` downsizes inputs to 256×256 and that 16,000 samples are selected per class.
- The backed-up training CSV contains flattened names such as `patient00001_study1_view1_frontal.jpg`, which matches `_generate_new_path()` output rather than the original nested CheXpert path.

## Remaining limits

- The actual preprocessed 256×256 training image directory was not preserved, so byte-for-byte comparison against the exact historical files is impossible.
- The script is interactive and does not preserve a preprocessing manifest. The conclusion is therefore based on code, README, CSV path structure, model shape, and a successful checkpoint smoke test.
- E2 must use the frozen preprocessing implementation and record its code/config hash. Changing equalization, interpolation, JPEG format, range, or label handling creates a new protocol version.

## Revalidation results

- `pytest`: 3/3 unit tests passed, including `tests/unit/test_chexpert_preprocessing.py`.
- DDIM smoke: 4/4 samples finite, no OOM/NaN.
- Mean runtime: 23.0635 seconds/image at noise level 500.
- Mean PSNR: 33.8917 dB.
- Mean image cosine: 0.999375.

These scientific values are observations, not pass thresholds.
