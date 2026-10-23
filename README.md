# retina_multimodal_reg_new

Affine registration of retinal **FAF** (fundus autofluorescence, 55° FOV, *moving*)
images onto **IR** (SLO infrared, 30° FOV, *fixed*) images, driven by vessel
masks and LoFTR feature matching. Includes a batch CLI with landmark-based
evaluation (MLE / success rate / AUC) and an ablation harness.

The `fafir_registration` package in the parent directory is a thin
compatibility shim that re-exports this package, so
`python -m fafir_registration.cli` and `python -m retina_multimodal_reg_new.cli`
run the same code.

## Pipeline

For each IR/FAF pair, `register()` ([pipeline.py](pipeline.py)) does:

1. **Load & resize** both images and vessel masks to a square working grid
   (`size`, default 224). Vessel masks are required; a missing mask raises
   `ValueError` (the CLI records the pair as `skipped`).
2. **Preprocess** (optional) — CLAHE + light Gaussian blur
   ([preprocessing.py](preprocessing.py)).
3. **FOV scale seed** — centre-anchored scale of `55/30 ≈ 1.833`
   (`utils.FOV_SCALE_RATIO`) to bring the FAF into the IR field of view.
4. **LoFTR on vessel masks** ([matching.py](matching.py),
   [registration.py](registration.py)) — matches the IR mask against the
   FOV-scaled FAF mask, once per closing kernel in `LOFTR_CLOSE_KSIZES`
   (3, 4), keeping the highest-inlier result. Both a 4-DoF similarity and a
   6-DoF affine are fitted with RANSAC; the one with lower vessel ASD wins.
   If inliers < `LOFTR_MIN_INLIERS` (20), it re-matches at confidence
   `LOFTR_RETRY_CONF` (0.2).
5. **Dense-flow refinement** (optional) — TV-L1 optical flow (Farneback
   fallback) fitted to an affine and kept only if it lowers ASD.
6. **Warp & score** — warps the FAF image/mask into the IR frame and computes
   before/after metrics.

If kornia/LoFTR is unavailable, the pipeline falls back to the FOV scale seed
(plus dense flow).

The winning path is reported in `RegistrationResult.label`, e.g.
`LoFTR@1.83`, `LoFTR@1.83+denseflow`, `full_affine_scale_best`, or
`annotation_landmarks` (when a transform is forced).

## Modules

| File | Purpose |
|---|---|
| [pipeline.py](pipeline.py) | Public `register()` API and `RegistrationResult` |
| [registration.py](registration.py) | FOV seed → LoFTR → dense-flow orchestration (`register_affine`) |
| [matching.py](matching.py) | LoFTR matching and affine fitting; optional keypoint blending |
| [models/loftr.py](models/loftr.py) | Lazy loader for the pretrained kornia LoFTR (`outdoor`) |
| [preprocessing.py](preprocessing.py) | CLAHE/blur, vessel closing, density and quality heuristics |
| [metrics.py](metrics.py) | Dice, NCC, SSD, HD95, ASD, Wasserstein, MI/NMI, SSIM, PSNR |
| [visualization.py](visualization.py) | Overlays, checkerboards, vessel overlap/mismatch, match plots, grids |
| [utils.py](utils.py) | Device, constants, warping, affine composition/decomposition |
| [io.py](io.py) | Image / vessel-mask loading |
| [cli.py](cli.py) | Batch runner, landmark evaluation, fallback, ablation |
| [check_results.py](check_results.py) | Shared constants (the analysis helpers are commented out) |

## Installation

```bash
pip install -r ../requirements.txt
# numpy scipy opencv-python scikit-image torch kornia pandas
```

Optional: `pot` (exact Wasserstein; otherwise a sliced approximation is used)
and `opencv-contrib-python` (TV-L1 flow; otherwise Farneback). A CUDA GPU is
used automatically if available.

## Python API

```python
from retina_multimodal_reg_new import register

res = register(
    ir_image="fixed/0001_SLO_IR-0_0.png",
    faf_image="moving/0001_AF_B-2_0.png",
    ir_vessel="vessels/fixed/0001_SLO_IR-0_0.png",
    faf_vessel="vessels/moving/0001_AF_B-2_0.png",
    size=224,
)

res.transform          # 2x3 affine, FAF -> IR, in size x size coordinates
res.registered_image   # warped FAF (RGB uint8)
res.metrics            # before/after metrics dict
res.label              # winning path
res.save("out/", name="0001")   # write all diagnostic images
```

Inputs may be paths or numpy arrays. `fixed_*` / `moving_*` keyword aliases
are accepted in place of `ir_*` / `faf_*`.

Useful keyword arguments (mostly for ablations):

| Argument | Default | Effect |
|---|---|---|
| `fov_scale` | 55/30 | FOV scale seed |
| `size` | 224 | Working resolution |
| `preprocess` | `True` | CLAHE/blur |
| `use_loftr` | `True` | Enable LoFTR |
| `affine_dof` | `None` | `None` = auto (ASD), `4` = similarity, `6` = full affine |
| `min_conf` | 0.4 | LoFTR confidence for the first pass |
| `use_denseflow` | `True` | Dense-flow refinement |
| `ransac_thresh` | `5.0 * size/224` | RANSAC reprojection threshold (px) |
| `use_ransac` | `True` | `False` uses LMEDS |
| `keypoints_fixed` / `keypoints_moving` | `None` | Annotation landmarks to guide matching |
| `forced_transform` | `None` | Skip registration and use this 2x3 matrix |

## Command line

Run from the directory that contains this package (`irfaf_registration/`):

```bash
python -m retina_multimodal_reg_new.cli image_paths_768.csv out_run
```

### Input CSV

One row per pair. Default column names:

```
moving,fixed,moving_vessel_mask,fixed_vessel_mask,annotation_csv
```

Override them with `--faf-col/--moving-col`, `--ir-col/--fixed-col`,
`--faf-vessel-col`, `--ir-vessel-col` and `--annotation-col`.

Annotation CSVs (optional, used for evaluation) contain matched landmarks:

```
point_id,FAF_x,FAF_y,IR_x,IR_y
```

Coordinates are in native image pixels (the `*_768` data assumes 768 px
images). Column names are configurable with `--landmark-{moving,fixed}-{x,y}-col`.

### Main options

| Option | Description |
|---|---|
| `--size N` | Working resolution (default 224) |
| `--fov-scale S` | FOV scale seed (default 1.833) |
| `--no-loftr` | Disable LoFTR |
| `--method {auto,loftr,full_affine}` | Force a registration path |
| `--no-preprocess` | Skip CLAHE/blur |
| `--no-images` | Write CSVs only, no diagnostic images |
| `--only-row N [N ...]` | Run selected 1-based CSV rows |
| `--keypoints-only` | Skip pairs without usable annotation landmarks |

### Run modes

- **Fully automatic** (default, `--no-fallback`) — landmarks are only used to
  score the result.
- **Semi-automatic fallback** (`--fallback`) — if automatic registration fails
  (exception, no LoFTR transform, or inliers ≤ `--fallback-inlier-threshold`,
  default 30), the transform is re-estimated from the annotation landmarks
  (RANSAC, 8 px tolerance). Fallback pairs are reported separately because
  their landmark score is in-sample. Add `--use-keypoints` to also blend
  landmarks into LoFTR matching for every pair.
- **Annotation alignment** (`--align-from-annotation`) — every transform is
  built from landmarks (not an automatic registration; useful as an upper
  bound).
- **Ablation** (`--ablation [--configs NAME ...]`) — runs each config below on
  every pair. `--ablation-keypoints-only` limits it to pairs with a valid
  landmark transform.

| Config | Change |
|---|---|
| `full` | Full method, fully automatic |
| `full_rescue` | Full method + annotation fallback |
| `with_keypoints` | Landmark-guided matching |
| `no_loftr` | FOV seed only |
| `no_fov_scale` | `fov_scale=1.0` |
| `no_retry` | No low-confidence re-match |
| `no_denseflow` | No dense-flow refinement |
| `no_ransac` | LMEDS instead of RANSAC |

### Outputs

| File | Contents |
|---|---|
| `results.csv` | Per-pair status, winning label, transform parameters, vessel/image/landmark metrics (before & after) |
| `matching.csv` | Per-pair LoFTR inliers / total matches |
| `summary.csv` | Aggregated metrics |
| `ablation_results.csv`, `ablation_summary.csv` | Ablation mode equivalents |
| `grid/`, `registered/`, `overlap_*/`, `keypoint_matches/`, ... | Diagnostic images (unless `--no-images`) |

Pairs are `status=ok`, `skipped` (e.g. missing vessel mask) or `error`; only
`ok` pairs are included in metrics.

### Landmark metrics

- **MLE** — mean landmark error in pixels after mapping FAF landmarks into the
  IR frame.
- **Success rate** — fraction of pairs with MLE ≤ 18.75 px
  (`LANDMARK_SR_THRESHOLD_PX`).
- **AUC@37.5px** — area under the cumulative-error curve up to 37.5 px
  (`LANDMARK_AUC_MAX_PX`).

## Tests

```bash
pytest ../tests
```
