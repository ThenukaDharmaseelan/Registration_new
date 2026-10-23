# from __future__ import annotations

# import argparse
# import sys
# import time
# from pathlib import Path
# from typing import Optional

# import cv2
# import numpy as np
# import pandas as pd

# from .pipeline import register, _as_vessel
# from . import utils, visualization



# # Registration success rate (Eq. 11): a pair is "successful" iff its landmark
# # MLE is <= T_SR. T_SR is defined as half of T_AUC (Eq. 10's integration
# # bound). The success-rate and AUC cutoffs are configured independently.
# LANDMARK_SR_THRESHOLD_PX = 18.75  # T_SR
# LANDMARK_AUC_MAX_PX = 37.5  # T_AUC

# # RANSAC inlier tolerance, in native image pixels, for the annotation-derived
# # fallback affine fit (see _estimate_affine_from_annotation) -- NOT the
# # size x size (224) working grid. Fitting on points already rescaled to the
# # 224 working grid and then applying this tolerance there would let it
# # silently balloon to ~10px of real reprojection slack at 768px, loose
# # enough that a clustered, inconsistent annotation click gets accepted as an
# # inlier instead of rejected. Assumes every pair's images are standardized
# # to 768px (image_paths_768.csv / annotations_768/) -- mixed-resolution
# # input would need this scaled per pair instead of used as a flat constant.
# ANNOTATION_RANSAC_PX = 8.0
# _AUC_LABEL = f"AUC@{LANDMARK_AUC_MAX_PX:.2f}px"
# _AUC_STEM = f"landmark_auc_{LANDMARK_AUC_MAX_PX:.2f}".replace(".", "_")

# # (display_name, preferred_stem, fallback_stem)
# SUMMARY_METRICS_VESSEL = [
#     ("dice", "vessel_dice", "dice"),
#     ("ncc", "vessel_ncc", "ncc"),
#     ("ssd", "vessel_ssd", "ssd"),
#     ("hd95", "vessel_hd95", "hd95"),
#     ("asd", "vessel_asd", "asd"),
#     ("wasserstein", "vessel_wasserstein", "wasserstein"),
# ]
# SUMMARY_METRICS_IMAGE = [
#     ("dice", "image_dice", None),
#     ("ncc", "image_ncc", None),
#     ("ssd", "image_ssd", None),
#     ("mi", "image_mi", "mi"),
#     ("nmi", "image_nmi", "nmi"),
#     ("ssim", "image_ssim", "ssim"),
#     ("psnr", "image_psnr", "psnr"),
# ]
# SUMMARY_METRICS_LANDMARK = [
#     ("MLE", "landmark_mle", None),
# ]
# SUMMARY_METRICS_LANDMARK_AUC = [
#     (_AUC_LABEL, _AUC_STEM, None),
# ]

# # Per-config ablation summary metrics written to ablation_summary.csv.
# # (source_stem, output_metric_name, higher_is_better, phase)
# ABLATION_METRICS = [
#     ("wasserstein", "wasserstein", False, "after"),
#     ("asd", "asd", False, "after"),
#     ("dice", "dice", True, "after"),
#     ("ncc", "ncc", True, "after"),
#     ("hd95", "hd95", False, "after"),
#     ("landmark_mle", "mle", False, "after"),
#     ("landmark_mle", "success_rate", True, "after"),
#     ("landmark_mle", "AUC@37.50px", True, "after"),
# ]

# # Each entry is (name, label, overrides). `overrides` is passed as
# # **kwargs into pipeline.register() -- see run_ablation() -- except for the
# # keys in _GLOBAL_OVERRIDE_ATTRS below, which set a `utils` module constant
# # for the duration of the call instead (those aren't register() params).
# ABLATIONS = [
#     ("full",          "Full method (w/o keypoint annotation)", {}),
#     ("full_rescue",   "Full method (w/ annotation fallback, matches run_batch/out_12)", dict(_rescue=True)),
#     ("with_keypoints", "Keypoint-annotation alignment (w/ keypoints)", dict(use_keypoints=True)),
#     ("no_loftr",      "w/o LoFTR (FOV seed only)",     dict(use_loftr=False)),
#     ("no_fov_scale",  "w/o FOV scale",                 dict(fov_scale=1.0)),
#     ("affine4",       "Similarity transform (4 DOF)",  dict(affine_dof=4)),
#     ("affine6",       "Full affine (6 DOF)",           dict(affine_dof=6)),
#     ("k1_only",       "closing k=1 only",              dict(loftr_close_ksizes=(1,))),
#     ("k4_only",       "closing k=4 only",              dict(loftr_close_ksizes=(4,))),
#     ("no_retry",      "w/o low-conf re-match",         dict(loftr_min_inliers=0)),
#     ("no_denseflow",  "w/o dense-flow refinement",     dict(use_denseflow=False)),
#     ("no_ransac",     "w/o RANSAC",                    dict(use_ransac=False)),
#     ("conf_020",      "LoFTR conf 0.2",                dict(min_conf=0.2)),
#     ("conf_060",      "LoFTR conf 0.6",                dict(min_conf=0.6)),
#     ("ransac_3",      "RANSAC 3 px",                   dict(ransac_thresh=3.0)),
#     ("ransac_8",      "RANSAC 8 px",                   dict(ransac_thresh=8.0)),
# ]

# # overrides keys that map to a `utils` module constant rather than a
# # register() kwarg -- popped out and monkeypatched around each call.
# _GLOBAL_OVERRIDE_ATTRS = {
#     "loftr_close_ksizes": "LOFTR_CLOSE_KSIZES",
#     "loftr_min_inliers": "LOFTR_MIN_INLIERS",
# }


# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(
#         prog="fafir-register",
#         description="Batch IR<->FAF registration over a CSV of image pairs.",
#     )
#     p.add_argument("csv", help="CSV listing image pairs (one per row).")
#     p.add_argument("output", help="Output directory for images + results.csv.")
#     p.add_argument("--ir-col", "--fixed-col", dest="ir_col", default="fixed",
#                    help="CSV column with the IR image path.")
#     p.add_argument("--faf-col", "--moving-col", dest="faf_col", default="moving",
#                    help="CSV column with the FAF image path.")
#     p.add_argument("--ir-vessel-col", "--fixed-vessel-col", dest="ir_vessel_col", default="fixed_vessel_mask",
#                    help="CSV column with the IR vessel-mask path.")
#     p.add_argument("--faf-vessel-col", "--moving-vessel-col", dest="faf_vessel_col", default="moving_vessel_mask",
#                    help="CSV column with the FAF vessel-mask path.")
#     p.add_argument("--fov-scale", type=float, default=55.0 / 30.0,
#                    help="FOV scale seed (default 55/30 ~= 1.833).")
#     p.add_argument("--size", type=int, default=224,
#                    help="Working resolution, multiple of 14 (default 224).")
#     p.add_argument("--no-loftr", action="store_true",
#                    help="Disable LoFTR feature matching.")
#     p.add_argument("--method", choices=["auto", "loftr", "full_affine"], default="auto",
#                    help="Registration method policy: auto (default), force LoFTR, or force full-affine.")
#     p.add_argument("--no-preprocess", action="store_true",
#                    help="Skip CLAHE/blur preprocessing.")
#     p.add_argument("--no-images", action="store_true",
#                    help="Only write results.csv; skip saving diagnostic images.")
#     p.add_argument("--only-row", type=int, nargs="+", default=None, metavar="N",
#                    help="Run only specific 1-based CSV row numbers.")
#     p.add_argument("--annotation-col", default="annotation_csv",
#                    help="CSV column containing annotation CSV path (default: annotation_csv).")
#     p.add_argument("--landmark-moving-image", choices=["ir", "faf"], default="faf",
#                    help="Which image owns moving landmark coordinates (default: faf).")
#     p.add_argument("--landmark-fixed-image", choices=["ir", "faf"], default="ir",
#                    help="Which image owns fixed landmark coordinates (default: ir).")
#     p.add_argument("--landmark-moving-x-col", default="FAF_x",
#                    help="Annotation CSV x-column for moving landmarks (default: FAF_x).")
#     p.add_argument("--landmark-moving-y-col", default="FAF_y",
#                    help="Annotation CSV y-column for moving landmarks (default: FAF_y).")
#     p.add_argument("--landmark-fixed-x-col", default="IR_x",
#                    help="Annotation CSV x-column for fixed landmarks (default: IR_x).")
#     p.add_argument("--landmark-fixed-y-col", default="IR_y",
#                    help="Annotation CSV y-column for fixed landmarks (default: IR_y).")
#     p.add_argument("--align-from-annotation", action="store_true",
#                    help="Use annotation keypoints to estimate alignment transform (replaces vessel/LoFTR transform).")

#     p.add_argument("--ablation", action="store_true",
#                    help="Run leave-one-out ablation study.")
#     p.add_argument("--configs", nargs="+", default=None, metavar="NAME",
#                    help=f"Ablation configs (default all): {[n for n, _, _ in ABLATIONS]}")
#     p.add_argument("--ablation-keypoints-only", action="store_true",
#                    help="Ablation only: run all configs only on rows with valid annotation keypoints.")
#     p.add_argument("--keypoints-only", action="store_true",
#                    help="Batch mode only: skip pairs with no (or too few) annotation landmarks "
#                         "instead of registering them without keypoint guidance.")
#     p.add_argument("--use-keypoints", action="store_true",
#                    help="Batch mode only: blend annotation landmarks into the LoFTR/vessel "
#                         "matching fit for every pair (same mechanism as the ablation harness's "
#                         "'with_keypoints' config), instead of using them only for evaluation.")
#     p.add_argument("--fallback-inlier-threshold", type=int, default=30, metavar="N",
#                    help="Batch mode only: treat a pair as a failed automatic registration "
#                         "(and substitute the landmark-derived transform, when available) "
#                         "whenever its LoFTR inlier count is <= N, not just when it found zero "
#                         "matches. N is a fixed count of matched keypoints, not the landmark "
#                         "ground truth, so raising it does not leak the evaluation metric. "
#                         "Default 30 (empirically: pairs at or below this inlier count fail far "
#                         "more often -- ~57%% failure / 128px+ mean landmark error -- than pairs "
#                         "above it -- ~17%% failure / ~6px mean error). Pass 0 to restore the "
#                         "strict zero-match-only trigger.")

#     # ── Held-out landmark fallback (leakage-free evaluation) ────────────────
#     p.add_argument("--fallback", dest="no_fallback", action="store_false",
#                    help="Batch mode only: opt back into the landmark/keypoint fallback and "
#                         "blending. Off by default -- the default run is fully automatic and "
#                         "never touches annotation landmarks to build a transform.")
#     p.add_argument("--no-fallback", dest="no_fallback", action="store_true", default=True,
#                    help="Batch mode only: fully automatic run (default). Never use annotation "
#                         "landmarks to build a transform (they are used only for evaluation).")
#     p.add_argument("--fallback-k-guide", type=int, default=7, metavar="K",
#                     help="Batch mode only: number of guide landmarks used for optional keypoint "
#                         "blending and held-out evaluation (default 7). Failed-case fallback fits "
#                         "from all valid annotation landmarks, independent of this setting.")
#     p.add_argument("--split-seed", type=int, default=0,
#                    help="Random seed for the guide/eval landmark split (default 0). Use the same "
#                         "seed for the automatic and semi-automatic runs so both are scored on "
#                         "identical held-out landmarks.")
#     p.add_argument("--eval-all-landmarks", action="store_true",
#                    help="Only valid with --no-fallback: score MLE on ALL landmarks instead of the "
#                         "held-out subset (safe because no landmark touches the transform).")
#     return p


# def _vessel_mask_missing(x) -> bool:
#     if x is None:
#         return True
#     if isinstance(x, str):
#         s = x.strip()
#         return (not s) or (not Path(s).exists())
#     if isinstance(x, Path):
#         return not x.exists()
#     return pd.isna(x)


# def _requested_use_loftr(args, default_use_loftr: bool) -> bool:
#     if args.method == "full_affine":
#         return False
#     if args.method == "loftr":
#         return True
#     return bool(default_use_loftr)


# def _assert_requested_method(args, winner_label: str) -> None:
#     if args.method == "loftr" and not str(winner_label).startswith("LoFTR@"):
#         raise RuntimeError(f"requested method=loftr but winner={winner_label}")


# def _iter_pairs(df: pd.DataFrame, args):
#     for n, (_, row) in enumerate(df.iterrows(), 1):
#         ir_path = row[args.ir_col]
#         faf_path = row[args.faf_col]
#         ir_v = row.get(args.ir_vessel_col, None)
#         faf_v = row.get(args.faf_vessel_col, None)
#         name = f"{n:04d}_{Path(str(faf_path)).stem}_to_{Path(str(ir_path)).stem}"
#         yield n, name, ir_path, faf_path, ir_v, faf_v


# def _estimate_affine_from_annotation(annotation_path, ir_path, faf_path, size: int, args):
#     """Estimate a 2x3 affine from annotation landmarks in working-grid coordinates.

#     Returns None if data is missing/insufficient. Returned transform always maps
#     FAF -> IR (registration direction).
#     """
#     if annotation_path is None or not str(annotation_path).strip():
#         return None

#     ann = Path(str(annotation_path))
#     if not ann.exists():
#         return None

#     try:
#         adf = pd.read_csv(ann)
#     except Exception:
#         return None
#     if adf.empty:
#         return None

#     needed = [
#         args.landmark_moving_x_col,
#         args.landmark_moving_y_col,
#         args.landmark_fixed_x_col,
#         args.landmark_fixed_y_col,
#     ]
#     if any(c not in adf.columns for c in needed):
#         return None

#     pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
#     if len(pts) < 2:
#         return None

#     moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
#     fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
#     if moving_img_path == fixed_img_path:
#         return None

#     moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
#     fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
#     if moving_raw is None or fixed_raw is None:
#         return None

#     mh, mw = moving_raw.shape[:2]
#     fh, fw = fixed_raw.shape[:2]
#     if min(mh, mw, fh, fw) <= 0:
#         return None

#     mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
#     my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
#     fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
#     fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)

#     # Fit RANSAC in native pixel space, NOT on points pre-rescaled to the
#     # size x size working grid -- a fixed reprojection threshold applied
#     # after that rescale would correspond to a much looser real-pixel
#     # tolerance on higher-resolution pairs, and could accept a genuinely
#     # inconsistent/mislabeled point as an inlier.
#     src_native = np.stack([mx, my], axis=1).astype(np.float32)
#     dst_native = np.stack([fx, fy], axis=1).astype(np.float32)

#     M_native, _ = cv2.estimateAffinePartial2D(
#         src_native,
#         dst_native,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=ANNOTATION_RANSAC_PX,
#         maxIters=5000,
#         confidence=0.999,
#     )
#     if M_native is None:
#         return None

#     if args.landmark_moving_image != "faf" or args.landmark_fixed_image != "ir":
#         return None

#     # Fold the native-space fit into the size x size working grid register()
#     # actually operates on -- the inverse of the scaling _landmark_metrics
#     # applies to go from working-grid back to native pixels.
#     sx_m, sy_m = float(size) / float(mw), float(size) / float(mh)
#     sx_f, sy_f = float(size) / float(fw), float(size) / float(fh)
#     S_f = np.diag([sx_f, sy_f])
#     S_m_inv = np.diag([1.0 / sx_m, 1.0 / sy_m])
#     M_native = M_native.astype(np.float64)
#     A_grid = S_f @ M_native[:2, :2] @ S_m_inv
#     t_grid = S_f @ M_native[:2, 2]
#     M_decl = np.hstack([A_grid, t_grid[:, None]]).astype(np.float32)
#     return M_decl


# def _read_annotation_keypoints(annotation_path, args, *, size: int, ir_path: str, faf_path: str):
#     """Return fixed/moving landmark arrays in registration working-grid coordinates.

#     The returned arrays are shaped ``(N, 2)`` and are the coordinates to feed into
#     the LoFTR feature matcher as keypoint-enhanced feature maps. If annotation data
#     is missing or insufficient, returns ``(None, None)``.
#     """
#     if annotation_path is None or not str(annotation_path).strip():
#         return None, None

#     ann = Path(str(annotation_path))
#     if not ann.exists():
#         return None, None

#     try:
#         adf = pd.read_csv(ann)
#     except Exception:
#         return None, None
#     if adf.empty:
#         return None, None

#     needed = [
#         args.landmark_moving_x_col,
#         args.landmark_moving_y_col,
#         args.landmark_fixed_x_col,
#         args.landmark_fixed_y_col,
#     ]
#     if any(c not in adf.columns for c in needed):
#         return None, None

#     pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
#     if pts.empty:
#         return None, None

#     moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
#     fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
#     if moving_img_path == fixed_img_path:
#         return None, None

#     moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
#     fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
#     if moving_raw is None or fixed_raw is None:
#         return None, None

#     mh, mw = moving_raw.shape[:2]
#     fh, fw = fixed_raw.shape[:2]
#     if min(mh, mw, fh, fw) <= 0:
#         return None, None

#     mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
#     my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
#     fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
#     fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)

#     if len(mx) < 2 or len(fx) < 2:
#         return None, None

#     moving_pts = np.stack([
#         mx * (float(size) / float(mw)),
#         my * (float(size) / float(mh)),
#     ], axis=1).astype(np.float32)
#     fixed_pts = np.stack([
#         fx * (float(size) / float(fw)),
#         fy * (float(size) / float(fh)),
#     ], axis=1).astype(np.float32)

#     if args.landmark_moving_image == "faf" and args.landmark_fixed_image == "ir":
#         return fixed_pts, moving_pts
#     return None, None


# def _farthest_point_indices(coords: np.ndarray, k: int, start_idx: int) -> list[int]:
#     """Greedily pick ``k`` row indices from ``coords`` that maximize spread.

#     Standard farthest-point sampling: start at ``start_idx``, then repeatedly
#     add whichever remaining point is farthest (by min distance) from every
#     point already chosen. A 2-3 point affine fit estimated from points
#     clustered close together is numerically unstable -- small annotation
#     click noise gets amplified into wildly wrong scale/rotation -- so this
#     keeps the guide points spread out instead of picking them uniformly at
#     random, which can (and empirically does) hand back a near-degenerate
#     triangle.
#     """
#     n = len(coords)
#     chosen = [int(start_idx)]
#     min_dist = np.linalg.norm(coords - coords[start_idx], axis=1)
#     min_dist[start_idx] = -1.0
#     while len(chosen) < k:
#         nxt = int(np.argmax(min_dist))
#         chosen.append(nxt)
#         min_dist = np.minimum(min_dist, np.linalg.norm(coords - coords[nxt], axis=1))
#         min_dist[chosen] = -1.0
#     return chosen


# def _split_annotation(annotation_path, out_dir: Path, name: str, args,
#                       k_guide: int = 3, seed: int = 0):
#     """Split one pair's landmarks into guide points and held-out evaluation points.

#     Guide points simulate a clinician clicking K corresponding landmarks.
#     Evaluation points are held out for automatic-registration scoring. Failed-
#     case landmark fallback instead fits from all valid annotation landmarks,
#     so its post-registration score is explicitly marked as in-sample.

#     Returns ``(guide_csv, eval_csv, n_guide, n_eval)``; paths are ``None``
#     when the pair has too few valid landmarks (< K + 1).
#     """
#     none = (None, None, 0, 0)
#     if annotation_path is None or not str(annotation_path).strip():
#         return none
#     ann = Path(str(annotation_path))
#     if not ann.exists():
#         return none
#     try:
#         adf = pd.read_csv(ann)
#     except Exception:
#         return none
#     cols = [args.landmark_moving_x_col, args.landmark_moving_y_col,
#             args.landmark_fixed_x_col, args.landmark_fixed_y_col]
#     if adf.empty or any(c not in adf.columns for c in cols):
#         return none

#     numeric = adf[cols].apply(pd.to_numeric, errors="coerce")
#     valid = adf[numeric.notna().all(axis=1)].reset_index(drop=True)
#     k_guide = int(k_guide)
#     if k_guide < 2:
#         return none

#     if len(valid) < k_guide + 1:
#         # Too few points to reserve any as a held-out eval set. Per explicit
#         # instruction: use every available point BOTH to fit the fallback
#         # transform AND to score it, rather than leaving the pair on a bad
#         # automatic fit. This is NOT leakage-free -- the reported MLE for a
#         # pair that ends up on the fallback here reflects the same points
#         # the transform was fit to -- but for a 2-3 point annotation there is
#         # no way to do both a fit and an honest held-out score.
#         if len(valid) < 2:
#             return none
#         split_dir = Path(out_dir) / "landmark_splits"
#         split_dir.mkdir(parents=True, exist_ok=True)
#         guide_csv = split_dir / f"{name}_guide_reused.csv"
#         valid.to_csv(guide_csv, index=False)
#         return str(guide_csv), str(guide_csv), len(valid), len(valid)

#     # Per-pair deterministic split: same seed + same pair -> same split in
#     # every run, so automatic and semi-automatic runs share eval points.
#     pair_seed = (int(seed) * 1_000_003 + sum(ord(c) for c in name)) % (2 ** 32)
#     rng = np.random.default_rng(pair_seed)

#     # Guide points are chosen to maximize spatial spread (farthest-point
#     # sampling), not picked uniformly at random -- see _farthest_point_indices.
#     # Only the random starting point is seeded by rng; the rest is greedy.
#     coords = valid[[args.landmark_moving_x_col, args.landmark_moving_y_col]].to_numpy(dtype=np.float64)
#     start_idx = int(rng.integers(len(valid)))
#     guide_idx = _farthest_point_indices(coords, k_guide, start_idx)
#     eval_idx = [i for i in range(len(valid)) if i not in guide_idx]

#     split_dir = Path(out_dir) / "landmark_splits"
#     split_dir.mkdir(parents=True, exist_ok=True)
#     guide_csv = split_dir / f"{name}_guide.csv"
#     eval_csv = split_dir / f"{name}_eval.csv"
#     valid.iloc[guide_idx].to_csv(guide_csv, index=False)
#     valid.iloc[eval_idx].to_csv(eval_csv, index=False)
#     return str(guide_csv), str(eval_csv), k_guide, len(valid) - k_guide


# def _warp_points_via_image(pts: np.ndarray, M: np.ndarray,
#                            in_hw: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
#     """Predicted landmark coordinates read off the actual warped image.

#     ``pts`` are in the moving image's own native pixel coordinates, and ``M``
#     maps moving-native pixels directly to fixed-native pixels (already
#     rescaled from the registration working-grid transform -- see
#     ``_landmark_metrics``). Each point is placed on a canvas at the moving
#     image's own resolution (``in_hw``) as a bilinear-splatted impulse (so the
#     pre-warp centroid is the exact sub-pixel point, not a rounded pixel),
#     warped with the same cv2.warpAffine call utils.warp_img uses for the real
#     registered image (now sized to the fixed image's own resolution,
#     ``out_hw``), and read back as the sub-pixel intensity centroid of the
#     result. This is the literal "coordinate after warping the image" rather
#     than an analytic M @ point re-derivation -- and it runs end-to-end in
#     original image pixels, with no intermediate resize to the registration
#     working grid.
#     """
#     in_h, in_w = in_hw
#     out_h, out_w = out_hw
#     out = np.full((len(pts), 2), np.nan, dtype=np.float32)
#     for i, (x, y) in enumerate(pts):
#         x0, y0 = int(np.floor(float(x))), int(np.floor(float(y)))
#         dx, dy = float(x) - x0, float(y) - y0
#         canvas = np.zeros((in_h, in_w), dtype=np.float32)
#         for xo, yo, wgt in (
#             (x0, y0, (1 - dx) * (1 - dy)),
#             (x0 + 1, y0, dx * (1 - dy)),
#             (x0, y0 + 1, (1 - dx) * dy),
#             (x0 + 1, y0 + 1, dx * dy),
#         ):
#             if 0 <= xo < in_w and 0 <= yo < in_h:
#                 canvas[yo, xo] += wgt
#         landed = False
#         if canvas.sum() > 1e-6:
#             warped = cv2.warpAffine(canvas, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
#                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
#             total = float(warped.sum())
#             if total > 1e-6:
#                 ys, xs = np.nonzero(warped)
#                 weights = warped[ys, xs]
#                 out[i, 0] = float(np.sum(xs * weights) / total)
#                 out[i, 1] = float(np.sum(ys * weights) / total)
#                 landed = True
#         if not landed:
#             # The point warped off the canvas entirely -- a genuine
#             # registration failure, not missing data. Fall back to the exact
#             # affine map so it still contributes a (large) error instead of
#             # silently dropping the pair from every metric's average.
#             out[i, 0] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
#             out[i, 1] = M[1, 0] * x + M[1, 1] * y + M[1, 2]
#     return out


# def _landmark_metrics(annotation_path, transform, ir_path, faf_path, size: int, args):
#     """Compute before/after landmark MLE for one pair, in original image pixels.

#     The registration transform is estimated on the ``size``x``size`` working
#     grid, but is rescaled here to map moving-native pixels directly to
#     fixed-native pixels before being applied to the landmark points -- so the
#     reported MLE is computed end-to-end in original image pixel space, with
#     no intermediate resize to the working grid. If moving/fixed landmark
#     ownership is IR->FAF, the inverse transform is used for landmark
#     transfer.
#     """
#     out = {
#         "landmark_count": 0,
#         "landmark_mle_before": None,
#         "landmark_mle_after": None,
#         "landmark_mle_delta": None,
#     }
#     if annotation_path is None or not str(annotation_path).strip():
#         return out

#     ann = Path(str(annotation_path))
#     if not ann.exists():
#         return out

#     try:
#         adf = pd.read_csv(ann)
#     except Exception:
#         return out
#     if adf.empty:
#         return out

#     needed = [
#         args.landmark_moving_x_col,
#         args.landmark_moving_y_col,
#         args.landmark_fixed_x_col,
#         args.landmark_fixed_y_col,
#     ]
#     if any(c not in adf.columns for c in needed):
#         return out

#     pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
#     if pts.empty:
#         return out

#     # Map user-declared moving/fixed landmarks to image files.
#     moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
#     fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
#     if moving_img_path == fixed_img_path:
#         return out

#     moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
#     fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
#     if moving_raw is None or fixed_raw is None:
#         return out

#     mh, mw = moving_raw.shape[:2]
#     fh, fw = fixed_raw.shape[:2]
#     if min(mh, mw, fh, fw) <= 0:
#         return out

#     mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
#     my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
#     fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
#     fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)
#     n = int(len(pts))
#     if n <= 0:
#         return out

#     # Registration transform is FAF->IR.
#     if args.landmark_moving_image != "faf" or args.landmark_fixed_image != "ir":
#         return out

#     # The registration transform maps size x size working-grid coordinates:
#     # moving-native -[scale by S_m]-> working grid -[M]-> working grid
#     # -[scale by S_f^-1]-> fixed-native. Fold the two scalings into M so it
#     # maps moving-native pixels directly to fixed-native pixels, letting the
#     # landmark points be warped without ever resizing through the working
#     # grid.
#     M_grid = np.asarray(transform, dtype=np.float64)
#     sx_m, sy_m = float(size) / float(mw), float(size) / float(mh)
#     sx_f, sy_f = float(size) / float(fw), float(size) / float(fh)
#     S_m = np.diag([sx_m, sy_m])
#     S_f_inv = np.diag([1.0 / sx_f, 1.0 / sy_f])
#     A = S_f_inv @ M_grid[:2, :2] @ S_m
#     t = S_f_inv @ M_grid[:2, 2]
#     M_use = np.hstack([A, t[:, None]]).astype(np.float32)

#     src = np.stack([mx, my], axis=1).astype(np.float32)

#     # Predicted coordinate read off the actual warped image (not an analytic
#     # M @ point re-derivation) -- see _warp_points_via_image. Canvas sizes
#     # are each image's own native resolution.
#     pred = _warp_points_via_image(src, M_use, in_hw=(mh, mw), out_hw=(fh, fw))
#     valid = ~np.isnan(pred).any(axis=1)
#     if not valid.any():
#         return out

#     pred_x = pred[valid, 0]
#     pred_y = pred[valid, 1]

#     # Baseline before-registration mapping by image-size ratio only.
#     base_x = mx[valid] * (float(fw) / float(mw))
#     base_y = my[valid] * (float(fh) / float(mh))

#     err_before = np.sqrt((base_x - fx[valid]) ** 2 + (base_y - fy[valid]) ** 2)
#     err_after = np.sqrt((pred_x - fx[valid]) ** 2 + (pred_y - fy[valid]) ** 2)

#     mle_before = float(np.mean(err_before))
#     mle_after = float(np.mean(err_after))
#     out.update({
#         "landmark_count": int(valid.sum()),
#         "landmark_mle_before": round(mle_before, 4),
#         "landmark_mle_after": round(mle_after, 4),
#         "landmark_mle_delta": round(mle_before - mle_after, 4),
#     })
#     return out


# def _print_pair_metrics(metrics: dict):
#     dice = metrics.get("vessel_dice_after", metrics.get("dice_after"))
#     asd = metrics.get("vessel_asd_after", metrics.get("asd_after"))
#     hd95 = metrics.get("vessel_hd95_after", metrics.get("hd95_after"))
#     ncc = metrics.get("image_ncc_after", metrics.get("ncc_after"))

#     def _fmt(v, nd=4):
#         return "NA" if v is None else f"{float(v):.{nd}f}"

#     print(
#         "  metrics:"
#         f" vessel_dice_after={_fmt(dice)}"
#         f" vessel_asd_after={_fmt(asd, 3)}"
#         f" vessel_hd95_after={_fmt(hd95, 3)}"
#         f" image_ncc_after={_fmt(ncc)}"
#     )


# def _pick_stem(sub: pd.DataFrame, preferred: str, fallback: str | None):
#     cb, ca = f"{preferred}_before", f"{preferred}_after"
#     if cb in sub.columns and ca in sub.columns:
#         return preferred
#     if fallback is not None:
#         cb, ca = f"{fallback}_before", f"{fallback}_after"
#         if cb in sub.columns and ca in sub.columns:
#             return fallback
#     return None


# def _append_rows(rows: list[dict], sub: pd.DataFrame, group: str, category: str,
#                  metrics: list[tuple[str, str, str | None]]):
#     for shown, preferred, fallback in metrics:
#         stem = _pick_stem(sub, preferred, fallback)
#         if stem is None:
#             continue
#         cb, ca = f"{stem}_before", f"{stem}_after"
#         b = pd.to_numeric(sub[cb], errors="coerce")
#         a = pd.to_numeric(sub[ca], errors="coerce")
#         valid = b.notna() & a.notna()
#         if not valid.any():
#             continue
#         b = b[valid]
#         a = a[valid]
#         d = a - b
#         rows.append(dict(
#             group=group,
#             category=category,
#             metric=shown,
#             metric_stem=stem,
#             n=int(valid.sum()),
#             before_mean=float(b.mean()),
#             before_std=float(b.std(ddof=0)),
#             after_mean=float(a.mean()),
#             after_std=float(a.std(ddof=0)),
#             delta_mean=float(d.mean()),
#             delta_std=float(d.std(ddof=0)),
#         ))


# def _landmark_auc(errors, max_threshold: float = LANDMARK_AUC_MAX_PX) -> Optional[float]:
#     values = pd.to_numeric(pd.Series(errors), errors="coerce").dropna().to_numpy(dtype=np.float32)
#     if values.size == 0:
#         return None
#     max_threshold = float(max_threshold)
#     if max_threshold <= 0:
#         return None
#     thresholds = np.linspace(0.0, max_threshold, 501, dtype=np.float32)
#     success = np.array([(values <= t).mean() for t in thresholds], dtype=np.float32)
#     _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 1.x and 2.x
#     auc = float(_trapz(success, thresholds) / max_threshold)
#     return round(auc, 4)


# def _append_landmark_auc_rows(rows: list[dict], sub: pd.DataFrame, group: str, category: str):
#     before_col = "landmark_mle_before" if "landmark_mle_before" in sub.columns else None
#     after_col = "landmark_mle_after" if "landmark_mle_after" in sub.columns else None
#     if before_col is None or after_col is None:
#         return

#     before = pd.to_numeric(sub[before_col], errors="coerce").dropna()
#     after = pd.to_numeric(sub[after_col], errors="coerce").dropna()
#     if before.empty or after.empty:
#         return

#     rows.append(dict(
#         group=group,
#         category=category,
#         metric=_AUC_LABEL,
#         metric_stem=_AUC_STEM,
#         n=int(min(len(before), len(after))),
#         before_mean=_landmark_auc(before),
#         before_std=0.0,
#         after_mean=_landmark_auc(after),
#         after_std=0.0,
#         delta_mean=None if (_landmark_auc(before) is None or _landmark_auc(after) is None)
#         else round(_landmark_auc(after) - _landmark_auc(before), 4),
#         delta_std=0.0,
#     ))


# def _stats(sub: pd.DataFrame, stem: str, phase: str = "after"):
#     col = f"{stem}_{phase}"
#     if col not in sub.columns:
#         return None
#     v = pd.to_numeric(sub[col], errors="coerce").dropna()
#     if v.empty:
#         return None
#     return float(v.mean()), float(v.std(ddof=0)), int(len(v))


# def summarize_ablation(df: pd.DataFrame, configs) -> pd.DataFrame:
#     """Summarize ablation results as config-level after-metric rows."""
#     ok = df[df["status"] == "ok"] if "status" in df.columns else df
#     rows: list[dict] = []
#     for name, label, _ in configs:
#         sub = ok[ok["config"] == name]
#         if "landmark_mle_after" in sub.columns:
#             sub = sub[pd.to_numeric(sub["landmark_mle_after"], errors="coerce").notna()]

#         for stem, out_name, higher, phase in ABLATION_METRICS:
#             if out_name == "success_rate":
#                 mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
#                 if mle.empty:
#                     continue
#                 success = float((mle <= LANDMARK_SR_THRESHOLD_PX).mean())
#                 rows.append(dict(
#                     config=name,
#                     label=label,
#                     metric=out_name,
#                     n=int(len(mle)),
#                     mean=round(success, 4),
#                     std=0.0,
#                     higher_is_better=higher,
#                 ))
#                 continue

#             if out_name == "AUC@37.50px":
#                 mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
#                 if mle.empty:
#                     continue
#                 auc = _landmark_auc(mle, max_threshold=LANDMARK_AUC_MAX_PX)
#                 if auc is None:
#                     continue
#                 rows.append(dict(
#                     config=name,
#                     label=label,
#                     metric=out_name,
#                     n=int(len(mle)),
#                     mean=round(auc, 4),
#                     std=0.0,
#                     higher_is_better=higher,
#                 ))
#                 continue

#             st = _stats(sub, stem, phase=phase)
#             if st is None:
#                 continue
#             mean, std, n = st
#             rows.append(dict(
#                 config=name,
#                 label=label,
#                 metric=out_name,
#                 n=n,
#                 mean=round(mean, 4),
#                 std=round(std, 4),
#                 higher_is_better=higher,
#             ))
#     return pd.DataFrame(rows)


# def _print_block(rows_df: pd.DataFrame, group: str, category: str):
#     block = rows_df[(rows_df["group"] == group) & (rows_df["category"] == category)]
#     if block.empty:
#         return
#     print(f"\n--- {category.title()} ---")
#     print(f"  {'metric':<12} {'before (mean ± std)':>26} {'after (mean ± std)':>26} {'score delta ± std':>24}")
#     for _, r in block.iterrows():
#         before = f"{r['before_mean']:.4f} ± {r['before_std']:.4f}"
#         after = f"{r['after_mean']:.4f} ± {r['after_std']:.4f}"
#         delta = f"{r['delta_mean']:+.4f} ± {r['delta_std']:.4f}"
#         print(f"  {r['metric']:<12} {before:>26} {after:>26} {delta:>24}")


# def summarize(df: pd.DataFrame):
#     if df.empty:
#         print("No rows to summarize.")
#         return None

#     ok = df[df["status"] == "ok"] if "status" in df.columns else df
#     if ok.empty:
#         print("No successful rows to summarize.")
#         return None

#     # Restrict every category (not just landmark) to pairs that actually
#     # have a landmark MLE, so vessel/image/landmark are all reported over
#     # the same apples-to-apples subset instead of vessel/image silently
#     # including extra pairs that have no annotation data to score.
#     if "landmark_mle_after" in ok.columns:
#         ok_lm = ok[pd.to_numeric(ok["landmark_mle_after"], errors="coerce").notna()]
#     else:
#         ok_lm = ok

#     rows: list[dict] = []

#     # Overall block
#     _append_rows(rows, ok_lm, "overall", "vessel", SUMMARY_METRICS_VESSEL)
#     _append_rows(rows, ok_lm, "overall", "image", SUMMARY_METRICS_IMAGE)
#     _append_rows(rows, ok_lm, "overall", "landmark", SUMMARY_METRICS_LANDMARK)
#     _append_landmark_auc_rows(rows, ok_lm, "overall", "landmark")

#     if not rows:
#         print("No comparable before/after metrics available.")
#         return None

#     out = pd.DataFrame(rows)

#     print("\nSummary metrics (mean ± std):")
#     print(f"  included (status=ok): {len(ok)}/{len(df)}")
#     print(f"  included (with landmark MLE): {len(ok_lm)}/{len(df)}")

#     for g in ["overall"]:
#         n_g = len(ok_lm)
#         print(f"\n=== {g.upper()} (n={n_g}) ===")
#         _print_block(out, g, "vessel")
#         _print_block(out, g, "image")
#         _print_block(out, g, "landmark")

#     def _print_landmark_success(label: str, sub: pd.DataFrame):
#         if "landmark_mle_after" not in sub.columns:
#             return
#         mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
#         # Eq. 11: a pair only counts as a landmark success if it has a valid
#         # MLE (excludes rows with too few correspondences to estimate a
#         # transform in the first place -- those already dropped by dropna)
#         # AND that MLE is <= T_SR.
#         if mle.empty:
#             return
#         n_mle = len(mle)
#         sr = int((mle <= LANDMARK_SR_THRESHOLD_PX).sum())
#         fr = n_mle - sr
#         print(f"\n{label} -- classified: {n_mle}")
#         print(f"  success (MLE <= {LANDMARK_SR_THRESHOLD_PX:.2f}px): {sr}/{n_mle} ({100.0*sr/n_mle:.2f}%)")
#         print(f"  failure (MLE >  {LANDMARK_SR_THRESHOLD_PX:.2f}px): {fr}/{n_mle} ({100.0*fr/n_mle:.2f}%)")

#     _print_landmark_success("Landmark MLE (overall)", ok)

#     return out


# # Legacy retry/upscaling block intentionally disabled.
# # The single registration pass is kept on the configured size and does not
# # automatically re-run at a higher pixel resolution based on Dice/Wasserstein.


# def run_batch(df: pd.DataFrame, out_dir: Path, args) -> int:
#     """Batch registration with a leakage-free landmark fallback.

#     Landmark roles per pair (see ``_split_annotation``):
#             * guide landmarks (K points)  -> optional keypoint guidance and the
#                 held-out evaluation split for automatic runs.
#       * eval landmarks (the rest)   -> used only to compute MLE; never used
#         to build any transform.

#     Modes:
#       * default (``--no-fallback``): fully automatic. No landmark ever
#         touches the transform. MLE on held-out eval landmarks (or on all
#         landmarks with ``--eval-all-landmarks``).
#       * ``--fallback``: automatic registration; if it fails (no LoFTR
#         transform, an exception, or LoFTR inliers <=
#         ``--fallback-inlier-threshold``), the transform is re-estimated from
#         all valid annotation landmarks.
#     The failure trigger uses only the method's own internal signals, never
#     the landmark ground truth.
#     """
#     print(f"fafir-register: {len(df)} pair(s) -> {out_dir.resolve()}\n")

#     if args.eval_all_landmarks and not args.no_fallback:
#         print("error: --eval-all-landmarks is only valid with --no-fallback", file=sys.stderr)
#         return 2
#     if args.align_from_annotation:
#         print("WARNING: --align-from-annotation builds EVERY transform from landmarks; "
#               "results are not an automatic registration.", file=sys.stderr)
#     if args.use_keypoints:
#         print("NOTE: --use-keypoints blends only the K guide landmarks "
#               "into matching; report this run as landmark-guided.", file=sys.stderr)

#     mode = "automatic" if args.no_fallback else "semi_automatic_fallback"
#     print(f"mode={mode}  eval_guide_k={args.fallback_k_guide}  split_seed={args.split_seed}  "
#           f"fallback_inlier_threshold={args.fallback_inlier_threshold}\n")

#     records = []
#     results_csv = out_dir / "results.csv"
#     matching_csv = out_dir / "matching.csv"

#     for n, name, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
#         if args.only_row and n not in args.only_row:
#             continue

#         print(f"[{n}/{len(df)}] {Path(str(faf_path)).name} -> {Path(str(ir_path)).name}")
#         base_record = dict(
#             name=name,
#             ir=ir_path,
#             faf=faf_path,
#             ir_name=Path(str(ir_path)).name,
#             faf_name=Path(str(faf_path)).name,
#             mode=mode,
#         )
#         ann_path = df.iloc[n - 1].get(args.annotation_col, None) if args.annotation_col in df.columns else None
#         base_record["annotation_csv"] = ann_path

#         # ── Split landmarks for held-out evaluation and optional keypoint guidance ──
#         guide_csv, eval_csv, n_guide, n_eval = _split_annotation(
#             ann_path, out_dir, name, args,
#             k_guide=args.fallback_k_guide, seed=args.split_seed,
#         )
#         if args.no_fallback and args.eval_all_landmarks:
#             score_csv = ann_path          # safe: landmarks never touch the transform
#             scored_on = "all_landmarks"
#         elif guide_csv is not None and eval_csv is not None and str(guide_csv) == str(eval_csv):
#             # Too few points (< K + 1, but >= 2) to hold any out. Per explicit
#             # instruction, the fallback is fit AND scored on the same points
#             # -- NOT leakage-free, unlike every other branch here, but the
#             # only option when there aren't enough points to split.
#             score_csv = eval_csv
#             scored_on = "guide_reused_not_leakage_free"
#         elif eval_csv is not None:
#             score_csv = eval_csv          # held-out landmarks only
#             scored_on = "heldout_landmarks"
#         else:
#             # Fewer than 2 valid points, or no annotation at all -- no guide
#             # set exists, so the fallback is impossible for this pair and no
#             # landmark can touch its transform -> scoring on all landmarks is
#             # leakage-free, and the pair is kept instead of silently dropped
#             # from the metrics. The same rule applies in every mode, so
#             # automatic and semi-automatic runs still score identical
#             # landmarks.
#             score_csv = ann_path
#             scored_on = "all_landmarks_no_split_possible"
#         base_record.update(
#             guide_csv=guide_csv, eval_csv=eval_csv, score_csv=score_csv,
#             scored_on=scored_on,
#             n_guide_landmarks=n_guide, n_eval_landmarks=n_eval,
#         )

#         if args.keypoints_only and score_csv is None:
#             records.append(dict(
#                 **base_record,
#                 status="skipped",
#                 skip_reason="insufficient_landmarks_for_split",
#                 excluded_from_metrics=True,
#                 exclusion_reason="skipped_by_keypoints_only_filter",
#             ))
#             pd.DataFrame(records).to_csv(results_csv, index=False)
#             continue

#         if _vessel_mask_missing(ir_v) or _vessel_mask_missing(faf_v):
#             records.append(dict(
#                 **base_record,
#                 status="skipped",
#                 skip_reason="missing_vessel_mask",
#                 excluded_from_metrics=True,
#                 exclusion_reason="system_failure",
#             ))
#             pd.DataFrame(records).to_csv(results_csv, index=False)
#             continue

#         try:
#             forced_transform = None
#             forced_label = None

#             # Guide landmarks are reserved for optional matching guidance.
#             guide_fixed = guide_moving = None
#             if guide_csv is not None:
#                 guide_fixed, guide_moving = _read_annotation_keypoints(
#                     annotation_path=guide_csv, args=args, size=args.size,
#                     ir_path=ir_path, faf_path=faf_path,
#                 )

#             if args.align_from_annotation and guide_csv is not None:
#                 forced_transform = _estimate_affine_from_annotation(
#                     annotation_path=guide_csv, ir_path=ir_path, faf_path=faf_path,
#                     size=args.size, args=args,
#                 )
#                 if forced_transform is not None:
#                     forced_label = "annotation_landmarks"

#             if args.use_keypoints and not args.no_fallback:
#                 blend_keypoints_fixed, blend_keypoints_moving = guide_fixed, guide_moving
#             else:
#                 blend_keypoints_fixed = blend_keypoints_moving = None

#             # Fallback fitting reads the full annotation file directly; it must
#             # not depend on the guide count used by evaluation/keypoint blending.
#             fallback_allowed = not args.no_fallback

#             def _all_landmark_fallback_transform():
#                 # Fit every valid annotation correspondence; RANSAC handles outliers.
#                 return _estimate_affine_from_annotation(
#                     annotation_path=ann_path, ir_path=ir_path, faf_path=faf_path,
#                     size=args.size, args=args,
#                 )

#             def _register_forced(M, label):
#                 return register(
#                     ir_image=ir_path, faf_image=faf_path,
#                     ir_vessel=ir_v, faf_vessel=faf_v,
#                     fov_scale=args.fov_scale, size=args.size,
#                     preprocess=not args.no_preprocess,
#                     use_loftr=False,
#                     forced_transform=M, forced_label=label,
#                 )

#             fallback_used = False
#             fallback_reason = ""
#             auto_label = None
#             auto_inliers = None

#             t0 = time.perf_counter()
#             try:
#                 result = register(
#                     ir_image=ir_path,
#                     faf_image=faf_path,
#                     ir_vessel=ir_v,
#                     faf_vessel=faf_v,
#                     fov_scale=args.fov_scale,
#                     size=args.size,
#                     preprocess=not args.no_preprocess,
#                     use_loftr=_requested_use_loftr(args, not args.no_loftr),
#                     forced_transform=forced_transform,
#                     forced_label=(forced_label or "annotation_landmarks"),
#                     keypoints_fixed=blend_keypoints_fixed,
#                     keypoints_moving=blend_keypoints_moving,
#                 )
#                 _assert_requested_method(args, result.label)
#             except Exception as auto_exc:
#                 if not (fallback_allowed and forced_transform is None):
#                     raise
#                 M_fb = _all_landmark_fallback_transform()
#                 if M_fb is None:
#                     raise
#                 result = _register_forced(M_fb, "annotation_landmarks_fallback")
#                 fallback_used = True
#                 fallback_reason = f"automatic_exception: {auto_exc}"
#                 auto_label = "exception"

#             # ── Failure detection: internal signals only (no ground truth) ──
#             if not fallback_used and forced_transform is None:
#                 auto_label = result.label
#                 auto_inliers = int((result.metrics or {}).get("keypoint_matches", 0) or 0)
#                 loftr_failed = not str(result.label).startswith("LoFTR@")
#                 low_inliers = auto_inliers <= args.fallback_inlier_threshold
#                 if fallback_allowed and (loftr_failed or low_inliers):
#                     M_fb = _all_landmark_fallback_transform()
#                     if M_fb is not None:
#                         result = _register_forced(M_fb, "annotation_landmarks_fallback")
#                         fallback_used = True
#                         fallback_reason = "no_loftr_transform" if loftr_failed \
#                             else f"low_inliers({auto_inliers}<={args.fallback_inlier_threshold})"

#             if fallback_used:
#                 score_csv = ann_path
#                 scored_on = "all_landmarks_in_sample_fallback"
#                 base_record.update(score_csv=score_csv, scored_on=scored_on)

#             # Fallback metrics are in-sample because the fit used all annotations;
#             # automatic-registration metrics use only the held-out evaluation split.
#             lm = _landmark_metrics(
#                 annotation_path=score_csv, transform=result.transform,
#                 ir_path=ir_path, faf_path=faf_path, size=args.size, args=args,
#             )

#             # "before" never touches the transform (it's the raw pre-registration
#             # distance), so scoring it on ALL landmarks instead of just the
#             # held-out subset doesn't leak anything -- and it makes the
#             # reported baseline comparable across modes (--no-fallback
#             # --eval-all-landmarks, ablation, and this fallback run all then
#             # report the same "before" for a given pair). "after"/AUC/SR stay
#             # held-out-only, since those DO depend on the fitted transform.
#             if lm.get("landmark_count", 0) > 0 and str(score_csv) != str(ann_path):
#                 lm_all = _landmark_metrics(
#                     annotation_path=ann_path, transform=result.transform,
#                     ir_path=ir_path, faf_path=faf_path, size=args.size, args=args,
#                 )
#                 full_before = lm_all.get("landmark_mle_before")
#                 if full_before is not None:
#                     lm["landmark_mle_before"] = full_before
#                     if lm.get("landmark_mle_after") is not None:
#                         lm["landmark_mle_delta"] = round(full_before - lm["landmark_mle_after"], 4)

#             runtime_sec = time.perf_counter() - t0
#         except Exception as e:
#             records.append(dict(
#                 **base_record,
#                 status="error",
#                 error=str(e),
#                 excluded_from_metrics=True,
#                 exclusion_reason="registration_error",
#             ))
#             pd.DataFrame(records).to_csv(results_csv, index=False)
#             continue

#         if not args.no_images:
#             result.save(out_dir, name=name)

#         if fallback_used:
#             annotation_used_for_transform = "forced_all_landmarks"
#         elif result.label == "annotation_landmarks":
#             annotation_used_for_transform = "forced_guide_only"
#         elif blend_keypoints_fixed is not None:
#             annotation_used_for_transform = "blended_guide_only"
#         else:
#             annotation_used_for_transform = "none"

#         records.append(dict(
#             **base_record,
#             status="ok",
#             label=result.label,
#             auto_label=auto_label,
#             auto_inliers=auto_inliers,
#             fallback_used=fallback_used,
#             fallback_reason=fallback_reason,
#             scale=result.scale,
#             runtime_sec=round(runtime_sec, 3),
#             annotation_used_for_transform=annotation_used_for_transform,
#             **result.metrics,
#             **lm,
#         ))
#         print(f"  runtime={runtime_sec:.2f}s")
#         _print_pair_metrics(result.metrics)
#         if fallback_used:
#             print(f"  FALLBACK used ({fallback_reason}) -> transform from all valid annotation landmarks")
#         if lm.get("landmark_count", 0) > 0:
#             print("  landmarks (held-out): "
#                   f"n={lm['landmark_count']} "
#                   f"mle_before={lm['landmark_mle_before']:.3f}px "
#                   f"mle_after={lm['landmark_mle_after']:.3f}px "
#                   f"delta={lm['landmark_mle_delta']:+.3f}px")
#         pd.DataFrame(records).to_csv(results_csv, index=False)

#     df_out = pd.DataFrame(records)
#     df_out.to_csv(results_csv, index=False)

#     cols = ["name", "ir_name", "faf_name", "status", "keypoint_matches", "keypoint_matches_total"]
#     match_rows = []
#     for r in records:
#         match_rows.append({
#             "name": r.get("name"),
#             "ir_name": r.get("ir_name"),
#             "faf_name": r.get("faf_name"),
#             "status": r.get("status"),
#             "keypoint_matches": r.get("keypoint_matches", 0),
#             "keypoint_matches_total": r.get("keypoint_matches_total", 0),
#         })
#     pd.DataFrame(match_rows, columns=cols).to_csv(matching_csv, index=False)

#     summary = summarize(df_out)
#     if summary is not None:
#         summary_csv = out_dir / "summary.csv"
#         numeric_cols = [
#             "before_mean", "before_std", "after_mean", "after_std", "delta_mean", "delta_std"
#         ]
#         summary_csv_df = summary.copy()
#         for c in numeric_cols:
#             if c in summary_csv_df.columns:
#                 summary_csv_df[c] = pd.to_numeric(summary_csv_df[c], errors="coerce").round(4)
#         summary_csv_df.to_csv(summary_csv, index=False)
#         print(f"\nSummary -> {summary_csv.resolve()}")

#     total = len(records)
#     ok_n = int((df_out["status"] == "ok").sum()) if total else 0
#     skipped = int((df_out["status"] == "skipped").sum()) if total else 0
#     errors = int((df_out["status"] == "error").sum()) if total else 0
#     excluded = total - ok_n
#     ok_df = df_out[df_out["status"] == "ok"] if "status" in df_out.columns else df_out

#     # ── Fallback report (automatic vs fallback subsets) ──
#     if "fallback_used" in ok_df.columns and not ok_df.empty:
#         fb = ok_df["fallback_used"].astype(bool)
#         n_fb = int(fb.sum())
#         print(f"\n=== Fallback report ({mode}) ===")
#         print(f"  automatic transform kept : {len(ok_df) - n_fb}/{len(ok_df)}")
#         print(f"  guide-landmark fallback  : {n_fb}/{len(ok_df)}")
#         if "landmark_mle_after" in ok_df.columns:
#             mle = pd.to_numeric(ok_df["landmark_mle_after"], errors="coerce")
#             for lbl, sel in (("automatic pairs", ~fb), ("fallback pairs", fb), ("all pairs", fb | ~fb)):
#                 v = mle[sel].dropna()
#                 if v.empty:
#                     continue
#                 sr = 100.0 * float((v <= LANDMARK_SR_THRESHOLD_PX).mean())
#                 print(f"  {lbl:<16} n={len(v):<3} MLE mean={v.mean():.2f} ± {v.std(ddof=0):.2f}  "
#                       f"median={v.median():.2f}  SR@{LANDMARK_SR_THRESHOLD_PX:.2f}px={sr:.1f}%  "
#                       f"{_AUC_LABEL}={_landmark_auc(v)}")

#     if "landmark_mle_after" in ok_df.columns:
#         lm_after = pd.to_numeric(ok_df["landmark_mle_after"], errors="coerce").dropna()
#         lm_before = pd.to_numeric(ok_df.get("landmark_mle_before"), errors="coerce").dropna()
#         if not lm_after.empty:
#             print(f"\nMean landmark error (after, held-out): {float(lm_after.mean()):.4f}px over {len(lm_after)} pair(s)")
#         if not lm_before.empty and len(lm_before) == len(lm_after):
#             print(f"Mean landmark error (before): {float(lm_before.mean()):.4f}px")
#             print(f"Mean landmark improvement (before-after): {float(lm_before.mean() - lm_after.mean()):+.4f}px")

#     if "runtime_sec" in ok_df.columns:
#         rt = pd.to_numeric(ok_df["runtime_sec"], errors="coerce").dropna()
#         if not rt.empty:
#             print(f"\nRuntime: mean={rt.mean():.2f}s  median={rt.median():.2f}s  "
#                   f"min={rt.min():.2f}s  max={rt.max():.2f}s  total={rt.sum():.1f}s over {len(rt)} pair(s)")

#     print(f"Done. included for metrics: {ok_n}/{total}; excluded: {excluded} (system_failure/skipped={skipped}, error={errors}). Metrics -> {results_csv.resolve()}")
#     return 0


# def run_ablation(df: pd.DataFrame, out_dir: Path, args) -> int:
#     configs = ABLATIONS
#     if args.configs:
#         known = {n for n, _, _ in ABLATIONS}
#         bad = [c for c in args.configs if c not in known]
#         if bad:
#             print(f"error: unknown config(s) {bad}; choices: {sorted(known)}", file=sys.stderr)
#             return 2
#         configs = [c for c in ABLATIONS if c[0] in args.configs]

#     records = []
#     results_csv = out_dir / "ablation_results.csv"
#     print(f"ablation: {len(df)} pair(s) x {len(configs)} config(s) -> {out_dir.resolve()}\n")

#     for n, pair, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
#         if args.only_row and n not in args.only_row:
#             continue
#         ann_path = df.iloc[n - 1].get(args.annotation_col, None) if args.annotation_col in df.columns else None

#         # Same held-out guide/eval split as run_batch (same k_guide/seed), so
#         # every ablation config -- "full" included -- is scored on exactly
#         # the same landmarks as a run_batch/out_11 automatic run, and the two
#         # are numerically comparable.
#         _, eval_csv, _, _ = _split_annotation(
#             ann_path, out_dir, pair, args,
#             k_guide=args.fallback_k_guide, seed=args.split_seed,
#         )
#         score_csv = eval_csv if eval_csv is not None else ann_path

#         # The annotation-landmark transform doesn't depend on engine knobs,
#         # so it's cached per working size and shared by every config that
#         # needs it -- via `use_keypoints`, --ablation-keypoints-only below,
#         # or the "full"/"with_keypoints" rescue further down.
#         kp_cache: dict = {}

#         def _kp_for_size(sz):
#             if sz not in kp_cache:
#                 kf, km = _read_annotation_keypoints(
#                     annotation_path=ann_path, args=args, size=sz,
#                     ir_path=ir_path, faf_path=faf_path,
#                 )
#                 kt = _estimate_affine_from_annotation(
#                     annotation_path=ann_path, ir_path=ir_path, faf_path=faf_path,
#                     size=sz, args=args,
#                 )
#                 has_fb = (
#                     kf is not None and km is not None
#                     and min(len(kf), len(km)) >= 2
#                 )
#                 kp_cache[sz] = (kf, km, kt, has_fb)
#             return kp_cache[sz]

#         if args.ablation_keypoints_only:
#             _, _, kp_transform, _ = _kp_for_size(args.size)
#             if kp_transform is None:
#                 for name, _label, _overrides in configs:
#                     records.append(dict(
#                         pair=pair,
#                         config=name,
#                         ir=ir_path,
#                         faf=faf_path,
#                         status="skipped",
#                         error="insufficient_annotation_landmarks",
#                     ))
#                 pd.DataFrame(records).to_csv(results_csv, index=False)
#                 continue
#         if _vessel_mask_missing(ir_v) or _vessel_mask_missing(faf_v):
#             for name, _label, _overrides in configs:
#                 records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
#                                     status="skipped", error="missing_vessel_mask"))
#             pd.DataFrame(records).to_csv(results_csv, index=False)
#             continue

#         for name, _label, overrides in configs:
#             # Base engine knobs, overridden per-config below (never passed
#             # twice -- `overrides` may redefine any of these, e.g. `size` or
#             # `use_loftr`).
#             call_kwargs = dict(
#                 fov_scale=args.fov_scale,
#                 size=args.size,
#                 preprocess=not args.no_preprocess,
#                 use_loftr=_requested_use_loftr(args, not args.no_loftr),
#             )
#             global_saved = {}
#             use_keypoints = False
#             use_rescue = False
#             for key, value in overrides.items():
#                 if key == "use_keypoints":
#                     use_keypoints = bool(value)
#                     continue
#                 if key == "_rescue":
#                     use_rescue = bool(value)
#                     continue
#                 attr = _GLOBAL_OVERRIDE_ATTRS.get(key)
#                 if attr is not None:
#                     global_saved[attr] = getattr(utils, attr)
#                     setattr(utils, attr, value)
#                 else:
#                     if key == "use_loftr" and args.method != "auto":
#                         continue
#                     call_kwargs[key] = value

#             if use_keypoints:
#                 k_fixed, k_moving, kp_transform, _ = _kp_for_size(call_kwargs["size"])
#                 if k_fixed is not None and k_moving is not None:
#                     call_kwargs["keypoints_fixed"] = k_fixed
#                     call_kwargs["keypoints_moving"] = k_moving
#                 if kp_transform is None and ("keypoints_fixed" not in call_kwargs or "keypoints_moving" not in call_kwargs):
#                     records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
#                                         status="skipped",
#                                         error="insufficient_annotation_landmarks"))
#                     for attr, value in global_saved.items():
#                         setattr(utils, attr, value)
#                     continue
#                 if kp_transform is not None and "keypoints_fixed" not in call_kwargs:
#                     call_kwargs["forced_transform"] = kp_transform
#                     call_kwargs["forced_label"] = "annotation_landmarks"

#             try:
#                 result = register(
#                     ir_image=ir_path,
#                     faf_image=faf_path,
#                     ir_vessel=ir_v,
#                     faf_vessel=faf_v,
#                     **call_kwargs,
#                 )
#                 _assert_requested_method(args, result.label)
#             except Exception as e:
#                 records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
#                                     status="error", error=str(e),
#                                     excluded_from_metrics=True,
#                                     exclusion_reason="registration_error"))
#                 continue
#             finally:
#                 for attr, value in global_saved.items():
#                     setattr(utils, attr, value)

#             # Landmark MLE/AUC against the held-out eval split (same
#             # k_guide/seed as run_batch, so "full" is numerically comparable
#             # to a run_batch/out_11 automatic run), regardless of whether
#             # this config used the keypoints to derive its transform -- lets
#             # every ablation be scored against the same ground truth. No
#             # post-hoc keypoint rescue here (unlike run_batch), for any
#             # config including "full" and "with_keypoints": the whole point
#             # of an ablation is each config's own raw performance, and
#             # rescuing would mask the very differences it exists to show --
#             # e.g. with_keypoints vs full is supposed to demonstrate whether
#             # blending keypoints into matching helps; giving both the same
#             # rescue erases that comparison exactly like it did for the
#             # other 17 knob configs.
#             lm = _landmark_metrics(
#                 annotation_path=score_csv, transform=result.transform,
#                 ir_path=ir_path, faf_path=faf_path, size=call_kwargs["size"], args=args,
#             )

#             # "full_rescue" reproduces run_batch's annotation-fallback safety
#             # net (see run_batch above) inside the ablation harness, so it is
#             # directly comparable to a production run like out_12: if the
#             # vessel/LoFTR result never matched at all, or "succeeded" but
#             # still landed outside the T_SR bound, retry with the
#             # annotation-derived transform and keep it only if it actually
#             # lowers landmark MLE.
#             if use_rescue:
#                 mle_after = lm.get("landmark_mle_after")
#                 not_matched = not str(result.label).startswith("LoFTR@")
#                 technically_ok_but_bad = (
#                     mle_after is None or mle_after > LANDMARK_SR_THRESHOLD_PX
#                 )
#                 if not_matched or technically_ok_but_bad:
#                     _, _, kp_transform, has_fb = _kp_for_size(call_kwargs["size"])
#                     if has_fb and kp_transform is not None:
#                         fallback_result = register(
#                             ir_image=ir_path,
#                             faf_image=faf_path,
#                             ir_vessel=ir_v,
#                             faf_vessel=faf_v,
#                             fov_scale=args.fov_scale,
#                             size=call_kwargs["size"],
#                             preprocess=not args.no_preprocess,
#                             use_loftr=False,
#                             forced_transform=kp_transform,
#                             forced_label="annotation_landmarks_fallback",
#                         )
#                         fallback_lm = _landmark_metrics(
#                             annotation_path=score_csv, transform=fallback_result.transform,
#                             ir_path=ir_path, faf_path=faf_path, size=call_kwargs["size"], args=args,
#                         )
#                         fallback_mle = fallback_lm.get("landmark_mle_after")
#                         if mle_after is None or (fallback_mle is not None and fallback_mle < mle_after):
#                             result, lm = fallback_result, fallback_lm

#             # Same categorization as run_batch: "forced" = transform came
#             # directly from annotation landmarks (with_keypoints' no-blend
#             # fallback, or full_rescue's not-matched rescue), "blended" =
#             # with_keypoints folded annotation points into the LoFTR/RANSAC
#             # fit, "none" = no annotation involvement in this transform.
#             if result.label in ("annotation_landmarks", "annotation_landmarks_fallback"):
#                 annotation_used_for_transform = "forced"
#             elif call_kwargs.get("keypoints_fixed") is not None:
#                 annotation_used_for_transform = "blended"
#             else:
#                 annotation_used_for_transform = "none"

#             records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
#                                 status="ok",
#                                 label=result.label, scale=result.scale,
#                                 annotation_used_for_transform=annotation_used_for_transform,
#                                 **result.metrics, **lm))

#         pd.DataFrame(records).to_csv(results_csv, index=False)

#     df_out = pd.DataFrame(records)
#     df_out.to_csv(results_csv, index=False)
#     print(f"Done. {(df_out['status'] == 'ok').sum() if not df_out.empty else 0}/{len(df_out)} runs ok.")
#     print(f"Per-run metrics -> {results_csv.resolve()}")

#     summary = summarize_ablation(df_out, configs)
#     if not summary.empty:
#         summary_csv = out_dir / "ablation_summary.csv"
#         summary.to_csv(summary_csv, index=False)
#         print(f"Summary         -> {summary_csv.resolve()}")
#     return 0


# def main(argv=None) -> int:
#     args = build_parser().parse_args(argv)

#     csv_path = Path(args.csv)
#     if not csv_path.exists():
#         print(f"error: CSV not found: {csv_path}", file=sys.stderr)
#         return 2

#     out_dir = Path(args.output)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(csv_path)
#     required = [args.ir_col, args.faf_col]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         print(f"error: CSV missing columns: {missing}\n       available: {list(df.columns)}", file=sys.stderr)
#         return 2

#     if args.ablation:
#         return run_ablation(df, out_dir, args)
#     return run_batch(df, out_dir, args)


# if __name__ == "__main__":
#     raise SystemExit(main())


from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .pipeline import register, _as_vessel
from . import utils, visualization



# Registration success rate (Eq. 11): a pair is "successful" iff its landmark
# MLE is <= T_SR. T_SR is defined as half of T_AUC (Eq. 10's integration
# bound). The success-rate and AUC cutoffs are configured independently.
LANDMARK_SR_THRESHOLD_PX = 18.75  # T_SR
LANDMARK_AUC_MAX_PX = 37.5  # T_AUC

# RANSAC inlier tolerance, in native image pixels, for the annotation-derived
# fallback affine fit (see _estimate_affine_from_annotation) -- NOT the
# size x size (224) working grid. Fitting on points already rescaled to the
# 224 working grid and then applying this tolerance there would let it
# silently balloon to ~10px of real reprojection slack at 768px, loose
# enough that a clustered, inconsistent annotation click gets accepted as an
# inlier instead of rejected. Assumes every pair's images are standardized
# to 768px (image_paths_768.csv / annotations_768/) -- mixed-resolution
# input would need this scaled per pair instead of used as a flat constant.
ANNOTATION_RANSAC_PX = 8.0
_AUC_LABEL = f"AUC@{LANDMARK_AUC_MAX_PX:.2f}px"
_AUC_STEM = f"landmark_auc_{LANDMARK_AUC_MAX_PX:.2f}".replace(".", "_")

# (display_name, preferred_stem, fallback_stem)
SUMMARY_METRICS_VESSEL = [
    ("dice", "vessel_dice", "dice"),
    ("ncc", "vessel_ncc", "ncc"),
    ("ssd", "vessel_ssd", "ssd"),
    ("hd95", "vessel_hd95", "hd95"),
    ("asd", "vessel_asd", "asd"),
    ("wasserstein", "vessel_wasserstein", "wasserstein"),
]
SUMMARY_METRICS_IMAGE = [
    ("dice", "image_dice", None),
    ("ncc", "image_ncc", None),
    ("ssd", "image_ssd", None),
    ("mi", "image_mi", "mi"),
    ("nmi", "image_nmi", "nmi"),
    ("ssim", "image_ssim", "ssim"),
    ("psnr", "image_psnr", "psnr"),
]
SUMMARY_METRICS_LANDMARK = [
    ("MLE", "landmark_mle", None),
]
SUMMARY_METRICS_LANDMARK_AUC = [
    (_AUC_LABEL, _AUC_STEM, None),
]

# Per-config ablation summary metrics written to ablation_summary.csv.
# (source_stem, output_metric_name, higher_is_better, phase)
ABLATION_METRICS = [
    ("wasserstein", "wasserstein", False, "after"),
    ("asd", "asd", False, "after"),
    ("dice", "dice", True, "after"),
    ("ncc", "ncc", True, "after"),
    ("hd95", "hd95", False, "after"),
    ("landmark_mle", "mle", False, "after"),
    ("landmark_mle", "success_rate", True, "after"),
    ("landmark_mle", "AUC@37.50px", True, "after"),
]

# Each entry is (name, label, overrides). `overrides` is passed as
# **kwargs into pipeline.register() -- see run_ablation() -- except for the
# keys in _GLOBAL_OVERRIDE_ATTRS below, which set a `utils` module constant
# for the duration of the call instead (those aren't register() params).
ABLATIONS = [
    ("full",          "Full method (fully automatic)", {}),
    ("full_rescue",   "Full method + annotation fallback (same trigger as run_batch --fallback)", dict(_rescue=True)),
    ("with_keypoints", "Keypoint-guided matching (annotation landmarks)", dict(use_keypoints=True)),
    ("no_loftr",      "w/o LoFTR (FOV seed only)",     dict(use_loftr=False)),
    ("no_fov_scale",  "w/o FOV scale",                 dict(fov_scale=1.0)),
    # Disabled parameter sweeps; keep the underlying registration options available.
    # ("affine4",       "Similarity transform (4 DOF)",  dict(affine_dof=4)),
    # ("affine6",       "Full affine (6 DOF)",           dict(affine_dof=6)),
    # ("k1_only",       "closing k=1 only",              dict(loftr_close_ksizes=(1,))),
    # ("k4_only",       "closing k=4 only",              dict(loftr_close_ksizes=(4,))),
    ("no_retry",      "w/o low-conf re-match",         dict(loftr_min_inliers=0)),
    ("no_denseflow",  "w/o dense-flow refinement",     dict(use_denseflow=False)),
    ("no_ransac",     "w/o RANSAC",                    dict(use_ransac=False)),
    # ("conf_020",      "LoFTR conf 0.2",                dict(min_conf=0.2)),
    # ("conf_060",      "LoFTR conf 0.6",                dict(min_conf=0.6)),
    # ("ransac_3",      "RANSAC 3 px",                   dict(ransac_thresh=3.0)),
    # ("ransac_8",      "RANSAC 8 px",                   dict(ransac_thresh=8.0)),
]

# overrides keys that map to a `utils` module constant rather than a
# register() kwarg -- popped out and monkeypatched around each call.
_GLOBAL_OVERRIDE_ATTRS = {
    "loftr_close_ksizes": "LOFTR_CLOSE_KSIZES",
    "loftr_min_inliers": "LOFTR_MIN_INLIERS",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fafir-register",
        description="Batch IR<->FAF registration over a CSV of image pairs.",
    )
    p.add_argument("csv", help="CSV listing image pairs (one per row).")
    p.add_argument("output", help="Output directory for images + results.csv.")
    p.add_argument("--ir-col", "--fixed-col", dest="ir_col", default="fixed",
                   help="CSV column with the IR image path.")
    p.add_argument("--faf-col", "--moving-col", dest="faf_col", default="moving",
                   help="CSV column with the FAF image path.")
    p.add_argument("--ir-vessel-col", "--fixed-vessel-col", dest="ir_vessel_col", default="fixed_vessel_mask",
                   help="CSV column with the IR vessel-mask path.")
    p.add_argument("--faf-vessel-col", "--moving-vessel-col", dest="faf_vessel_col", default="moving_vessel_mask",
                   help="CSV column with the FAF vessel-mask path.")
    p.add_argument("--fov-scale", type=float, default=55.0 / 30.0,
                   help="FOV scale seed (default 55/30 ~= 1.833).")
    p.add_argument("--size", type=int, default=224,
                   help="Working resolution, multiple of 14 (default 224).")
    p.add_argument("--no-loftr", action="store_true",
                   help="Disable LoFTR feature matching.")
    p.add_argument("--method", choices=["auto", "loftr", "full_affine"], default="auto",
                   help="Registration method policy: auto (default), force LoFTR, or force full-affine.")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Skip CLAHE/blur preprocessing.")
    p.add_argument("--no-images", action="store_true",
                   help="Only write results.csv; skip saving diagnostic images.")
    p.add_argument("--only-row", type=int, nargs="+", default=None, metavar="N",
                   help="Run only specific 1-based CSV row numbers.")
    p.add_argument("--annotation-col", default="annotation_csv",
                   help="CSV column containing annotation CSV path (default: annotation_csv).")
    p.add_argument("--landmark-moving-image", choices=["ir", "faf"], default="faf",
                   help="Which image owns moving landmark coordinates (default: faf).")
    p.add_argument("--landmark-fixed-image", choices=["ir", "faf"], default="ir",
                   help="Which image owns fixed landmark coordinates (default: ir).")
    p.add_argument("--landmark-moving-x-col", default="FAF_x",
                   help="Annotation CSV x-column for moving landmarks (default: FAF_x).")
    p.add_argument("--landmark-moving-y-col", default="FAF_y",
                   help="Annotation CSV y-column for moving landmarks (default: FAF_y).")
    p.add_argument("--landmark-fixed-x-col", default="IR_x",
                   help="Annotation CSV x-column for fixed landmarks (default: IR_x).")
    p.add_argument("--landmark-fixed-y-col", default="IR_y",
                   help="Annotation CSV y-column for fixed landmarks (default: IR_y).")
    p.add_argument("--align-from-annotation", action="store_true",
                   help="Use annotation landmarks to estimate the alignment transform "
                        "(replaces vessel/LoFTR transform).")

    p.add_argument("--ablation", action="store_true",
                   help="Run leave-one-out ablation study.")
    p.add_argument("--configs", nargs="+", default=None, metavar="NAME",
                   help=f"Ablation configs (default all): {[n for n, _, _ in ABLATIONS]}")
    p.add_argument("--ablation-keypoints-only", action="store_true",
                   help="Ablation only: run all configs only on rows with a valid annotation-landmark transform.")
    p.add_argument("--keypoints-only", action="store_true",
                   help="Batch mode only: skip pairs with no (or too few) annotation landmarks "
                        "instead of registering them without keypoint guidance.")
    p.add_argument("--use-keypoints", action="store_true",
                   help="Batch mode only (with --fallback): blend all annotation landmarks into "
                        "the LoFTR/vessel matching fit for every pair (same mechanism as the "
                        "ablation harness's 'with_keypoints' config).")
    p.add_argument("--fallback-inlier-threshold", type=int, default=30, metavar="N",
                   help="Treat a pair as a failed automatic registration (and substitute the "
                        "annotation-landmark transform, when available) whenever its LoFTR inlier "
                        "count is <= N. N is a count of matched keypoints, not the landmark "
                        "ground truth. Choose N on a separate validation split (or report a "
                        "sensitivity sweep), not on the test set. Pass 0 for the strict "
                        "zero-match-only trigger. Used by run_batch --fallback and by the "
                        "ablation 'full_rescue' config.")

    # ── Annotation-landmark fallback ─────────────────────────────────────────
    p.add_argument("--fallback", dest="no_fallback", action="store_false",
                   help="Batch mode only: enable the annotation-landmark fallback (semi-automatic). "
                        "When automatic registration fails (exception, no LoFTR transform, or "
                        "inliers <= --fallback-inlier-threshold) the transform is re-estimated "
                        "from all valid annotation landmarks. Off by default.")
    p.add_argument("--no-fallback", dest="no_fallback", action="store_true", default=True,
                   help="Batch mode only: fully automatic run (default). Never use annotation "
                        "landmarks to build a transform (they are used only for evaluation).")
    return p


def _vessel_mask_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip()
        return (not s) or (not Path(s).exists())
    if isinstance(x, Path):
        return not x.exists()
    return pd.isna(x)


def _requested_use_loftr(args, default_use_loftr: bool) -> bool:
    if args.method == "full_affine":
        return False
    if args.method == "loftr":
        return True
    return bool(default_use_loftr)


def _assert_requested_method(args, winner_label: str) -> None:
    if args.method == "loftr" and not str(winner_label).startswith("LoFTR@"):
        raise RuntimeError(f"requested method=loftr but winner={winner_label}")


def _iter_pairs(df: pd.DataFrame, args):
    for n, (_, row) in enumerate(df.iterrows(), 1):
        ir_path = row[args.ir_col]
        faf_path = row[args.faf_col]
        ir_v = row.get(args.ir_vessel_col, None)
        faf_v = row.get(args.faf_vessel_col, None)
        name = f"{n:04d}_{Path(str(faf_path)).stem}_to_{Path(str(ir_path)).stem}"
        yield n, name, ir_path, faf_path, ir_v, faf_v


def _estimate_affine_from_annotation(annotation_path, ir_path, faf_path, size: int, args):
    """Estimate a 2x3 affine from annotation landmarks in working-grid coordinates.

    Returns None if data is missing/insufficient. Returned transform always maps
    FAF -> IR (registration direction).
    """
    if annotation_path is None or not str(annotation_path).strip():
        return None

    ann = Path(str(annotation_path))
    if not ann.exists():
        return None

    try:
        adf = pd.read_csv(ann)
    except Exception:
        return None
    if adf.empty:
        return None

    needed = [
        args.landmark_moving_x_col,
        args.landmark_moving_y_col,
        args.landmark_fixed_x_col,
        args.landmark_fixed_y_col,
    ]
    if any(c not in adf.columns for c in needed):
        return None

    pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pts) < 2:
        return None

    moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
    fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
    if moving_img_path == fixed_img_path:
        return None

    moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
    fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
    if moving_raw is None or fixed_raw is None:
        return None

    mh, mw = moving_raw.shape[:2]
    fh, fw = fixed_raw.shape[:2]
    if min(mh, mw, fh, fw) <= 0:
        return None

    mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
    my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
    fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
    fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)

    # Fit RANSAC in native pixel space, NOT on points pre-rescaled to the
    # size x size working grid -- a fixed reprojection threshold applied
    # after that rescale would correspond to a much looser real-pixel
    # tolerance on higher-resolution pairs, and could accept a genuinely
    # inconsistent/mislabeled point as an inlier.
    src_native = np.stack([mx, my], axis=1).astype(np.float32)
    dst_native = np.stack([fx, fy], axis=1).astype(np.float32)

    M_native, _ = cv2.estimateAffinePartial2D(
        src_native,
        dst_native,
        method=cv2.RANSAC,
        ransacReprojThreshold=ANNOTATION_RANSAC_PX,
        maxIters=5000,
        confidence=0.999,
    )
    if M_native is None:
        return None

    if args.landmark_moving_image != "faf" or args.landmark_fixed_image != "ir":
        return None

    # Fold the native-space fit into the size x size working grid register()
    # actually operates on -- the inverse of the scaling _landmark_metrics
    # applies to go from working-grid back to native pixels.
    sx_m, sy_m = float(size) / float(mw), float(size) / float(mh)
    sx_f, sy_f = float(size) / float(fw), float(size) / float(fh)
    S_f = np.diag([sx_f, sy_f])
    S_m_inv = np.diag([1.0 / sx_m, 1.0 / sy_m])
    M_native = M_native.astype(np.float64)
    A_grid = S_f @ M_native[:2, :2] @ S_m_inv
    t_grid = S_f @ M_native[:2, 2]
    M_decl = np.hstack([A_grid, t_grid[:, None]]).astype(np.float32)
    return M_decl


def _read_annotation_keypoints(annotation_path, args, *, size: int, ir_path: str, faf_path: str):
    """Return fixed/moving landmark arrays in registration working-grid coordinates.

    The returned arrays are shaped ``(N, 2)`` and are the coordinates to feed into
    the LoFTR feature matcher as keypoint-enhanced feature maps. If annotation data
    is missing or insufficient, returns ``(None, None)``.
    """
    if annotation_path is None or not str(annotation_path).strip():
        return None, None

    ann = Path(str(annotation_path))
    if not ann.exists():
        return None, None

    try:
        adf = pd.read_csv(ann)
    except Exception:
        return None, None
    if adf.empty:
        return None, None

    needed = [
        args.landmark_moving_x_col,
        args.landmark_moving_y_col,
        args.landmark_fixed_x_col,
        args.landmark_fixed_y_col,
    ]
    if any(c not in adf.columns for c in needed):
        return None, None

    pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if pts.empty:
        return None, None

    moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
    fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
    if moving_img_path == fixed_img_path:
        return None, None

    moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
    fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
    if moving_raw is None or fixed_raw is None:
        return None, None

    mh, mw = moving_raw.shape[:2]
    fh, fw = fixed_raw.shape[:2]
    if min(mh, mw, fh, fw) <= 0:
        return None, None

    mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
    my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
    fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
    fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)

    if len(mx) < 2 or len(fx) < 2:
        return None, None

    moving_pts = np.stack([
        mx * (float(size) / float(mw)),
        my * (float(size) / float(mh)),
    ], axis=1).astype(np.float32)
    fixed_pts = np.stack([
        fx * (float(size) / float(fw)),
        fy * (float(size) / float(fh)),
    ], axis=1).astype(np.float32)

    if args.landmark_moving_image == "faf" and args.landmark_fixed_image == "ir":
        return fixed_pts, moving_pts
    return None, None


def _warp_points_via_image(pts: np.ndarray, M: np.ndarray,
                           in_hw: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    """Predicted landmark coordinates read off the actual warped image.

    ``pts`` are in the moving image's own native pixel coordinates, and ``M``
    maps moving-native pixels directly to fixed-native pixels (already
    rescaled from the registration working-grid transform -- see
    ``_landmark_metrics``). Each point is placed on a canvas at the moving
    image's own resolution (``in_hw``) as a bilinear-splatted impulse (so the
    pre-warp centroid is the exact sub-pixel point, not a rounded pixel),
    warped with the same cv2.warpAffine call utils.warp_img uses for the real
    registered image (now sized to the fixed image's own resolution,
    ``out_hw``), and read back as the sub-pixel intensity centroid of the
    result. This is the literal "coordinate after warping the image" rather
    than an analytic M @ point re-derivation -- and it runs end-to-end in
    original image pixels, with no intermediate resize to the registration
    working grid.
    """
    in_h, in_w = in_hw
    out_h, out_w = out_hw
    out = np.full((len(pts), 2), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        x0, y0 = int(np.floor(float(x))), int(np.floor(float(y)))
        dx, dy = float(x) - x0, float(y) - y0
        canvas = np.zeros((in_h, in_w), dtype=np.float32)
        for xo, yo, wgt in (
            (x0, y0, (1 - dx) * (1 - dy)),
            (x0 + 1, y0, dx * (1 - dy)),
            (x0, y0 + 1, (1 - dx) * dy),
            (x0 + 1, y0 + 1, dx * dy),
        ):
            if 0 <= xo < in_w and 0 <= yo < in_h:
                canvas[yo, xo] += wgt
        landed = False
        if canvas.sum() > 1e-6:
            warped = cv2.warpAffine(canvas, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            total = float(warped.sum())
            if total > 1e-6:
                ys, xs = np.nonzero(warped)
                weights = warped[ys, xs]
                out[i, 0] = float(np.sum(xs * weights) / total)
                out[i, 1] = float(np.sum(ys * weights) / total)
                landed = True
        if not landed:
            # The point warped off the canvas entirely -- a genuine
            # registration failure, not missing data. Fall back to the exact
            # affine map so it still contributes a (large) error instead of
            # silently dropping the pair from every metric's average.
            out[i, 0] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            out[i, 1] = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return out


def _landmark_metrics(annotation_path, transform, ir_path, faf_path, size: int, args):
    """Compute before/after landmark MLE for one pair, in original image pixels.

    The registration transform is estimated on the ``size``x``size`` working
    grid, but is rescaled here to map moving-native pixels directly to
    fixed-native pixels before being applied to the landmark points -- so the
    reported MLE is computed end-to-end in original image pixel space, with
    no intermediate resize to the working grid. If moving/fixed landmark
    ownership is IR->FAF, the inverse transform is used for landmark
    transfer.
    """
    out = {
        "landmark_count": 0,
        "landmark_mle_before": None,
        "landmark_mle_after": None,
        "landmark_mle_delta": None,
    }
    if annotation_path is None or not str(annotation_path).strip():
        return out

    ann = Path(str(annotation_path))
    if not ann.exists():
        return out

    try:
        adf = pd.read_csv(ann)
    except Exception:
        return out
    if adf.empty:
        return out

    needed = [
        args.landmark_moving_x_col,
        args.landmark_moving_y_col,
        args.landmark_fixed_x_col,
        args.landmark_fixed_y_col,
    ]
    if any(c not in adf.columns for c in needed):
        return out

    pts = adf[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if pts.empty:
        return out

    # Map user-declared moving/fixed landmarks to image files.
    moving_img_path = ir_path if args.landmark_moving_image == "ir" else faf_path
    fixed_img_path = ir_path if args.landmark_fixed_image == "ir" else faf_path
    if moving_img_path == fixed_img_path:
        return out

    moving_raw = cv2.imread(str(moving_img_path), cv2.IMREAD_UNCHANGED)
    fixed_raw = cv2.imread(str(fixed_img_path), cv2.IMREAD_UNCHANGED)
    if moving_raw is None or fixed_raw is None:
        return out

    mh, mw = moving_raw.shape[:2]
    fh, fw = fixed_raw.shape[:2]
    if min(mh, mw, fh, fw) <= 0:
        return out

    mx = pts[args.landmark_moving_x_col].to_numpy(dtype=np.float32)
    my = pts[args.landmark_moving_y_col].to_numpy(dtype=np.float32)
    fx = pts[args.landmark_fixed_x_col].to_numpy(dtype=np.float32)
    fy = pts[args.landmark_fixed_y_col].to_numpy(dtype=np.float32)
    n = int(len(pts))
    if n <= 0:
        return out

    # Registration transform is FAF->IR.
    if args.landmark_moving_image != "faf" or args.landmark_fixed_image != "ir":
        return out

    # The registration transform maps size x size working-grid coordinates:
    # moving-native -[scale by S_m]-> working grid -[M]-> working grid
    # -[scale by S_f^-1]-> fixed-native. Fold the two scalings into M so it
    # maps moving-native pixels directly to fixed-native pixels, letting the
    # landmark points be warped without ever resizing through the working
    # grid.
    M_grid = np.asarray(transform, dtype=np.float64)
    sx_m, sy_m = float(size) / float(mw), float(size) / float(mh)
    sx_f, sy_f = float(size) / float(fw), float(size) / float(fh)
    S_m = np.diag([sx_m, sy_m])
    S_f_inv = np.diag([1.0 / sx_f, 1.0 / sy_f])
    A = S_f_inv @ M_grid[:2, :2] @ S_m
    t = S_f_inv @ M_grid[:2, 2]
    M_use = np.hstack([A, t[:, None]]).astype(np.float32)

    src = np.stack([mx, my], axis=1).astype(np.float32)

    # Predicted coordinate read off the actual warped image (not an analytic
    # M @ point re-derivation) -- see _warp_points_via_image. Canvas sizes
    # are each image's own native resolution.
    pred = _warp_points_via_image(src, M_use, in_hw=(mh, mw), out_hw=(fh, fw))
    valid = ~np.isnan(pred).any(axis=1)
    if not valid.any():
        return out

    pred_x = pred[valid, 0]
    pred_y = pred[valid, 1]

    # Baseline before-registration mapping by image-size ratio only.
    base_x = mx[valid] * (float(fw) / float(mw))
    base_y = my[valid] * (float(fh) / float(mh))

    err_before = np.sqrt((base_x - fx[valid]) ** 2 + (base_y - fy[valid]) ** 2)
    err_after = np.sqrt((pred_x - fx[valid]) ** 2 + (pred_y - fy[valid]) ** 2)

    mle_before = float(np.mean(err_before))
    mle_after = float(np.mean(err_after))
    out.update({
        "landmark_count": int(valid.sum()),
        "landmark_mle_before": round(mle_before, 4),
        "landmark_mle_after": round(mle_after, 4),
        "landmark_mle_delta": round(mle_before - mle_after, 4),
    })
    return out


def _print_pair_metrics(metrics: dict):
    dice = metrics.get("vessel_dice_after", metrics.get("dice_after"))
    asd = metrics.get("vessel_asd_after", metrics.get("asd_after"))
    hd95 = metrics.get("vessel_hd95_after", metrics.get("hd95_after"))
    ncc = metrics.get("image_ncc_after", metrics.get("ncc_after"))

    def _fmt(v, nd=4):
        return "NA" if v is None else f"{float(v):.{nd}f}"

    print(
        "  metrics:"
        f" vessel_dice_after={_fmt(dice)}"
        f" vessel_asd_after={_fmt(asd, 3)}"
        f" vessel_hd95_after={_fmt(hd95, 3)}"
        f" image_ncc_after={_fmt(ncc)}"
    )


def _pick_stem(sub: pd.DataFrame, preferred: str, fallback: str | None):
    cb, ca = f"{preferred}_before", f"{preferred}_after"
    if cb in sub.columns and ca in sub.columns:
        return preferred
    if fallback is not None:
        cb, ca = f"{fallback}_before", f"{fallback}_after"
        if cb in sub.columns and ca in sub.columns:
            return fallback
    return None


def _append_rows(rows: list[dict], sub: pd.DataFrame, group: str, category: str,
                 metrics: list[tuple[str, str, str | None]]):
    for shown, preferred, fallback in metrics:
        stem = _pick_stem(sub, preferred, fallback)
        if stem is None:
            continue
        cb, ca = f"{stem}_before", f"{stem}_after"
        b = pd.to_numeric(sub[cb], errors="coerce")
        a = pd.to_numeric(sub[ca], errors="coerce")
        valid = b.notna() & a.notna()
        if not valid.any():
            continue
        b = b[valid]
        a = a[valid]
        d = a - b
        rows.append(dict(
            group=group,
            category=category,
            metric=shown,
            metric_stem=stem,
            n=int(valid.sum()),
            before_mean=float(b.mean()),
            before_std=float(b.std(ddof=0)),
            after_mean=float(a.mean()),
            after_std=float(a.std(ddof=0)),
            delta_mean=float(d.mean()),
            delta_std=float(d.std(ddof=0)),
        ))


def _landmark_auc(errors, max_threshold: float = LANDMARK_AUC_MAX_PX) -> Optional[float]:
    values = pd.to_numeric(pd.Series(errors), errors="coerce").dropna().to_numpy(dtype=np.float32)
    if values.size == 0:
        return None
    max_threshold = float(max_threshold)
    if max_threshold <= 0:
        return None
    thresholds = np.linspace(0.0, max_threshold, 501, dtype=np.float32)
    success = np.array([(values <= t).mean() for t in thresholds], dtype=np.float32)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 1.x and 2.x
    auc = float(_trapz(success, thresholds) / max_threshold)
    return round(auc, 4)


def _append_landmark_auc_rows(rows: list[dict], sub: pd.DataFrame, group: str, category: str):
    before_col = "landmark_mle_before" if "landmark_mle_before" in sub.columns else None
    after_col = "landmark_mle_after" if "landmark_mle_after" in sub.columns else None
    if before_col is None or after_col is None:
        return

    before = pd.to_numeric(sub[before_col], errors="coerce").dropna()
    after = pd.to_numeric(sub[after_col], errors="coerce").dropna()
    if before.empty or after.empty:
        return

    rows.append(dict(
        group=group,
        category=category,
        metric=_AUC_LABEL,
        metric_stem=_AUC_STEM,
        n=int(min(len(before), len(after))),
        before_mean=_landmark_auc(before),
        before_std=0.0,
        after_mean=_landmark_auc(after),
        after_std=0.0,
        delta_mean=None if (_landmark_auc(before) is None or _landmark_auc(after) is None)
        else round(_landmark_auc(after) - _landmark_auc(before), 4),
        delta_std=0.0,
    ))


def _stats(sub: pd.DataFrame, stem: str, phase: str = "after"):
    col = f"{stem}_{phase}"
    if col not in sub.columns:
        return None
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    if v.empty:
        return None
    return float(v.mean()), float(v.std(ddof=0)), int(len(v))


def summarize_ablation(df: pd.DataFrame, configs) -> pd.DataFrame:
    """Summarize ablation results as config-level after-metric rows."""
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    rows: list[dict] = []
    for name, label, _ in configs:
        sub = ok[ok["config"] == name]
        if "landmark_mle_after" in sub.columns:
            sub = sub[pd.to_numeric(sub["landmark_mle_after"], errors="coerce").notna()]

        for stem, out_name, higher, phase in ABLATION_METRICS:
            if out_name == "success_rate":
                mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
                if mle.empty:
                    continue
                success = float((mle <= LANDMARK_SR_THRESHOLD_PX).mean())
                rows.append(dict(
                    config=name,
                    label=label,
                    metric=out_name,
                    n=int(len(mle)),
                    mean=round(success, 4),
                    std=0.0,
                    higher_is_better=higher,
                ))
                continue

            if out_name == "AUC@37.50px":
                mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
                if mle.empty:
                    continue
                auc = _landmark_auc(mle, max_threshold=LANDMARK_AUC_MAX_PX)
                if auc is None:
                    continue
                rows.append(dict(
                    config=name,
                    label=label,
                    metric=out_name,
                    n=int(len(mle)),
                    mean=round(auc, 4),
                    std=0.0,
                    higher_is_better=higher,
                ))
                continue

            st = _stats(sub, stem, phase=phase)
            if st is None:
                continue
            mean, std, n = st
            rows.append(dict(
                config=name,
                label=label,
                metric=out_name,
                n=n,
                mean=round(mean, 4),
                std=round(std, 4),
                higher_is_better=higher,
            ))
    return pd.DataFrame(rows)


def _print_block(rows_df: pd.DataFrame, group: str, category: str):
    block = rows_df[(rows_df["group"] == group) & (rows_df["category"] == category)]
    if block.empty:
        return
    print(f"\n--- {category.title()} ---")
    print(f"  {'metric':<12} {'before (mean ± std)':>26} {'after (mean ± std)':>26} {'score delta ± std':>24}")
    for _, r in block.iterrows():
        before = f"{r['before_mean']:.4f} ± {r['before_std']:.4f}"
        after = f"{r['after_mean']:.4f} ± {r['after_std']:.4f}"
        delta = f"{r['delta_mean']:+.4f} ± {r['delta_std']:.4f}"
        print(f"  {r['metric']:<12} {before:>26} {after:>26} {delta:>24}")


def summarize(df: pd.DataFrame):
    if df.empty:
        print("No rows to summarize.")
        return None

    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok.empty:
        print("No successful rows to summarize.")
        return None

    # Restrict every category (not just landmark) to pairs that actually
    # have a landmark MLE, so vessel/image/landmark are all reported over
    # the same apples-to-apples subset instead of vessel/image silently
    # including extra pairs that have no annotation data to score.
    if "landmark_mle_after" in ok.columns:
        ok_lm = ok[pd.to_numeric(ok["landmark_mle_after"], errors="coerce").notna()]
    else:
        ok_lm = ok

    rows: list[dict] = []

    # Overall block
    _append_rows(rows, ok_lm, "overall", "vessel", SUMMARY_METRICS_VESSEL)
    _append_rows(rows, ok_lm, "overall", "image", SUMMARY_METRICS_IMAGE)
    _append_rows(rows, ok_lm, "overall", "landmark", SUMMARY_METRICS_LANDMARK)
    _append_landmark_auc_rows(rows, ok_lm, "overall", "landmark")

    if not rows:
        print("No comparable before/after metrics available.")
        return None

    out = pd.DataFrame(rows)

    print("\nSummary metrics (mean ± std):")
    print(f"  included (status=ok): {len(ok)}/{len(df)}")
    print(f"  included (with landmark MLE): {len(ok_lm)}/{len(df)}")

    for g in ["overall"]:
        n_g = len(ok_lm)
        print(f"\n=== {g.upper()} (n={n_g}) ===")
        _print_block(out, g, "vessel")
        _print_block(out, g, "image")
        _print_block(out, g, "landmark")

    def _print_landmark_success(label: str, sub: pd.DataFrame):
        if "landmark_mle_after" not in sub.columns:
            return
        mle = pd.to_numeric(sub["landmark_mle_after"], errors="coerce").dropna()
        # Eq. 11: a pair only counts as a landmark success if it has a valid
        # MLE (excludes rows with too few correspondences to estimate a
        # transform in the first place -- those already dropped by dropna)
        # AND that MLE is <= T_SR.
        if mle.empty:
            return
        n_mle = len(mle)
        sr = int((mle <= LANDMARK_SR_THRESHOLD_PX).sum())
        fr = n_mle - sr
        print(f"\n{label} -- classified: {n_mle}")
        print(f"  success (MLE <= {LANDMARK_SR_THRESHOLD_PX:.2f}px): {sr}/{n_mle} ({100.0*sr/n_mle:.2f}%)")
        print(f"  failure (MLE >  {LANDMARK_SR_THRESHOLD_PX:.2f}px): {fr}/{n_mle} ({100.0*fr/n_mle:.2f}%)")

    _print_landmark_success("Landmark MLE (overall)", ok)

    return out


# Legacy retry/upscaling block intentionally disabled.
# The single registration pass is kept on the configured size and does not
# automatically re-run at a higher pixel resolution based on Dice/Wasserstein.


def run_batch(df: pd.DataFrame, out_dir: Path, args) -> int:
    """Batch registration with an annotation-landmark fallback.

    Modes:
      * default (``--no-fallback``): fully automatic. No landmark ever touches
        the transform; landmarks are used only to compute MLE.
      * ``--fallback``: automatic registration first. If it fails (an
        exception, no LoFTR transform, or LoFTR inliers <=
        ``--fallback-inlier-threshold``), the transform is re-estimated from
        all valid annotation landmarks.

    The failure trigger uses only the method's own internal signals
    (exception, label, inlier count), never the landmark error. Every pair is
    scored on all valid landmarks; for fallback pairs this score is in-sample
    (the same landmarks built the transform), so automatic and fallback pairs
    are reported separately in the fallback report.
    """
    print(f"fafir-register: {len(df)} pair(s) -> {out_dir.resolve()}\n")

    if args.align_from_annotation:
        print("WARNING: --align-from-annotation builds EVERY transform from annotation "
              "landmarks; results are not an automatic registration.", file=sys.stderr)
    if args.use_keypoints:
        if args.no_fallback:
            print("NOTE: --use-keypoints is ignored without --fallback (fully automatic run).",
                  file=sys.stderr)
        else:
            print("NOTE: --use-keypoints blends all annotation landmarks into matching; "
                  "report this run as landmark-guided.", file=sys.stderr)

    mode = "automatic" if args.no_fallback else "semi_automatic_fallback"
    print(f"mode={mode}  fallback_inlier_threshold={args.fallback_inlier_threshold}\n")

    records = []
    results_csv = out_dir / "results.csv"
    matching_csv = out_dir / "matching.csv"

    for n, name, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
        if args.only_row and n not in args.only_row:
            continue

        print(f"[{n}/{len(df)}] {Path(str(faf_path)).name} -> {Path(str(ir_path)).name}")
        base_record = dict(
            name=name,
            ir=ir_path,
            faf=faf_path,
            ir_name=Path(str(ir_path)).name,
            faf_name=Path(str(faf_path)).name,
            mode=mode,
        )
        ann_path = df.iloc[n - 1].get(args.annotation_col, None) if args.annotation_col in df.columns else None
        base_record["annotation_csv"] = ann_path

        if args.keypoints_only:
            kf, km = _read_annotation_keypoints(
                ann_path, args, size=args.size, ir_path=ir_path, faf_path=faf_path,
            )
            if kf is None or km is None:
                records.append(dict(
                    **base_record,
                    status="skipped",
                    skip_reason="no_annotation_keypoints",
                    excluded_from_metrics=True,
                    exclusion_reason="skipped_by_keypoints_only_filter",
                ))
                pd.DataFrame(records).to_csv(results_csv, index=False)
                continue

        if _vessel_mask_missing(ir_v) or _vessel_mask_missing(faf_v):
            records.append(dict(
                **base_record,
                status="skipped",
                skip_reason="missing_vessel_mask",
                excluded_from_metrics=True,
                exclusion_reason="system_failure",
            ))
            pd.DataFrame(records).to_csv(results_csv, index=False)
            continue

        try:
            forced_transform = None
            forced_label = None
            fallback_allowed = not args.no_fallback

            if args.align_from_annotation:
                forced_transform = _estimate_affine_from_annotation(
                    annotation_path=ann_path, ir_path=ir_path, faf_path=faf_path,
                    size=args.size, args=args,
                )
                if forced_transform is not None:
                    forced_label = "annotation_landmarks"

            if args.use_keypoints and fallback_allowed:
                blend_keypoints_fixed, blend_keypoints_moving = _read_annotation_keypoints(
                    ann_path, args, size=args.size, ir_path=ir_path, faf_path=faf_path,
                )
            else:
                blend_keypoints_fixed = blend_keypoints_moving = None

            def _annotation_fallback_transform():
                # Fit every valid annotation correspondence; RANSAC handles outliers.
                return _estimate_affine_from_annotation(
                    annotation_path=ann_path, ir_path=ir_path, faf_path=faf_path,
                    size=args.size, args=args,
                )

            def _register_forced(M, label):
                return register(
                    ir_image=ir_path, faf_image=faf_path,
                    ir_vessel=ir_v, faf_vessel=faf_v,
                    fov_scale=args.fov_scale, size=args.size,
                    preprocess=not args.no_preprocess,
                    use_loftr=False,
                    forced_transform=M, forced_label=label,
                )

            fallback_used = False
            fallback_reason = ""
            auto_label = None
            auto_inliers = None

            t0 = time.perf_counter()
            try:
                result = register(
                    ir_image=ir_path,
                    faf_image=faf_path,
                    ir_vessel=ir_v,
                    faf_vessel=faf_v,
                    fov_scale=args.fov_scale,
                    size=args.size,
                    preprocess=not args.no_preprocess,
                    use_loftr=_requested_use_loftr(args, not args.no_loftr),
                    forced_transform=forced_transform,
                    forced_label=(forced_label or "annotation_landmarks"),
                    keypoints_fixed=blend_keypoints_fixed,
                    keypoints_moving=blend_keypoints_moving,
                )
                _assert_requested_method(args, result.label)
            except Exception as auto_exc:
                if not (fallback_allowed and forced_transform is None):
                    raise
                M_fb = _annotation_fallback_transform()
                if M_fb is None:
                    raise
                result = _register_forced(M_fb, "annotation_landmarks_fallback")
                fallback_used = True
                fallback_reason = f"automatic_exception: {auto_exc}"
                auto_label = "exception"

            # ── Failure detection: internal signals only (no ground truth) ──
            if not fallback_used and forced_transform is None:
                auto_label = result.label
                auto_inliers = int((result.metrics or {}).get("keypoint_matches", 0) or 0)
                loftr_failed = not str(result.label).startswith("LoFTR@")
                low_inliers = auto_inliers <= args.fallback_inlier_threshold
                if fallback_allowed and (loftr_failed or low_inliers):
                    M_fb = _annotation_fallback_transform()
                    if M_fb is not None:
                        result = _register_forced(M_fb, "annotation_landmarks_fallback")
                        fallback_used = True
                        fallback_reason = "no_loftr_transform" if loftr_failed \
                            else f"low_inliers({auto_inliers}<={args.fallback_inlier_threshold})"

            # Every pair is scored on all valid annotation landmarks.
            lm = _landmark_metrics(
                annotation_path=ann_path, transform=result.transform,
                ir_path=ir_path, faf_path=faf_path, size=args.size, args=args,
            )

            runtime_sec = time.perf_counter() - t0
        except Exception as e:
            records.append(dict(
                **base_record,
                status="error",
                error=str(e),
                excluded_from_metrics=True,
                exclusion_reason="registration_error",
            ))
            pd.DataFrame(records).to_csv(results_csv, index=False)
            continue

        if not args.no_images:
            result.save(out_dir, name=name)

        if fallback_used:
            annotation_used_for_transform = "fallback_all_landmarks"
        elif result.label == "annotation_landmarks":
            annotation_used_for_transform = "forced_all_landmarks"
        elif blend_keypoints_fixed is not None:
            annotation_used_for_transform = "blended_all_landmarks"
        else:
            annotation_used_for_transform = "none"
        # In-sample = the landmarks being scored also built/guided the transform.
        scored_on = "all_landmarks" if annotation_used_for_transform == "none" \
            else "all_landmarks_in_sample"

        records.append(dict(
            **base_record,
            status="ok",
            label=result.label,
            auto_label=auto_label,
            auto_inliers=auto_inliers,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            scale=result.scale,
            runtime_sec=round(runtime_sec, 3),
            annotation_used_for_transform=annotation_used_for_transform,
            scored_on=scored_on,
            **result.metrics,
            **lm,
        ))
        print(f"  runtime={runtime_sec:.2f}s")
        _print_pair_metrics(result.metrics)
        if fallback_used:
            print(f"  FALLBACK used ({fallback_reason}) -> transform from all valid annotation landmarks")
        if lm.get("landmark_count", 0) > 0:
            print(f"  landmarks ({scored_on}): "
                  f"n={lm['landmark_count']} "
                  f"mle_before={lm['landmark_mle_before']:.3f}px "
                  f"mle_after={lm['landmark_mle_after']:.3f}px "
                  f"delta={lm['landmark_mle_delta']:+.3f}px")
        pd.DataFrame(records).to_csv(results_csv, index=False)

    df_out = pd.DataFrame(records)
    df_out.to_csv(results_csv, index=False)

    cols = ["name", "ir_name", "faf_name", "status", "keypoint_matches", "keypoint_matches_total"]
    match_rows = []
    for r in records:
        match_rows.append({
            "name": r.get("name"),
            "ir_name": r.get("ir_name"),
            "faf_name": r.get("faf_name"),
            "status": r.get("status"),
            "keypoint_matches": r.get("keypoint_matches", 0),
            "keypoint_matches_total": r.get("keypoint_matches_total", 0),
        })
    pd.DataFrame(match_rows, columns=cols).to_csv(matching_csv, index=False)

    summary = summarize(df_out)
    if summary is not None:
        summary_csv = out_dir / "summary.csv"
        numeric_cols = [
            "before_mean", "before_std", "after_mean", "after_std", "delta_mean", "delta_std"
        ]
        summary_csv_df = summary.copy()
        for c in numeric_cols:
            if c in summary_csv_df.columns:
                summary_csv_df[c] = pd.to_numeric(summary_csv_df[c], errors="coerce").round(4)
        summary_csv_df.to_csv(summary_csv, index=False)
        print(f"\nSummary -> {summary_csv.resolve()}")

    total = len(records)
    ok_n = int((df_out["status"] == "ok").sum()) if total else 0
    skipped = int((df_out["status"] == "skipped").sum()) if total else 0
    errors = int((df_out["status"] == "error").sum()) if total else 0
    excluded = total - ok_n
    ok_df = df_out[df_out["status"] == "ok"] if "status" in df_out.columns else df_out

    # ── Fallback report (automatic vs fallback subsets) ──
    if "fallback_used" in ok_df.columns and not ok_df.empty:
        fb = ok_df["fallback_used"].astype(bool)
        n_fb = int(fb.sum())
        print(f"\n=== Fallback report ({mode}) ===")
        print(f"  automatic transform kept : {len(ok_df) - n_fb}/{len(ok_df)}")
        print(f"  annotation fallback      : {n_fb}/{len(ok_df)}  "
              f"(MLE in-sample: same landmarks built the transform)")
        if "landmark_mle_after" in ok_df.columns:
            mle = pd.to_numeric(ok_df["landmark_mle_after"], errors="coerce")
            for lbl, sel in (("automatic pairs", ~fb), ("fallback pairs", fb), ("all pairs", fb | ~fb)):
                v = mle[sel].dropna()
                if v.empty:
                    continue
                sr = 100.0 * float((v <= LANDMARK_SR_THRESHOLD_PX).mean())
                print(f"  {lbl:<16} n={len(v):<3} MLE mean={v.mean():.2f} ± {v.std(ddof=0):.2f}  "
                      f"median={v.median():.2f}  SR@{LANDMARK_SR_THRESHOLD_PX:.2f}px={sr:.1f}%  "
                      f"{_AUC_LABEL}={_landmark_auc(v)}")

    if "landmark_mle_after" in ok_df.columns:
        lm_after = pd.to_numeric(ok_df["landmark_mle_after"], errors="coerce").dropna()
        lm_before = pd.to_numeric(ok_df.get("landmark_mle_before"), errors="coerce").dropna()
        if not lm_after.empty:
            print(f"\nMean landmark error (after): {float(lm_after.mean()):.4f}px over {len(lm_after)} pair(s)")
        if not lm_before.empty and len(lm_before) == len(lm_after):
            print(f"Mean landmark error (before): {float(lm_before.mean()):.4f}px")
            print(f"Mean landmark improvement (before-after): {float(lm_before.mean() - lm_after.mean()):+.4f}px")

    if "runtime_sec" in ok_df.columns:
        rt = pd.to_numeric(ok_df["runtime_sec"], errors="coerce").dropna()
        if not rt.empty:
            print(f"\nRuntime: mean={rt.mean():.2f}s  median={rt.median():.2f}s  "
                  f"min={rt.min():.2f}s  max={rt.max():.2f}s  total={rt.sum():.1f}s over {len(rt)} pair(s)")

    print(f"Done. included for metrics: {ok_n}/{total}; excluded: {excluded} (system_failure/skipped={skipped}, error={errors}). Metrics -> {results_csv.resolve()}")
    return 0


def run_ablation(df: pd.DataFrame, out_dir: Path, args) -> int:
    configs = ABLATIONS
    if args.configs:
        known = {n for n, _, _ in ABLATIONS}
        bad = [c for c in args.configs if c not in known]
        if bad:
            print(f"error: unknown config(s) {bad}; choices: {sorted(known)}", file=sys.stderr)
            return 2
        configs = [c for c in ABLATIONS if c[0] in args.configs]

    records = []
    results_csv = out_dir / "ablation_results.csv"
    print(f"ablation: {len(df)} pair(s) x {len(configs)} config(s) -> {out_dir.resolve()}\n")

    for n, pair, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
        if args.only_row and n not in args.only_row:
            continue
        ann_path = df.iloc[n - 1].get(args.annotation_col, None) if args.annotation_col in df.columns else None

        # Annotation keypoints/transform don't depend on engine knobs, so
        # they're cached per working size and shared by every config that
        # needs them (with_keypoints, full_rescue, --ablation-keypoints-only).
        kp_cache: dict = {}

        def _kp_for_size(sz):
            if sz not in kp_cache:
                kf, km = _read_annotation_keypoints(
                    annotation_path=ann_path, args=args, size=sz,
                    ir_path=ir_path, faf_path=faf_path,
                )
                kt = _estimate_affine_from_annotation(
                    annotation_path=ann_path, ir_path=ir_path, faf_path=faf_path,
                    size=sz, args=args,
                )
                kp_cache[sz] = (kf, km, kt)
            return kp_cache[sz]

        if args.ablation_keypoints_only:
            _, _, kp_transform = _kp_for_size(args.size)
            if kp_transform is None:
                for name, _label, _overrides in configs:
                    records.append(dict(
                        pair=pair,
                        config=name,
                        ir=ir_path,
                        faf=faf_path,
                        status="skipped",
                        error="insufficient_annotation_landmarks",
                    ))
                pd.DataFrame(records).to_csv(results_csv, index=False)
                continue
        if _vessel_mask_missing(ir_v) or _vessel_mask_missing(faf_v):
            for name, _label, _overrides in configs:
                records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
                                    status="skipped", error="missing_vessel_mask"))
            pd.DataFrame(records).to_csv(results_csv, index=False)
            continue

        for name, _label, overrides in configs:
            # Base engine knobs, overridden per-config below (never passed
            # twice -- `overrides` may redefine any of these, e.g. `size` or
            # `use_loftr`).
            call_kwargs = dict(
                fov_scale=args.fov_scale,
                size=args.size,
                preprocess=not args.no_preprocess,
                use_loftr=_requested_use_loftr(args, not args.no_loftr),
            )
            global_saved = {}
            use_keypoints = False
            use_rescue = False
            for key, value in overrides.items():
                if key == "use_keypoints":
                    use_keypoints = bool(value)
                    continue
                if key == "_rescue":
                    use_rescue = bool(value)
                    continue
                attr = _GLOBAL_OVERRIDE_ATTRS.get(key)
                if attr is not None:
                    global_saved[attr] = getattr(utils, attr)
                    setattr(utils, attr, value)
                else:
                    if key == "use_loftr" and args.method != "auto":
                        continue
                    call_kwargs[key] = value

            if use_keypoints:
                k_fixed, k_moving, kp_transform = _kp_for_size(call_kwargs["size"])
                if k_fixed is not None and k_moving is not None:
                    call_kwargs["keypoints_fixed"] = k_fixed
                    call_kwargs["keypoints_moving"] = k_moving
                if kp_transform is None and ("keypoints_fixed" not in call_kwargs or "keypoints_moving" not in call_kwargs):
                    records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
                                        status="skipped",
                                        error="insufficient_annotation_landmarks"))
                    for attr, value in global_saved.items():
                        setattr(utils, attr, value)
                    continue
                if kp_transform is not None and "keypoints_fixed" not in call_kwargs:
                    call_kwargs["forced_transform"] = kp_transform
                    call_kwargs["forced_label"] = "annotation_landmarks"

            try:
                result = register(
                    ir_image=ir_path,
                    faf_image=faf_path,
                    ir_vessel=ir_v,
                    faf_vessel=faf_v,
                    **call_kwargs,
                )
                _assert_requested_method(args, result.label)
            except Exception as e:
                records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
                                    status="error", error=str(e),
                                    excluded_from_metrics=True,
                                    exclusion_reason="registration_error"))
                continue
            finally:
                for attr, value in global_saved.items():
                    setattr(utils, attr, value)

            # "full_rescue" mirrors run_batch --fallback exactly: the trigger
            # uses only the registration's own internal signals (no LoFTR
            # transform, or LoFTR inliers <= --fallback-inlier-threshold) and
            # never looks at landmark error.
            fallback_used = False
            if use_rescue:
                n_inl = int((result.metrics or {}).get("keypoint_matches", 0) or 0)
                not_matched = not str(result.label).startswith("LoFTR@")
                low_inliers = n_inl <= args.fallback_inlier_threshold
                if not_matched or low_inliers:
                    _, _, kp_transform = _kp_for_size(call_kwargs["size"])
                    if kp_transform is not None:
                        result = register(
                            ir_image=ir_path,
                            faf_image=faf_path,
                            ir_vessel=ir_v,
                            faf_vessel=faf_v,
                            fov_scale=args.fov_scale,
                            size=call_kwargs["size"],
                            preprocess=not args.no_preprocess,
                            use_loftr=False,
                            forced_transform=kp_transform,
                            forced_label="annotation_landmarks_fallback",
                        )
                        fallback_used = True

            # Landmark MLE/AUC on all valid annotation landmarks, every config.
            lm = _landmark_metrics(
                annotation_path=ann_path, transform=result.transform,
                ir_path=ir_path, faf_path=faf_path, size=call_kwargs["size"], args=args,
            )

            if result.label in ("annotation_landmarks", "annotation_landmarks_fallback"):
                annotation_used_for_transform = "forced_all_landmarks"
            elif call_kwargs.get("keypoints_fixed") is not None:
                annotation_used_for_transform = "blended_all_landmarks"
            else:
                annotation_used_for_transform = "none"
            scored_on = "all_landmarks" if annotation_used_for_transform == "none" \
                else "all_landmarks_in_sample"

            records.append(dict(pair=pair, config=name, ir=ir_path, faf=faf_path,
                                status="ok",
                                label=result.label, scale=result.scale,
                                fallback_used=fallback_used,
                                annotation_used_for_transform=annotation_used_for_transform,
                                scored_on=scored_on,
                                **result.metrics, **lm))

        pd.DataFrame(records).to_csv(results_csv, index=False)

    df_out = pd.DataFrame(records)
    df_out.to_csv(results_csv, index=False)
    print(f"Done. {(df_out['status'] == 'ok').sum() if not df_out.empty else 0}/{len(df_out)} runs ok.")
    print(f"Per-run metrics -> {results_csv.resolve()}")

    summary = summarize_ablation(df_out, configs)
    if not summary.empty:
        summary_csv = out_dir / "ablation_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"Summary         -> {summary_csv.resolve()}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = [args.ir_col, args.faf_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"error: CSV missing columns: {missing}\n       available: {list(df.columns)}", file=sys.stderr)
        return 2

    if args.ablation:
        return run_ablation(df, out_dir, args)
    return run_batch(df, out_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())

