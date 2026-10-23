
from __future__ import annotations

import numpy as np
import cv2

from . import utils
from .matching import loftr_to_affine
from .metrics import hausdorff_vessel_distance
from .models import loftr
from .preprocessing import (
    vessel_density,
)


def _run_tvl1(src8, dst8):
    """Compute dense optical flow and fall back to Farneback if TV-L1 is unavailable."""
    try:
        of = cv2.optflow.DualTVL1OpticalFlow_create()
        return of.calc(src8, dst8, None)
    except Exception:
        return cv2.calcOpticalFlowFarneback(src8, dst8, None,
                                            0.5, 3, 25, 5, 7, 1.5, 0)


def _flow_to_affine_from_field(fv, flow, M_init, use_ransac=True):
    """Fit a local affine from dense optical flow and compose it onto the current transform."""
    H, W = fv.shape
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    q25 = float(np.percentile(mag, 25))
    q75 = float(np.percentile(mag, 75))
    iqr = q75 - q25
    iqr_bound = float(np.median(mag)) + 1.5 * iqr
    max_disp = float(max(iqr_bound, 1.0))
    flow = cv2.GaussianBlur(flow, (31, 31), 10)
    flow[..., 0] = np.clip(flow[..., 0], -max_disp, max_disp)
    flow[..., 1] = np.clip(flow[..., 1], -max_disp, max_disp)

    vessel_mask = (fv > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.dilate(vessel_mask, k).astype(np.float32)
    flow[..., 0] *= vmask
    flow[..., 1] *= vmask

    step = 10
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    sp = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)
    fx = flow[::step, ::step, 0].ravel()
    fy = flow[::step, ::step, 1].ravel()
    dp = sp + np.stack([fx, fy], -1).astype(np.float32)
    valid = np.sqrt(fx ** 2 + fy ** 2) > 0.3
    if valid.sum() < 6:
        return None

    method = cv2.RANSAC if use_ransac else cv2.LMEDS
    Mf, _ = cv2.estimateAffine2D(sp[valid], dp[valid], method=method,
                                 ransacReprojThreshold=3.0,
                                 maxIters=2000, confidence=0.999)
    if Mf is None:
        return None
    return utils.compose(M_init, Mf.astype(np.float32))


def _asd_score(M, fv, mv):
    H, W = fv.shape
    warped = utils.warp_mask(mv, M, H, W)
    score = hausdorff_vessel_distance(fv, warped)
    return score.get("masd")


def _denseflow_refine(fv, mv, M_in, use_ransac=True):
    """Refine the affine transform with dense optical flow if it improves the ASD score."""
    H, W = fv.shape
    fv8 = (fv.astype(np.float32) * 255).astype(np.uint8)
    mv8 = (mv.astype(np.float32) * 255).astype(np.uint8)
    mv_w = cv2.warpAffine(mv8, M_in, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    flow_full = _run_tvl1(mv_w, fv8)
    M_full = _flow_to_affine_from_field(fv, flow_full, M_in, use_ransac=use_ransac)
    if M_full is None:
        return M_in, False

    base_score = _asd_score(M_in, fv, mv)
    flow_score = _asd_score(M_full, fv, mv)
    if base_score is None or flow_score is None:
        return M_in, False

    improved = flow_score < base_score
    if improved:
        print(f"  [DenseFlow] ASD {base_score:.3f} -> {flow_score:.3f}")
        return M_full, True
    return M_in, False


def _keypoint_match_meta(fi, mi_sc, matches, n_inliers, kp_fixed=None, kp_moving_sc=None):
    """Package a LoFTR match result into reportable metadata.

    Keeps ``fi``/``mi_sc`` (the fixed image and the moving image warped to
    this candidate's scale) plus the raw ``src``/``dst``/``inlier_mask``
    correspondences, so the winning candidate's keypoint matches can be
    rendered as a diagnostic image (see ``visualization.make_keypoint_match_image``).

    ``kp_fixed``/``kp_moving_sc`` are the raw annotation landmarks (moving
    side already carried through the candidate's ``M_sc`` scale, same frame
    as ``mi_sc``) so the diagnostic image can show them distinctly from the
    LoFTR-derived correspondences -- otherwise there is no visible evidence
    that the annotation keypoints (as opposed to the vessel masks) fed into
    the match at all.
    """
    if matches is None:
        return None
    src = np.asarray(matches.get("src", []), dtype=np.float32)
    dst = np.asarray(matches.get("dst", []), dtype=np.float32)
    return dict(
        n_matches=len(src),
        n_inliers=int(n_inliers),
        fi=fi, mi_sc=mi_sc,
        src=src, dst=dst,
        inlier_mask=matches.get("inlier_mask"),
        kp_fixed=kp_fixed, kp_moving_sc=kp_moving_sc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Active registration flow (full 6-DoF affine + global dense-flow refinement)
# ─────────────────────────────────────────────────────────────────────────────
#   IR / FAF images + vessel masks
#                |
#                v
#       preprocess / density check
#                |
#                v
#       fixed FOV seed -> full affine initialization
#                |
#                v
#      LoFTR vessel matches -> RANSAC affine fit (6-DoF)
#                |
#                v
#   dense optical flow over the whole image -> global affine update
#                |
#                v
#      accept only if ASD improves; otherwise keep current affine
#                |
#                v
#                  best full-affine candidate
# ─────────────────────────────────────────────────────────────────────────────

def fov_scale_candidate(fv, mv):
    H, W = fv.shape
    s = float(utils.FOV_SCALE_RATIO)
    cx, cy = W / 2.0, H / 2.0

    M = utils.make_scale_matrix(s, cx, cy)
    print(f"  [FOVScale] fixed FOV seed {s:.3f}x")
    return M, s


def _loftr_best_over_ksize(fv, mv_sc, **kw):
    """LoFTR once per closing kernel size; keep the highest-inlier candidate."""
    best_M, best_n, best_k, best_matches = None, 0, None, None
    k_orig = utils.FAF_CLOSE_KSIZE
    try:
        for k in utils.LOFTR_CLOSE_KSIZES:
            utils.FAF_CLOSE_KSIZE = k
            M, n, matches = loftr_to_affine(fv, mv_sc, **kw)
            if M is None:
                continue
            if n > best_n:
                best_M, best_n, best_k, best_matches = M, n, k, matches
    finally:
        utils.FAF_CLOSE_KSIZE = k_orig
    if best_k is not None and len(utils.LOFTR_CLOSE_KSIZES) > 1:
        print(f"  [LoFTR] close ksize={best_k} won (inliers={best_n})")
    return best_M, best_n, best_matches


def _transform_points(pts, M):
    """Forward-transform (N, 2) points through affine matrix ``M`` (or pass through None)."""
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return pts
    return cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def _loftr_from_scale(fv, mv, M_sc, scale_lbl, H, W,
                      any_mask_missing=False, affine_dof=None, min_conf=0.4,
                      ransac_thresh=5.0,
                      use_ransac=True, fi=None, mi=None,
                      keypoints_fixed=None, keypoints_moving=None):
    if not loftr.LOFTR_AVAILABLE:
        return None, None
    mv_sc = utils.warp_mask(mv, M_sc, H, W)
    mi_sc = utils.warp_img(mi, M_sc, H, W) if mi is not None else None

    # keypoints_moving is given in the original (pre-M_sc) moving-image frame,
    # but mv_sc and the LoFTR input live in the FOV-scaled frame -- carry the points through
    # the same M_sc warp so the blended peaks land on the right vessels.
    keypoints_moving_sc = _transform_points(keypoints_moving, M_sc)

    # Mask usability heuristics intentionally removed: the registration path
    # should proceed whenever the vessel masks are present, rather than rejecting
    # them early based on connected-component size or low-pixel thresholds.
    if any_mask_missing:
        print(f"  [LoFTR@{scale_lbl}] skipped (missing vessel mask)")
        return None, None

    M_loftr, n_loftr, matches_v = _loftr_best_over_ksize(
        fv, mv_sc, affine_dof=affine_dof, min_conf=min_conf,
        ransac_thresh=ransac_thresh, use_ransac=use_ransac,
        keypoints0=keypoints_fixed, keypoints1=keypoints_moving_sc,
    )
    if M_loftr is None or n_loftr < utils.LOFTR_MIN_INLIERS:
        M2, n2, matches2 = loftr_to_affine(
            fv, mv_sc, min_conf=utils.LOFTR_RETRY_CONF, affine_dof=affine_dof,
            ransac_thresh=ransac_thresh,
            use_ransac=use_ransac,
            keypoints0=keypoints_fixed, keypoints1=keypoints_moving_sc,
        )
        if M2 is not None and n2 > (n_loftr or 0):
            print(f"  [LoFTR@{scale_lbl}] retry conf="
                  f"{utils.LOFTR_RETRY_CONF}: n {n_loftr} -> {n2}")
            M_loftr, n_loftr, matches_v = M2, n2, matches2

    if M_loftr is None:
        print(f"  [LoFTR@{scale_lbl}] no matches")
        return None, None

    M_full = utils.compose(M_sc, M_loftr)
    if not utils.sane(M_full):
        print(f"  [LoFTR@{scale_lbl}] insane affine — skipped")
        return None, None

    print(f"  [LoFTR@{scale_lbl}] n={n_loftr}")
    return M_full, _keypoint_match_meta(fi, mi_sc, matches_v, n_loftr,
                                        kp_fixed=keypoints_fixed,
                                        kp_moving_sc=keypoints_moving_sc)


def register_affine(fv, mv, H, W, no_loftr=False, any_mask_missing=False, affine_dof=None,
                    min_conf=0.4, use_denseflow=True,
                    ransac_thresh=5.0, use_ransac=True, fi=None, mi=None,
                    keypoints_fixed=None, keypoints_moving=None):
    """Full affine registration path with a dense-flow refinement pass.

    The affine estimate remains the primary solution, and an optional dense-flow
    stage is applied only if it improves the vessel ASD distance.

    ``affine_dof`` (``None``, ``4``, or ``6``) forces the LoFTR fit to a
    similarity or full affine transform instead of the default ASD-based
    auto-selection; see :func:`matching.loftr_to_affine`.

    ``min_conf`` is the LoFTR confidence threshold for the initial matching
    pass (the low-confidence retry still uses ``utils.LOFTR_RETRY_CONF``).

    ``use_denseflow=False`` skips the dense-flow refinement pass entirely
    (used for ablations).

    ``ransac_thresh`` is the RANSAC reprojection threshold (pixels) used to
    fit the LoFTR-derived transform -- see :func:`matching.loftr_to_affine`.

    ``use_ransac=False`` switches LoFTR affine fitting to LMEDS.
    """
    fv_density = vessel_density(fv)
    mv_density = vessel_density(mv)
    print(f"  [Density] IR={fv_density:.4f}  FAF={mv_density:.4f}")

    M_scale_best, best_s = fov_scale_candidate(fv, mv)
    final_label = "full_affine_scale_best"
    final_M = M_scale_best
    final_match_meta = None

    if not no_loftr:
        M_loftr, match_meta = _loftr_from_scale(
            fv, mv, M_scale_best, f"{best_s:.2f}", H, W,
            any_mask_missing=any_mask_missing, affine_dof=affine_dof,
            min_conf=min_conf,
            ransac_thresh=ransac_thresh, use_ransac=use_ransac,
            fi=fi, mi=mi,
            keypoints_fixed=keypoints_fixed,
            keypoints_moving=keypoints_moving,
        )
        if M_loftr is not None:
            final_label = f"LoFTR@{best_s:.2f}"
            final_M = M_loftr
            final_match_meta = match_meta

    if use_denseflow:
        refined_M, improved = _denseflow_refine(fv, mv, final_M, use_ransac=use_ransac)
        if improved:
            final_label = f"{final_label}+denseflow"
            final_M = refined_M

    print(f"  [Winner] {final_label}")
    return final_label, final_M, best_s, final_match_meta
