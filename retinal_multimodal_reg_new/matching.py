from __future__ import annotations

import cv2
import numpy as np

from . import utils
from .metrics import hausdorff_vessel_distance
from .models import loftr
from .preprocessing import close_faf_vessel


# ─────────────────────────────────────────────────────────────────────────────
# LoFTR matching
# ─────────────────────────────────────────────────────────────────────────────

def _blend_keypoints_into_feature_map(feature_map: np.ndarray, keypoints: np.ndarray | None,
                                     sigma: float = 4.0, strength: float = 1.0) -> np.ndarray:
    """Inject landmark points as soft Gaussian peaks into the LoFTR feature map.

    The default vessel-only behavior is unchanged when ``keypoints`` is None.
    When keypoints are provided, they are added as weak, local peaks that guide
    LoFTR without fully replacing the vessel representation.
    """
    if keypoints is None:
        return np.asarray(feature_map, dtype=np.float32)

    pts = np.asarray(keypoints, dtype=np.float32)
    if pts.size == 0:
        return np.asarray(feature_map, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"keypoints must have shape (N, 2), got {pts.shape!r}")

    h, w = feature_map.shape[:2]
    blended = np.asarray(feature_map, dtype=np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y in pts:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        heat = np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2.0 * sigma * sigma))
        blended = np.maximum(blended, heat * strength)
    return blended.astype(np.float32)


def _resolve_keypoints(keypoints, fixed=None, moving=None):
    """Normalize the `keypoints` dict/tuple shorthand into explicit (fixed, moving) arrays.

    Explicit `fixed`/`moving` arguments win over whatever `keypoints` supplies,
    so callers can override just one side without repackaging `keypoints`.
    """
    if keypoints is None:
        return fixed, moving
    if isinstance(keypoints, dict):
        fixed = keypoints.get("fixed") if fixed is None else fixed
        moving = keypoints.get("moving") if moving is None else moving
        return fixed, moving
    if isinstance(keypoints, (tuple, list)) and len(keypoints) == 2:
        fixed = keypoints[0] if fixed is None else fixed
        moving = keypoints[1] if moving is None else moving
        return fixed, moving
    raise ValueError("keypoints must be a dict or a 2-item tuple/list of (fixed, moving)")


def loftr_match(fv, mv_aligned, min_conf=0.4, keypoints=None, keypoints0=None, keypoints1=None):
    """Run LoFTR on the fixed/moving vessel representations and return confident correspondences.

    Matches raw binary vessel masks; the FAF (moving) side receives FAF
    closing morphology first. Optionally blends annotation `keypoints` as soft
    peaks, then keeps only matches above `min_conf`. Returns `(None, None)` if
    LoFTR isn't available or fewer than 8 matches clear the threshold.
    """
    if keypoints is not None:
        keypoints0, keypoints1 = _resolve_keypoints(keypoints, keypoints0, keypoints1)
    if not loftr.LOFTR_AVAILABLE or loftr.loftr_matcher is None:
        return None, None
    import torch as th
    mv_closed = close_faf_vessel(mv_aligned)
    fixed_input = np.asarray(fv, dtype=np.float32)
    moving_input = mv_closed.astype(np.float32)
    if keypoints0 is not None:
        fixed_input = _blend_keypoints_into_feature_map(fixed_input, keypoints0)
    if keypoints1 is not None:
        moving_input = _blend_keypoints_into_feature_map(moving_input, keypoints1)
    f_t = th.from_numpy(fixed_input).unsqueeze(0).unsqueeze(0).to(utils.DEVICE)
    m_t = th.from_numpy(moving_input).unsqueeze(0).unsqueeze(0).to(utils.DEVICE)
    with th.no_grad():
        out = loftr.loftr_matcher({"image0": f_t, "image1": m_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    mask = conf > min_conf
    if mask.sum() < 8:
        print(f"  [LoFTR] only {int(mask.sum())} matches (conf>{min_conf})")
        return None, None
    print(f"  [LoFTR] {int(mask.sum())} confident matches")
    return kp1[mask].astype(np.float32), kp0[mask].astype(np.float32)


def loftr_to_affine(fv, mv_aligned, min_conf=0.4, params=None, affine_dof=None,
                    ransac_thresh=5.0, use_ransac=True, keypoints=None, keypoints0=None, keypoints1=None,
                    keypoint_repeat=8):
    """Fit an affine from vessel-mask LoFTR correspondences.

    By default (``affine_dof=None``) fits both a full 6-DoF affine and a
    4-DoF similarity transform (rotation + uniform scale + translation) via
    RANSAC on the same correspondences, then picks whichever actually warps
    the vessel mask closer to the fixed one (lower ASD), rather than
    whichever has more RANSAC inliers. Inlier count is not a reliable proxy
    here: with sparse correspondences the full affine's extra shear/
    anisotropic-scale freedom can match one more (noisy) point while
    producing a far worse warp overall -- even an exact inlier tie can hide
    a large ASD gap between the two models.

    Passing ``affine_dof=6`` or ``affine_dof=4`` forces the full affine or
    the similarity transform respectively, skipping the ASD comparison
    (used for ablations).

    ``ransac_thresh`` is the reprojection threshold (pixels, default 5.0)
    used when ``use_ransac=True``.

    ``use_ransac=False`` switches the robust solver to LMEDS (no RANSAC).

    ``keypoints``/``keypoints0``/``keypoints1`` (annotation landmarks) are
    blended into the LoFTR feature maps as Gaussian peaks (see
    :func:`loftr_match`) *and* their exact correspondences are concatenated
    directly into the RANSAC correspondence pool, each repeated
    ``keypoint_repeat`` times so a handful of annotated points (5-40, say)
    aren't diluted into invisibility among hundreds of vessel matches and
    actually carry weight in the final least-squares affine refit. Blending
    alone only nudges LoFTR toward the right match with no guarantee it's
    found there -- the direct correspondences guarantee they're used even
    if LoFTR's own matching ignores them. When LoFTR finds no vessel
    matches at all, the annotation correspondences alone are used to fit
    the affine (needs >= 3 points for the 6-DoF fit).

    Returns ``(M, n_inliers, matches)`` where ``matches`` is a dict with
    ``src``/``dst`` keypoint arrays and the RANSAC ``inlier_mask`` (or
    ``None`` when no affine could be fit).
    """
    if affine_dof not in (None, 4, 6):
        raise ValueError(f"affine_dof must be None, 4, or 6, got {affine_dof!r}")

    keypoints0, keypoints1 = _resolve_keypoints(keypoints, keypoints0, keypoints1)

    src, dst = loftr_match(
        fv,
        mv_aligned,
        min_conf=min_conf,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
    )

    loftr_src, loftr_dst = src, dst
    n_loftr = 0 if loftr_src is None else len(loftr_src)

    kp_src = kp_dst = None
    if keypoints0 is not None and keypoints1 is not None:
        kp_dst_all = np.asarray(keypoints0, dtype=np.float32)
        kp_src_all = np.asarray(keypoints1, dtype=np.float32)
        n_kp = min(len(kp_src_all), len(kp_dst_all))
        if n_kp > 0:
            kp_src, kp_dst = kp_src_all[:n_kp], kp_dst_all[:n_kp]
    n_kp = 0 if kp_src is None else len(kp_src)

    if loftr_src is None and kp_src is None:
        return None, 0, None
    if loftr_src is None:
        fit_src, fit_dst = kp_src, kp_dst
    elif kp_src is not None:
        # Repeated `keypoint_repeat` times so a handful of annotation points
        # aren't diluted into invisibility among many vessel matches in the
        # final least-squares refit -- this padded pool is only used to fit
        # the transform, never reported (see report_src/report_dst below),
        # so it doesn't inflate the match/inlier counts callers see.
        fit_src = np.concatenate([loftr_src] + [kp_src] * keypoint_repeat, axis=0)
        fit_dst = np.concatenate([loftr_dst] + [kp_dst] * keypoint_repeat, axis=0)
    else:
        fit_src, fit_dst = loftr_src, loftr_dst
    if len(fit_src) < 3:
        return None, 0, None

    method = cv2.RANSAC if use_ransac else cv2.LMEDS
    M_aff = im_aff = M_sim = im_sim = None
    if affine_dof in (None, 6):
        M_aff, im_aff = cv2.estimateAffine2D(
            fit_src,
            fit_dst,
            method=method,
            ransacReprojThreshold=ransac_thresh,
            maxIters=5000,
            confidence=0.999,
        )
    if affine_dof in (None, 4):
        M_sim, im_sim = cv2.estimateAffinePartial2D(
            fit_src,
            fit_dst,
            method=method,
            ransacReprojThreshold=ransac_thresh,
            maxIters=5000,
            confidence=0.999,
        )

    if affine_dof == 6:
        M, im = M_aff, im_aff
    elif affine_dof == 4:
        M, im = M_sim, im_sim
    else:
        H, W = fv.shape

        def _warp_asd(M):
            if M is None:
                return None
            warped = cv2.warpAffine(mv_aligned.astype(np.float32), M, (W, H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            warped = (warped >= 0.45).astype(np.float32)
            return hausdorff_vessel_distance(fv, warped).get("masd")

        asd_aff = _warp_asd(M_aff)
        asd_sim = _warp_asd(M_sim)

        if asd_aff is not None and (asd_sim is None or asd_aff <= asd_sim):
            M, im = M_aff, im_aff
        else:
            M, im = M_sim, im_sim

    if M is None:
        return None, 0, None
    if not utils.sane(M, params):
        return None, 0, None

    # Report each correspondence once: the LoFTR matches as-is, plus one
    # inlier flag per keypoint (its first repeat -- all repeats of the same
    # point get the same in/out decision) rather than `keypoint_repeat`
    # duplicate entries that would inflate n_matches/n_inliers. `fit_src`
    # always has the (unrepeated) LoFTR matches first, so slicing off the
    # first repeat's worth of keypoints gives one copy of each in all cases.
    report_src = fit_src[:n_loftr + n_kp]
    report_dst = fit_dst[:n_loftr + n_kp]
    if im is not None:
        report_inlier_mask = np.concatenate([im[:n_loftr], im[n_loftr:n_loftr + n_kp]])
    else:
        report_inlier_mask = None
    n_inliers = int(report_inlier_mask.sum()) if report_inlier_mask is not None else 0
    matches = dict(src=report_src, dst=report_dst, inlier_mask=report_inlier_mask)
    return M.astype(np.float32), n_inliers, matches

