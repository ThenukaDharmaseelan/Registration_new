from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import metrics as M
from . import utils, visualization
from .io import load_image, load_vessel
from .models import loftr
from .preprocessing import (
    preprocess_image,
)
from .registration import register_affine

ArrayOrPath = Union[np.ndarray, str, Path]

__all__ = ["register", "RegistrationResult", "load_image", "load_vessel", "preprocess_image"]


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    """Everything a caller needs from one registration, and nothing they don't.

    Attributes
    ----------
    transform : np.ndarray
        Final 2x3 affine mapping the moving (FAF) image onto the fixed (IR)
        frame. Apply with ``cv2.warpAffine(img, transform, (W, H))``.
    registered_image : np.ndarray
        Moving image warped into the fixed frame (RGB, uint8).
    registered_vessel : np.ndarray
        Moving vessel mask warped into the fixed frame (float32, 0/1).
    metrics : dict
        Before/after registration-quality metrics.
    overlay, checkerboard, vessel_overlap, vessel_mismatch : np.ndarray
        Diagnostic views.
    label : str
        Which internal path won (e.g. ``"LoFTR@1.83"``).
    scale : float
        The effective FOV scale used.
    """

    transform: np.ndarray
    registered_image: np.ndarray
    registered_vessel: np.ndarray
    metrics: dict = field(default_factory=dict)
    overlay: np.ndarray = None
    checkerboard: np.ndarray = None
    vessel_overlap: np.ndarray = None
    vessel_mismatch: np.ndarray = None
    label: str = ""
    scale: float = 0.0
    match_meta: dict = None
    # Loaded/resized inputs kept for convenience (e.g. saving grids).
    fixed_image: np.ndarray = None
    moving_image: np.ndarray = None
    fixed_vessel: np.ndarray = None
    moving_vessel: np.ndarray = None

    def save(self, output_dir: Union[str, Path], name: str = "pair") -> Path:
        """Write the full diagnostic image set (registered image, overlays,
        checkerboard, vessel views and the annotated grid) under
        ``output_dir``."""
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        H, W = self.fixed_image.shape[:2]
        M_scale = utils.make_scale_matrix(self.scale, W / 2.0, H / 2.0)
        decomp = utils.decompose_affine(self.transform)
        _args = (od, name,
                 self.fixed_image, self.fixed_vessel,
                 self.moving_image, self.moving_vessel,
                 self.registered_image, self.registered_vessel,
                 M_scale)
        _kw = dict(decomp=decomp,
                   stage=self.label, best_scale=self.scale,
                   scale_cluster="fov", match_meta=self.match_meta)
        visualization.save_outputs(*_args, **_kw)
        return od


# ─────────────────────────────────────────────────────────────────────────────
# Input loading
# ─────────────────────────────────────────────────────────────────────────────

def _as_image(x: ArrayOrPath, h: int, w: int) -> np.ndarray:
    """Return an (h, w, 3) uint8 RGB image from an array or a path."""
    if isinstance(x, (str, Path)):
        return load_image(x, h, w)
    arr = np.asarray(x)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr] * 3, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr


def _vessel_mask_missing(x) -> bool:
    """Return whether a CSV/path vessel-mask value is intentionally missing."""
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip() or not Path(x).exists()
    if isinstance(x, Path):
        return not x.exists()
    # Pandas represents blank CSV cells as float NaN.
    return isinstance(x, (float, np.floating)) and bool(np.isnan(x))


def _as_vessel(x: Optional[ArrayOrPath], h: int, w: int) -> np.ndarray:
    """Return an (h, w) float32 {0,1} vessel mask from an array or a path."""
    if isinstance(x, (str, Path)):
        return load_vessel(x, h, w)
    arr = np.asarray(x)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    thr = 0.5 if arr.max() <= 1.0 else 127
    return (arr > thr).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics assembly
# ─────────────────────────────────────────────────────────────────────────────

def _collect_metrics(fi, fv, mi, mv, wi, wv, transform, scale) -> dict:
    """Before/after quality metrics for one registered pair."""
    H, W = fv.shape

    dsc_before = round(M.dice(fv, mv), 4)
    dsc_after  = round(M.dice(fv, wv), 4)
    ncc_before = round(M.ncc(fv, mv), 4)
    ncc_after  = round(M.ncc(fv, wv), 4)

    hd_before  = M.hausdorff_vessel_distance(fv, mv)
    hd_after   = M.hausdorff_vessel_distance(fv, wv)
    wass_before = M.wasserstein_vessel_distance(fv, mv)
    wass_after = M.wasserstein_vessel_distance(fv, wv)
    ssd_before = M.ssd_vessel_distance(fv, mv)
    ssd_after  = M.ssd_vessel_distance(fv, wv)

    # image-intensity (greyscale, inside common FOV)
    M_scale = utils.make_scale_matrix(scale, W / 2.0, H / 2.0)
    mi_scaled = utils.warp_img(mi, M_scale, H, W)
    fov_before = ((utils.get_fov_mask(fi) > 0) & (utils.get_fov_mask(mi_scaled) > 0)).astype(np.float32)
    fov_after  = ((utils.get_fov_mask(fi) > 0) & (utils.get_fov_mask(wi) > 0)).astype(np.float32)
    img_before = M.image_intensity_metrics(fi, mi_scaled, fov_mask=fov_before)
    img_after  = M.image_intensity_metrics(fi, wi, fov_mask=fov_after)

    def _image_binary_dice(a_rgb, b_rgb, fov_mask):
        """Dice on binarized grayscale images inside a common FOV mask."""
        if fov_mask is None or float(fov_mask.sum()) < 1.0:
            return None
        a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        m = fov_mask > 0
        if int(m.sum()) < 100:
            return None

        # Zero out outside-FOV so thresholding is driven by valid retina area.
        a_m = a.copy(); a_m[~m] = 0
        b_m = b.copy(); b_m[~m] = 0
        _, a_bin = cv2.threshold(a_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b_bin = cv2.threshold(b_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        aa = (a_bin > 0) & m
        bb = (b_bin > 0) & m
        denom = float(aa.sum() + bb.sum())
        if denom < 1.0:
            return None
        return round(float(2.0 * (aa & bb).sum() / denom), 4)

    image_dice_before = _image_binary_dice(fi, mi_scaled, fov_before)
    image_dice_after = _image_binary_dice(fi, wi, fov_after)
    image_ncc_before = img_before.get("ncc_img")
    image_ncc_after = img_after.get("ncc_img")

    def _image_ssd(a_rgb, b_rgb, fov_mask):
        if fov_mask is None:
            return None
        m = fov_mask > 0
        if int(m.sum()) < 16:
            return None
        a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        ssd = float(np.sum((a[m] - b[m]) ** 2))
        return round(ssd, 4)

    image_ssd_before = _image_ssd(fi, mi_scaled, fov_before)
    image_ssd_after = _image_ssd(fi, wi, fov_after)

    vessel_ssd_before = ssd_before["ssd"]
    vessel_ssd_after = ssd_after["ssd"]

    def _d(b, a, higher=True):
        if b is None or a is None:
            return None
        return round((a - b) if higher else (b - a), 4)

    wass_after = (wass_after.get("wass_w2") if wass_after.get("wass_w2") is not None
                  else wass_after.get("wass_w1"))
    wass_bef   = (wass_before.get("wass_w2") if wass_before.get("wass_w2") is not None
                  else wass_before.get("wass_w1"))

    return dict(
        # --- explicit vessel metrics requested ---
        vessel_dice_before=dsc_before,
        vessel_dice_after=dsc_after,
        vessel_dice_delta=_d(dsc_before, dsc_after, higher=True),
        vessel_ncc_before=ncc_before,
        vessel_ncc_after=ncc_after,
        vessel_ncc_delta=_d(ncc_before, ncc_after, higher=True),
        vessel_ssd_before=vessel_ssd_before,
        vessel_ssd_after=vessel_ssd_after,
        vessel_ssd_delta=_d(vessel_ssd_before, vessel_ssd_after, higher=False),
        # --- explicit image metrics requested ---
        image_dice_before=image_dice_before,
        image_dice_after=image_dice_after,
        image_dice_delta=_d(image_dice_before, image_dice_after, higher=True),
        image_ncc_before=image_ncc_before,
        image_ncc_after=image_ncc_after,
        image_ncc_delta=_d(image_ncc_before, image_ncc_after, higher=True),
        image_ssd_before=image_ssd_before,
        image_ssd_after=image_ssd_after,
        image_ssd_delta=_d(image_ssd_before, image_ssd_after, higher=False),
        # --- vessel-mask overlap (higher = better) ---
        dice_before=dsc_before, dice_after=dsc_after,
        dice_delta=_d(dsc_before, dsc_after, higher=True),
        ncc_before=ncc_before, ncc_after=ncc_after,
        ncc_delta=_d(ncc_before, ncc_after, higher=True),
        # --- surface / distribution distances (lower = better) ---
        hd95_before=hd_before.get("hd95"), hd95_after=hd_after.get("hd95"),
        hd95_delta=_d(hd_before.get("hd95"), hd_after.get("hd95"), higher=False),
        asd_before=hd_before.get("masd"), asd_after=hd_after.get("masd"),
        asd_delta=_d(hd_before.get("masd"), hd_after.get("masd"), higher=False),
        wasserstein_before=wass_bef, wasserstein_after=wass_after,
        wasserstein_delta=_d(wass_bef, wass_after, higher=False),
        ssd_before=vessel_ssd_before, ssd_after=vessel_ssd_after,
        ssd_delta=_d(vessel_ssd_before, vessel_ssd_after, higher=False),
        # --- image intensity (cross-modal; MI/NMI primary) ---
        mi_before=img_before["mi"], mi_after=img_after["mi"],
        nmi_before=img_before["nmi"], nmi_after=img_after["nmi"],
        ssim_before=img_before["ssim"], ssim_after=img_after["ssim"],
        psnr_before=img_before["psnr"], psnr_after=img_after["psnr"],
        # --- transform decomposition ---
        **{f"transform_{k}": round(v, 4)
           for k, v in utils.decompose_affine(transform).items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# The one public entry point
# ─────────────────────────────────────────────────────────────────────────────

def register(
    ir_image: Optional[ArrayOrPath] = None,
    faf_image: Optional[ArrayOrPath] = None,
    ir_vessel: Optional[ArrayOrPath] = None,
    faf_vessel: Optional[ArrayOrPath] = None,
    *,
    fov_scale: Optional[float] = None,
    size: int = 224,
    preprocess: bool = True,
    use_loftr: bool = True,
    affine_dof: Optional[int] = None,
    min_conf: float = 0.4,
    use_denseflow: bool = True,
    ransac_thresh: Optional[float] = None,
    use_ransac: bool = True,
    compute_metrics: bool = True,
    forced_transform: Optional[np.ndarray] = None,
    forced_label: str = "annotation_landmarks",
    keypoints_fixed: Optional[np.ndarray] = None,
    keypoints_moving: Optional[np.ndarray] = None,
    fixed_image: Optional[ArrayOrPath] = None,
    moving_image: Optional[ArrayOrPath] = None,
    fixed_vessel: Optional[ArrayOrPath] = None,
    moving_vessel: Optional[ArrayOrPath] = None,
) -> RegistrationResult:
    """Register a moving (FAF) image onto a fixed (IR) image.

    Parameters
    ----------
    ir_image, faf_image, ir_vessel, faf_vessel
        numpy arrays or paths. IR = fixed (30°), FAF = moving (55°).
    fov_scale, size, preprocess, use_loftr, compute_metrics
        Engine knobs (see module docstring).
    affine_dof
        ``None`` (default) auto-selects between a 4-DoF similarity and a
        6-DoF affine per pair by ASD; pass ``4`` or ``6`` to force one
        (used for ablations).
    min_conf
        LoFTR confidence threshold for the initial matching pass (default
        0.4; used for ablations).
    use_denseflow
        ``False`` skips the dense-flow refinement pass entirely (used for
        ablations).
    ransac_thresh
        RANSAC reprojection threshold in pixels, in ``size``-grid units.
        Default (``None``) is 5.0px scaled to the working grid (5.0 *
        size/224) -- tuned at size=224, a fixed absolute-pixel threshold
        would be proportionally far tighter on a larger grid (e.g. only
        ~1.5px-equivalent at size=768), rejecting good matches as outliers
        and degrading the fit. Pass an explicit value to override (used for
        ablations, which always run at the default size).
    use_ransac
        When False, LoFTR affine fitting uses LMEDS instead of RANSAC.

    Returns
    -------
    RegistrationResult
    """
    if ransac_thresh is None:
        ransac_thresh = 5.0 * (float(size) / 224.0)

    if ir_image is None:
        ir_image = fixed_image
    if faf_image is None:
        faf_image = moving_image
    if ir_vessel is None:
        ir_vessel = fixed_vessel
    if faf_vessel is None:
        faf_vessel = moving_vessel

    if ir_image is None or faf_image is None:
        raise TypeError("register() requires ir_image and faf_image")

    # Configure the (module-global) engine knobs for this call.
    if fov_scale is not None:
        utils.FOV_SCALE_RATIO = float(fov_scale)

    h = w = size

    fi = _as_image(ir_image, h, w)
    mi = _as_image(faf_image, h, w)

    _ir_vessel_missing = _vessel_mask_missing(ir_vessel)
    _faf_vessel_missing = _vessel_mask_missing(faf_vessel)
    if _ir_vessel_missing or _faf_vessel_missing:
        missing = [side for side, is_missing in (("ir", _ir_vessel_missing),
                                                 ("faf", _faf_vessel_missing))
                   if is_missing]
        raise ValueError(f"missing vessel mask(s): {', '.join(missing)}")

    fv = _as_vessel(ir_vessel, h, w)
    mv = _as_vessel(faf_vessel, h, w)

    fi_raw = fi.copy()
    mi_raw = mi.copy()

    if preprocess:
        kw = dict(clip_limit=8.0, tile_grid=(4, 4), blur_ksize=3, blur_sigma=0.5)
        fi = preprocess_image(fi, **kw)
        mi = preprocess_image(mi, **kw)

    if use_loftr and not loftr.LOFTR_AVAILABLE:
        loftr.init_loftr()

    if forced_transform is not None:
        transform = np.asarray(forced_transform, dtype=np.float32)
        if transform.shape != (2, 3):
            raise ValueError(f"forced_transform must have shape (2, 3), got {transform.shape}")
        label = str(forced_label)
        scale = float(utils.FOV_SCALE_RATIO)
        match_meta = dict(n_matches=0, n_inliers=0)
    else:
        label, transform, scale, match_meta = register_affine(
            fv, mv, h, w,
            no_loftr=not use_loftr,
            affine_dof=affine_dof,
            min_conf=min_conf,
            use_denseflow=use_denseflow,
            ransac_thresh=ransac_thresh,
            use_ransac=use_ransac,
            fi=fi, mi=mi,
            keypoints_fixed=keypoints_fixed,
            keypoints_moving=keypoints_moving,
        )

    wi = utils.warp_img(mi_raw, transform, h, w)
    wv = utils.warp_mask(mv, transform, h, w)

    metrics = (_collect_metrics(fi_raw, fv, mi_raw, mv, wi, wv, transform, scale)
               if compute_metrics else {})
    if compute_metrics:
        metrics["keypoint_matches"] = int(match_meta["n_inliers"]) if match_meta else 0
        metrics["keypoint_matches_total"] = int(match_meta["n_matches"]) if match_meta else 0

    return RegistrationResult(
        transform=transform,
        registered_image=wi,
        registered_vessel=wv,
        metrics=metrics,
        overlay=visualization.make_overlap_image(fi_raw, wi),
        checkerboard=visualization.make_checkerboard(fi_raw, wi),
        vessel_overlap=visualization.make_overlap_vessels(fv, wv, fi_raw),
        vessel_mismatch=visualization.make_vessel_mismatch(fv, wv, fi_raw),
        label=label,
        scale=scale,
        match_meta=match_meta,
        fixed_image=fi_raw, moving_image=mi_raw,
        fixed_vessel=fv, moving_vessel=mv,
    )
