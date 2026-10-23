from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from . import utils

# Optional POT (python-optimal-transport) for exact Wasserstein.
try:
    import ot as _ot
    _POT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ot = None
    _POT_AVAILABLE = False

# Optional scikit-image for SSIM.
try:
    from skimage.metrics import structural_similarity as _sk_ssim
    _SSIM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SSIM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Simple overlap metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice(a, b):
    return float(2 * (a * b).sum() / (a.sum() + b.sum() + 1e-8))


def ncc(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean();                 b = b - b.mean()
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


# ─────────────────────────────────────────────────────────────────────────────
# Whole-structure vessel distances
# ─────────────────────────────────────────────────────────────────────────────

def hausdorff_vessel_distance(fv, wv, max_pts=3000):
    H, W     = fv.shape
    img_diag = float(np.sqrt(H ** 2 + W ** 2))
    rng      = np.random.default_rng()

    def _mask_pts(v, n):
        pts = np.argwhere(np.asarray(v) > 0).astype(np.float32)
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            pts = pts[idx]
        return pts

    pts_f = _mask_pts(fv, max_pts)
    pts_w = _mask_pts(wv, max_pts)
    if len(pts_f) < 3 or len(pts_w) < 3:
        return dict(hd95=None, masd=None)
    D      = cdist(pts_f, pts_w)
    nn_fw  = D.min(axis=1)
    nn_bw  = D.min(axis=0)
    pooled = np.concatenate([nn_fw, nn_bw])
    hd95     = float(np.percentile(pooled, 95))
    masd     = float(pooled.mean())     # symmetric (pooled) surface distance == ASD
    return dict(
        hd95 = round(hd95, 2),
        masd = round(masd, 2),
    )


def _sliced_wasserstein_fallback(pts_f, pts_w, n_proj=200):
    rng  = np.random.default_rng()
    dirs = rng.standard_normal((n_proj, 2)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sw = 0.0
    for d in dirs:
        p1 = pts_f @ d
        p2 = pts_w @ d
        p2r = rng.choice(p2, len(p1), replace=len(p2) < len(p1))
        sw += float(np.mean(np.abs(np.sort(p1) - np.sort(p2r))))
    return sw / n_proj


def wasserstein_vessel_distance(fv, wv, max_pts=2000, n_proj=200):
    H, W     = fv.shape
    img_diag = float(np.sqrt(H ** 2 + W ** 2))

    def _mask_pts(v, n):
        pts = np.argwhere(np.asarray(v) > 0).astype(np.float32)
        if len(pts) > n:
            idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
            pts = pts[idx]
        return pts

    pts_f = _mask_pts(fv, max_pts)
    pts_w = _mask_pts(wv, max_pts)
    null = dict(wass_w1=None, wass_w2=None,
                wass_w1_norm=None, wass_w2_norm=None)
    if len(pts_f) < 3 or len(pts_w) < 3:
        return null
    sliced = _sliced_wasserstein_fallback(pts_f, pts_w, n_proj=n_proj)
    if _POT_AVAILABLE:
        try:
            nf, nw  = len(pts_f), len(pts_w)
            a = np.ones(nf, dtype=np.float64) / nf
            b = np.ones(nw, dtype=np.float64) / nw
            M1  = cdist(pts_f, pts_w, metric="cityblock").astype(np.float64)
            w1  = float(_ot.emd2(a, b, M1))
            M2  = cdist(pts_f, pts_w, metric="sqeuclidean").astype(np.float64)
            w2  = float(np.sqrt(max(_ot.emd2(a, b, M2), 0.0)))
            return dict(
                wass_w1      = round(w1, 2),
                wass_w2      = round(w2, 2),
                wass_w1_norm = round(w1 / img_diag, 4),
                wass_w2_norm = round(w2 / img_diag, 4),
            )
        except Exception as e:  # pragma: no cover
            print(f"  [Wasserstein] POT failed ({e}) — using sliced fallback")
    return dict(
        wass_w1      = round(sliced, 2),
        wass_w2      = None,
        wass_w1_norm = round(sliced / img_diag, 4),
        wass_w2_norm = None,
    )


def ssd_vessel_distance(fv, X):
    H, W = fv.shape
    diff = fv.astype(np.float32) - X.astype(np.float32)
    ssd  = float(np.sum(diff ** 2))
    return dict(ssd=round(ssd, 2), ssd_mse=round(ssd / float(H * W), 6))


# ─────────────────────────────────────────────────────────────────────────────
# Image-intensity metrics (computed on greyscale images, not masks)
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(img):
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    return g.astype(np.float32) / 255.0


def _mutual_information(a, b, bins=64):
    if a.size < 16:
        return None, None
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pab = hist / (hist.sum() + 1e-12)
    pa  = pab.sum(axis=1)
    pb  = pab.sum(axis=0)
    nz  = pab > 0
    h_ab = -np.sum(pab[nz] * np.log2(pab[nz]))
    h_a  = -np.sum(pa[pa > 0]  * np.log2(pa[pa > 0]))
    h_b  = -np.sum(pb[pb > 0]  * np.log2(pb[pb > 0]))
    mi   = h_a + h_b - h_ab
    nmi  = (h_a + h_b) / (h_ab + 1e-12)
    return float(mi), float(nmi)


def image_intensity_metrics(fixed_rgb, moving_rgb, fov_mask=None, bins=64):
    fg = _to_gray(fixed_rgb)
    mg = _to_gray(moving_rgb)

    if fov_mask is not None:
        m = fov_mask > 0
    else:
        m = np.ones(fg.shape, dtype=bool)
    if m.sum() < 16:
        return dict(mi=None, nmi=None, ncc_img=None, ssim=None,
                    mse=None, rmse=None, psnr=None, intensity_fov_frac=0.0)

    fa = fg[m].ravel()
    ma = mg[m].ravel()

    mi, nmi = _mutual_information(fa, ma, bins=bins)

    fz = fa - fa.mean(); mz = ma - ma.mean()
    df = float(np.sqrt((fz * fz).sum())); dm = float(np.sqrt((mz * mz).sum()))
    ncc_img = float((fz * mz).sum() / (df * dm)) if df > 1e-12 and dm > 1e-12 else 0.0

    mse  = float(np.mean((fa - ma) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 1e-12 else None  # data range = 1.0

    if _SSIM_AVAILABLE:
        ssim = float(_sk_ssim(fg, mg, data_range=1.0))
    else:
        ssim = None

    return dict(
        mi                 = round(mi,   4) if mi   is not None else None,
        nmi                = round(nmi,  4) if nmi  is not None else None,
        ncc_img            = round(ncc_img, 4),
        ssim               = round(ssim, 4) if ssim is not None else None,
        mse                = round(mse,  6),
        rmse               = round(rmse, 6),
        psnr               = round(psnr, 3) if psnr is not None else None,
        intensity_fov_frac = round(float(m.mean()), 4),
    )
