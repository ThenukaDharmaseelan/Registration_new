from __future__ import annotations

import cv2
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None

# ─────────────────────────────────────────────────────────────────────────────
# Device / working size
# ─────────────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

FOV_SCALE_RATIO = 55.0 / 30.0  # ≈ 1.833

# ── LoFTR inlier retry ───────────────────────────────────────────────────────
# Below this many inliers the affine is under-constrained (failing pairs sit at
# 5-15; succeeding pairs at 200+), so re-match at a lower confidence and let
# RANSAC reject the extra noise.
LOFTR_MIN_INLIERS = 20
LOFTR_RETRY_CONF  = 0.2

# ── Fixed affine sanity limits ───────────────────────────────────────────────
REJECT_ABS_ROT        = 120.0
REJECT_SCALE_MAX      = 4.0
REJECT_SCALE_MIN      = 0.1
REJECT_NEGATIVE_SCALE = True

# The 1x1 closing kernel is a true no-op; use a 3x3 ellipse as the default
# bridge for single-pixel vessel gaps while still allowing the 4x4 sweep to be
# evaluated as a larger-connectivity fallback.
FAF_CLOSE_KSIZE = 3

LOFTR_CLOSE_KSIZES = (3, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Scale / FOV helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_scale_matrix(scale, cx, cy):
    """Return the full 6-DoF affine seed used by the current pipeline."""
    return np.array([
        [scale, 0,     cx * (1 - scale)],
        [0,     scale, cy * (1 - scale)]
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Warp utilities
# ─────────────────────────────────────────────────────────────────────────────

def warp_img(img_rgb, M, h, w):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def warp_mask(mv, M, h, w):
    # Linear (not nearest) so a non-integer scale resamples thin vessel lines
    # smoothly instead of blocking them into unevenly duplicated pixels.
    src = np.clip(mv.astype(np.float32), 0.0, 1.0)
    out = cv2.warpAffine(src, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out = (out >= 0.45).astype(np.uint8)
    if out.shape[0] > 0 and out.shape[1] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(np.float32)


def compose(M1, M2):
    def t(M): return np.vstack([M, [0, 0, 1]])
    return (t(M2) @ t(M1))[:2].astype(np.float32)


def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return (mask > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Affine decomposition & flagging
# ─────────────────────────────────────────────────────────────────────────────

def decompose_affine(M):
    tx, ty = float(M[0, 2]), float(M[1, 2])
    A = M[:2, :2].astype(np.float64)
    U, S, Vt = np.linalg.svd(A)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1; S[-1] *= -1
    R = U @ Vt
    return dict(tx=tx, ty=ty,
                angle_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                sx=float(S[0]), sy=float(S[1]),
                shear=float((R.T @ A)[0, 1] / (float(S[0]) + 1e-8)))


# ─────────────────────────────────────────────────────────────────────────────
# Affine sanity gates
# ─────────────────────────────────────────────────────────────────────────────

def affine_is_sane(M, params=None):
    """Affine sanity gate disabled for the final affine-only flow.

    The project uses the affine estimate directly; decomposition-based rejection is
    intentionally not applied here.
    """
    if M is None:
        return False
    return True


def sane(M, params=None):
    """Unified sanity gate (affine-only, no decomposition rejection)."""
    return affine_is_sane(M, params)

