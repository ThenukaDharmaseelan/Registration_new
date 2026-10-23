
from __future__ import annotations

import cv2
import numpy as np

from . import utils
from .preprocessing import close_faf_vessel



# ─────────────────────────────────────────────────────────────────────────────
# Intensity + mismatch views
# ─────────────────────────────────────────────────────────────────────────────

def enhance_display_image(img_rgb, clip_limit=4.0, tile_grid=(8, 8), gamma=0.85,
                          gain=1.0):
    img8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    fov = utils.get_fov_mask(img8) > 0
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)

    vals = l_eq[fov] if np.any(fov) else l_eq.reshape(-1)
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.5))
    if hi <= lo:
        l_norm = l_eq.astype(np.float32) / 255.0
    else:
        l_norm = np.clip((l_eq.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    l_gamma = np.power(l_norm, gamma)
    l_out = np.clip(l_gamma * 255.0 * gain, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2RGB)
    enhanced[~fov] = 0
    return enhanced


def enhance_faf_display(img_rgb):
    # Slightly stronger lift for FAF-family views so dim scans remain readable.
    return enhance_display_image(img_rgb, clip_limit=5.5, tile_grid=(8, 8),
                                 gamma=0.75, gain=1.10)

def make_overlap_image(fi, wi):
    fov = utils.get_fov_mask(fi)

    def n(x):
        g = cv2.cvtColor(enhance_display_image(x), cv2.COLOR_RGB2GRAY).astype(np.float32)
        vals = g[fov > 0] if np.any(fov > 0) else g.reshape(-1)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        return np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    f, w = n(fi), n(wi)
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = f
    out[..., 1] = w
    out[..., 2] = f
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)

def make_overlap_image_redblue(fi, wi):
    """Raw-intensity overlay, red/blue scheme: fixed FAF = RED, warped IR = BLUE.
    Overlap (well-aligned) regions read as purple/magenta; misaligned structure
    shows as pure red or pure blue fringes — often easier to read than the
    magenta/green scheme since R and B don't share a channel."""
    fov = utils.get_fov_mask(fi)
    def n(x):
        g = cv2.cvtColor(enhance_display_image(x), cv2.COLOR_RGB2GRAY).astype(np.float32)
        vals = g[fov > 0] if np.any(fov > 0) else g.reshape(-1)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        return np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    f, w = n(fi), n(wi)
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = f   # R = fixed
    out[..., 1] = 0.0
    out[..., 2] = w   # B = warped
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)


def make_checkerboard(fi, wi, tiles=8):
    fov = utils.get_fov_mask(fi)
    H, W = fi.shape[:2]
    f = cv2.cvtColor(enhance_display_image(fi, clip_limit=5.0, tile_grid=(6, 6)), cv2.COLOR_RGB2GRAY)
    w = cv2.cvtColor(enhance_display_image(wi, clip_limit=5.0, tile_grid=(6, 6)), cv2.COLOR_RGB2GRAY)
    ty, tx = max(H // tiles, 1), max(W // tiles, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    board  = (((yy // ty) + (xx // tx)) % 2).astype(np.uint8)
    out = np.where(board > 0, f, w).astype(np.float32) * fov
    return np.stack([out.astype(np.uint8)] * 3, -1)


def make_vessel_mismatch(fv, wv, fi, tol=1):
    fov = utils.get_fov_mask(fi)
    f = (fv > 0).astype(np.float32)
    w = (wv > 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    f_d = (cv2.dilate(f, k) > 0).astype(np.float32)     # fixed vessels, tol-fattened
    w_d = (cv2.dilate(w, k) > 0).astype(np.float32)     # warped vessels, tol-fattened
    only_f  = f * (1.0 - w_d)                            # fixed with no warped nearby
    only_w  = w * (1.0 - f_d)                            # warped with no fixed nearby
    matched = np.clip(f * w_d + w * f_d, 0, 1)          # vessels with a partner
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = only_f + 0.30 * matched               # R
    out[..., 1] = only_w + 0.30 * matched               # G
    out[..., 2] = only_f + 0.30 * matched               # B (only_f => magenta)
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)


def make_overlap_vessels(fv, wv, fi, tol_px=1):
    """Create an RGB overlay of warped (R) and fixed (G) vessel masks.

    A small tolerance makes this overlay less sensitive to 1-pixel edge
    differences caused by resampling.
    """
    fov = utils.get_fov_mask(fi)
    H, W = fv.shape
    out = np.zeros((H, W, 3), np.uint8)

    f = (fv > 0).astype(np.uint8)
    w = (wv > 0).astype(np.uint8)
    if int(tol_px) > 0:
        r = int(tol_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        f_d = (cv2.dilate(f, k) > 0).astype(np.uint8)
        w_d = (cv2.dilate(w, k) > 0).astype(np.uint8)
        matched = ((f & w_d) | (w & f_d)).astype(np.uint8)
    else:
        matched = (f & w).astype(np.uint8)

    only_w = (w & (1 - matched)).astype(np.uint8)
    only_f = (f & (1 - matched)).astype(np.uint8)
    out[..., 0] = ((only_w | matched) * 255).astype(np.uint8)
    out[..., 1] = ((only_f | matched) * 255).astype(np.uint8)
    return (out * fov[..., None]).astype(np.uint8)


def make_scaled_overlap_vessels(fv, mv, M_scale, fi):
    H, W = fv.shape; fov = utils.get_fov_mask(fi)
    mv_sc = utils.warp_mask(mv, M_scale, H, W)
    out = np.zeros((H, W, 3), np.uint8)
    out[..., 0] = (mv_sc * 255).astype(np.uint8)
    out[..., 1] = (fv * 255).astype(np.uint8)
    return (out * fov[..., None]).astype(np.uint8)


def mask_to_rgb(m):
    g = (m * 255).astype(np.uint8); return np.stack([g, g, g], -1)


def vessel_on_image(base_rgb, vessel_mask, color=(0, 255, 0), alpha=0.85):
    base = np.clip(base_rgb, 0, 255).astype(np.uint8)
    out = base.copy()
    m = vessel_mask > 0
    if np.any(m):
        c = np.array(color, dtype=np.float32)
        blended = (1.0 - alpha) * out[m].astype(np.float32) + alpha * c
        out[m] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def add_label(img, t, fs=0.40):
    out = img.copy()
    for c, th in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
        cv2.putText(out, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    return out


def make_keypoint_match_image(fi, mi_sc, src, dst, inlier_mask=None, max_matches=300,
                              kp_fixed=None, kp_moving_sc=None):
    """Side-by-side fixed/moving images with lines joining LoFTR keypoints.

    ``dst`` points are in ``fi``'s frame; ``src`` points are in ``mi_sc``'s
    frame (the moving image already warped by the candidate's scale matrix
    onto the same canvas). Inliers (from the affine RANSAC fit) are drawn in
    green. Outliers are omitted when ``inlier_mask`` is given.

    ``kp_fixed``/``kp_moving_sc`` are the raw annotation landmarks (moving
    side already in ``mi_sc``'s frame) that were blended into the LoFTR
    feature maps as Gaussian peaks. When given, they're drawn as yellow
    dots -- distinct from the green LoFTR match lines -- so it's visible
    that annotation keypoints (not just the vessel masks) fed into this
    match, rather than having to infer it from the match count alone.
    """
    H, W = fi.shape[:2]
    left = enhance_display_image(fi)
    right = enhance_faf_display(mi_sc)
    gap = 2
    canvas = np.concatenate(
        [left, np.full((H, gap, 3), 40, np.uint8), right], axis=1)
    n = len(dst)
    if inlier_mask is not None:
        inl = np.asarray(inlier_mask).reshape(-1).astype(bool)
    else:
        inl = np.ones(n, dtype=bool)
    idx = np.arange(n)
    if n > max_matches:
        idx = np.random.RandomState(0).choice(n, size=max_matches, replace=False)
    x_off = W + gap
    for i in idx:
        if not inl[i]:
            continue
        p0 = (int(round(float(dst[i][0]))), int(round(float(dst[i][1]))))
        p1 = (int(round(float(src[i][0]))) + x_off, int(round(float(src[i][1]))))
        color = (60, 220, 60)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    n_inl = int(inl.sum())

    n_kp = 0
    if kp_fixed is not None and kp_moving_sc is not None:
        kp_fixed = np.asarray(kp_fixed, dtype=np.float32)
        kp_moving_sc = np.asarray(kp_moving_sc, dtype=np.float32)
        n_kp = min(len(kp_fixed), len(kp_moving_sc))
        for x, y in kp_fixed[:n_kp]:
            cv2.circle(canvas, (int(round(float(x))), int(round(float(y)))),
                      4, (255, 220, 0), -1, cv2.LINE_AA)
        for x, y in kp_moving_sc[:n_kp]:
            cv2.circle(canvas, (int(round(float(x))) + x_off, int(round(float(y)))),
                      4, (255, 220, 0), -1, cv2.LINE_AA)

    label = f"KeyPoint Matches: {n_inl}/{n} inliers"
    if n_kp:
        label += f"  |  annotation landmarks: {n_kp} (yellow)"
    return add_label(canvas, label, fs=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair grid
# ─────────────────────────────────────────────────────────────────────────────

def make_row_grid(fi, fv, mi, mv, wi, wv, M_scale,
                  row_label=None,
                  decomp=None, stage=None, best_scale=None,
                  scale_cluster=None, params=None):
    H, W2 = fi.shape[:2]; GAP = 5
    gv = np.ones((H, GAP, 3), np.uint8) * 60
    mi_sc = utils.warp_img(mi, M_scale, H, W2); mv_sc = utils.warp_mask(mv, M_scale, H, W2)
    fv_closed = close_faf_vessel(fv)
    fi_disp = enhance_display_image(fi)
    mi_disp = enhance_faf_display(mi)
    mi_sc_disp = enhance_faf_display(mi_sc)
    wi_disp = enhance_faf_display(wi)

    def vessel_or_image(base, mask, min_px=100):
        return mask_to_rgb(mask) if int((mask > 0).sum()) >= min_px else base

    panels = [
        add_label(fi_disp,               "IR Fixed"),
        add_label(vessel_or_image(fi_disp, fv), "IR Vessel"),
        add_label(vessel_or_image(fi_disp, fv_closed), "IR Closed"),
        add_label(mi_disp,               "FAF Moving"),
        add_label(vessel_or_image(mi_disp, mv), "FAF Vessel"),
        add_label(mi_sc_disp,            "FAF Scaled"),
        add_label(vessel_or_image(mi_sc_disp, mv_sc), "FAF Vessel Scaled"),
        add_label(make_scaled_overlap_vessels(fv, mv, M_scale, fi), "Scale Overlap"),
        add_label(wi_disp,               "Registered"),
        add_label(make_overlap_image(fi, wi), "IR+Reg Overlay"),
        add_label(make_checkerboard(fi, wi), "Checkerboard"),
        add_label(make_overlap_vessels(fv, wv, fi), "Vessel Overlap"),
    ]
    row = np.concatenate([x for p in panels for x in [p, gv]][:-1], axis=1)
    SB = 300; sb = np.full((H, SB, 3), (30, 30, 30), np.uint8)
    lines = []
    if row_label     is not None: lines.append(row_label)
    if stage         is not None: lines.append(f"Stage:{stage[:22]}")
    if best_scale    is not None: lines.append(f"BestScale:{best_scale:.2f}x")
    if scale_cluster is not None: lines.append(f"ScaleCluster:{scale_cluster}")
    if decomp is not None:
        lines += ["---",
                  f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}",
                  f"rot:{decomp['angle_deg']:+.2f}deg",
                  f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}"]
    if params is not None:
        lines.append("---")
        st = params.get("sparse_thresh")
        line = f"sparse:{params.get('sparse')}"
        if st is not None:
            line += f"(t={st:.5f})"
        lines.append(line)
    fc = (200, 200, 200)
    for li, line in enumerate(lines):
        y = 14 + li * 13
        if y > H - 4:
            break
        col = (60, 200, 255) if line.startswith("!") else fc
        cv2.putText(sb, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, col, 1, cv2.LINE_AA)
    grid = np.concatenate([sb, np.ones((H, GAP, 3), np.uint8) * 60, row], axis=1)
    return grid


def make_failure_grid(fi, fv, mi, mv, row_label=None, stage=None, flags=None):
    """Compact 4-panel grid for a pair that failed before registration was
    attempted: IR image, FAF image, IR vessel, FAF vessel. A vessel panel
    whose mask is missing (all-zero) renders fully black."""
    H, W = fi.shape[:2]
    GAP = 5
    gv = np.ones((H, GAP, 3), np.uint8) * 60
    panels = [
        add_label(enhance_display_image(fi), "IR"),
        add_label(enhance_faf_display(mi),   "FAF"),
        add_label(mask_to_rgb(fv),           "IR Vessel"),
        add_label(mask_to_rgb(mv),           "FAF Vessel"),
    ]
    row = np.concatenate([x for p in panels for x in [p, gv]][:-1], axis=1)
    SB = 220; sb = np.full((H, SB, 3), (30, 30, 30), np.uint8)
    lines = []
    if row_label is not None: lines.append(row_label)
    if stage     is not None: lines.append(f"Stage:{stage[:22]}")
    if flags:
        lines.append("WARN:")
        for f in flags:
            lines.append(f"  {f[:30]}")
    fc = (80, 80, 255) if flags else (200, 200, 200)
    for li, line in enumerate(lines):
        y = 14 + li * 13
        if y > H - 4:
            break
        col = fc if (line.startswith("WARN") or line.startswith("  ")) else (200, 200, 200)
        cv2.putText(sb, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, col, 1, cv2.LINE_AA)
    grid = np.concatenate([sb, np.ones((H, GAP, 3), np.uint8) * 60, row], axis=1)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Disk output
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(od, name, fi, fv, mi, mv, wi, wv, M_scale,
                 pair_index=None,
                 decomp=None, stage=None, best_scale=None,
                 scale_cluster=None, params=None, match_meta=None):
    H, W2 = fi.shape[:2]

    def wr(cat, img):
        d = od / cat; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def wm(cat, m):
        d = od / cat; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), (m * 255).astype(np.uint8))

    def wgray(cat, g):
        d = od / cat; d.mkdir(parents=True, exist_ok=True)
        gray = np.asarray(g, dtype=np.float32)
        if gray.ndim == 2:
            img = np.stack([gray, gray, gray], axis=-1)
        else:
            img = gray
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(np.stack([img]*3,-1), cv2.COLOR_RGB2BGR))

    mi_sc = utils.warp_img(mi, M_scale, H, W2); mv_sc = utils.warp_mask(mv, M_scale, H, W2)
    wr("images/fixed",  enhance_display_image(fi))
    wr("images/moving", enhance_faf_display(mi))
    wr("registered",    enhance_faf_display(wi))
    wm("registered_vessel",     wv)
    wr("ir_scaled",             enhance_faf_display(mi_sc))
    wm("ir_vessel_scaled",      mv_sc)
    wr("overlap_image",           make_overlap_image(fi, wi))
    wr("overlap_image_redblue",   make_overlap_image_redblue(fi, wi))
    wr("overlap_mismatch",        make_vessel_mismatch(fv, wv, fi))
    wr("overlap_checker",         make_checkerboard(fi, wi))
    wr("overlap_vessels",       make_overlap_vessels(fv, wv, fi))
    wr("overlap_scaled_vessels",        make_scaled_overlap_vessels(fv, mv, M_scale, fi))

    if match_meta and match_meta.get("src") is not None:
        wr("keypoint_matches", make_keypoint_match_image(
            match_meta["fi"], match_meta["mi_sc"],
            match_meta["src"], match_meta["dst"],
            inlier_mask=match_meta.get("inlier_mask"),
            kp_fixed=match_meta.get("kp_fixed"),
            kp_moving_sc=match_meta.get("kp_moving_sc")))

    row = make_row_grid(fi, fv, mi, mv, wi, wv, M_scale,
                        row_label=f"#{pair_index}" if pair_index else None,
                        decomp=decomp, stage=stage, best_scale=best_scale,
                        scale_cluster=scale_cluster, params=params)
    wr("grid", row)
    print(f"  Saved: {name}.png  (categories)")
    return row
