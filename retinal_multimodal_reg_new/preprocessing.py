
"""irfaf_registration.preprocessing — image/mask preparation and density rules.

Intensity preprocessing (CLAHE + blur), FAF vessel closing, and vessel-density
quality tiering.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# Intensity preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_rgb, clip_limit=3.0, tile_grid=(8, 8),
                     blur_ksize=5, blur_sigma=1.2,
                     auto_brighten=True, luma_target=105.0):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)

    # Mildly lift underexposed images for more stable feature matching.
    if auto_brighten:
        fov = (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) > 10)
        vals = l_eq[fov] if np.any(fov) else l_eq.reshape(-1)
        cur = float(np.mean(vals))
        if cur < luma_target:
            gamma = np.clip(cur / (luma_target + 1e-8), 0.70, 1.0)
            gain = np.clip(luma_target / (cur + 1e-8), 1.0, 1.12)
            l_norm = l_eq.astype(np.float32) / 255.0
            l_eq = np.clip((np.power(l_norm, gamma) * gain) * 255.0,
                           0, 255).astype(np.uint8)

    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    return cv2.GaussianBlur(enhanced, (ksize, ksize), blur_sigma)


def close_faf_vessel(fv, ksize=None):
    if ksize is None:
        ksize = utils.FAF_CLOSE_KSIZE
    fv = np.asarray(fv, dtype=np.float32)
    if ksize is not None and int(ksize) <= 1:
        return fv.copy()
    b = (fv * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    return (cv2.morphologyEx(b, cv2.MORPH_CLOSE, k) > 0).astype(np.float32)


def vessel_density(v):
    return float(v.sum()) / (v.shape[0] * v.shape[1] + 1e-8)


def estimate_vessel_mask(img_rgb,
                         percentile=93.0,
                         min_pixels=32,
                         min_component_pixels=20,
                         min_vesselness=6.0):
    """Create a provisional vessel mask when no segmentation file exists.

    This compatibility helper mirrors the legacy API expected by the tests and
    the CLI wrappers. It extracts dark, thin structures after CLAHE and a
    morphological black-hat filter so the normal vessel-based pipeline can still
    be attempted when a proper vessel mask file is absent.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fov = utils.get_fov_mask(img_rgb) > 0
    if int(np.count_nonzero(fov)) < 64:
        return np.zeros(gray.shape, dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vesselness = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    values = vesselness[fov]
    threshold = float(np.percentile(values, percentile))
    threshold = max(threshold, float(min_vesselness))
    mask = ((vesselness >= threshold) & fov).astype(np.uint8)

    if int(np.count_nonzero(mask)) > 0 and int(min_component_pixels) > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_component_pixels):
                keep[labels == i] = 1
        mask = keep

    if int(np.count_nonzero(mask)) < int(min_pixels):
        return np.zeros(gray.shape, dtype=np.float32)
    return mask.astype(np.float32)


def estimate_optic_disc_center(img_rgb,
                               min_area_frac=0.0008,
                               max_area_frac=0.08,
                               min_circularity=0.35):
    """Estimate an optic-disc center from a retinal image.

    This matches the legacy API the rest of the codebase and tests expect.
    """
    if img_rgb is None:
        return None, 0.0
    img = np.asarray(img_rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        return None, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fov = utils.get_fov_mask(img) > 0
    if not np.any(fov):
        return None, 0.0

    vals = gray[fov]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fov_area = int(np.count_nonzero(fov))
    if fov_area <= 0:
        return None, 0.0

    lo, hi = np.percentile(vals, (0.5, 99.5))
    if float(hi - lo) < 8.0:
        return None, 0.0

    best_center, best_score = None, 0.0
    for is_bright, threshold in ((True, hi), (False, lo)):
        cand = ((gray >= threshold) if is_bright else (gray <= threshold))
        cand = (cand & fov).astype(np.uint8)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
        num, labels, stats, centers = cv2.connectedComponentsWithStats(
            cand, connectivity=8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            area_frac = area / float(fov_area)
            if area_frac < min_area_frac or area_frac > max_area_frac:
                continue
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            peri = float(cv2.arcLength(cnts[0], True))
            if peri <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (peri * peri + 1e-8))
            if circularity < min_circularity:
                continue
            contrast = abs(float(gray[labels == i].mean()) - float(np.median(vals)))
            score = circularity * contrast / (float(hi - lo) + 1e-8)
            if score > best_score:
                best_center = tuple(float(v) for v in centers[i])
                best_score = float(score)

    return best_center, best_score


def has_optic_disc_candidate(img_rgb, **kwargs):
    """Return whether :func:`estimate_optic_disc_center` found a candidate."""
    center, _ = estimate_optic_disc_center(img_rgb, **kwargs)
    return center is not None


def heuristic_params(fv, mv):
    """Compatibility params mirror the legacy preprocessing API."""
    H, W = fv.shape
    diag = float(np.sqrt(H ** 2 + W ** 2))
    fd = vessel_density(fv)
    md = vessel_density(mv)

    sparse_thresh = 0.01
    density_ratio = min(fd, md) / max(fd, md + 1e-8)
    sparse = (min(fd, md) < sparse_thresh) and (density_ratio < 0.30)

    if sparse:
        topo_weights = dict(w_cd=0.30, w_sd=0.25, w_bf=0.45)
    elif fd > 0.05 or md > 0.05:
        topo_weights = dict(w_cd=0.50, w_sd=0.40, w_bf=0.10)
    else:
        topo_weights = dict(w_cd=0.45, w_sd=0.35, w_bf=0.20)

    return dict(
        topo_weights=topo_weights,
        sparse=sparse,
        sparse_thresh=round(sparse_thresh, 6),
        density_ratio=round(density_ratio, 4),
        diag=round(diag, 4),
        fd=round(fd, 6),
        md=round(md, 6),
    )


def _vessel_component_stats(v,
                            min_component_pixels=24,
                            min_largest_component_pixels=48):
    """Summarize whether a vessel mask contains meaningful structure.

    Tiny isolated dots can have non-zero pixels yet still be unusable for
    registration. This helper measures connected-component size so callers can
    reject dot-only masks.
    """
    b = (np.asarray(v) > 0).astype(np.uint8)
    px = int(np.count_nonzero(b))
    if px == 0:
        return dict(px=0, largest=0, n_components=0, n_large=0, dot_like=True)

    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(b, connectivity=8)
    if num <= 1:
        return dict(px=px, largest=0, n_components=0, n_large=0, dot_like=True)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    largest = int(areas.max())
    n_components = int(len(areas))
    n_large = int(np.sum(areas >= int(min_component_pixels)))
    dot_like = (largest < int(min_largest_component_pixels)) or (n_large == 0)
    return dict(
        px=px,
        largest=largest,
        n_components=n_components,
        n_large=n_large,
        dot_like=bool(dot_like),
    )


def calibrate_quality_thresholds(density_values, success_labels):
    """Fit ``bad``/``good`` vessel-density cutoffs from labeled outcomes,
    instead of hand-picking them.

    ``density_values`` is the per-pair ``min(fixed_density, moving_density)``
    (what :func:`classify_pair_quality` calls ``lo``) and ``success_labels``
    is the matching boolean array of whether that pair's registration
    actually succeeded (e.g. landmark MLE <= T_SR). Each cutoff is chosen by
    Youden's J (sensitivity + specificity - 1) over every candidate density
    value in the data: the ``bad`` cutoff is the threshold that best
    separates *all* pairs into fail/succeed; the ``good`` cutoff repeats the
    same sweep restricted to the pairs already above the ``bad`` cutoff, to
    find where "reliably succeeds" starts.

    This ties the tiers to what they're actually meant to predict
    (registration success) rather than to a density number nobody can
    justify -- but it needs a reasonable number of labeled pairs (dozens,
    not a handful) to not be noise, and should be refit whenever the
    dataset changes meaningfully rather than treated as permanent.

    Returns ``(bad_vessel_density, good_vessel_density)``.
    """
    d = np.asarray(density_values, dtype=np.float64)
    s = np.asarray(success_labels, dtype=bool)
    if d.size == 0 or d.size != s.size:
        raise ValueError("density_values and success_labels must be equal-length and non-empty")

    def _best_threshold(dd, ss):
        best_j, best_t = -1.0, float(np.median(dd))
        for t in np.unique(dd):
            pred_bad = dd < t
            tp = int(np.sum(pred_bad & ~ss))
            fn = int(np.sum(~pred_bad & ~ss))
            tn = int(np.sum(~pred_bad & ss))
            fp = int(np.sum(pred_bad & ss))
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            j = sens + spec - 1.0
            if j > best_j:
                best_j, best_t = j, float(t)
        return best_t, best_j

    bad_t, _ = _best_threshold(d, s)
    not_bad = d >= bad_t
    if int(np.sum(not_bad)) >= 2 and len(set(s[not_bad].tolist())) > 1:
        good_t, _ = _best_threshold(d[not_bad], s[not_bad])
    else:
        good_t = float(np.percentile(d, 75))
    return round(bad_t, 4), round(max(good_t, bad_t), 4)


def auto_detect_quality_thresholds(density_values, n_bins=256):
    """Unsupervised ``bad``/``good`` vessel-density cutoffs via multi-level
    Otsu thresholding -- no success/failure labels needed at all.

    Finds the two thresholds that split ``density_values`` into 3 groups by
    maximizing between-class variance (equivalently minimizing within-class
    variance), the standard generalization of Otsu's binarization threshold
    to 3 classes. This is purely a statement about the *shape* of the
    density distribution (where its natural clusters/gaps fall) -- unlike
    :func:`calibrate_quality_thresholds`, it has no way to know whether
    those clusters actually correspond to "registration succeeds here."

    Use this when you don't have (or don't trust) labeled outcomes yet; use
    the outcome-calibrated thresholds instead once you do, since those are
    fit to what the tiers are actually supposed to predict.

    Returns ``(bad_vessel_density, good_vessel_density)``.
    """
    d = np.asarray(density_values, dtype=np.float64)
    if d.size < 3:
        raise ValueError("need at least 3 density values")

    lo, hi = float(d.min()), float(d.max())
    if hi <= lo:
        return round(lo, 4), round(lo, 4)

    # Histogram once; work on bin centers/counts rather than raw values so
    # the O(n_bins^2) search over threshold pairs stays cheap regardless of
    # how many pairs are in the dataset.
    counts, edges = np.histogram(d, bins=n_bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total_w = counts.sum()
    total_mean = float((counts * centers).sum() / total_w)

    best_var, best_pair = -1.0, (int(n_bins / 3), int(2 * n_bins / 3))
    cum_w = np.cumsum(counts)
    cum_wm = np.cumsum(counts * centers)
    for i in range(1, n_bins - 1):
        w0 = cum_w[i - 1]
        if w0 == 0:
            continue
        for j in range(i + 1, n_bins):
            w1 = cum_w[j - 1] - cum_w[i - 1]
            w2 = total_w - cum_w[j - 1]
            if w1 == 0 or w2 == 0:
                continue
            m0 = cum_wm[i - 1] / w0
            m1 = (cum_wm[j - 1] - cum_wm[i - 1]) / w1
            m2 = (cum_wm[-1] - cum_wm[j - 1]) / w2
            between_var = (
                w0 * (m0 - total_mean) ** 2
                + w1 * (m1 - total_mean) ** 2
                + w2 * (m2 - total_mean) ** 2
            )
            if between_var > best_var:
                best_var, best_pair = between_var, (i, j)

    i, j = best_pair
    return round(float(centers[i]), 4), round(float(centers[j]), 4)


def classify_pair_quality(fv, mv, bad_vessel_density, good_vessel_density,
                          fi=None, mi=None,
                          min_usable_vessel_pixels=24,
                          min_component_pixels=24,
                          min_largest_component_pixels=48):
    """Vessel-density tier for a pair, applied *before* registration is
    attempted (fv/mv are the vessel masks). Three tiers, driven by the density
    of the sparser of the two masks:

    - ``"good"``     — density >= ``good_vessel_density``: dense retinal
      vessel network, suitable for reliable vessel-based registration.
    - ``"moderate"`` — ``bad_vessel_density`` <= density < ``good_vessel_density``:
      sparse retinal vessel network; registration may be less reliable.
    - ``"bad"``      — density < ``bad_vessel_density``, or a mask that is
      effectively empty / dot-like (no coherent vessel tree at all):
      insufficient vessel information for registration to be expected to
      succeed.

    ``bad_vessel_density``/``good_vessel_density`` are required, deliberately
    -- there is no built-in fallback number. Fit them per dataset with
    :func:`calibrate_quality_thresholds` (needs labeled success/failure
    outcomes -- e.g. landmark MLE <= T_SR) or, lacking labels,
    :func:`auto_detect_quality_thresholds` (unsupervised multi-Otsu, weaker
    since it only reflects the density distribution's own shape, not
    whether that shape predicts anything). A single hardcoded pair of
    numbers doesn't transfer across datasets, which is why this function no
    longer ships with one.

    Optic-disc presence is not considered here — it was too unreliable a
    signal and was wrongly suppressing good vessel-based registrations for
    pairs that had perfectly usable vessel masks.
    """
    fv_stats = _vessel_component_stats(
        fv,
        min_component_pixels=min_component_pixels,
        min_largest_component_pixels=min_largest_component_pixels,
    )
    mv_stats = _vessel_component_stats(
        mv,
        min_component_pixels=min_component_pixels,
        min_largest_component_pixels=min_largest_component_pixels,
    )
    fv_pixels = int(fv_stats["px"])
    mv_pixels = int(mv_stats["px"])
    fd = float(vessel_density(fv))
    md = float(vessel_density(mv))
    lo = min(fd, md)
    reasons = []

    dot_like = bool(fv_stats["dot_like"] or mv_stats["dot_like"])
    if (min(fv_pixels, mv_pixels) < int(min_usable_vessel_pixels)) or dot_like:
        label = "bad"
        reasons.append(
            f"no_vessels(fv_px={fv_pixels},mv_px={mv_pixels},"
            f"fv_largest={int(fv_stats['largest'])},mv_largest={int(mv_stats['largest'])},"
            f"fv={fd:.4f},mv={md:.4f})"
        )
    elif lo < bad_vessel_density:
        label = "bad"
        reasons.append(f"low_density(fv={fd:.4f},mv={md:.4f},thresh={bad_vessel_density})")
    elif lo < good_vessel_density:
        label = "moderate"
        reasons.append(f"sparse_vessels(fv={fd:.4f},mv={md:.4f})")
    else:
        label = "good"
    return label, reasons
