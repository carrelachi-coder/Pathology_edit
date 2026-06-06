"""Preview a warped reference: reference appearance rearranged into the
 target layout via same-class mask correspondence.

This script keeps only the original mean / soft / hard pixel-level baselines.

The HARD mode is the intended warped_ref candidate: it performs same-class
nearest-neighbor pixel copy from the reference into the target layout. Patch
warp has been intentionally removed because it can introduce quilting artifacts.

Important design choice
-----------------------
Matching features are mask-derived only. Do NOT put VGG/UNI into the matching
feature for this zero-training preview: target inference has only masks, while
reference has RGB. Deep RGB features on only one side would reintroduce a domain
mismatch. VGG/UNI are better used later as payloads to warp or as evaluation
features, not as first-pass matching features.

Usage
-----
Real data:
  python warp_preview_hard.py \
      --ref-image ref.png --ref-tissue ref_tissue.png --ref-nuclei ref_nuclei.png \
      --tar-image tar.png --tar-tissue tar_tissue.png --tar-nuclei tar_nuclei.png \
      --out warp_preview.png --corr-size 256 --gate tissue

Synthetic demo:
  python warp_preview_hard.py --demo --out warp_preview.png

Recommended real-data first try:
  python warp_preview_hard.py \
      --ref-image ref.png --ref-tissue ref_tissue.png --ref-nuclei ref_nuclei.png \
      --tar-tissue tar_tissue.png --tar-nuclei tar_nuclei.png \
      --out warp_preview.png \
      --corr-size 256 --baseline-gate tissue_nucbin --smooth 3 --no-soft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as cc_label,
    median_filter,
)


# ---------------------------------------------------------------------------
# IO / resize helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _load_label(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:  # color-coded mask -> take channel 0; adjust if yours differs
        arr = arr[..., 0]
    return arr.astype(np.int64)


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(_to_uint8(arr)).resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.float32) / 255.0


def _resize_label(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.int32), mode="I").resize((size, size), Image.NEAREST)
    return np.asarray(img).astype(np.int64)


def _save_rgb(path: str | Path, arr: np.ndarray) -> None:
    Image.fromarray(_to_uint8(np.clip(arr, 0, 1))).save(path)


def _save_gray(path: str | Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, np.float32)
    arr = arr / (float(arr.max()) + 1e-6)
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).save(path)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (float(x.max()) + 1e-6)


def _standardize_features(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Channelwise robust-ish standardization for one image's descriptors."""
    feat = feat.astype(np.float32)
    flat = feat.reshape(-1, feat.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    return ((flat - mean) / (std + eps)).reshape(feat.shape).astype(np.float32)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Mask descriptors and class gates
# ---------------------------------------------------------------------------

def _remap_tissue_pair(ref_t: np.ndarray, tar_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([ref_t.reshape(-1), tar_t.reshape(-1)]))
    mapping = {int(c): i for i, c in enumerate(classes)}
    ref = np.vectorize(mapping.get)(ref_t).astype(np.int64)
    tar = np.vectorize(mapping.get)(tar_t).astype(np.int64)
    return ref, tar, classes.astype(np.int64)


def _make_gate_labels(tissue: np.ndarray, nuclei: np.ndarray, *, gate: str, nuc_classes: int) -> np.ndarray:
    """Labels used only for hard gating reference candidates.

    gate='tissue' is usually the best first choice for pathology warp because it
    avoids over-fragmenting the match by exact nucleus type/pixel label.
    """
    if gate == "tissue":
        return tissue.astype(np.int64)
    if gate == "joint":
        return (tissue.astype(np.int64) * int(nuc_classes) + nuclei.astype(np.int64)).astype(np.int64)
    if gate == "tissue_nucbin":
        return (tissue.astype(np.int64) * 2 + (nuclei > 0).astype(np.int64)).astype(np.int64)
    raise ValueError(f"unknown gate={gate!r}")


def _boundary_map(label: np.ndarray) -> np.ndarray:
    b = np.zeros(label.shape, bool)
    b[:-1, :] |= label[:-1, :] != label[1:, :]
    b[1:, :] |= label[:-1, :] != label[1:, :]
    b[:, :-1] |= label[:, :-1] != label[:, 1:]
    b[:, 1:] |= label[:, :-1] != label[:, 1:]
    return b.astype(np.float32)


def _component_local_geometry_features(
    tissue: np.ndarray,
    nuclei: np.ndarray,
    *,
    tissue_classes: np.ndarray,
    density_sigma: float = 3.0,
) -> np.ndarray:
    """Build a mask-only descriptor for matching target tokens to reference tokens.

    The descriptor mixes:
      - component-local x/y coordinates within each tissue connected component,
      - relative position to component centroid,
      - distance inside tissue component and distance to centroid,
      - local nuclei binary/density/distance,
      - tissue boundary distance/orientation cues.

    It intentionally does not include RGB/VGG/UNI, so target and reference remain
    in a shared mask-derived space.
    """
    H, W = tissue.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    local_x = np.zeros((H, W), np.float32)
    local_y = np.zeros((H, W), np.float32)
    rel_cx = np.zeros((H, W), np.float32)
    rel_cy = np.zeros((H, W), np.float32)
    dist_in = np.zeros((H, W), np.float32)
    dist_centroid = np.zeros((H, W), np.float32)
    comp_scale = np.zeros((H, W), np.float32)

    for c in tissue_classes:
        mask_c = tissue == int(c)
        if not mask_c.any():
            continue
        labs, n_lab = cc_label(mask_c)
        for lab_id in range(1, n_lab + 1):
            m = labs == lab_id
            if not m.any():
                continue
            ys, xs = np.where(m)
            y0, y1 = float(ys.min()), float(ys.max())
            x0, x1 = float(xs.min()), float(xs.max())
            h_span = max(y1 - y0, 1.0)
            w_span = max(x1 - x0, 1.0)
            cy, cx = float(ys.mean()), float(xs.mean())
            scale = max(np.sqrt(float(m.sum())), 1.0)

            local_x[m] = ((xx[m] - x0) / w_span) * 2.0 - 1.0
            local_y[m] = ((yy[m] - y0) / h_span) * 2.0 - 1.0
            rel_cx[m] = (xx[m] - cx) / scale
            rel_cy[m] = (yy[m] - cy) / scale
            d = distance_transform_edt(m)
            dist_in[m] = d[m] / (float(d.max()) + 1e-6)
            dist_centroid[m] = np.sqrt((xx[m] - cx) ** 2 + (yy[m] - cy) ** 2) / scale
            comp_scale[m] = np.log1p(scale) / np.log1p(max(H, W))

    nuc_bin = (nuclei > 0).astype(np.float32)
    nuc_density = gaussian_filter(nuc_bin, sigma=float(density_sigma))
    nuc_density = _norm01(nuc_density)
    if nuc_bin.any() and (~nuc_bin.astype(bool)).any():
        # distance from each point to nearest nucleus; invert-ish by including raw normed distance
        nuc_dist = _norm01(distance_transform_edt(~nuc_bin.astype(bool)))
    else:
        nuc_dist = np.zeros((H, W), np.float32)

    tissue_boundary = _boundary_map(tissue)
    boundary_density = gaussian_filter(tissue_boundary, sigma=1.0)
    boundary_density = _norm01(boundary_density)
    boundary_dist = _norm01(distance_transform_edt(~tissue_boundary.astype(bool))) if (~tissue_boundary.astype(bool)).any() else np.zeros((H, W), np.float32)
    # Low-weight global coordinates help break perfect symmetries in large flat regions.
    gx = (xx / max(W - 1, 1)) * 2.0 - 1.0
    gy = (yy / max(H - 1, 1)) * 2.0 - 1.0

    feat = np.stack(
        [
            local_x,
            local_y,
            rel_cx,
            rel_cy,
            dist_in,
            dist_centroid,
            comp_scale,
            nuc_bin,
            nuc_density,
            nuc_dist,
            boundary_density,
            boundary_dist,
            0.25 * gx,
            0.25 * gy,
        ],
        axis=-1,
    ).astype(np.float32)
    return _standardize_features(feat)


# ---------------------------------------------------------------------------
# Baseline pixel warps
# ---------------------------------------------------------------------------

def compute_warp(
    ref_rgb: np.ndarray,
    ref_tissue: np.ndarray,
    ref_nuclei: np.ndarray,
    tar_tissue: np.ndarray,
    tar_nuclei: np.ndarray,
    *,
    mode: str = "hard",
    corr_size: int = 192,
    tau: float = 0.02,
    max_ref: int = 4096,
    chunk: int = 2048,
    smooth: int = 3,
    gate: str = "joint",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (warped_rgb [corr_size,corr_size,3], validity [corr_size,corr_size])."""
    if mode not in {"mean", "soft", "hard"}:
        raise ValueError(f"mode must be one of mean/soft/hard, got {mode!r}")
    rng = np.random.default_rng(seed)
    h = w = int(corr_size)

    ref_rgb_s = _resize_rgb(ref_rgb, h)
    ref_t0 = _resize_label(ref_tissue, h)
    ref_n = _resize_label(ref_nuclei, h)
    tar_t0 = _resize_label(tar_tissue, h)
    tar_n = _resize_label(tar_nuclei, h)
    ref_t, tar_t, tissue_classes = _remap_tissue_pair(ref_t0, tar_t0)

    nuc_classes = int(max(ref_n.max(), tar_n.max())) + 1
    tar_gate = _make_gate_labels(tar_t, tar_n, gate=gate, nuc_classes=nuc_classes).reshape(-1)
    ref_gate = _make_gate_labels(ref_t, ref_n, gate=gate, nuc_classes=nuc_classes).reshape(-1)

    ref_rgb_flat = ref_rgb_s.reshape(-1, 3)
    global_mean = ref_rgb_flat.mean(0)
    tar_feat = _component_local_geometry_features(tar_t, tar_n, tissue_classes=np.arange(len(tissue_classes))).reshape(-1, 14)
    ref_feat = _component_local_geometry_features(ref_t, ref_n, tissue_classes=np.arange(len(tissue_classes))).reshape(-1, 14)

    n = h * w
    warped = np.zeros((n, 3), np.float32)
    validity = np.zeros((n,), np.float32)
    match_y = np.full((n,), -1, np.int64)
    match_x = np.full((n,), -1, np.int64)

    for cls in np.unique(tar_gate):
        tar_idx = np.where(tar_gate == cls)[0]
        ref_idx = np.where(ref_gate == cls)[0]

        if ref_idx.size == 0:
            warped[tar_idx] = global_mean
            validity[tar_idx] = 0.0
            continue
        validity[tar_idx] = 1.0

        if ref_idx.size > max_ref:
            ref_idx = rng.choice(ref_idx, size=max_ref, replace=False)
        ref_rgb_c = ref_rgb_flat[ref_idx]

        if mode == "mean":
            warped[tar_idx] = ref_rgb_c.mean(0)
            continue

        ref_feat_c = ref_feat[ref_idx]
        ref_y = ref_idx // w
        ref_x = ref_idx % w

        for s in range(0, tar_idx.size, chunk):
            ti = tar_idx[s : s + chunk]
            tf = tar_feat[ti]
            sq = ((tf[:, None, :] - ref_feat_c[None, :, :]) ** 2).sum(-1)
            if mode == "soft":
                wgt = _softmax(-sq / max(tau, 1e-6), axis=1)
                warped[ti] = wgt @ ref_rgb_c
            elif mode == "hard":
                nn = sq.argmin(axis=1)
                match_y[ti] = ref_y[nn]
                match_x[ti] = ref_x[nn]

    if mode == "hard":
        my = match_y.reshape(h, w).astype(np.float32)
        mx = match_x.reshape(h, w).astype(np.float32)
        hole = match_y.reshape(h, w) < 0
        filter_size = max(int(smooth), 1)
        if filter_size > 1:
            # Simple global smoothing. Patch warp below does class-aware grid smoothing.
            my = median_filter(my, size=filter_size)
            mx = median_filter(mx, size=filter_size)
        my_i = np.clip(np.round(my), 0, h - 1).astype(np.int64)
        mx_i = np.clip(np.round(mx), 0, w - 1).astype(np.int64)
        gathered = ref_rgb_s[my_i, mx_i]
        gathered[hole] = global_mean
        warped = gathered.reshape(-1, 3)

    return warped.reshape(h, w, 3), validity.reshape(h, w)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(
    ref_rgb,
    ref_t,
    ref_n,
    tar_rgb,
    tar_t,
    tar_n,
    warped_mean,
    warped_soft,
    warped_hard,
    validity,
    out_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 4, figsize=(15.5, 11.5))
    mask_kw = dict(cmap="tab10", vmin=0, vmax=9, interpolation="nearest")

    ax[0, 0].imshow(ref_rgb); ax[0, 0].set_title("reference RGB\n(texture source)")
    ax[0, 1].imshow(ref_t, **mask_kw); ax[0, 1].set_title("reference tissue mask")
    ax[0, 2].imshow(ref_n, **mask_kw); ax[0, 2].set_title("reference nuclei mask")
    ax[0, 3].axis("off")

    if tar_rgb is None:
        ax[1, 0].axis("off")
        ax[1, 0].text(0.5, 0.5, "target RGB unavailable\n(generation-only case)", ha="center", va="center")
    else:
        ax[1, 0].imshow(tar_rgb)
    ax[1, 0].set_title("TARGET RGB\n(preview only)")
    ax[1, 1].imshow(tar_t, **mask_kw); ax[1, 1].set_title("TARGET tissue mask")
    ax[1, 2].imshow(tar_n, **mask_kw); ax[1, 2].set_title("TARGET nuclei mask")
    ax[1, 3].imshow(validity, cmap="gray", vmin=0, vmax=1); ax[1, 3].set_title("HARD validity")

    ax[2, 0].imshow(np.clip(warped_mean, 0, 1)); ax[2, 0].set_title("MEAN\nper-class color")
    ax[2, 1].imshow(np.clip(warped_soft, 0, 1)); ax[2, 1].set_title("SOFT\nfull avg; texture dies")
    ax[2, 2].imshow(np.clip(warped_hard, 0, 1)); ax[2, 2].set_title("HARD pixel copy\nuse as warped_ref")
    ax[2, 3].imshow(np.clip(np.abs(warped_hard - warped_mean) * 2.0, 0, 1)); ax[2, 3].set_title("HARD - MEAN\nlocal variation")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        "Warp preview: mean / soft / hard pixel-level copy only",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

def _stamp_disks(shape, centers, radius):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    m = np.zeros(shape, bool)
    for cy, cx in centers:
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return m


def _sample_nuclei(tissue, tumor_label, n_tumor, n_other, radius, rng):
    H, W = tissue.shape
    pts = []
    ys, xs = np.where(tissue == tumor_label)
    if len(ys):
        sel = rng.choice(len(ys), size=min(n_tumor, len(ys)), replace=False)
        pts += list(zip(ys[sel], xs[sel]))
    ys, xs = np.where(tissue != tumor_label)
    if len(ys):
        sel = rng.choice(len(ys), size=min(n_other, len(ys)), replace=False)
        pts += list(zip(ys[sel], xs[sel]))
    return _stamp_disks((H, W), pts, radius).astype(np.int64)


def make_synthetic(size=256, seed=0):
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W]

    # ---- target: tumor blob on the LEFT, plus a small necrosis (class 3) ----
    tar_t = np.ones((H, W), np.int64)  # 1 = stroma
    tar_t[((xx - W * 0.30) ** 2) / (W * 0.22) ** 2 + ((yy - H * 0.45) ** 2) / (H * 0.26) ** 2 < 1] = 2
    tar_t[((xx - W * 0.72) ** 2) / (W * 0.09) ** 2 + ((yy - H * 0.74) ** 2) / (H * 0.09) ** 2 < 1] = 3
    tar_n = _sample_nuclei(tar_t, 2, n_tumor=90, n_other=25, radius=3, rng=rng)

    # ---- reference: tumor blob on the RIGHT (different layout), NO necrosis ----
    ref_t = np.ones((H, W), np.int64)
    ref_t[((xx - W * 0.70) ** 2) / (W * 0.24) ** 2 + ((yy - H * 0.50) ** 2) / (H * 0.30) ** 2 < 1] = 2
    ref_n = _sample_nuclei(ref_t, 2, n_tumor=110, n_other=30, radius=3, rng=rng)

    # ---- reference appearance: distinct color per class + visible texture ----
    colors = {1: (232, 170, 200), 2: (150, 92, 172), 3: (110, 80, 70)}
    ref_rgb = np.zeros((H, W, 3), np.float32)
    for c, col in colors.items():
        ref_rgb[ref_t == c] = col
    field = gaussian_filter(rng.standard_normal((H, W, 1)), sigma=(6, 6, 0))
    field = field / (np.abs(field).max() + 1e-6)
    streaks = 0.12 * np.sin((xx + 2 * yy) / 7.0)[..., None]
    fine = 0.06 * rng.standard_normal((H, W, 1))
    ref_rgb = ref_rgb * (1.0 + 0.35 * field + streaks + fine)
    ref_rgb[ref_n > 0] = (52, 36, 96)
    ref_rgb = np.clip(ref_rgb / 255.0, 0, 1)
    return ref_rgb, ref_t, ref_n, tar_t, tar_n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Preview a warped reference from masks.")
    p.add_argument("--demo", action="store_true", help="run on synthetic data")
    p.add_argument("--ref-image"); p.add_argument("--ref-tissue"); p.add_argument("--ref-nuclei")
    p.add_argument("--tar-image", default=None, help="Optional target RGB image shown for preview comparison.")
    p.add_argument("--tar-tissue"); p.add_argument("--tar-nuclei")
    p.add_argument("--out", default="warp_preview.png")
    p.add_argument("--save-prefix", default=None, help="Optional prefix to save hard/mean/validity PNGs separately.")
    p.add_argument("--corr-size", type=int, default=192)
    p.add_argument("--tau", type=float, default=0.02, help="temperature for the baseline soft warp")
    p.add_argument("--smooth", type=int, default=3, help="median filter size for baseline hard copy")
    p.add_argument("--baseline-gate", choices=["tissue", "tissue_nucbin", "joint"], default="joint")
    p.add_argument("--baseline-max-ref", type=int, default=2048, help="max reference pixels for baseline mean/soft/hard modes")
    p.add_argument("--no-soft", action="store_true", help="skip expensive full softmax baseline; reuse mean panel instead")
    p.add_argument("--density-sigma", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.demo:
        ref_rgb, ref_t, ref_n, tar_t, tar_n = make_synthetic(seed=args.seed)
        tar_rgb = None
    else:
        for need in ("ref_image", "ref_tissue", "ref_nuclei", "tar_tissue", "tar_nuclei"):
            if getattr(args, need) is None:
                raise SystemExit(f"--{need.replace('_','-')} is required (or use --demo).")
        ref_rgb = _load_rgb(args.ref_image)
        ref_t = _load_label(args.ref_tissue); ref_n = _load_label(args.ref_nuclei)
        tar_t = _load_label(args.tar_tissue); tar_n = _load_label(args.tar_nuclei)
        tar_rgb = _load_rgb(args.tar_image) if args.tar_image else None

    common = dict(corr_size=args.corr_size, tau=args.tau, smooth=args.smooth, gate=args.baseline_gate, seed=args.seed, max_ref=args.baseline_max_ref)
    warped_mean, _ = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="mean", **common)
    if args.no_soft:
        warped_soft = warped_mean.copy()
    else:
        warped_soft, _ = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="soft", **common)
    warped_hard, validity = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="hard", **common)
    ref_t_s = _resize_label(ref_t, args.corr_size); ref_n_s = _resize_label(ref_n, args.corr_size)
    tar_t_s = _resize_label(tar_t, args.corr_size); tar_n_s = _resize_label(tar_n, args.corr_size)
    ref_rgb_s = _resize_rgb(ref_rgb, args.corr_size)
    tar_rgb_s = _resize_rgb(tar_rgb, args.corr_size) if tar_rgb is not None else None
    visualize(
        ref_rgb_s,
        ref_t_s,
        ref_n_s,
        tar_rgb_s,
        tar_t_s,
        tar_n_s,
        warped_mean,
        warped_soft,
        warped_hard,
        validity,
        args.out,
    )

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        _save_rgb(prefix.with_name(prefix.name + "_hard.png"), warped_hard)
        _save_gray(prefix.with_name(prefix.name + "_validity.png"), validity)
        _save_rgb(prefix.with_name(prefix.name + "_mean.png"), warped_mean)
        if not args.no_soft:
            _save_rgb(prefix.with_name(prefix.name + "_soft.png"), warped_soft)
        print(f"saved extra files with prefix {prefix}")


if __name__ == "__main__":
    main()
