#!/usr/bin/env python3
"""
Depth keypoint descriptor similarity debugger.

Goal:
    - Inspect how depth-layout descriptors behave across reference/query frames.
    - For multiple (grid_x, grid_y, depth_bins) configurations:
        * Build descriptors from depth keypoints.
        * Compute a ref x query similarity/distance matrix on subsets.
        * Save matrices and a heatmap for inspection.

Assumptions:
    - Keypoint files:
        ref-kp-dir   / depth_{idx:06d}.kps.npz
        query-kp-dir / depth_{idx:06d}.kps.npz
    - Each .kps.npz contains at least:
        x      (N,) float32
        y      (N,) float32
        depth  (N,) float32

Matrix shape:
    - rows    = references (subset)
    - columns = queries    (subset)

We compute:
    dist[r, q] = L1( desc_ref[r], desc_query[q] )
    sim[r, q]  = 1 - (dist - min_dist) / (max_dist - min_dist)
                 (normalized to [0,1], NaNs left as NaN)
"""

import argparse
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Descriptor construction
# ----------------------------------------------------------------------

def build_depth_layout_descriptor(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    grid_x: int,
    grid_y: int,
    depth_bins: int,
    depth_min: float,
    depth_max: float,
    min_kps: int,
    flat_std_thresh: float,
):
    """
    Build a depth-layout descriptor from keypoints.

    Steps:
        - Filter out non-finite depths.
        - Require at least min_kps keypoints.
        - Reject frame if depth std < flat_std_thresh (too flat).
        - Approximate image size from keypoints (max x,y).
        - Normalise x,y to [0,1), then bin into (grid_x, grid_y) cells.
        - Clip depths to [depth_min, depth_max], normalise to [0,1),
          bin into depth_bins.
        - Accumulate a 3D histogram (gy, gx, depth_bin).
        - L1-normalise and flatten.

    Returns:
        1D np.ndarray (grid_y * grid_x * depth_bins,) or None.
    """
    if x.size == 0 or depth.size == 0:
        return None

    mask = np.isfinite(depth)
    if not np.any(mask):
        return None

    x = x[mask]
    y = y[mask]
    depth = depth[mask]

    if x.size < min_kps:
        return None

    if float(np.std(depth)) < flat_std_thresh:
        return None

    W = float(x.max() + 1.0)
    H = float(y.max() + 1.0)
    if W <= 1.0 or H <= 1.0:
        return None

    # Normalised coordinates [0,1)
    u = np.clip(x / (W - 1.0), 0.0, 0.999999)
    v = np.clip(y / (H - 1.0), 0.0, 0.999999)

    gx = (u * grid_x).astype(np.int32)  # 0..grid_x-1
    gy = (v * grid_y).astype(np.int32)  # 0..grid_y-1

    if depth_max <= depth_min:
        return None

    depth_clipped = np.clip(depth, depth_min, depth_max)
    depth_norm = (depth_clipped - depth_min) / (depth_max - depth_min + 1e-8)
    db = np.clip((depth_norm * depth_bins).astype(np.int32), 0, depth_bins - 1)

    hist = np.zeros((grid_y, grid_x, depth_bins), dtype=np.float32)
    for yy, xx, dd in zip(gy, gx, db):
        hist[yy, xx, dd] += 1.0

    total = hist.sum()
    if total <= 0.0:
        return None

    hist = hist.reshape(-1)
    hist /= total
    return hist


def load_descriptor_for_index(
    root_dir: Path,
    idx: int,
    pattern: str,
    grid_x: int,
    grid_y: int,
    depth_bins: int,
    depth_min: float,
    depth_max: float,
    min_kps: int,
    flat_std_thresh: float,
):
    """
    Load keypoints from .kps.npz for a given index and build descriptor.

    Returns:
        descriptor (np.ndarray or None)
    """
    fname = pattern.format(idx)
    kp_path = root_dir / fname

    if not kp_path.exists():
        return None

    try:
        data = np.load(str(kp_path))
    except Exception:
        return None

    if "x" not in data or "y" not in data or "depth" not in data:
        return None

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    depth = data["depth"].astype(np.float32)

    desc = build_depth_layout_descriptor(
        x=x,
        y=y,
        depth=depth,
        grid_x=grid_x,
        grid_y=grid_y,
        depth_bins=depth_bins,
        depth_min=depth_min,
        depth_max=depth_max,
        min_kps=min_kps,
        flat_std_thresh=flat_std_thresh,
    )
    return desc


def build_all_descriptors(
    indices: np.ndarray,
    kp_root: Path,
    pattern: str,
    grid_x: int,
    grid_y: int,
    depth_bins: int,
    depth_min: float,
    depth_max: float,
    min_kps: int,
    flat_std_thresh: float,
    label: str,
):
    """
    Build descriptors for the provided frame indices.

    indices: array of ints (frame indices)
    Returns:
        descs : list of length len(indices), each entry is np.ndarray or None
    """
    descs = []
    num_valid = 0
    num_missing = 0
    num_flat_or_small = 0

    print(f"[INFO] Building descriptors for {label} from {kp_root}")
    print(f"[INFO] Indices range: {indices.min()}..{indices.max()} "
          f"(count={len(indices)})")

    for idx in tqdm(indices, desc=f"Descriptors {label}"):
        fname = pattern.format(idx)
        kp_path = kp_root / fname

        if not kp_path.exists():
            descs.append(None)
            num_missing += 1
            continue

        desc = load_descriptor_for_index(
            root_dir=kp_root,
            idx=idx,
            pattern=pattern,
            grid_x=grid_x,
            grid_y=grid_y,
            depth_bins=depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            min_kps=min_kps,
            flat_std_thresh=flat_std_thresh,
        )
        if desc is None:
            descs.append(None)
            num_flat_or_small += 1
        else:
            descs.append(desc)
            num_valid += 1

    print(f"[INFO] {label}: {num_valid} valid descriptors, "
          f"{num_missing} missing files, "
          f"{num_flat_or_small} flat/too-few-kps.")
    return descs


# ----------------------------------------------------------------------
# Similarity matrix building (on subsets)
# ----------------------------------------------------------------------

def select_indices_from_dir(kp_root: Path, pattern: str):
    """
    Inspect kp_root for files matching the given pattern and return
    sorted array of frame indices present.

    pattern: e.g. 'depth_{:06d}.kps.npz' or 'frame_{:06d}.kps.npz'
    """
    if "{" not in pattern or "}" not in pattern:
        raise ValueError(f"kp-pattern '{pattern}' must contain a '{{}}' placeholder")

    prefix, rest = pattern.split("{", 1)
    _, suffix = rest.split("}", 1)

    prefix_re = re.escape(prefix)
    suffix_re = re.escape(suffix)
    regex = re.compile(rf"^{prefix_re}(\d+){suffix_re}$")

    indices = []
    for p in kp_root.iterdir():
        if not p.is_file():
            continue
        m = regex.fullmatch(p.name)
        if m:
            idx = int(m.group(1))
            indices.append(idx)

    if not indices:
        raise RuntimeError(f"No keypoint files matching pattern '{pattern}' "
                           f"found in {kp_root}")

    indices = np.array(sorted(indices), dtype=int)
    print(f"[INFO] Found {len(indices)} keypoint files in {kp_root} "
          f"(min idx={indices.min()}, max idx={indices.max()})")
    return indices


def choose_subset_indices(all_indices: np.ndarray, max_items: int):
    """
    Select up to max_items *actual frame indices* from all_indices.

    If max_items <= 0 or >= len(all_indices): return all_indices.
    Otherwise, pick evenly spaced indices across the array.
    """
    N = len(all_indices)
    if max_items <= 0 or max_items >= N:
        return all_indices
    pos = np.linspace(0, N - 1, max_items, dtype=int)
    return all_indices[pos]


def build_descriptor_matrix(descs: list):
    """
    Turn a list of descriptors (np.ndarray or None) into:

        mat   : (N, D) float32 (N = len(descs), D = descriptor dim or 0)
        valid : (N,) bool

    For None entries, row is zeros and valid[i] = False.
    If all descs are None, mat has shape (N, 0).
    """
    N = len(descs)
    valid = np.array([d is not None for d in descs], dtype=bool)

    if not np.any(valid):
        # No valid descriptors at all
        return np.zeros((N, 0), dtype=np.float32), valid

    # Index of first valid descriptor
    first_idx = int(np.argmax(valid))
    D = descs[first_idx].shape[0]

    mat = np.zeros((N, D), dtype=np.float32)
    for i, d in enumerate(descs):
        if d is not None:
            if d.shape[0] != D:
                raise ValueError("Inconsistent descriptor dimensionality.")
            mat[i] = d
    return mat, valid


def compute_l1_distance_matrix_sub(
    ref_mat: np.ndarray,
    ref_valid: np.ndarray,
    query_mat: np.ndarray,
    query_valid: np.ndarray,
):
    """
    Compute a distance matrix using L1 distance between descriptors:

        dist[r_idx, q_idx] = L1( ref_desc[r], query_desc[q] )

    - If either descriptor is invalid, the distance is NaN.
    """
    R = ref_mat.shape[0]
    Q = query_mat.shape[0]

    if ref_mat.shape[1] == 0 or query_mat.shape[1] == 0:
        return np.full((R, Q), np.nan, dtype=np.float32)

    diff = np.abs(ref_mat[:, None, :] - query_mat[None, :, :])
    dist = diff.sum(axis=-1).astype(np.float32)  # (R, Q)

    for i in range(R):
        if not ref_valid[i]:
            dist[i, :] = np.nan
    for j in range(Q):
        if not query_valid[j]:
            dist[:, j] = np.nan

    return dist


def normalize_to_similarity(dist: np.ndarray):
    """
    Convert a distance matrix to [0,1] similarity for plotting:

        sim = 1 - (dist - min_dist) / (max_dist - min_dist)

    - Ignores NaNs when computing min/max.
    - Leaves NaNs as NaNs.
    """
    sim = np.full_like(dist, np.nan, dtype=np.float32)
    mask = np.isfinite(dist)
    if not np.any(mask):
        return sim

    d_min = float(dist[mask].min())
    d_max = float(dist[mask].max())
    if d_max <= d_min + 1e-12:
        sim[mask] = 1.0
        return sim

    sim[mask] = 1.0 - (dist[mask] - d_min) / (d_max - d_min)
    return sim


# ----------------------------------------------------------------------
# CLI and main
# ----------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Visualize depth keypoint descriptor similarity matrices "
                    "for different (grid_x, grid_y, depth_bins) settings."
    )

    p.add_argument(
        "--ref-kp-dir",
        type=str,
        required=True,
        help="Directory with reference keypoint .kps.npz files, "
             "e.g. depth_{idx:06d}.kps.npz",
    )
    p.add_argument(
        "--query-kp-dir",
        type=str,
        required=True,
        help="Directory with query keypoint .kps.npz files, "
             "e.g. depth_{idx:06d}.kps.npz",
    )
    p.add_argument(
        "--kp-pattern",
        type=str,
        default="depth_{:06d}.kps.npz",
        help="Filename pattern for keypoint files (single integer placeholder), "
             "e.g. 'depth_{:06d}.kps.npz' or 'frame_{:06d}.kps.npz'.",
    )

    # Parameter sweeps
    p.add_argument(
        "--grid-configs",
        type=str,
        default="4x3,6x4,12x8,18x12",
        help="Comma-separated list of grid configs, e.g. '4x3,6x4'.",
    )
    p.add_argument(
        "--depth-bins-list",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated list of depth_bins values, e.g. '2,4,8'.",
    )

    # Descriptor thresholds
    p.add_argument("--depth-min", type=float, default=0.0,
                   help="Minimum depth for clipping.")
    p.add_argument("--depth-max", type=float, default=80.0,
                   help="Maximum depth for clipping.")
    p.add_argument("--min-kps", type=int, default=3,
                   help="Minimum keypoints required to build a descriptor.")
    p.add_argument("--flat-std-thresh", type=float, default=0.01,
                   help="If std(depth) below this, treat as flat/uninformative.")

    # Subset sizes for plotting
    p.add_argument("--max-refs", type=int, default=256,
                   help="Max references to include in similarity matrix "
                        "(0 or negative = use all found).")
    p.add_argument("--max-queries", type=int, default=256,
                   help="Max queries to include in similarity matrix "
                        "(0 or negative = use all found).")

    # Output
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for matrices and plots.",
    )

    return p


def main():
    args = build_argparser().parse_args()

    ref_kp_root = Path(args.ref_kp_dir)
    query_kp_root = Path(args.query_kp_dir)
    if not ref_kp_root.exists():
        raise FileNotFoundError(ref_kp_root)
    if not query_kp_root.exists():
        raise FileNotFoundError(query_kp_root)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Infer available indices from folders
    ref_all_indices = select_indices_from_dir(ref_kp_root, args.kp_pattern)
    query_all_indices = select_indices_from_dir(query_kp_root, args.kp_pattern)

    # Choose subset indices for plotting (actual frame indices)
    ref_indices_sub = choose_subset_indices(ref_all_indices, args.max_refs)
    query_indices_sub = choose_subset_indices(query_all_indices, args.max_queries)
    print(f"[INFO] Using {len(ref_indices_sub)} refs (subset) and "
          f"{len(query_indices_sub)} queries (subset) for matrices.")

    # Prepare grids and depth_bins to sweep
    grid_cfgs = []
    for token in args.grid_configs.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token.lower():
            raise ValueError(f"Bad grid config: '{token}', expected 'WxH'")
        gx_str, gy_str = token.lower().split("x")
        grid_cfgs.append((int(gx_str), int(gy_str)))

    depth_bins_list = [int(x) for x in args.depth_bins_list.split(",") if x.strip()]

    print(f"[INFO] Grid configs: {grid_cfgs}")
    print(f"[INFO] Depth bins:   {depth_bins_list}")

    # Loop over descriptor configs
    for (grid_x, grid_y) in grid_cfgs:
        for depth_bins in depth_bins_list:
            cfg_name = f"grid{grid_x}x{grid_y}_bins{depth_bins}"
            print("\n" + "=" * 80)
            print(f"[CFG] {cfg_name}")
            print("=" * 80)

            # Build descriptors for subset indices for this config
            ref_descs = build_all_descriptors(
                indices=ref_indices_sub,
                kp_root=ref_kp_root,
                pattern=args.kp_pattern,
                grid_x=grid_x,
                grid_y=grid_y,
                depth_bins=depth_bins,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                min_kps=args.min_kps,
                flat_std_thresh=args.flat_std_thresh,
                label=f"refs_{cfg_name}",
            )

            query_descs = build_all_descriptors(
                indices=query_indices_sub,
                kp_root=query_kp_root,
                pattern=args.kp_pattern,
                grid_x=grid_x,
                grid_y=grid_y,
                depth_bins=depth_bins,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                min_kps=args.min_kps,
                flat_std_thresh=args.flat_std_thresh,
                label=f"queries_{cfg_name}",
            )

            ref_mat, ref_valid = build_descriptor_matrix(ref_descs)
            query_mat, query_valid = build_descriptor_matrix(query_descs)

            print(f"[INFO] Descriptor dim for this config: {ref_mat.shape[1]}")

            # Build distance matrix
            dist = compute_l1_distance_matrix_sub(
                ref_mat=ref_mat,
                ref_valid=ref_valid,
                query_mat=query_mat,
                query_valid=query_valid,
            )

            sim = normalize_to_similarity(dist)

            # Save matrices
            np.save(outdir / f"dist_{cfg_name}.npy", dist)
            np.save(outdir / f"sim_{cfg_name}.npy", sim)
            print(f"[INFO] Saved dist/sim matrices for {cfg_name} to {outdir}")

            # Plot similarity matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                sim,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
            )
            fig.colorbar(im, ax=ax, label="Similarity (normalized)")

            ax.set_xlabel(f"Query subset indices (size={len(query_indices_sub)})")
            ax.set_ylabel(f"Ref subset indices (size={len(ref_indices_sub)})")
            ax.set_title(f"Depth descriptor similarity: {cfg_name}")

            plt.tight_layout()
            fig_path = outdir / f"sim_{cfg_name}.png"
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"[INFO] Saved similarity heatmap to {fig_path}")


if __name__ == "__main__":
    main()
