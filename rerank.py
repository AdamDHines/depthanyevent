#!/usr/bin/env python3
"""
Geometric Re-ranking for VPR (RANSAC-based).

Methodology:
    1. Load the base Appearance Distance Matrix.
    2. For every query, look at the Top-K reference candidates.
    3. LOAD keypoints (AKAZE) and depth for the query and those K refs.
    4. PERFORM 3D RANSAC (Affine3D) to count inliers.
    5. UPDATE the distance: NewDist = BaseDist - (Weight * Num_Inliers).
    6. Compute Recall@K on the new matrix.

Performance Note:
    This is computationally heavier than histogram matching. 
    It uses parallel processing (joblib) to speed up the Top-K checks.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from prettytable import PrettyTable
from skimage.transform import resize
from joblib import Parallel, delayed
import os

# ----------------------------------------------------------------------
# I/O & Data Loading
# ----------------------------------------------------------------------

def load_data_for_index(root_dir: Path, idx: int, pattern: str):
    """
    Loads x, y, depth, desc for a specific index.
    Returns dict or None if failed.
    """
    fname = pattern.format(idx)
    path = root_dir / fname
    
    if not path.exists():
        return None

    try:
        data = np.load(str(path))
        # Basic checks
        if "x" not in data or "desc" not in data:
            return None
        
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        desc = data["desc"]
        
        # Handle depth
        if "depth" in data:
            depth = data["depth"].astype(np.float32)
        else:
            # If depth missing, we can't do 3D ransac
            return None

        # Handle empty descriptors
        if desc is None or desc.size == 0:
            return None
        
        if desc.ndim == 1:
            desc = desc.reshape(1, -1)

        return {"x": x, "y": y, "z": depth, "desc": desc}

    except Exception:
        return None

# ----------------------------------------------------------------------
# Geometric Core
# ----------------------------------------------------------------------

def compute_inliers(q_data, r_data, matcher, ransac_thresh):
    """
    Matches descriptors and runs 3D RANSAC.
    Returns: number of inliers (int).
    """
    if q_data is None or r_data is None:
        return 0

    desc_q = q_data["desc"]
    desc_r = r_data["desc"]

    # 1. Descriptor Matching
    # NORM_HAMMING for AKAZE/ORB (Binary)
    try:
        matches = matcher.knnMatch(desc_q, desc_r, k=2)
    except cv2.error:
        return 0

    good = []
    # Ratio Test (0.8 is standard for Lowe's test)
    for m_n in matches:
        if len(m_n) == 2 and m_n[0].distance < 0.8 * m_n[1].distance:
            good.append(m_n[0])

    if len(good) < 4:
        return 0

    # 2. Back-project to 3D
    # Approximated intrinsics (relative changes matter more than absolute here)
    fx = fy = 320.0
    cx = cy = 128.0 

    src_pts = []
    dst_pts = []

    for m in good:
        q_idx = m.queryIdx
        r_idx = m.trainIdx

        zq = q_data["z"][q_idx]
        zr = r_data["z"][r_idx]

        # Skip invalid depth
        if np.isnan(zq) or np.isnan(zr) or zq <= 0.001 or zr <= 0.001:
            continue

        uq, vq = q_data["x"][q_idx], q_data["y"][q_idx]
        ur, vr = r_data["x"][r_idx], r_data["y"][r_idx]

        # Pinhole unprojection
        xq = (uq - cx) * zq / fx
        yq = (vq - cy) * zq / fx
        
        xr = (ur - cx) * zr / fx
        yr = (vr - cy) * zr / fx

        src_pts.append([xq, yq, zq])
        dst_pts.append([xr, yr, zr])

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    if len(src_pts) < 4:
        return 0

    # 3. Geometric Verification (Affine3D)
    success, _, inliers = cv2.estimateAffine3D(
        src_pts, dst_pts, 
        ransacThreshold=ransac_thresh,
        confidence=0.99
    )

    if not success or inliers is None:
        return 0
    
    return int(np.sum(inliers))


def process_single_query(
    q_idx, 
    base_dists, 
    top_k, 
    q_data, 
    ref_kp_dir, 
    kp_pattern, 
    ransac_thresh,
    inlier_weight
):
    """
    Worker function for parallel processing.
    Process one query against its Top-K candidates.
    """
    # Identify Top-K candidates from base distances (lowest is best)
    # Argpartition is faster than argsort for top-k
    if top_k >= len(base_dists):
        top_indices = np.arange(len(base_dists))
    else:
        # Get indices of k smallest values
        part_indices = np.argpartition(base_dists, top_k)[:top_k]
        # Sort them to process best first (optional, mostly for debug)
        top_indices = part_indices[np.argsort(base_dists[part_indices])]

    # Create a local matcher for this thread
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Copy base distances to modify
    new_dists_col = base_dists.copy()

    # If query data is bad, we can't re-rank, return original
    if q_data is None:
        return q_idx, new_dists_col

    for r_idx in top_indices:
        # Load ref data on demand to save RAM
        # (OS file caching makes this reasonably fast on repeated hits)
        r_data = load_data_for_index(ref_kp_dir, r_idx, kp_pattern)
        
        num_inliers = compute_inliers(q_data, r_data, matcher, ransac_thresh)

        if num_inliers > 0:
            # RE-RANKING FORMULA:
            # NewDist = BaseDist - (Weight * Inliers)
            # 
            # Logic: BaseDist is usually [0.0 to 1.0] or [0.0 to 2.0]
            # Inliers can be [10 to 100+]
            # We want a confirmed match (high inliers) to become a very small (or negative) distance.
            # This pushes it to the front of the sort order.
            
            modifier = num_inliers * inlier_weight
            new_dists_col[r_idx] = base_dists[r_idx] - modifier

    return q_idx, new_dists_col


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def create_GTtol(GT: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0: return GT
    GT = (GT > 0).astype(np.uint8)
    R, Q = GT.shape
    GTtol = GT.copy()
    ones = np.argwhere(GT > 0)
    for r, c in ones:
        r0, r1 = max(0, r - tolerance), min(R, r + tolerance + 1)
        c0, c1 = max(0, c - tolerance), min(Q, c + tolerance + 1)
        GTtol[r0:r1, c0:c1] = 1
    return GTtol

def recallAtK(S, GT, K=1):
    # S is similarity (higher better). If using distance, convert before calling.
    j = GT.sum(0) > 0 
    S = S[:, j]
    GT = GT[:, j]
    if GT.shape[1] == 0: return 0.0
    
    # Argsort descending
    i = S.argsort(0)[-K:, :]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    hits = GT[i, j]
    return np.sum(hits.sum(0) > 0) / GT.shape[1]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-matrix", type=str, required=True)
    parser.add_argument("--output-matrix", type=str, default=None)
    parser.add_argument("--query-kp-dir", type=str, required=True)
    parser.add_argument("--ref-kp-dir", type=str, required=True)
    parser.add_argument("--gt-path", type=str, required=True)
    
    parser.add_argument("--kp-pattern", type=str, default="depth_{:06d}.kps.npz")
    parser.add_argument("--top-k", type=int, default=100, help="Only verify top-K candidates. RANSAC is expensive!")
    parser.add_argument("--ransac-thresh", type=float, default=0.10, help="3D RANSAC Inlier threshold")
    parser.add_argument("--inlier-weight", type=float, default=0.05, 
                        help="How much distance to subtract per inlier. "
                             "E.g. 0.05 * 20 inliers = -1.0 distance boost.")
    
    parser.add_argument("--gt-tolerance", type=int, default=2)
    parser.add_argument("--matrix-type", type=str, default="distance", choices=["distance", "similarity"])
    parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 = all)")

    args = parser.parse_args()

    # 1. Load Matrix
    print(f"[INFO] Loading distance matrix: {args.distance_matrix}")
    dist_matrix = np.load(args.distance_matrix)
    R, Q = dist_matrix.shape
    print(f"[INFO] Matrix Shape: {R} Refs x {Q} Queries")

    # 2. Pre-load Queries (Small enough to fit in RAM usually)
    # We don't pre-load Refs because R might be huge. We load Refs on demand in workers.
    print(f"[INFO] Pre-loading {Q} query keypoints...")
    query_kp_root = Path(args.query_kp_dir)
    ref_kp_root = Path(args.ref_kp_dir)
    
    queries_data = []
    for i in tqdm(range(Q)):
        queries_data.append(load_data_for_index(query_kp_root, i, args.kp_pattern))

    # 3. Parallel Re-ranking
    print(f"[INFO] Starting Geometric Re-ranking on Top-{args.top_k} candidates...")
    print(f"[INFO] Using {args.jobs} cores.")
    
    # Run parallel jobs
    results = Parallel(n_jobs=args.jobs)(
        delayed(process_single_query)(
            q_idx=i,
            base_dists=dist_matrix[:, i],
            top_k=args.top_k,
            q_data=queries_data[i],
            ref_kp_dir=ref_kp_root,
            kp_pattern=args.kp_pattern,
            ransac_thresh=args.ransac_thresh,
            inlier_weight=args.inlier_weight
        ) 
        for i in tqdm(range(Q), desc="Re-ranking")
    )

    # Reconstruct matrix
    new_dist_matrix = dist_matrix.copy()
    for q_idx, new_col in results:
        new_dist_matrix[:, q_idx] = new_col

    # Save
    if args.output_matrix:
        out_path = Path(args.output_matrix)
    else:
        out_path = Path(args.distance_matrix).with_name(Path(args.distance_matrix).stem + "_geometric.npy")
    
    np.save(out_path, new_dist_matrix)
    print(f"[INFO] Saved re-ranked matrix to {out_path}")

    # 4. Evaluate
    print("[INFO] Loading GT...")
    gt = np.load(args.gt_path)
    if gt.shape != dist_matrix.shape:
        print(f"[WARN] Resizing GT from {gt.shape} to {dist_matrix.shape}")
        gt = resize(gt, dist_matrix.shape, order=0, preserve_range=True, anti_aliasing=False)
    
    gt_bin = (gt > 0.5).astype(np.int32)
    gt_tol = create_GTtol(gt_bin, args.gt_tolerance)

    # Convert to similarity for Recall calc
    if args.matrix_type == "distance":
        # Invert distance: Max - Dist
        # Note: geometric re-ranking might make distances negative (Base - 0.05*Inliers). 
        # This works perfectly fine, the algebraic order is preserved.
        sim_base = dist_matrix.max() - dist_matrix
        sim_new = new_dist_matrix.max() - new_dist_matrix
    else:
        sim_base = dist_matrix
        sim_new = new_dist_matrix

    table = PrettyTable()
    table.field_names = ["K", "Recall (Base)", "Recall (Geometric)"]
    
    for k in [1, 5, 10, 20]:
        r_b = recallAtK(sim_base, gt_tol, k)
        r_n = recallAtK(sim_new, gt_tol, k)
        table.add_row([k, f"{r_b:.4f}", f"{r_n:.4f}"])

    print(table)

if __name__ == "__main__":
    main()