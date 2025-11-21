#!/usr/bin/env python3
"""
Visualise GEOMETRICALLY VERIFIED depth keypoint matches.

Updates:
- Added --start-index to skip early frames.
- Added --ransac-thresh to tune geometric strictness.
- Output filenames now use the absolute query index (q{i}) rather than a counter.
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


# ----------------------------- I/O helpers ----------------------------- #

def list_kp_files(root: Path):
    files = sorted(root.glob("*.kps.npz"))
    return files


def load_kp_npz(path: Path):
    """
    Load x, y, depth, desc from a .kps.npz file.
    """
    data = np.load(str(path))

    if "x" not in data or "y" not in data:
        return np.empty(0), np.empty(0), np.empty(0), None

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    
    # Load depth if available, else return None
    if "depth" in data:
        depth = data["depth"].astype(np.float32)
    else:
        depth = None

    desc = None
    if "desc" in data:
        desc = data["desc"]
        if desc is None or desc.size == 0:
            desc = None
        else:
            if desc.ndim == 1:
                desc = desc.reshape(1, -1)
            desc = desc.astype(np.uint8)

    return x, y, depth, desc


def load_all_data(kp_dir: Path, label: str):
    kp_files = list_kp_files(kp_dir)
    if not kp_files:
        raise FileNotFoundError(f"No .kps.npz files found in {kp_dir}")

    print(f"[INFO] {label}: found {len(kp_files)} keypoint files.")

    data_list = []
    num_valid = 0

    for p in tqdm(kp_files, desc=f"Loading {label}"):
        x, y, z, desc = load_kp_npz(p)
        # Store as a dict for cleaner access
        data_list.append({
            "path": p,
            "x": x,
            "y": y,
            "z": z,
            "desc": desc
        })
        if desc is not None:
            num_valid += 1

    print(f"[INFO] {label}: {num_valid}/{len(kp_files)} have valid descriptors.")
    return data_list


def find_depth_image_for_kp(kp_path: Path, kp_root: Path, depth_root: Path):
    rel = kp_path.relative_to(kp_root)
    rel_str = rel.as_posix()
    
    # Strip extensions
    if rel_str.endswith(".kps.npz"):
        base = rel_str[:-len(".kps.npz")]
    else:
        base = rel_str.rsplit(".", 1)[0]

    # Try common extensions
    for ext in (".png", ".npy", ".jpg", ".jpeg"):
        candidate = depth_root / f"{base}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_depth_image_for_vis(path: Path):
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None: return None
        return img

    if ext == ".npy":
        depth = np.load(str(path)).astype(np.float32)
        if depth.ndim == 3: depth = np.squeeze(depth)
        
        valid = depth[np.isfinite(depth)]
        if valid.size > 0:
            dmin, dmax = float(valid.min()), float(valid.max())
            depth_norm = (depth - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(depth)
        else:
            depth_norm = np.zeros_like(depth)

        img8 = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(img8, cv2.COLORMAP_MAGMA)
    return None


# ----------------------------- Geometric Logic ----------------------------- #

def perform_geometric_verification(
    kps_q, depth_q, 
    kps_r, depth_r, 
    matches,
    ransac_thresh
):
    """
    Filters matches using 3D RANSAC (Affine3D).
    """
    if len(matches) < 4:
        return np.zeros((len(matches), 1), dtype=np.uint8)

    # --- 1. Back-project pixels to 3D points ---
    fx = fy = 320.0
    cx = cy = 128.0 

    src_pts_3d = []
    dst_pts_3d = []
    valid_indices = []

    for i, m in enumerate(matches):
        q_idx = m.queryIdx
        r_idx = m.trainIdx

        zq = depth_q[q_idx] if depth_q is not None else np.nan
        zr = depth_r[r_idx] if depth_r is not None else np.nan

        if np.isnan(zq) or np.isnan(zr) or zq <= 1e-3 or zr <= 1e-3:
            continue

        uq, vq = kps_q[q_idx]
        ur, vr = kps_r[r_idx]

        xq = (uq - cx) * zq / fx
        yq = (vq - cy) * zq / fx
        
        xr = (ur - cx) * zr / fx
        yr = (vr - cy) * zr / fx

        src_pts_3d.append([xq, yq, zq])
        dst_pts_3d.append([xr, yr, zr])
        valid_indices.append(i)

    src_pts_3d = np.array(src_pts_3d, dtype=np.float32)
    dst_pts_3d = np.array(dst_pts_3d, dtype=np.float32)

    if len(src_pts_3d) < 4:
        return np.zeros((len(matches), 1), dtype=np.uint8)

    # --- 2. Run RANSAC (Affine3D) ---
    success, transform, inliers = cv2.estimateAffine3D(
        src_pts_3d, dst_pts_3d, 
        ransacThreshold=ransac_thresh,
        confidence=0.99
    )

    # --- 3. Construct final mask ---
    full_mask = np.zeros((len(matches), 1), dtype=np.uint8)
    
    if success and inliers is not None:
        for k, valid_idx in enumerate(valid_indices):
            if inliers[k] > 0:
                full_mask[valid_idx] = 1
                
    return full_mask


# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-kp-dir", type=str, required=True)
    parser.add_argument("--ref-kp-dir", type=str, required=True)
    parser.add_argument("--query-depth-root", type=str, required=True)
    parser.add_argument("--ref-depth-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    
    parser.add_argument("--start-index", type=int, default=200, 
                        help="Index of query to start processing from (skip early frames).")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of query frames to process after start-index.")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.8, help="Descriptor ratio test")
    
    parser.add_argument("--ransac-thresh", type=float, default=0.05,
                        help="Inlier threshold for 3D RANSAC (units of depth map). "
                             "Increase if depth is unscaled or matches are too strict.")
    
    args = parser.parse_args()

    q_root = Path(args.query_kp_dir)
    r_root = Path(args.ref_kp_dir)
    qd_root = Path(args.query_depth_root)
    rd_root = Path(args.ref_depth_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Data
    queries = load_all_data(q_root, "queries")
    refs = load_all_data(r_root, "refs")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    processed_count = 0

    # Enumerate allows us to track the ACTUAL index 'i' 
    for i, q_data in enumerate(queries):
        
        # Skip until we hit start index
        if i < args.start_index:
            continue
            
        # Stop if we've done enough
        if processed_count >= args.num_queries:
            break

        desc_q = q_data["desc"]
        if desc_q is None: continue

        # 1. Match against ALL refs to find candidates
        candidate_scores = []

        for r_idx, r_data in enumerate(refs):
            desc_r = r_data["desc"]
            if desc_r is None: 
                candidate_scores.append(0)
                continue

            knn = bf.knnMatch(desc_q, desc_r, k=2)
            good = []
            for m_n in knn:
                if len(m_n) == 2 and m_n[0].distance < args.ratio * m_n[1].distance:
                    good.append(m_n[0])
            
            candidate_scores.append(len(good))

        # 2. Select Top-K candidates
        candidate_indices = np.argsort(candidate_scores)[::-1][:args.topk]
        
        # 3. Verify and Visualize
        img_q = None 
        kp_q_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=7) for x, y in zip(q_data["x"], q_data["y"])]
        
        for r_idx in candidate_indices:
            if candidate_scores[r_idx] < 5: continue 

            r_data = refs[r_idx]
            desc_r = r_data["desc"]

            knn = bf.knnMatch(desc_q, desc_r, k=2)
            matches = []
            for m_n in knn:
                if len(m_n) == 2 and m_n[0].distance < args.ratio * m_n[1].distance:
                    matches.append(m_n[0])

            # --- GEOMETRIC VERIFICATION ---
            kps_q_arr = np.stack([q_data["x"], q_data["y"]], axis=1)
            kps_r_arr = np.stack([r_data["x"], r_data["y"]], axis=1)

            inlier_mask = perform_geometric_verification(
                kps_q_arr, q_data["z"],
                kps_r_arr, r_data["z"],
                matches,
                ransac_thresh=args.ransac_thresh
            )
            
            num_inliers = np.sum(inlier_mask)
            
            # Visualization
            if img_q is None:
                p = find_depth_image_for_kp(q_data["path"], q_root, qd_root)
                img_q = load_depth_image_for_vis(p) if p else None
            
            p_r = find_depth_image_for_kp(r_data["path"], r_root, rd_root)
            img_r = load_depth_image_for_vis(p_r) if p_r else None

            if img_q is not None and img_r is not None:
                kp_r_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=7) for x, y in zip(r_data["x"], r_data["y"])]

                vis = cv2.drawMatches(
                    img_q, kp_q_cv,
                    img_r, kp_r_cv,
                    matches, None,
                    matchColor=(0, 255, 0),       
                    singlePointColor=(0, 0, 255), 
                    matchesMask=inlier_mask.ravel().tolist(),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                # Use 'i' (absolute index) in filename, not processed_count
                fname = f"q{i:06d}_{q_data['path'].stem}_vs_{r_data['path'].stem}_inliers{num_inliers}.png"
                cv2.imwrite(str(outdir / fname), vis)
                print(f"[VIS] Saved {fname} (Inliers: {num_inliers}/{len(matches)})")

        processed_count += 1

if __name__ == "__main__":
    main()