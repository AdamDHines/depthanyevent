#!/usr/bin/env python3
"""
Depth Keypoint Extraction Script (Enhanced with AKAZE & Surface Normals)

This script:
- Loads depth maps (PNG magma or .npy).
- Computes Surface Normals to create "geometric texture" from smooth depth.
- Detects keypoints using AKAZE, ORB, GFTT, or Canny Edges.
- Enforces depth-balanced sampling.
- Saves keypoints (.npz) and visualizations (.png).

Usage:
    python extract_kps.py --input-dir /path/to/depth --output-dir /path/to/kps --method akaze --save-vis
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ==========================
# Depth & Geometry Helpers
# ==========================

def load_depth_from_file(path: Path):
    """
    Load a depth map.
    Returns:
        depth_raw : 2D float32 array (arbitrary units)
        depth_norm: 2D float32 array in [0,1]
        vis_img   : 3-channel uint8 image (Magma colormap) for visualisation
    """
    ext = path.suffix.lower()
    
    if ext == ".npy":
        depth_raw = np.load(str(path)).astype(np.float32)
        if depth_raw.ndim == 3:
            depth_raw = np.squeeze(depth_raw)
            
        # Normalize per-image to [0,1] for processing
        valid = depth_raw[np.isfinite(depth_raw)]
        if valid.size > 0:
            dmin, dmax = float(valid.min()), float(valid.max())
            if dmax > dmin:
                depth_norm = (depth_raw - dmin) / (dmax - dmin)
            else:
                depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
        else:
            depth_norm = np.zeros_like(depth_raw, dtype=np.float32)

        # Create Magma vis
        img_gray = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        vis_img = cv2.applyColorMap(img_gray, cv2.COLORMAP_MAGMA)

        return depth_raw, depth_norm, vis_img

    elif ext in [".png", ".jpg", ".jpeg"]:
        # Assume image is already a colormap or grayscale proxy for depth
        img_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_color is None:
            raise IOError(f"Failed to read image {path}")
        
        vis_img = img_color.copy()
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        depth_norm = img_gray.astype(np.float32) / 255.0
        depth_raw = depth_norm.copy()

        return depth_raw, depth_norm, vis_img

    else:
        raise ValueError(f"Unsupported depth file extension: {path}")


def compute_surface_normals(depth: np.ndarray):
    """
    Compute surface normals from a normalized depth map.
    This converts smooth gradients into texture that detectors (ORB/AKAZE) can see.
    """
    # 1. Slight smooth to reduce noise
    depth = cv2.GaussianBlur(depth, (3, 3), 0)

    # 2. Compute Gradients
    dz_dx = cv2.Scharr(depth, cv2.CV_32F, 1, 0)
    dz_dy = cv2.Scharr(depth, cv2.CV_32F, 0, 1)

    # 3. Auto-scale Z component
    # If the depth map is very flat (low gradients), we amplify the Z-component
    # logic so the surface normals don't just look like a solid color.
    grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    mean_grad = np.mean(grad_mag)
    
    # Tunable sensitivity: lower z_factor = more exaggerated slopes
    z_factor = 1.0 if mean_grad < 1e-6 else (mean_grad * 2.0)

    # Construct normal vector (-dz/dx, -dz/dy, z_factor)
    normal = np.dstack((-dz_dx, -dz_dy, np.full_like(depth, z_factor)))

    # 4. Normalize vector length
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1e-6 
    normal_unit = normal / norm

    # 5. Map [-1, 1] to [0, 255] BGR image
    normal_vis = ((normal_unit + 1) * 127.5).astype(np.uint8)
    normal_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
    
    return normal_bgr


# ==========================
# Keypoint Detectors
# ==========================

def detect_akaze(depth_norm: np.ndarray, max_kp: int):
    """
    AKAZE detector on Surface Normals.
    Robust to non-linear scale changes and smooth surfaces.
    """
    # Detect on Normals, not raw depth
    img_for_det = compute_surface_normals(depth_norm)
    
    detector = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=0.0005,   # Low threshold = more keypoints
        nOctaves=4,
        nOctaveLayers=4
    )

    keypoints, desc = detector.detectAndCompute(img_for_det, None)

    if not keypoints or desc is None:
        return np.empty(0), np.empty(0), np.empty(0), None

    # Sort by response
    idx_sorted = np.argsort([-kp.response for kp in keypoints])
    # Limit BEFORE processing to save time, but we usually limit after balancing
    # Here we just ensure we don't return millions
    keep_n = max(max_kp * 5, len(keypoints)) 
    idx_keep = idx_sorted[:keep_n]

    keypoints = [keypoints[i] for i in idx_keep]
    desc = desc[idx_keep]

    xs = np.array([kp.pt[0] for kp in keypoints], dtype=np.float32)
    ys = np.array([kp.pt[1] for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    return xs, ys, responses, desc


def detect_orb(depth_norm: np.ndarray, max_kp: int):
    """
    ORB detector on Surface Normals with lowered thresholds.
    """
    img_for_det = compute_surface_normals(depth_norm)

    orb = cv2.ORB_create(
        nfeatures=max_kp * 5,   # Oversample
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=10,       # Reduced from 31 to find points near edges
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=0         # 0 allows detecting very subtle curvature
    )

    keypoints, desc = orb.detectAndCompute(img_for_det, None)

    if not keypoints or desc is None:
        return np.empty(0), np.empty(0), np.empty(0), None

    idx_sorted = np.argsort([-kp.response for kp in keypoints])
    keypoints = [keypoints[i] for i in idx_sorted]
    desc = desc[idx_sorted]

    xs = np.array([kp.pt[0] for kp in keypoints], dtype=np.float32)
    ys = np.array([kp.pt[1] for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    return xs, ys, responses, desc


def detect_gftt(depth_norm: np.ndarray, max_kp: int):
    """
    GFTT on Surface Normals.
    """
    img_for_det = compute_surface_normals(depth_norm)
    img_gray = cv2.cvtColor(img_for_det, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        img_gray,
        maxCorners=max_kp * 2,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        useHarrisDetector=False,
    )

    if corners is None:
        return np.empty(0), np.empty(0), np.empty(0), None

    corners = corners.reshape(-1, 2)
    xs = corners[:, 0]
    ys = corners[:, 1]
    
    # Fake response using Sobel magnitude
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    
    responses = []
    H, W = img_gray.shape
    for x, y in zip(xs, ys):
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            responses.append(float(grad_mag[iy, ix]))
        else:
            responses.append(0.0)
            
    return xs, ys, np.array(responses, dtype=np.float32), None


def detect_edges(depth_norm: np.ndarray, max_kp: int):
    """
    Canny Edges directly on depth (finds depth discontinuities).
    """
    img8 = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    edges = cv2.Canny(img8, 50, 150)
    
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return np.empty(0), np.empty(0), np.empty(0), None

    # Random subsample if too many
    if xs.size > max_kp * 4:
        idx = np.random.choice(xs.size, size=max_kp * 4, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    return xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32), None


# ==========================
# Sampling
# ==========================

def depth_balanced_sampling(xs, ys, responses, depth_vals, max_kp, depth_bins, per_bin_kp, desc=None):
    """
    Ensures keypoints are distributed across different depth ranges.
    """
    N = xs.size
    if N == 0 or max_kp <= 0:
        return xs, ys, responses, desc

    # Sort by response (strongest first)
    order = np.argsort(-responses)
    xs = xs[order]
    ys = ys[order]
    resp = responses[order]
    depth = depth_vals[order]
    desc = desc[order] if desc is not None else None

    if depth_bins <= 1:
        keep = min(max_kp, N)
        return xs[:keep], ys[:keep], resp[:keep], (desc[:keep] if desc is not None else None)

    # Percentile binning
    valid_depths = depth[np.isfinite(depth)]
    if valid_depths.size == 0:
         keep = min(max_kp, N)
         return xs[:keep], ys[:keep], resp[:keep], (desc[:keep] if desc is not None else None)

    percentiles = np.linspace(0, 100, depth_bins + 1)
    edges = np.percentile(valid_depths, percentiles)
    
    # Ensure strictly increasing edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-5

    if per_bin_kp is None:
        per_bin_kp = max(1, max_kp // depth_bins)

    indices = []
    for b in range(depth_bins):
        lo, hi = edges[b], edges[b+1]
        # Find indices in the sorted arrays that fall in this depth bin
        in_bin = np.where((depth >= lo) & (depth < hi))[0]
        if in_bin.size > 0:
            indices.append(in_bin[:per_bin_kp])

    if not indices:
        idx_final = np.arange(min(max_kp, N))
    else:
        idx_final = np.concatenate(indices)
        # If we gathered too many (rare), clip; if too few, that's fine.
        if idx_final.size > max_kp:
            # If we need to cut, re-sort by response? 
            # They are already sorted by response globally, so just taking the head is okay-ish,
            # but ideally we balance the cut. For simplicity, just cut.
            idx_final = idx_final[:max_kp]

    return (xs[idx_final], ys[idx_final], resp[idx_final], 
            (desc[idx_final] if desc is not None else None))


# ==========================
# Main
# ==========================

def process_depth_file(path: Path, args):
    try:
        depth_raw, depth_norm, vis_img = load_depth_from_file(path)
    except Exception as e:
        print(f"[ERR] {path}: {e}")
        return

    H, W = depth_norm.shape

    # 1. Detect
    if args.method == "akaze":
        xs, ys, responses, desc = detect_akaze(depth_norm, args.max_kp)
    elif args.method == "orb":
        xs, ys, responses, desc = detect_orb(depth_norm, args.max_kp)
    elif args.method == "gftt":
        xs, ys, responses, desc = detect_gftt(depth_norm, args.max_kp)
    elif args.method == "edges":
        xs, ys, responses, desc = detect_edges(depth_norm, args.max_kp)
    else:
        raise ValueError(f"Unknown method {args.method}")

    if xs.size == 0:
        if args.verbose:
            print(f"[WARN] No keypoints found in {path.name}")
        return

    # 2. Extract Depth Values for Keypoints
    # (We need integer coordinates for lookups)
    ixs = np.round(xs).astype(int)
    iys = np.round(ys).astype(int)
    
    # Clip to bounds
    ixs = np.clip(ixs, 0, W-1)
    iys = np.clip(iys, 0, H-1)
    
    kps_depth = depth_raw[iys, ixs]

    # 3. Depth Balanced Sampling
    xs_sel, ys_sel, resp_sel, desc_sel = depth_balanced_sampling(
        xs, ys, responses, kps_depth,
        max_kp=args.max_kp,
        depth_bins=args.depth_bins,
        per_bin_kp=args.per_bin_kp,
        desc=desc
    )

    # Refetch depths for selected final set
    ixs_sel = np.round(xs_sel).astype(int).clip(0, W-1)
    iys_sel = np.round(ys_sel).astype(int).clip(0, H-1)
    final_depths = depth_raw[iys_sel, ixs_sel]

    # 4. Save Results
    rel = path.relative_to(args.input_dir)
    out_npz = Path(args.output_dir) / rel.with_suffix(".kps.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "x": xs_sel,
        "y": ys_sel,
        "depth": final_depths,
        "response": resp_sel,
        "method": np.array(args.method)
    }
    if desc_sel is not None:
        save_dict["desc"] = desc_sel

    np.savez_compressed(out_npz, **save_dict)

    # 5. Visualisation (Optional)
    if args.save_vis:
        # Draw green circles
        vis_out = vis_img.copy()
        for x, y in zip(xs_sel, ys_sel):
            cv2.circle(vis_out, (int(x), int(y)), 3, (0, 255, 0), 1, cv2.LINE_AA)
        
        out_png = Path(args.output_dir) / "vis" / rel
        out_png.parent.mkdir(parents=True, exist_ok=True)
        
        # We append the method/count to filename for easier debugging
        out_vis_path = out_png.with_name(f"{out_png.stem}_{args.method}_{len(xs_sel)}kp.png")
        cv2.imwrite(str(out_vis_path), vis_out)

    if args.verbose:
        print(f"[INFO] {path.name}: {len(xs_sel)} kps saved.")


def main():
    parser = argparse.ArgumentParser(description="Extract Keypoints from Depth Maps (AKAZE/ORB/GFTT)")
    
    parser.add_argument("--input-dir", type=str, required=True, help="Input root directory (depth maps)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root directory (npz files)")
    
    parser.add_argument("--method", type=str, default="akaze", choices=["akaze", "orb", "gftt", "edges"],
                        help="Detector method. AKAZE/ORB use Surface Normals.")
    
    parser.add_argument("--max-kp", type=int, default=1000, help="Max keypoints to keep")
    
    parser.add_argument("--depth-bins", type=int, default=3, 
                        help="Number of depth bins for balancing (0 to disable)")
    parser.add_argument("--per-bin-kp", type=int, default=None, 
                        help="Target keypoints per bin")
    
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir {input_dir} does not exist.")

    files = []
    for ext in ["*.npy", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(list(input_dir.rglob(ext)))
    
    files = sorted(list(set(files))) # dedup

    print(f"[INIT] Found {len(files)} files in {input_dir}")
    print(f"[INIT] Method: {args.method} | Max KP: {args.max_kp}")

    for f in tqdm(files, desc="Processing"):
        process_depth_file(f, args)

if __name__ == "__main__":
    main()