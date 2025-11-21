#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# --------------------
# Dataset discovery
# --------------------

def find_event_datasets(f: h5py.File):
    """
    Find x, y, t, p datasets inside an HDF5 file.
    Assumes a group 'events' with datasets 'x', 'y', 't', 'p'.
    """
    if "events" not in f or not isinstance(f["events"], h5py.Group):
        raise RuntimeError("Expected a group 'events' in the HDF5 file.")

    g = f["events"]

    for key in ["x", "y", "t", "p"]:
        if key not in g:
            raise RuntimeError(f"Expected dataset 'events/{key}' in the HDF5 file.")

    x_dset = g["x"]
    y_dset = g["y"]
    t_dset = g["t"]
    p_dset = g["p"]

    return x_dset, y_dset, t_dset, p_dset


def infer_resolution_stream(x_dset, y_dset, chunk_size: int = 200_000):
    """
    Infer sensor resolution by streaming through x/y once.
    No tqdm – just a single pass.
    """
    N = len(x_dset)
    H_max, W_max = 0, 0

    if N == 0:
        raise ValueError("No events in dataset; cannot infer resolution.")

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        xs = x_dset[start:end][:]
        ys = y_dset[start:end][:]

        if xs.size == 0:
            continue

        W_max = max(W_max, int(xs.max()) + 1)
        H_max = max(H_max, int(ys.max()) + 1)

    print(f"[INFO] Inferred resolution HxW = {H_max}x{W_max}")
    return H_max, W_max


# --------------------
# Tencode core
# --------------------

def tencode_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    white_frame: bool = False,
    normalize: bool = False,
):
    """
    NumPy implementation of the Tencode mapping used in DepthAnyEvent/event_representation.

    - x, y, t, p are 1D arrays for a single temporal chunk (window)
    - height, width: frame size
    - white_frame: if True, initialise with white (255) instead of black
    - normalize: if True, output in [0,1]; otherwise [0,255]
    """
    assert x.ndim == y.ndim == t.ndim == p.ndim == 1
    n = x.shape[0]
    base_val = 255.0 if white_frame else 0.0

    if n == 0:
        frame = np.full((3, height, width), base_val, dtype=np.float32)
        return frame / 255.0 if normalize else frame

    # Convert to correct dtypes
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    t = t.astype(np.float64)
    p = p.astype(np.float64)

    # Sort by time so "last event wins" at each pixel
    order = np.argsort(t)
    x = x[order]
    y = y[order]
    t = t[order]
    p = p[order]

    # Polarity to {0,1}
    if p.min() < 0:
        pol = (p > 0).astype(np.float32)
    else:
        pol = (p > 0).astype(np.float32)

    # Normalise time to [0,1] within this window: t_norm[0]=0 (oldest), t_norm[-1]=1 (newest)
    if t[-1] != t[0]:
        t_norm = (t - t[0]) / (t[-1] - t[0])
    else:
        t_norm = np.zeros_like(t, dtype=np.float32)

    tencode = np.full((3, height, width), base_val, dtype=np.float32)

    # Valid indices
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return tencode / 255.0 if normalize else tencode

    xv = x[valid]
    yv = y[valid]
    polv = pol[valid]
    tn = t_norm[valid]

    # R: 255 for positive polarity, else 0
    tencode[0, yv, xv] = 255.0 * polv
    # G: 255 * (1 - t_norm) -> newest events darker
    tencode[1, yv, xv] = 255.0 * (1.0 - tn)
    # B: 255 for negative polarity, else 0
    tencode[2, yv, xv] = 255.0 * (1.0 - polv)

    if normalize:
        tencode = tencode / 255.0

    return tencode


def save_tencode_png(tencode: np.ndarray, out_path: Path, normalize: bool):
    """
    Save a Tencode frame (3,H,W) as an RGB PNG (H,W,3).
    """
    if normalize:
        img = np.clip(tencode * 255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(tencode, 0, 255).astype(np.uint8)

    img = np.transpose(img, (1, 2, 0))  # (3,H,W) -> (H,W,3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


# --------------------
# Main
# --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Tencode images from an event HDF5 file (streaming)."
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default="/media/adam/vprdatasets/data/event-datasets/brisbane_event/sunrise/sunrise.hdf5",
        help="Path to input .hdf5 event file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./tencode_out",
        help="Output directory for Tencode PNGs",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=30.0,
        help="Temporal window size in milliseconds (e.g., 30 ms).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1e-9,
        help=(
            "Scale factor to convert raw timestamps to seconds. "
            "Use 1e-9 for ns, 1e-6 for us, 1.0 for seconds."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help=(
            "Optional absolute start time in seconds (after scaling) "
            "for the first window. If earlier than the file's first "
            "timestamp, the script starts at the first event instead."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum number of Tencode frames to generate. "
            "If None, generate all frames."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Number of events to read per HDF5 chunk.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Override sensor height; if 0, infer from data.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Override sensor width; if 0, infer from data.",
    )
    parser.add_argument(
        "--white-frame",
        action="store_true",
        help="Initialise frame as white (255) instead of black (0).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize output to [0,1] internally (rescaled to [0,255] for PNG).",
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    out_dir = Path(args.out_dir)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        N = len(t_dset)
        print(f"[INFO] Number of events: {N}")

        # Resolution
        if args.height > 0 and args.width > 0:
            H, W = args.height, args.width
            print(f"[INFO] Using provided resolution HxW = {H}x{W}")
        else:
            H, W = infer_resolution_stream(x_dset, y_dset, chunk_size=args.chunk_size)

        # Time range (scaled)
        t_raw_start = t_dset[0]
        t_raw_end = t_dset[N - 1]
        t_start = float(t_raw_start) * args.time_scale
        t_end = float(t_raw_end) * args.time_scale
        dt = args.dt_ms / 1000.0

        print(f"[INFO] Raw time range: [{t_raw_start}, {t_raw_end}]")
        print(f"[INFO] Scaled time range (s): [{t_start:.6f}, {t_end:.6f}]")
        print(f"[INFO] Using fixed window dt = {dt*1000:.1f} ms")

        # Determine actual window start
        if args.start_time is not None:
            window_start = max(args.start_time, t_start)
            print(
                f"[INFO] Requested start_time={args.start_time:.6f} s, "
                f"using window_start={window_start:.6f} s"
            )
        else:
            window_start = t_start
            print(f"[INFO] Using first event time as window_start={window_start:.6f} s")

        # Estimate frames from chosen window_start
        if t_end > window_start:
            n_est_frames = int(np.ceil((t_end - window_start) / dt))
        else:
            n_est_frames = 1

        if args.max_frames is None:
            total_frames = n_est_frames
            print(f"[INFO] Generating all frames: {total_frames}")
        else:
            total_frames = min(n_est_frames, args.max_frames)
            print(f"[INFO] Estimated frames: {n_est_frames}, capped at {total_frames}")

        # Buffers for streaming
        x_buf = np.empty(0, dtype=np.int64)
        y_buf = np.empty(0, dtype=np.int64)
        t_buf = np.empty(0, dtype=np.float64)  # seconds
        p_buf = np.empty(0, dtype=np.int8)

        read_idx = 0
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Tencode frames")

        while window_start < t_end and frame_idx < total_frames:
            window_end = window_start + dt

            # Fill buffer with new data until we've read past window_end or hit EOF
            while True:
                if t_buf.size > 0 and t_buf[-1] >= window_end:
                    break
                if read_idx >= N:
                    break

                end_idx = min(N, read_idx + args.chunk_size)
                x_chunk = x_dset[read_idx:end_idx][:]
                y_chunk = y_dset[read_idx:end_idx][:]
                t_raw_chunk = t_dset[read_idx:end_idx][:]
                p_chunk = p_dset[read_idx:end_idx][:]

                t_chunk = t_raw_chunk.astype(np.float64) * args.time_scale

                if x_buf.size == 0:
                    x_buf = x_chunk.astype(np.int64)
                    y_buf = y_chunk.astype(np.int64)
                    t_buf = t_chunk
                    p_buf = p_chunk.astype(np.int8)
                else:
                    x_buf = np.concatenate((x_buf, x_chunk.astype(np.int64)))
                    y_buf = np.concatenate((y_buf, y_chunk.astype(np.int64)))
                    t_buf = np.concatenate((t_buf, t_chunk))
                    p_buf = np.concatenate((p_buf, p_chunk.astype(np.int8)))

                read_idx = end_idx

            # Drop any events before current window_start
            if t_buf.size > 0:
                valid_mask = t_buf >= window_start
                x_buf = x_buf[valid_mask]
                y_buf = y_buf[valid_mask]
                t_buf = t_buf[valid_mask]
                p_buf = p_buf[valid_mask]

            if t_buf.size == 0 and read_idx >= N:
                # No more events for remaining windows
                break

            # Events for current window: t in [window_start, window_end)
            mask_window = (t_buf >= window_start) & (t_buf < window_end)
            x_win = x_buf[mask_window]
            y_win = y_buf[mask_window]
            t_win = t_buf[mask_window]
            p_win = p_buf[mask_window]

            # Leftover for future windows: t >= window_end
            mask_leftover = t_buf >= window_end
            x_buf = x_buf[mask_leftover]
            y_buf = y_buf[mask_leftover]
            t_buf = t_buf[mask_leftover]
            p_buf = p_buf[mask_leftover]

            tencode = tencode_numpy(
                x_win,
                y_win,
                t_win,
                p_win,
                height=H,
                width=W,
                white_frame=args.white_frame,
                normalize=args.normalize,
            )

            out_path = out_dir / f"tencode_{frame_idx:04d}.png"
            save_tencode_png(tencode, out_path, normalize=args.normalize)

            frame_idx += 1
            window_start = window_end
            pbar.update(1)

        pbar.close()
        print(f"[INFO] Generated {frame_idx} Tencode frames.")


if __name__ == "__main__":
    main()
