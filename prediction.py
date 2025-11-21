#!/usr/bin/env python3
"""
Event-based Depth Prediction Script (Tencode + DAv2/RecDAv2)

- Loads a DAv2 / RecDAv2 depth model from checkpoint/config
- Reads event-based .hdf5 files
- Converts events to Tencode representation in fixed time windows
- Runs depth inference and saves depth images as PNGs

Assumes:
- HDF5 layout: group "events" with datasets "x", "y", "t", "p"
- Timestamps in nanoseconds (default time_scale=1e-9)
"""

from __future__ import print_function
import argparse
from html import parser
import os
from pathlib import Path
import json
import random

import h5py
import numpy as np
import cv2
import torch
from torch import autocast
from tqdm import tqdm
import cmapy

from models import fetch_model  # from the DepthAnyEvent repo


# ==========================
# HDF5 + Tencode utilities
# ==========================

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


def tencode_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    white_frame: bool = False,
    normalize: bool = True,
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

    # Normalise time to [0,1] within this window
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


def save_depth_png(depth_map: np.ndarray,
                   out_path: Path,
                   clip_distance: float,
                   gamma: float):
    """
    depth_map: linear depth in meters (2D array) after conversion.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    depth = np.nan_to_num(depth_map, nan=0.0, posinf=clip_distance, neginf=0.0)
    depth = np.clip(depth, 0.0, clip_distance)

    # normalize and apply gamma
    depth_norm = depth / clip_distance if clip_distance > 0 else depth
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_norm = depth_norm ** gamma

    img_gray = (depth_norm * 255.0).astype(np.uint8)
    # use the same magma colormap as the repo
    img_color = cv2.applyColorMap(img_gray, cmapy.cmap('magma'))

    ok = cv2.imwrite(str(out_path), img_color)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")

# ==========================
# Config / model utilities
# ==========================

def load_and_merge_config(args):
    """
    Load model checkpoint and configuration, merging with command line arguments.
    Returns: (ckpt, config_dict)
    """
    if args.loadmodel is not None:
        print(f"[INFO] Loading model from {args.loadmodel}")
        ckpt = torch.load(args.loadmodel, map_location='cpu')

        # External config (optional override)
        external_config = {}
        if args.config is not None:
            with open(args.config, 'r') as f:
                external_config = json.load(f)

        # Config from checkpoint or model folder
        if 'config' in ckpt:
            config = ckpt['config']
        else:
            model_folder = os.path.dirname(args.loadmodel)
            config_file = os.path.join(model_folder, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError("No config file found in model folder and none in checkpoint.")

        # Merge external config (excluding model section)
        for key in external_config:
            if key != 'model':
                config[key] = external_config[key]

        # Update model checkpoint path
        config['model']['checkpoint_path'] = args.loadmodel

    else:
        ckpt = None
        if args.config is None:
            raise ValueError("Either --loadmodel or --config must be specified")
        with open(args.config, 'r') as f:
            config = json.load(f)

    if 'trainer' in config and not args.discard_train_args:
        tr = config['trainer']
        args.use_logdepth   = tr.get('use_logdepth',   args.use_logdepth)
        args.reg_factor     = tr.get('reg_factor',     args.reg_factor)
        args.clip_distance  = tr.get('clip_distance',  args.clip_distance)

    return ckpt, config


def setup_device_and_seeds(args):
    """
    Setup device (CPU/CUDA) and random seeds for reproducibility.
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"[INFO] Using device {device}")
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    return device, autocast_device

def convert_pred_for_vis(prediction: np.ndarray,
                         use_logdepth: bool,
                         clip_distance: float,
                         reg_factor: float = 3.70378) -> np.ndarray:
    """
    Convert network prediction to linear depth in meters for visualisation.

    - If use_logdepth: treat prediction as log-depth in [0,1] and convert
      like evaluation.prepare_prediction_data (but without any GT).
    - Otherwise: assume prediction is already linear depth.
    """
    pred = prediction.astype(np.float32)

    if use_logdepth:
        # from prepare_prediction_data
        # 1) log-depth -> normalized linear depth
        pred = np.exp(reg_factor * (pred - 1.0))

        # 2) normalize by its own max (scale-invariant)
        valid = pred[~np.isnan(pred)]
        max_val = valid.max() if valid.size > 0 else 0.0
        if max_val > 0:
            pred = pred / max_val

        # 3) scale to clip_distance (meters)
        pred = pred * clip_distance

    else:
        # treat as already-linear depth
        pred = np.clip(pred, 0.0, clip_distance)

    return pred

# ==========================
# Core processing
# ==========================

@torch.no_grad()
def process_hdf5_file(
    hdf5_path: Path,
    model,
    model_name: str,
    device: torch.device,
    autocast_device: str,
    args,
):
    """
    For a single HDF5 event file:
    - stream events
    - build Tencode frames over time windows
    - run depth prediction
    - save PNG per frame
    """
    print(f"[INFO] Processing {hdf5_path}")
    out_root = Path(args.outdir)
    file_stem = hdf5_path.stem
    file_outdir = out_root / file_stem

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

        # Time range (scaled to seconds)
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

        # Streaming buffers
        x_buf = np.empty(0, dtype=np.int64)
        y_buf = np.empty(0, dtype=np.int64)
        t_buf = np.empty(0, dtype=np.float64)  # seconds
        p_buf = np.empty(0, dtype=np.int8)

        read_idx = 0
        frame_idx = 0
        prev_states = None  # for RecDAv2

        pbar = tqdm(total=total_frames, desc=f"{file_stem}: frames")

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

            # Drop events before window_start
            if t_buf.size > 0:
                valid_mask = t_buf >= window_start
                x_buf = x_buf[valid_mask]
                y_buf = y_buf[valid_mask]
                t_buf = t_buf[valid_mask]
                p_buf = p_buf[valid_mask]

            if t_buf.size == 0 and read_idx >= N:
                # No more events
                break

            # Events for current window: t in [window_start, window_end)
            mask_window = (t_buf >= window_start) & (t_buf < window_end)
            x_win = x_buf[mask_window]
            y_win = y_buf[mask_window]
            t_win = t_buf[mask_window]
            p_win = p_buf[mask_window]

            # Leftover: t >= window_end
            mask_leftover = t_buf >= window_end
            x_buf = x_buf[mask_leftover]
            y_buf = y_buf[mask_leftover]
            t_buf = t_buf[mask_leftover]
            p_buf = p_buf[mask_leftover]

            # Build Tencode
            tencode = tencode_numpy(
                x_win,
                y_win,
                t_win,
                p_win,
                height=H,
                width=W,
                white_frame=args.white_frame,
                normalize=True,  # model expects normalized representation
            )

            # To torch
            ev_tensor = torch.from_numpy(tencode).float().unsqueeze(0).to(device)  # (1,3,H,W)

            # Inference (no grad, via decorator)
            with autocast(autocast_device, enabled=args.mixed_precision):
                if model_name == 'DAv2':
                    pred = model.infer_image(ev_tensor)  # (1,1,H,W)
                elif model_name == 'RecDAv2':
                    pred, prev_states = model.infer_image(ev_tensor, prev_states=prev_states)
                else:
                    raise ValueError(f"Model {model_name} not implemented in this script.")

            # raw network output (log-depth or linear)
            pred_np_raw = pred.squeeze().detach().cpu().numpy()

            # convert to linear depth in meters, respecting use_logdepth/reg_factor/clip_distance
            depth_for_vis = convert_pred_for_vis(
                pred_np_raw,
                use_logdepth=args.use_logdepth,
                clip_distance=args.clip_distance,
                reg_factor=args.reg_factor,
            )

            out_path = file_outdir / f"depth_{frame_idx:06d}.png"
            save_depth_png(
                depth_for_vis,
                out_path,
                clip_distance=args.clip_distance,
                gamma=args.gamma,
            )

            # Free per-frame tensors explicitly (belt-and-braces)
            del ev_tensor, pred
            torch.cuda.empty_cache() if device.type == 'cuda' else None

            frame_idx += 1
            window_start = window_end
            pbar.update(1)

        pbar.close()
        print(f"[INFO] {hdf5_path}: generated {frame_idx} depth frames into {file_outdir}")


# ==========================
# Argument parser + main
# ==========================

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Event-based Depth Prediction from HDF5 (Tencode + DAv2/RecDAv2)')

    # Data / model
    parser.add_argument('--hdf5-path', type=str, required=True,
                        help='Path to a .hdf5 file or a directory containing .hdf5 files')
    parser.add_argument('--loadmodel', default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file. If not specified, config from model folder/checkpoint is used')

    # Output
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for depth PNGs')

    # Event / time windowing
    parser.add_argument('--dt-ms', type=float, default=30.0,
                        help='Temporal window size in milliseconds (e.g., 30 ms).')
    parser.add_argument('--time-scale', type=float, default=1e-9,
                        help='Scale factor to convert raw timestamps to seconds (1e-9 for ns).')
    parser.add_argument('--start-time', type=float, default=None,
                        help='Optional absolute start time in seconds (after scaling).')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames per file. If None, generate all.')
    parser.add_argument('--chunk-size', type=int, default=200_000,
                        help='Number of events to read per HDF5 chunk.')

    # Depth visualization
    parser.add_argument('--clip-distance', type=float, default=80.0,
                        help='Max depth distance for visualization clipping.')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Gamma for depth visualization.')
    parser.add_argument('--use_logdepth', action='store_true', default=False,
                        help='Model outputs log-depth in [0,1] instead of linear meters')
    parser.add_argument('--reg_factor', type=float, default=3.70378,
                        help='Regularization factor for log-depth conversion')
    parser.add_argument('--discard_train_args', action='store_true', default=False,
                        help='Ignore trainer settings in config (use CLI values instead)')

    # Sensor resolution override
    parser.add_argument('--height', type=int, default=0,
                        help='Override sensor height; if 0, infer from data.')
    parser.add_argument('--width', type=int, default=0,
                        help='Override sensor width; if 0, infer from data.')

    # System / misc
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA even if available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed for reproducibility')

    parser.add_argument('--white-frame', action='store_true',
                        help='Use white background for Tencode where no events.')


    return parser


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    os.makedirs(args.outdir, exist_ok=True)

    # Device / seeds
    device, autocast_device = setup_device_and_seeds(args)

    # Load model + config
    ckpt, config = load_and_merge_config(args)
    model = fetch_model(config['model'], args, device, test=True, _state_dict=ckpt)
    model_name = config['model']['model_type']
    model.eval()

    # Determine list of HDF5 files
    hdf5_path = Path(args.hdf5_path)
    if hdf5_path.is_dir():
        hdf5_files = sorted(hdf5_path.glob("*.hdf5"))
    else:
        hdf5_files = [hdf5_path]

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found at {hdf5_path}")

    for fpath in hdf5_files:
        process_hdf5_file(
            fpath,
            model,
            model_name,
            device,
            autocast_device,
            args,
        )


if __name__ == '__main__':
    main()
