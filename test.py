"""
Event-based Depth Estimation Testing Script

This script performs testing/evaluation of depth estimation models on event-based datasets.
It supports both DAv2 and RecDAv2 models and can process various input types (events, RGB).
The script evaluates models on different datasets and outputs metrics and visualizations.
"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import cmapy
import tqdm
import json

from torch import autocast

from losses import normalized_depth_scale_and_shift
from dataset import fetch_dataloader
from models import fetch_model
from evaluation import add_to_metrics, prepare_prediction_data, prepare_target_data, prepare_target_data_torch

# Constants for normalization (ImageNet standard values)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@torch.no_grad()
def run(data, model, global_context):
    """
    Run inference on a batch of data using the specified model.
    
    Args:
        data (dict): Input data batch containing depth, events, RGB, etc.
        model: The neural network model for depth estimation
        global_context (dict): Context containing args, model config, device info, etc.
        
    Returns:
        tuple: (metrics, prediction, maemap) - evaluation metrics, depth prediction, and error map
    """
    model.eval()

    # Extract configuration from global context
    args = global_context['args']
    model_config = global_context['model']
    model_name = model_config['model_type']
    input_type = model_config['input_type'] if 'input_type' in model_config else 'events'
    device = global_context['device']
    autocast_device = global_context['autocast_device']

    # Validate input type
    assert input_type in ['events', 'rgb'], f"Input type {input_type} not implemented yet"
    input_key = {'events': 'depth_aligned_events', 'rgb': 'rgb'}[input_type]

    # Apply input scaling if specified
    if args.iscale != 1:
        for key in ['rgb', 'depth_aligned_events', 'rgb_aligned_events']:
            if key in data:
                B, T, C, H, W = data[key].shape
                data[key] = F.interpolate(
                    data[key].reshape(B*T, C, H, W), 
                    scale_factor=1./args.iscale, 
                    mode='nearest'
                )
                new_H, new_W = data[key].shape[-2], data[key].shape[-1]
                data[key] = data[key].reshape(B, T, C, new_H, new_W)

    # Apply output scaling if specified
    if args.oscale != 1:
        for key in ['depth']:
            if key in data:
                B, T, C, H, W = data[key].shape
                data[key] = F.interpolate(
                    data[key].reshape(B*T, C, H, W), 
                    scale_factor=1./args.oscale, 
                    mode='nearest'
                )
                new_H, new_W = data[key].shape[-2], data[key].shape[-1]
                data[key] = data[key].reshape(B, T, C, new_H, new_W)

    # Move data to device
    for key in ['rgb', 'depth_aligned_events', 'rgb_aligned_events', 'depth']:
        if key in data:
            data[key] = data[key].to(device)

    # Apply padding (currently set to 0)
    pad_height, pad_width = 0, 0
    padding = [pad_width//2, pad_width - pad_width//2, pad_height//2, pad_height - pad_height//2]
    
    for key in ['depth_aligned_events', 'rgb', 'rgb_aligned_events']:
        if key in data:
            B, T, C, H, W = data[key].shape
            data[key] = F.pad(data[key].reshape(B*T, C, H, W), padding, mode='replicate')
            new_H, new_W = data[key].shape[-2], data[key].shape[-1]
            data[key] = data[key].reshape(B, T, C, new_H, new_W)

    # Model inference based on model type
    if model_name == 'DAv2':
        # DAv2 model inference (assumes batch size = 1 for testing)
        with autocast(autocast_device, enabled=args.mixed_precision):
            B, T, C, H, W = data[input_key].shape
            input_tensor = torch.clone(data[input_key]).reshape(B*T, C, H, W).to(device)
            prediction = model.infer_image(input_tensor)
    elif model_name == 'RecDAv2':
        # RecDAv2 model inference (recurrent model with state)
        with autocast(autocast_device, enabled=args.mixed_precision):
            if 'recdav2_state' not in global_context['tmp']:
                global_context['tmp']['recdav2_state'] = None

            B, T, C, H, W = data[input_key].shape
            prediction_list = []
            for t in range(T):
                pred, prev_states = model.infer_image(
                    torch.clone(data[input_key][:, t]).to(device), 
                    prev_states=global_context['tmp']['recdav2_state']
                )
                prediction_list.append(pred)
                global_context['tmp']['recdav2_state'] = prev_states
            
            prediction = torch.stack(prediction_list, dim=1)
            B, T, C, H, W = prediction.shape
            prediction = prediction.reshape(B*T, C, H, W)
    else:
        raise ValueError(f"Model {model_name} not implemented yet")

    # Remove padding from prediction
    height, width = prediction.shape[-2:]
    crop_coords = [padding[2], height-padding[3], padding[0], width-padding[1]]
    prediction = prediction[..., crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

    # Remove padding from input data
    for key in ['depth_aligned_events', 'rgb', 'rgb_aligned_events']:
        if key in data:
            data[key] = data[key][..., crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

    # Interpolate prediction to match target depth size if needed
    if args.iscale != 1 and args.iscale/args.oscale != 1:
        prediction = F.interpolate(
            prediction, 
            (data['depth'].shape[-2], data['depth'].shape[-1]), 
            mode='bilinear'
        )

    # Initialize metrics dictionary
    metrics = {}

    # Apply scale and shift normalization if enabled
    if args.use_scaleshift:
        B, T, C, H, W = data['depth'].shape
        target_depth = prepare_target_data_torch(
            data['depth'].reshape(B*T, C, H, W).squeeze(1).detach(), 
            args.clip_distance
        )
        scale, shift = normalized_depth_scale_and_shift(
            prediction.squeeze(1), target_depth, target_depth > 0
        )
        prediction = scale * prediction + shift

    # Prepare prediction data (log depth or clipped)
    if args.use_logdepth:
        pred_numpy = prepare_prediction_data(
            prediction.detach().squeeze().cpu().numpy(), 
            args.clip_distance, 
            args.reg_factor
        )
    else:
        pred_numpy = np.clip(
            prediction.detach().squeeze().cpu().numpy(), 
            0, 
            args.clip_distance
        )

    # Prepare target depth data
    B, T, C, H, W = data['depth'].shape
    target_depth = prepare_target_data(
        data['depth'].reshape(B*T, C, H, W).detach().squeeze().cpu().numpy(), 
        args.clip_distance
    )

    # Create depth mask (currently allows all pixels)
    depth_mask = (np.ones_like(target_depth) > 0)

    # Calculate overall metrics
    metrics = add_to_metrics(-1, metrics, target_depth, pred_numpy, depth_mask, prefix="_") 

    # Calculate metrics for different depth thresholds
    for depth_threshold in [10, 20, 30]:
        threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
        metrics = add_to_metrics(
            -1, metrics, target_depth, pred_numpy, 
            depth_mask & threshold_mask, prefix=f"_{depth_threshold}_"
        )
    
    # Calculate mean absolute error map
    error_map = np.abs(target_depth - pred_numpy)
   
    return metrics, prediction, error_map

def write_csv_header(file, metrics):
    """
    Write CSV header with dataset information and metric names.
    
    Args:
        file: File handle for writing
        metrics (dict): Dictionary of metrics to include in header
    """
    header = "DATASET,DATAPATH,SEQ,MODEL,MODEL_PATH,ISCALE,OSCALE,"
    keys = list(metrics.keys())
    if len(keys) > 0:
        for k in keys[:-1]:
            header += f"{k.upper()},"
        header += f"{keys[-1].upper()}\n"

    file.write(header)


def write_csv_row(file, args, config, metrics, seq):
    """
    Write a single row of results to CSV file.
    
    Args:
        file: File handle for writing
        args: Command line arguments
        config (dict): Configuration dictionary
        metrics (dict): Dictionary of metric values
        seq (str): Sequence name
    """
    # Extract configuration parameters
    datapath = config['data_loader']['datapath'] if 'datapath' in config['data_loader'] else args.datapath
    model_config = config['model']
    model_name = model_config['model_type']
    checkpoint_path = model_config['checkpoint_path'] if 'checkpoint_path' in model_config else args.loadmodel
    
    # Build parameter string
    parameters = f"{config['data_loader']['dataset']},{datapath},{seq},{model_name},{checkpoint_path},{args.iscale},{args.oscale},"
    keys = list(metrics.keys())

    # Add metric values
    if len(keys) > 0:
        for k in keys[:-1]: 
            parameters += f"{metrics[k]:.6f},"
        parameters += f"{metrics[keys[-1]]:.6f}\n"

    file.write(parameters)


def print_and_save(args, config, metrics, seq="MEAN"):
    """
    Print metrics to console and optionally save to CSV file.
    
    Args:
        args: Command line arguments
        config (dict): Configuration dictionary
        metrics (dict): Dictionary of metric values
        seq (str): Sequence name (default: "MEAN")
    """
    print(f"{seq} Metrics:")

    # Print metric names
    metric_names = ''
    for k in metrics:
        metric_names += f" {k.upper()} &"
    print(metric_names)

    # Print metric values
    metric_values = ''
    for k in metrics:
        metric_values += f" {metrics[k]:.6f} &"
    print(metric_values)

    # Check if the parent dir for CSV path exists
    if args.csv_path is not None:
        parent_dir = os.path.dirname(args.csv_path)
        if parent_dir != '' and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    # Save to CSV if path specified
    if args.csv_path is not None:
        if os.path.exists(args.csv_path):
            csv_file = open(args.csv_path, "a")
        else:
            csv_file = open(args.csv_path, "w")
            write_csv_header(csv_file, metrics)
        
        write_csv_row(csv_file, args, config, metrics, seq)
        csv_file.close()    

def setup_argument_parser():
    """
    Set up command line argument parser with all available options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Event-based Depth Estimation Testing')

    # Data and model paths
    parser.add_argument('--datapath', default=None, help='Path to dataset')
    parser.add_argument('--loadmodel', default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file. If not specified, config from model folder is used')
    
    # Output settings
    parser.add_argument('--outdir', default=None, help='Output directory for visualizations')
    parser.add_argument('--csv_path', default=None, help='Path to CSV file for saving results')
    parser.add_argument('--erase_csv', action='store_true', default=False,
                        help='Erase existing CSV file')
    
    # Processing parameters
    parser.add_argument('--iscale', type=int, default=1, help='Input downsampling factor')
    parser.add_argument('--oscale', type=int, default=1, help='Output downsampling factor')
    parser.add_argument('--dilation', type=int, default=0, help='Dilation factor for depth visualization')
    
    # Evaluation settings
    parser.add_argument('--valsize', default=0, type=int, help='Number of frames to evaluate (0 = all)')
    parser.add_argument('--seqid', type=int, default=-1, help='Select specific sequence ID (-1 = all)')
    parser.add_argument('--clip_distance', type=float, default=80.0,
                        help='Maximum depth distance for clipping')
    
    # Depth processing options
    parser.add_argument('--reg_factor', type=float, default=3.70378,
                        help='Log depth scale factor')
    parser.add_argument('--use_logdepth', action='store_true', default=False,
                        help='Use logarithmic depth processing')
    parser.add_argument('--use_scaleshift', action='store_true', default=False,
                        help='Use scale and shift normalization')
    
    # Training and system settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA even if available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed for reproducibility')
    
    # Debugging and development
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='Overfitting mode (for debugging)')
    parser.add_argument('--discard_train_args', action='store_true', default=False,
                        help='Ignore training arguments from config')

    return parser


def load_and_merge_config(args):
    """
    Load model checkpoint and configuration, merging with command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (checkpoint_dict, merged_config)
    """
    if args.loadmodel is not None:
        print(f"Loading model from {args.loadmodel}")
        
        # Load checkpoint
        ckpt = torch.load(args.loadmodel, map_location='cpu')
        
        # Load external config if provided
        external_config = {}
        if args.config is not None:
            with open(args.config, 'r') as f:
                external_config = json.load(f)
        
        # Get config from checkpoint or model folder
        if 'config' in ckpt:
            config = ckpt['config']
        else:
            # Try to find config in model folder
            model_folder = os.path.dirname(args.loadmodel)
            config_file = os.path.join(model_folder, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError("No config file found in model folder")

        # Merge external config (excluding model section)
        for key in external_config:
            if key not in ['model']:
                config[key] = external_config[key]

        # Update model checkpoint path
        config['model']['checkpoint_path'] = args.loadmodel

        # Merge training arguments if not discarded
        if 'trainer' in config and not args.discard_train_args:
            args.use_logdepth = config['trainer'].get('use_logdepth', args.use_logdepth)
            args.use_scaleshift = config['trainer'].get('use_scaleshift', args.use_scaleshift)
            args.reg_factor = config['trainer'].get('reg_factor', args.reg_factor)
            args.clip_distance = config['trainer'].get('clip_distance', args.clip_distance)
    else:
        ckpt = None
        
        # Load config file
        if args.config is not None:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError("Either --loadmodel or --config must be specified")

    return ckpt, config


def setup_device_and_seeds(args):
    """
    Setup device (CPU/CUDA) and random seeds for reproducibility.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (device, autocast_device_string)
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"Using device {device}")
    autocast_device = str(device).split(':')[0]
    
    return device, autocast_device


def create_output_directories(args):
    """
    Create output directory structure for saving results.
    
    Args:
        args: Command line arguments
    """
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


def save_batch_outputs(args, datablob, prediction, error_map, batch_idx, dataset_name):
    """
    Save visualization outputs for a single batch.
    
    Args:
        args: Command line arguments
        datablob (dict): Input data batch
        prediction (torch.Tensor): Model prediction
        error_map (np.ndarray): Error map
        batch_idx (int): Batch index for naming files
        dataset_name (str): Name of dataset for gamma correction
    """
    if args.outdir is None:
        return
        
    # Create subdirectories
    subdirs = ['prediction', 'rgb', 'depth', 'evrep', 'raw', 'maemap']
    for dirname in subdirs:
        subdir_path = os.path.join(args.outdir, dirname)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    # Save RGB image if available
    if 'rgb' in datablob:
        rgb_image = cv2.cvtColor(
            (255 * datablob['rgb']).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8), 
            cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(os.path.join(args.outdir, "rgb", f'{batch_idx}.png'), rgb_image)

    # Save ground truth depth
    depth_gt = datablob['depth'].squeeze().detach().cpu().numpy()
    
    # Apply dilation if specified
    if args.dilation > 0:
        kernel = np.ones((args.dilation, args.dilation))
        depth_gt = cv2.dilate(depth_gt, kernel)
    
    depth_colormap = cv2.applyColorMap(
        (255 * depth_gt / depth_gt.max()).astype(np.uint8), 
        cmapy.cmap('magma')
    )
    cv2.imwrite(os.path.join(args.outdir, "depth", f'{batch_idx}.png'), depth_colormap)
    
    # Save error map
    error_squeezed = error_map.squeeze()
    error_colormap = cv2.applyColorMap(
        (255 * (np.clip(error_squeezed, 0, depth_gt.max()) / depth_gt.max())).astype(np.uint8), 
        cmapy.cmap('Spectral_r')
    )
    error_colormap[depth_gt == 0] = 0
    error_colormap[depth_gt > args.clip_distance] = 0
    cv2.imwrite(os.path.join(args.outdir, "maemap", f'{batch_idx}.png'), error_colormap)

    # Save prediction
    pred_numpy = prediction.squeeze().detach().cpu().numpy()
    
    # Save raw prediction as numpy array
    np.save(os.path.join(args.outdir, "raw", f'event_tensor_{batch_idx:010d}.npy'), pred_numpy)
    
    # Create colorized prediction
    gamma = 0.2 if dataset_name == 'mvsec' else 1.0
    pred_colormap = cv2.applyColorMap(
        (255 * (np.clip(pred_numpy, 0, args.clip_distance) / args.clip_distance) ** gamma).astype(np.uint8), 
        cmapy.cmap('magma')
    )
    cv2.imwrite(os.path.join(args.outdir, "prediction", f'{batch_idx}.png'), pred_colormap)

    # Save event representation
    event_repr = datablob['depth_aligned_events'].squeeze().detach().cpu().numpy()
    if event_repr.shape[0] != 3:
        event_repr = np.mean(event_repr, axis=0)        
        event_colormap = cv2.applyColorMap(
            (255 * event_repr / event_repr.max()).astype(np.uint8), 
            cmapy.cmap('Spectral_r')
        )
    else:
        event_colormap = event_repr.transpose(1, 2, 0) * 255

    cv2.imwrite(os.path.join(args.outdir, "evrep", f'{batch_idx}.png'), event_colormap)


def main():
    """Main function for running depth estimation evaluation."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Clean up CSV file if requested
    if args.csv_path is not None and args.erase_csv:
        if os.path.exists(args.csv_path):
            os.remove(args.csv_path)

    # Setup device and reproducibility
    device, autocast_device = setup_device_and_seeds(args)
    
    # Create output directories
    create_output_directories(args)

    # Load model and configuration
    ckpt, config = load_and_merge_config(args)

    # Initialize data loaders and model
    data_loaders_dict = fetch_dataloader(config['data_loader'], args, test=True)
    model = fetch_model(config['model'], args, device, test=True, _state_dict=ckpt)        
    dataset_name = config['data_loader']['dataset']

    # Storage for sequence metrics
    metrics_sequence_dict = {}

    # Create global context for model inference
    global_context = {
        'args': args,
        'model': config['model'],
        'model_name': config['model']['model_type'],
        'device': device,
        'autocast_device': autocast_device,
        'tmp': {'reset_state': True}
    }

    # Filter sequences if specific sequence ID requested
    if args.seqid >= 0:
        seqs = list(data_loaders_dict.keys())
        seq = seqs[args.seqid]
        data_loaders_dict = {seq: data_loaders_dict[seq]}

    # Process each sequence
    for seq in data_loaders_dict:
        data_loader = data_loaders_dict[seq]
        pbar = tqdm.tqdm(total=len(data_loader), desc=f"Processing {seq}")
        val_len = min(len(data_loader), args.valsize) if args.valsize > 0 else len(data_loader)
        
        # Initialize metrics storage for this sequence
        metrics_sequence = {}
        
        # Reset context memory for each sequence
        global_context['tmp'] = {'reset_state': True}

        # Process each batch in the sequence
        for batch_idx, datablob in enumerate(data_loader):
            if batch_idx >= val_len:
                break
            
            # Run model inference
            metrics, prediction, error_map = run(datablob, model, global_context)

            # Save outputs if requested
            save_batch_outputs(args, datablob, prediction, error_map, batch_idx, dataset_name)

            # Accumulate metrics
            for k in metrics:
                if k not in metrics_sequence:
                    metrics_sequence[k] = []
                metrics_sequence[k].append(metrics[k])

                if args.verbose:
                    print(f"{batch_idx}) {k}: {metrics[k]}")

            pbar.update(1)
        pbar.close()

        # Calculate average metrics for this sequence
        for k in metrics_sequence:
            metrics_sequence[k] = np.nanmean(np.array(metrics_sequence[k]))

        metrics_sequence_dict[seq] = metrics_sequence
        print_and_save(args, config, metrics_sequence, seq)

    # Calculate overall average metrics across all sequences
    metrics_mean = {}
    for seq in metrics_sequence_dict:
        for k in metrics_sequence_dict[seq]:
            if k not in metrics_mean:
                metrics_mean[k] = []
            metrics_mean[k].append(np.nanmean(np.array(metrics_sequence_dict[seq][k])))
    
    for k in metrics_mean:
        metrics_mean[k] = np.nanmean(metrics_mean[k])

    print_and_save(args, config, metrics_mean)   


if __name__ == '__main__':
    main()
