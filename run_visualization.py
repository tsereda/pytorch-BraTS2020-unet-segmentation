#!/usr/bin/env python3
"""
Enhanced Visualization for Brain Tumor Segmentation

This script provides improved visualization for brain tumor segmentation results,
with better color mapping to distinguish between tumor regions and background.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path


def visualize_tumor_segmentation(
    model_path: str,
    data_path: str,
    output_path: str,
    num_samples: int = 3,
    use_cuda: bool = True
):
    """
    Main function to visualize tumor segmentation results
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to dataset
        output_path: Path to save visualizations
        num_samples: Number of samples to visualize
        use_cuda: Whether to use CUDA if available
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Import required modules here to avoid import errors
    try:
        from unet3d_model import UNet3D
        from dataset import BraTSDataset
    except ImportError:
        print("Error: Could not import required modules.")
        print("Make sure unet3d_model.py and dataset.py are in the current directory or PYTHONPATH.")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize model
    model = UNet3D(in_channels=3, num_classes=4)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        mean_iou = checkpoint.get('mean_iou', 'unknown')
        print(f"Loaded model from epoch {epoch} with mean IoU: {mean_iou}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model = model.to(device)
    model.eval()
    
    # Visualize samples
    print(f"Generating visualizations for {num_samples} samples...")
    for i in range(num_samples):
        print(f"Visualizing sample {i+1}/{num_samples}...")
        visualize_sample(
            model=model,
            data_path=data_path,
            output_path=output_path,
            sample_idx=i,
            device=device
        )
    
    print(f"Visualizations saved to {output_path}")


def visualize_sample(model, data_path, output_path, sample_idx=0, device=None):
    """
    Visualize a sample prediction with enhanced color mapping to better show class differences
    
    Args:
        model: Trained model
        data_path: Path to dataset
        output_path: Path to save visualization
        sample_idx: Index of sample to visualize
        device: Torch device (will use CUDA if available when None)
    """
    # Import BraTSDataset
    try:
        from dataset import BraTSDataset
    except ImportError:
        print("Error: Could not import dataset module.")
        return
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a sample from validation set
    try:
        val_dataset = BraTSDataset(data_path, mode='val', transform=None)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Check if the dataset path is correct and structured properly.")
        return
    
    if not val_dataset.img_files:
        print("No validation samples found.")
        return
    
    # Get sample
    sample_idx = min(sample_idx, len(val_dataset) - 1)
    try:
        img, mask = val_dataset[sample_idx]
    except Exception as e:
        print(f"Error loading sample {sample_idx}: {e}")
        return
    
    # Add batch dimension and move to device
    img = img.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        try:
            output = model(img)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return
    
    # Convert tensors to numpy arrays
    img = img.squeeze().cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Define a custom colormap for better visualization
    # 0: Background (dark blue)
    # 1: Necrotic core (yellow)
    # 2: Edema (green)
    # 3: Enhancing tumor (red)
    colors = [
        [0.1, 0.1, 0.3, 1.0],  # Dark blue (background)
        [1.0, 0.8, 0.0, 1.0],  # Yellow (necrotic core)
        [0.0, 0.8, 0.0, 1.0],  # Green (edema)
        [1.0, 0.0, 0.0, 1.0]   # Red (enhancing tumor)
    ]
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create multi-slice visualization
    num_slices = 3  # Number of slices to show
    slice_indices = []
    
    # Find slices containing tumor if possible
    tumor_slices = np.where(np.sum(mask > 0, axis=(1, 2)) > 100)[0]
    
    if len(tumor_slices) >= num_slices:
        # Use tumor-containing slices
        slice_spacing = len(tumor_slices) // num_slices
        slice_indices = tumor_slices[::slice_spacing][:num_slices]
    else:
        # Fallback to evenly spaced slices
        slice_spacing = mask.shape[0] // (num_slices + 1)
        slice_indices = [slice_spacing * (i + 1) for i in range(num_slices)]
    
    # Add middle slice if not already included
    middle_slice = mask.shape[0] // 2
    if middle_slice not in slice_indices and len(slice_indices) > 0:
        slice_indices[-1] = middle_slice
    
    # Create figure
    fig, axes = plt.subplots(num_slices, 5, figsize=(20, 4*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    # Add a title to the entire figure
    fig.suptitle(f'Brain Tumor Segmentation Results (Sample {sample_idx})', fontsize=16)
    
    for i, slice_idx in enumerate(slice_indices):
        # Display FLAIR
        axes[i, 0].imshow(img[0, slice_idx], cmap='gray')
        axes[i, 0].set_title(f'FLAIR (Slice {slice_idx})')
        axes[i, 0].axis('off')
        
        # Display T1CE
        axes[i, 1].imshow(img[1, slice_idx], cmap='gray')
        axes[i, 1].set_title(f'T1CE (Slice {slice_idx})')
        axes[i, 1].axis('off')
        
        # Display T2
        axes[i, 2].imshow(img[2, slice_idx], cmap='gray')
        axes[i, 2].set_title(f'T2 (Slice {slice_idx})')
        axes[i, 2].axis('off')
        
        # Display Ground Truth with custom colormap
        gt_img = axes[i, 3].imshow(mask[slice_idx], cmap=custom_cmap, vmin=0, vmax=3)
        axes[i, 3].set_title(f'Ground Truth (Slice {slice_idx})')
        axes[i, 3].axis('off')
        
        # Display Prediction with the same custom colormap
        pred_img = axes[i, 4].imshow(pred[slice_idx], cmap=custom_cmap, vmin=0, vmax=3)
        axes[i, 4].set_title(f'Prediction (Slice {slice_idx})')
        axes[i, 4].axis('off')
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(gt_img, cax=cbar_ax)
    cbar.set_ticks([0.4, 1.2, 2.0, 2.8])
    cbar.set_ticklabels(['Background', 'Necrotic Core', 'Edema', 'Enhancing Tumor'])
    
    # Add a text summary showing class distribution
    gt_stats = np.bincount(mask.flatten(), minlength=4) / mask.size * 100
    pred_stats = np.bincount(pred.flatten(), minlength=4) / pred.size * 100
    
    stats_text = f"Class Distribution (%):\n"
    stats_text += f"Background: GT={gt_stats[0]:.1f}%, Pred={pred_stats[0]:.1f}%\n"
    stats_text += f"Necrotic Core: GT={gt_stats[1]:.1f}%, Pred={pred_stats[1]:.1f}%\n"
    stats_text += f"Edema: GT={gt_stats[2]:.1f}%, Pred={pred_stats[2]:.1f}%\n"
    stats_text += f"Enhancing Tumor: GT={gt_stats[3]:.1f}%, Pred={pred_stats[3]:.1f}%"
    
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Adjust layout to make room for colorbar
    
    # Save plot
    output_file = output_path / f"enhanced_prediction_sample_{sample_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()
    
    # Additionally create individual slice visualizations for detailed view
    detailed_output_dir = output_path / f"sample_{sample_idx}_detailed"
    detailed_output_dir.mkdir(exist_ok=True)
    
    # Find slices with the most tumor content
    tumor_content = np.sum(mask > 0, axis=(1, 2))
    top_slices = np.argsort(tumor_content)[-5:]  # Top 5 slices with most tumor
    
    for slice_idx in top_slices:
        if tumor_content[slice_idx] > 0:  # Only visualize if there's actually tumor
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Display FLAIR
            axes[0].imshow(img[0, slice_idx], cmap='gray')
            axes[0].set_title(f'FLAIR (Slice {slice_idx})')
            axes[0].axis('off')
            
            # Display T1CE
            axes[1].imshow(img[1, slice_idx], cmap='gray')
            axes[1].set_title(f'T1CE (Slice {slice_idx})')
            axes[1].axis('off')
            
            # Display T2
            axes[2].imshow(img[2, slice_idx], cmap='gray')
            axes[2].set_title(f'T2 (Slice {slice_idx})')
            axes[2].axis('off')
            
            # Create overlay of FLAIR with ground truth
            axes[3].imshow(img[0, slice_idx], cmap='gray')
            mask_overlay = np.ma.masked_where(mask[slice_idx] == 0, mask[slice_idx])
            axes[3].imshow(mask_overlay, cmap=custom_cmap, alpha=0.7, vmin=0, vmax=3)
            axes[3].set_title(f'Ground Truth Overlay (Slice {slice_idx})')
            axes[3].axis('off')
            
            # Create overlay of FLAIR with prediction
            axes[4].imshow(img[0, slice_idx], cmap='gray')
            pred_overlay = np.ma.masked_where(pred[slice_idx] == 0, pred[slice_idx])
            axes[4].imshow(pred_overlay, cmap=custom_cmap, alpha=0.7, vmin=0, vmax=3)
            axes[4].set_title(f'Prediction Overlay (Slice {slice_idx})')
            axes[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(detailed_output_dir / f"slice_{slice_idx}.png", dpi=150)
            plt.close()
    
    return output_file


def get_model_info(model_path):
    """Get information about a model checkpoint"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Load checkpoint
    device = torch.device("cpu")  # Use CPU for metadata
    try:
        checkpoint = torch.load(model_path, map_location=device)
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
            'mean_iou': checkpoint.get('mean_iou', 'unknown')
        }
        return info
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize brain tumor segmentation results')
    parser.add_argument('--model', type=str, default='output/improved_model/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/input_data_128',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='output/visualizations',
                        help='Path to save visualizations')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--info', action='store_true',
                        help='Only show model info without visualization')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    args = parser.parse_args()
    
    # Show model info if requested
    if args.info:
        model_info = get_model_info(args.model)
        if model_info:
            print(f"Model: {args.model}")
            print(f"Epoch: {model_info['epoch']}")
            print(f"Loss: {model_info['loss']}")
            print(f"Mean IoU: {model_info['mean_iou']}")
        sys.exit(0)
    
    # Run visualization
    visualize_tumor_segmentation(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        num_samples=args.samples,
        use_cuda=not args.cpu
    )