import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import glob


class BraTSDataset(Dataset):
    """
    Dataset for BraTS2020 data
    """
    def __init__(self, data_dir: str, mode: str = 'train'):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Base directory containing image and mask directories
            mode (str): 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        # Get file paths
        self.img_dir = self.data_dir / mode / 'images'
        self.mask_dir = self.data_dir / mode / 'masks'
        
        self.img_files = sorted(list(self.img_dir.glob('*.npy')))
        self.mask_files = sorted(list(self.mask_dir.glob('*.npy')))
        
        # Sanity check - make sure we have matching numbers of images and masks
        assert len(self.img_files) == len(self.mask_files), \
            f"Number of images ({len(self.img_files)}) doesn't match number of masks ({len(self.mask_files)})"
        
        print(f"Found {len(self.img_files)} {mode} samples")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor (C,D,H,W) and mask tensor (D,H,W)
        """
        try:
            # Load image and mask
            img = np.load(self.img_files[idx])
            mask = np.load(self.mask_files[idx])
            
            # Extract just the integer label from the mask (if one-hot encoded)
            if len(mask.shape) == 4 and mask.shape[3] == 4:  # One-hot encoded mask
                mask = np.argmax(mask, axis=3)
            
            # Check for NaN values
            if np.isnan(img).any():
                print(f"Warning: NaN values found in image at index {idx}. Replacing with zeros.")
                img = np.nan_to_num(img)
            
            # Check range - normalize if needed
            if img.max() > 10:  # If not already normalized
                img = img / 255.0
                
            # Convert to PyTorch tensors
            # Image: (D,H,W,C) -> (C,D,H,W)
            img = torch.from_numpy(img).float().permute(3, 0, 1, 2)
            mask = torch.from_numpy(mask).long()
            
            # Final sanity check
            assert not torch.isnan(img).any(), f"NaN values in tensor after conversion at index {idx}"
            assert mask.max() <= 3, f"Mask has invalid class values at index {idx}: {mask.max()}"
            
            return img, mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a simple dummy sample instead of crashing
            dummy_img = torch.zeros((3, 128, 128, 128), dtype=torch.float32)
            dummy_mask = torch.zeros((128, 128, 128), dtype=torch.long)
            return dummy_img, dummy_mask


def get_data_loaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation
    """
    print("-" * 30) # Separator for clarity in logs
    print("DEBUGGING get_data_loaders:")
    print(f"Input data_dir: {data_dir}")

    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")


    img_dir_train = train_dir / 'images'
    mask_dir_train = train_dir / 'masks'
    img_dir_val = val_dir / 'images'
    mask_dir_val = val_dir / 'masks'

    print(f"Train images directory: {img_dir_train}")
    print(f"Train masks directory: {mask_dir_train}")
    print(f"Val images directory: {img_dir_val}")
    print(f"Val masks directory: {mask_dir_val}")


    # Create datasets
    train_dataset = BraTSDataset(data_dir, mode='train') # Note: data_dir is base, mode='train'
    val_dataset = BraTSDataset(data_dir, mode='val')   # Note: data_dir is base, mode='val'

    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of val_dataset: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Length of train_loader: {len(train_loader)}") # Batches in loader
    print(f"Length of val_loader: {len(val_loader)}")   # Batches in loader
    print("-" * 30) # Separator

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage and testing
    data_dir = "data/input_data_128"
    
    # Test dataset
    train_dataset = BraTSDataset(data_dir, mode='train')
    print(f"Dataset length: {len(train_dataset)}")
    
    # Test a sample
    sample_img, sample_mask = train_dataset[0]
    print(f"Image shape: {sample_img.shape}")
    print(f"Mask shape: {sample_mask.shape}")
    print(f"Unique mask values: {torch.unique(sample_mask)}")
    
    # Test data loader
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=1)
    print(f"Number of batches in training loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")
    
    # Load a batch
    batch_img, batch_mask = next(iter(train_loader))
    print(f"Batch image shape: {batch_img.shape}")
    print(f"Batch mask shape: {batch_mask.shape}")