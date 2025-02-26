import os
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json
import glob
import shutil
from typing import Tuple, List, Dict
import splitfolders
import multiprocessing
from functools import partial
import time
import sys


def process_single_case(case_data, output_path, min_label_ratio=0.01):
    """
    Process a single case with optimized operations
    
    Args:
        case_data (tuple): Tuple containing (case_idx, flair_path, t1ce_path, t2_path, mask_path)
        output_path (str): Path to save preprocessed data
        min_label_ratio (float): Minimum ratio of non-zero labels required
        
    Returns:
        tuple: (status, case_id) where status is True if valid, False if error, None if skipped
    """
    case_idx, flair_path, t1ce_path, t2_path, mask_path = case_data
    
    # Extract case_id from filename (pattern: BratsXX_Training_XXX_t2.nii)
    filename = os.path.basename(t2_path)
    case_id = filename.split('_t2.nii')[0]  # Remove the modality suffix
    
    try:
        # For better output from multiple processes
        sys.stdout.write(f"Starting to process {case_id}...\n")
        sys.stdout.flush()
        
        # Load modalities and explicitly convert to the right types
        temp_image_flair = nib.load(flair_path).get_fdata()
        temp_image_t1ce = nib.load(t1ce_path).get_fdata()
        temp_image_t2 = nib.load(t2_path).get_fdata()
        temp_mask = nib.load(mask_path).get_fdata()
        
        # Explicitly convert to float32 (important for in-place operations)
        temp_image_flair = temp_image_flair.astype(np.float32)
        temp_image_t1ce = temp_image_t1ce.astype(np.float32)
        temp_image_t2 = temp_image_t2.astype(np.float32)
        
        # Convert mask to uint8 for memory efficiency
        temp_mask = temp_mask.astype(np.uint8)
        
        # Remap label 4 to 3 (following BraTS convention)
        temp_mask[temp_mask == 4] = 3
        
        # Pre-crop to reduce memory footprint before normalization
        temp_image_flair = temp_image_flair[56:184, 56:184, 13:141]
        temp_image_t1ce = temp_image_t1ce[56:184, 56:184, 13:141]
        temp_image_t2 = temp_image_t2[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
        
        # Check if case has enough non-zero labels early to avoid unnecessary processing
        val, counts = np.unique(temp_mask, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) <= min_label_ratio:
            sys.stdout.write(f"Case {case_id} skipped: insufficient non-zero labels\n")
            sys.stdout.flush()
            return None, case_id  # Not enough non-zero labels
        
        # Optimize normalization using vectorized operations
        # Use in-place operations to reduce memory usage
        for img in [temp_image_flair, temp_image_t1ce, temp_image_t2]:
            # Verify data type to prevent errors
            if not np.issubdtype(img.dtype, np.floating):
                img = img.astype(np.float32)
                
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:  # Avoid division by zero
                img -= min_val
                img /= (max_val - min_val)
        
        # Stack modalities (flair, t1ce, t2) - more memory efficient than separate operations
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        # Save the processed files
        np.save(
            Path(output_path) / 'images' / f'image_{case_id}.npy',
            temp_combined_images
        )
        np.save(
            Path(output_path) / 'masks' / f'mask_{case_id}.npy',
            temp_mask
        )
        
        sys.stdout.write(f"Case {case_id} processed successfully\n")
        sys.stdout.flush()
        return True, case_id
        
    except Exception as e:
        sys.stdout.write(f"Error processing {case_id}: {str(e)}\n")
        sys.stdout.flush()
        return False, case_id


def preprocess_single_case_sequential(case_data, output_path, min_label_ratio=0.01):
    """
    Sequentially process a list of cases - used as fallback if parallel processing fails
    
    Args:
        case_data (list): List of case data tuples
        output_path (str): Path to save preprocessed data
        min_label_ratio (float): Minimum ratio of non-zero labels required
        
    Returns:
        dict: Dictionary containing valid and skipped cases
    """
    processed_files = {'valid_cases': [], 'skipped_cases': []}
    
    for i, data in enumerate(case_data):
        print(f"Processing case {i+1}/{len(case_data)}...")
        status, case_id = process_single_case(data, output_path, min_label_ratio)
        
        if status is True:
            processed_files['valid_cases'].append(case_id)
        else:
            processed_files['skipped_cases'].append(case_id)
    
    return processed_files


def preprocess_brats2020(input_path: str, output_path: str, num_workers: int = None, sequential: bool = False):
    """
    Preprocess BraTS2020 dataset with parallel processing for speed
    
    Args:
        input_path (str): Path to raw BraTS2020 dataset
        output_path (str): Path to save preprocessed data
        num_workers (int): Number of parallel workers (defaults to CPU count - 1)
        sequential (bool): If True, process cases sequentially instead of in parallel
    """
    print("Starting preprocessing of BraTS2020 dataset...")
    
    # Determine number of workers (use 1 less than CPU count to avoid system freeze)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output directories
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'masks').mkdir(exist_ok=True)
    
    # Get all subject folders using the correct file pattern
    print("Scanning for input files...")
    t2_list = sorted(glob.glob(f'{input_path}/*/*_t2.nii*'))
    t1ce_list = sorted(glob.glob(f'{input_path}/*/*_t1ce.nii*'))
    flair_list = sorted(glob.glob(f'{input_path}/*/*_flair.nii*'))
    mask_list = sorted(glob.glob(f'{input_path}/*/*_seg.nii*'))
    
    print(f"Found files: T2={len(t2_list)}, T1CE={len(t1ce_list)}, FLAIR={len(flair_list)}, MASK={len(mask_list)}")
    
    # Create a dictionary to map case IDs to their file paths
    case_files = {}
    
    # Process T2 files
    for path in t2_list:
        # Extract case_id from filename (pattern: BratsXX_Training_XXX_t2.nii)
        filename = os.path.basename(path)
        case_id = filename.split('_t2.nii')[0]  # Remove the modality suffix
        
        # Skip the problematic case
        if case_id == "BraTS20_Training_355":
            print(f"Skipping known problematic case: {case_id}")
            continue
            
        if case_id not in case_files:
            case_files[case_id] = {'t2': path}
        else:
            case_files[case_id]['t2'] = path
    
    # Process T1CE files
    for path in t1ce_list:
        filename = os.path.basename(path)
        case_id = filename.split('_t1ce.nii')[0]
        
        # Skip the problematic case
        if case_id == "BraTS20_Training_355":
            continue
            
        if case_id not in case_files:
            case_files[case_id] = {'t1ce': path}
        else:
            case_files[case_id]['t1ce'] = path
    
    # Process FLAIR files
    for path in flair_list:
        filename = os.path.basename(path)
        case_id = filename.split('_flair.nii')[0]
        
        # Skip the problematic case
        if case_id == "BraTS20_Training_355":
            continue
            
        if case_id not in case_files:
            case_files[case_id] = {'flair': path}
        else:
            case_files[case_id]['flair'] = path
    
    # Process MASK files
    for path in mask_list:
        filename = os.path.basename(path)
        case_id = filename.split('_seg.nii')[0]
        
        if case_id not in case_files:
            case_files[case_id] = {'mask': path}
        else:
            case_files[case_id]['mask'] = path
    
    # Find cases with all required modalities
    complete_cases = []
    incomplete_cases = []
    
    for case_id, files in case_files.items():
        if all(modality in files for modality in ['t2', 't1ce', 'flair', 'mask']):
            complete_cases.append(case_id)
        else:
            incomplete_cases.append((case_id, files.keys()))
    
    print(f"Complete cases: {len(complete_cases)}")
    if incomplete_cases:
        print(f"Found {len(incomplete_cases)} incomplete cases (missing one or more modalities):")
        for case_id, available_modalities in incomplete_cases[:5]:  # Show first 5 only
            print(f"  - {case_id}: has {list(available_modalities)}")
        if len(incomplete_cases) > 5:
            print(f"  - ... and {len(incomplete_cases) - 5} more")
    
    # Prepare data for processing
    case_data = []
    for idx, case_id in enumerate(complete_cases):
        files = case_files[case_id]
        case_data.append((idx, files['flair'], files['t1ce'], files['t2'], files['mask']))
    
    processed_files = {'valid_cases': [], 'skipped_cases': []}
    
    # Process cases (either in parallel or sequentially)
    print(f"Processing {len(case_data)} cases...")
    
    if sequential:
        print("Using sequential processing mode...")
        processed_files = preprocess_single_case_sequential(case_data, output_path)
    else:
        print(f"Using parallel processing with {num_workers} workers...")
        
        # Process first case separately to catch any setup issues early
        print("Processing first case to check for issues...")
        first_status, first_case_id = process_single_case(case_data[0], output_path)
        
        if first_status is True:
            processed_files['valid_cases'].append(first_case_id)
        else:
            processed_files['skipped_cases'].append(first_case_id)
        
        # Process remaining cases in parallel
        remaining_case_data = case_data[1:]
        print(f"Processing remaining {len(remaining_case_data)} cases in parallel...")
        
        # Set up the parallel processing function
        process_func = partial(process_single_case, output_path=output_path)
        
        # Process in smaller batches to avoid memory issues
        batch_size = min(50, len(remaining_case_data))
        batches = [remaining_case_data[i:i+batch_size] for i in range(0, len(remaining_case_data), batch_size)]
        
        total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
        
        for batch_idx, batch in enumerate(batches):
            # Use a context manager to ensure proper cleanup of resources
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use a simple map instead of tqdm
                batch_results = pool.map(process_func, batch)
            
            # Process batch results
            valid_in_batch = 0
            for status, case_id in batch_results:
                if status is True:
                    processed_files['valid_cases'].append(case_id)
                    valid_in_batch += 1
                else:
                    processed_files['skipped_cases'].append(case_id)
            
            total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
            progress_percent = (total_processed / len(case_data)) * 100
            
            print(f"Batch {batch_idx+1}/{len(batches)}: {valid_in_batch}/{len(batch)} valid cases - Overall: {total_processed}/{len(case_data)} ({progress_percent:.1f}%)")
    
    # Save processing results
    with open(output_path / 'processing_results.json', 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    print(f"Preprocessing complete. Processed {len(processed_files['valid_cases'])} valid cases.")
    print(f"Skipped {len(processed_files['skipped_cases'])} cases.")
    
    return processed_files


def split_dataset(input_folder: str, output_folder: str, train_ratio: float = 0.75):
    """
    Split preprocessed dataset into training and validation sets
    
    Args:
        input_folder (str): Path to preprocessed data
        output_folder (str): Path to save split data
        train_ratio (float): Ratio of training data
    """
    print(f"Splitting dataset with train ratio: {train_ratio}")
    
    # Check if there are actually files to split
    image_count = len(glob.glob(f"{input_folder}/images/*.npy"))
    mask_count = len(glob.glob(f"{input_folder}/masks/*.npy"))
    
    if image_count == 0 or mask_count == 0:
        print(f"WARNING: No files found to split! Images: {image_count}, Masks: {mask_count}")
        return
    
    # Split with a ratio
    splitfolders.ratio(
        input_folder, 
        output=output_folder, 
        seed=42, 
        ratio=(train_ratio, 1-train_ratio), 
        group_prefix=None
    )
    
    # Count files in each split
    train_images = len(glob.glob(f"{output_folder}/train/images/*.npy"))
    val_images = len(glob.glob(f"{output_folder}/val/images/*.npy"))
    
    print(f"Dataset split complete:")
    print(f"  Training: {train_images} images")
    print(f"  Validation: {val_images} images")


def print_dataset_info(path: str):
    """
    Print information about the dataset structure and files
    """
    print(f"\nDataset information for: {path}")
    
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    
    # Check for nifti files with the actual naming pattern
    nifti_files = glob.glob(f"{path}/*/*_*.nii*")
    print(f"Total .nii files found: {len(nifti_files)}")
    
    # Get some sample filenames
    if nifti_files:
        print("Sample filenames:")
        for f in nifti_files[:3]:
            print(f"  - {os.path.basename(f)}")
    
    # Check directory structure
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Subdirectories: {len(subdirs)} folders")
    
    if subdirs:
        sample_subdir = os.path.join(path, subdirs[0])
        files = os.listdir(sample_subdir)
        print(f"Sample subdir '{subdirs[0]}' contents: {len(files)} files")


def create_dataset_for_training(
    raw_data_path: str, 
    processed_path: str, 
    final_data_path: str,
    train_ratio: float = 0.75,
    num_workers: int = None,
    sequential: bool = False
):
    """
    Complete pipeline to prepare BraTS2020 dataset for training
    
    Args:
        raw_data_path (str): Path to raw BraTS2020 dataset
        processed_path (str): Path to save preprocessed data
        final_data_path (str): Path to save split data
        train_ratio (float): Ratio of training data
        num_workers (int): Number of parallel workers
        sequential (bool): If True, process cases sequentially
    """
    # Print dataset information to help diagnose issues
    print_dataset_info(raw_data_path)
    
    # Step 1: Preprocess the dataset
    preprocess_brats2020(raw_data_path, processed_path, num_workers, sequential)
    
    # Step 2: Split into train/val
    split_dataset(processed_path, final_data_path, train_ratio)
    
    print(f"Dataset preparation complete. Data ready for training at: {final_data_path}")


if __name__ == "__main__":
    # Example usage
    RAW_DATASET_PATH = "data/MICCAI_BraTS2020_TrainingData"
    PROCESSED_PATH = "data/processed_data"
    FINAL_DATA_PATH = "data/input_data_128"
    
    import time
    start_time = time.time()
    
    create_dataset_for_training(
        raw_data_path=RAW_DATASET_PATH,
        processed_path=PROCESSED_PATH,
        final_data_path=FINAL_DATA_PATH,
        train_ratio=0.75,
        num_workers=12, 
        sequential=False  # Set to True for sequential processing
    )
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")