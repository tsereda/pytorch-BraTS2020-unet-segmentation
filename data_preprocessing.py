import os
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json
from typing import Tuple, List, Dict


class BraTS20DataPreprocessor:
    """Handles preprocessing of BraTS2020 dataset"""
    def __init__(self, data_path: str, output_path: str):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to raw BraTS2020 dataset
            output_path (str): Path to save preprocessed data
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.scaler = MinMaxScaler()
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / 'images').mkdir(exist_ok=True)
        (self.output_path / 'masks').mkdir(exist_ok=True)
        
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize the volume using MinMaxScaler
        
        Args:
            volume (np.ndarray): Input volume to normalize
            
        Returns:
            np.ndarray: Normalized volume
        """
        shaped_volume = volume.reshape(-1, volume.shape[-1])
        normalized = self.scaler.fit_transform(shaped_volume)
        return normalized.reshape(volume.shape)

    def load_case(self, case_folder: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single case
        
        Args:
            case_folder (Path): Path to the case folder
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed image and mask arrays
        """
        # Load modalities
        flair = nib.load(str(case_folder / f'{case_folder.name}_flair.nii')).get_fdata()
        t1ce = nib.load(str(case_folder / f'{case_folder.name}_t1ce.nii')).get_fdata()
        t2 = nib.load(str(case_folder / f'{case_folder.name}_t2.nii')).get_fdata()
        
        # Normalize each modality
        flair_norm = self.normalize_volume(flair)
        t1ce_norm = self.normalize_volume(t1ce)
        t2_norm = self.normalize_volume(t2)
        
        # Load and process mask
        mask_file = list(case_folder.glob('*seg.nii'))[0]
        mask = nib.load(str(mask_file)).get_fdata().astype(np.uint8)
        
        # Remap label 4 to 3 (following BraTS convention)
        mask[mask == 4] = 3
        
        # Stack modalities (flair, t1ce, t2)
        combined = np.stack([flair_norm, t1ce_norm, t2_norm], axis=3)
        
        # Crop to 128x128x128 (standard size for processing)
        combined = combined[56:184, 56:184, 13:141]
        mask = mask[56:184, 56:184, 13:141]
        
        return combined, mask

    def process_dataset(self) -> Dict[str, List[str]]:
        """
        Process entire dataset and save as numpy files
        
        Returns:
            Dict[str, List[str]]: Dictionary containing lists of processed and skipped cases
        """
        case_folders = sorted([f for f in self.data_path.iterdir() if f.is_dir()])
        processed_files = {'valid_cases': [], 'skipped_cases': []}
        
        total_cases = len(case_folders)
        for idx, case_folder in enumerate(case_folders, 1):
            print(f"Processing case {idx}/{total_cases}: {case_folder.name}")
            
            try:
                combined, mask = self.load_case(case_folder)
                
                # Check if case has enough non-zero labels (1% threshold)
                val, counts = np.unique(mask, return_counts=True)
                if (1 - (counts[0]/counts.sum())) > 0.01:
                    # Save the processed files
                    np.save(
                        self.output_path / 'images' / f'image_{case_folder.name}.npy', 
                        combined
                    )
                    np.save(
                        self.output_path / 'masks' / f'mask_{case_folder.name}.npy', 
                        mask
                    )
                    processed_files['valid_cases'].append(case_folder.name)
                    print(f"Successfully processed {case_folder.name}")
                else:
                    processed_files['skipped_cases'].append(case_folder.name)
                    print(f"Skipped {case_folder.name} - insufficient non-zero labels")
                    
            except Exception as e:
                print(f"Error processing {case_folder.name}: {str(e)}")
                processed_files['skipped_cases'].append(case_folder.name)
                
        # Save processing results
        with open(self.output_path / 'processing_results.json', 'w') as f:
            json.dump(processed_files, f, indent=2)
            
        return processed_files


if __name__ == "__main__":
    # Example usage
    TRAIN_DATASET_PATH = "data/MICCAI_BraTS2020_TrainingData"
    OUTPUT_PATH = "processed_data"
    
    # Initialize and run preprocessing
    preprocessor = BraTS20DataPreprocessor(TRAIN_DATASET_PATH, OUTPUT_PATH)
    results = preprocessor.process_dataset()
    
    # Print summary
    print("\nPreprocessing Summary:")
    print(f"Processed {len(results['valid_cases'])} valid cases")
    print(f"Skipped {len(results['skipped_cases'])} cases")