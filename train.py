import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from unet3d_model import UNet3D
from dataset import get_data_loaders


class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation with improved numerical stability"""
    def __init__(self, alpha=0.25, gamma=2.0, weights: Optional[torch.Tensor] = None, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.epsilon = epsilon  # Increased epsilon for stability
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs shape: [B, C, D, H, W], targets shape: [B, D, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Apply softmax to get probabilities with clipping for stability
        inputs_softmax = F.softmax(inputs, dim=1).clamp(min=self.epsilon, max=1-self.epsilon)
        
        # Focal loss formula with improved stability
        ce_loss = -targets_one_hot * torch.log(inputs_softmax)
        
        # Apply class weights if provided
        if self.weights is not None:
            weight_tensor = self.weights.view(1, -1, 1, 1, 1).to(inputs.device)
            ce_loss = ce_loss * weight_tensor
            
        # Calculate focal term with clipping
        pt = (inputs_softmax * targets_one_hot + (1 - inputs_softmax) * (1 - targets_one_hot)).clamp(min=self.epsilon)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply focal term and alpha balancing
        loss = self.alpha * focal_term * ce_loss
        
        # Return mean loss, handling potential NaN values
        loss_sum = loss.sum(dim=1)
        if torch.isnan(loss_sum).any():
            print(f"Warning: NaN values in focal loss. Using fallback loss.")
            return torch.tensor(1.0, requires_grad=True, device=inputs.device)
        
        return loss_sum.mean()


class DiceLoss(nn.Module):
    """Dice loss for 3D segmentation with improved numerical stability"""
    def __init__(self, weights: Optional[torch.Tensor] = None, smooth: float = 1e-4):
        super().__init__()
        self.weights = weights
        self.smooth = smooth  # Increased smooth value for stability
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to get probabilities with clipping for stability
        predictions = F.softmax(logits, dim=1).clamp(min=1e-6, max=1-1e-6)
        
        # predictions shape: (batch_size, num_classes, D, H, W)
        # targets shape: (batch_size, D, H, W)
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Flatten predictions and targets
        predictions = predictions.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, -1)
        
        # One-hot encode targets
        targets = F.one_hot(targets, num_classes=num_classes)
        targets = targets.permute(0, 2, 1).float()
        
        # Calculate Dice score with improved stability
        intersection = torch.sum(predictions * targets, dim=2) + self.smooth
        union = torch.sum(predictions, dim=2) + torch.sum(targets, dim=2) + self.smooth
        dice_score = (2.0 * intersection) / union
        
        if self.weights is not None:
            dice_score = dice_score * self.weights.to(dice_score.device)
        
        # Handle potential NaN values
        if torch.isnan(dice_score).any():
            print(f"Warning: NaN values in dice score. Using fallback loss.")
            return torch.tensor(1.0, requires_grad=True, device=logits.device)
            
        return 1 - dice_score.mean()


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict[str, Any] = None,
        learning_rate: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        model_params = model_params or {}
        self.model = UNet3D(**model_params)
        self.model.initialize_weights()  # Use He uniform initialization
        
        # Loss functions
        self.dice_loss = DiceLoss(weights=class_weights)
        self.focal_loss = FocalLoss(weights=class_weights)
        
        # Metrics
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Debug: Check for problematic input values
        if torch.isnan(x).any():
            print(f"Warning: NaN values in input data (batch {batch_idx})")
            # Handle NaN inputs
            x = torch.nan_to_num(x, nan=0.0)
        
        y_hat = self(x)
        
        # Debug: Check for NaN in model output
        if torch.isnan(y_hat).any():
            print(f"Warning: NaN values in model output (batch {batch_idx})")
            # Use a valid loss value to continue training
            return torch.tensor(1.0, requires_grad=True, device=self.device)
        
        # Calculate losses
        dice_loss = self.dice_loss(y_hat, y)
        focal_loss = self.focal_loss(y_hat, y)
        
        # Check for NaN losses and handle them
        if torch.isnan(dice_loss):
            print(f"Warning: NaN dice loss in batch {batch_idx}")
            dice_loss = torch.tensor(0.5, requires_grad=True, device=self.device)
            
        if torch.isnan(focal_loss):
            print(f"Warning: NaN focal loss in batch {batch_idx}")
            focal_loss = torch.tensor(0.5, requires_grad=True, device=self.device)
        
        total_loss = dice_loss + focal_loss
        
        # Log metrics
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_focal_loss', focal_loss, on_step=True, on_epoch=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Handle potential NaN inputs
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        y_hat = self(x)
        
        # Handle potential NaN outputs
        if torch.isnan(y_hat).any():
            return torch.tensor(1.0, requires_grad=True, device=self.device)
        
        # Calculate losses
        dice_loss = self.dice_loss(y_hat, y)
        focal_loss = self.focal_loss(y_hat, y)
        
        # Handle NaN losses
        if torch.isnan(dice_loss) or torch.isnan(focal_loss):
            print(f"Warning: NaN loss in validation batch {batch_idx}")
            return torch.tensor(1.0, requires_grad=True, device=self.device)
        
        total_loss = dice_loss + focal_loss
        
        # Calculate IoU for each class
        y_pred = torch.argmax(y_hat, dim=1)
        ious = []
        for cls in range(4):  # 4 classes
            intersection = torch.sum((y_pred == cls) & (y == cls))
            union = torch.sum((y_pred == cls) | (y == cls))
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou)
        
        # Log metrics
        self.log('val_dice_loss', dice_loss, on_epoch=True, prog_bar=True)
        self.log('val_focal_loss', focal_loss, on_epoch=True)
        self.log('val_total_loss', total_loss, on_epoch=True)
        
        for cls in range(4):
            self.log(f'val_iou_class_{cls}', ious[cls], on_epoch=True)
        self.log('val_mean_iou', torch.mean(torch.stack(ious)), on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss"
            }
        }


def calculate_class_weights(data_path: str, use_equal_weights: bool = False) -> torch.Tensor:
    """
    Calculate class weights based on class distribution in dataset
    
    Args:
        data_path: Path to dataset
        use_equal_weights: If True, use equal weights (0.25, 0.25, 0.25, 0.25) like in Keras
        
    Returns:
        torch.Tensor of shape (4,) with class weights
    """
    if use_equal_weights:
        print("Using equal class weights: [0.25, 0.25, 0.25, 0.25]")
        return torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    print("Calculating class weights...")
    # Find mask files in both train and val directories
    train_masks = list(Path(data_path).glob('train/masks/*.npy'))
    val_masks = list(Path(data_path).glob('val/masks/*.npy'))
    mask_files = train_masks + val_masks
    
    if len(mask_files) == 0:
        print(f"Warning: No mask files found in {data_path}. Using equal weights.")
        return torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    class_counts = torch.zeros(4)
    
    for mask_file in mask_files:
        mask = np.load(mask_file)
        mask_flat = mask.flatten()
        for cls in range(4):
            class_counts[cls] += np.sum(mask_flat == cls)
    
    # Ensure no zeros in class_counts
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Calculate weights with clamping to prevent extreme values
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 4  # Normalize and scale
    weights = torch.clamp(weights, min=0.01, max=10.0)  # Clamp to reasonable range
    
    print(f"Class weights: {weights}")
    return weights


def train_model(
    data_path: str,
    output_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    use_equal_weights: bool = False,
    gradient_clip_val: float = 1.0
):
    """
    Main training function with improved stability and handling
    
    Args:
        data_path: Path to dataset containing train and val subdirectories
        output_path: Path to save model and logs
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_epochs: Maximum number of epochs to train
        learning_rate: Learning rate for optimizer
        use_equal_weights: If True, use equal class weights like in Keras
        gradient_clip_val: Value for gradient clipping (helps prevent NaN)
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data_path, use_equal_weights)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Debug: Check the number of samples in the loaders
    print(f"Training samples: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'unknown'}")
    print(f"Validation samples: {len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'unknown'}")
    
    # Handle case where validation loader is empty
    if len(val_loader) == 0:
        print("Warning: Validation loader is empty. Using a portion of training data for validation.")
        # Simple solution: split training loader
        train_size = int(0.8 * len(train_loader))
        val_size = len(train_loader) - train_size
        train_loader, val_loader = torch.utils.data.random_split(train_loader, [train_size, val_size])
    
    # Initialize model
    model = BrainTumorSegmentation(
        model_params={
            'in_channels': 3,
            'num_classes': 4,
            'init_features': 16  # Same as Keras model
        },
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / 'checkpoints',
        filename='brain_tumor_seg_{epoch:02d}_{val_mean_iou:.3f}',
        monitor='val_mean_iou',
        mode='max',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_total_loss',
        patience=15,
        mode='min'
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_path,
        name='logs',
        default_hp_metric=False
    )
    
    # Initialize trainer with gradient clipping to prevent NaN
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        gradient_clip_val=gradient_clip_val  # Add gradient clipping
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


def visualize_sample(model, data_path, sample_idx=0):
    """
    Visualize a sample prediction from the validation set
    
    Args:
        model: Trained model
        data_path: Path to dataset
        sample_idx: Index of sample to visualize
    """
    from matplotlib import pyplot as plt
    
    # Load a sample from validation set
    val_img_dir = Path(data_path) / 'val' / 'images'
    val_mask_dir = Path(data_path) / 'val' / 'masks'
    
    img_files = sorted(list(val_img_dir.glob('*.npy')))
    mask_files = sorted(list(val_mask_dir.glob('*.npy')))
    
    if not img_files or not mask_files:
        print("No validation samples found.")
        return
    
    # Get sample
    sample_idx = min(sample_idx, len(img_files) - 1)
    img = np.load(img_files[sample_idx])
    mask = np.load(mask_files[sample_idx])
    
    # Convert to torch tensors
    img_tensor = torch.from_numpy(img).float().permute(3, 0, 1, 2).unsqueeze(0)  # Add batch dimension
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    
    # Convert mask to label if one-hot encoded
    if len(mask.shape) == 4:
        mask = np.argmax(mask, axis=3)
    
    # Visualize a slice
    slice_idx = mask.shape[2] // 2  # Middle slice
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.title('FLAIR')
    plt.imshow(img[:, :, slice_idx, 0], cmap='gray')
    
    plt.subplot(142)
    plt.title('T1CE')
    plt.imshow(img[:, :, slice_idx, 1], cmap='gray')
    
    plt.subplot(143)
    plt.title('Ground Truth')
    plt.imshow(mask[:, :, slice_idx])
    
    plt.subplot(144)
    plt.title('Prediction')
    plt.imshow(prediction[:, :, slice_idx])
    
    plt.tight_layout()
    plt.savefig(Path(data_path).parent / 'prediction_sample.png')
    plt.close()
    
    print(f"Visualization saved to {Path(data_path).parent / 'prediction_sample.png'}")


if __name__ == "__main__":
    # Training configuration
    DATA_PATH = "data/input_data_128"
    OUTPUT_PATH = "output"
    
    # Train model with improvements
    model, trainer = train_model(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        batch_size=1,  # Use 1 for 3D volumes due to GPU memory constraints
        num_workers=4,
        max_epochs=100,
        learning_rate=1e-4,
        use_equal_weights=True,  # Use equal weights like in Keras implementation
        gradient_clip_val=1.0  # Add gradient clipping to prevent NaN
    )
    
    # Visualize a sample prediction
    # visualize_sample(model, DATA_PATH)