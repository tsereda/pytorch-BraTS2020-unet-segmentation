import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from model import UNet3D
from dataset import get_data_loaders


class DiceLoss(nn.Module):
    """Dice loss for segmentation with class weights support"""
    def __init__(self, weights: Optional[torch.Tensor] = None, smooth: float = 1e-5):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Predictions should be softmaxed
        batch_size = predictions.size(0)
        
        # Flatten predictions and targets
        predictions = predictions.view(batch_size, predictions.size(1), -1)
        targets = F.one_hot(targets, num_classes=predictions.size(1))
        targets = targets.view(batch_size, targets.size(-1), -1)
        targets = targets.permute(0, 2, 1)
        
        # Calculate Dice score for each class
        intersection = torch.sum(predictions * targets, dim=2)
        union = torch.sum(predictions, dim=2) + torch.sum(targets, dim=2)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.weights is not None:
            dice_score = dice_score * self.weights.to(dice_score.device)
            
        # Return mean Dice loss
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
        
        # Loss functions
        self.dice_loss = DiceLoss(weights=class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate losses
        dice_loss = self.dice_loss(y_hat, y)
        ce_loss = self.ce_loss(y_hat, y)
        total_loss = dice_loss + ce_loss
        
        # Log metrics
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate losses
        dice_loss = self.dice_loss(y_hat, y)
        ce_loss = self.ce_loss(y_hat, y)
        total_loss = dice_loss + ce_loss
        
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
        self.log('val_ce_loss', ce_loss, on_epoch=True)
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


def calculate_class_weights(data_path: str) -> torch.Tensor:
    """Calculate class weights based on class distribution in dataset"""
    print("Calculating class weights...")
    mask_files = sorted(list(Path(data_path).glob('masks/*.npy')))
    class_counts = torch.zeros(4)
    
    for mask_file in mask_files:
        mask = np.load(mask_file)
        for cls in range(4):
            class_counts[cls] += np.sum(mask == cls)
    
    # Calculate weights as inverse of frequency
    weights = 1.0 / class_counts
    weights = weights / weights.sum()  # Normalize
    print("Class weights:", weights)
    return weights


def train_model(
    data_path: str,
    output_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4
):
    """Main training function"""
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data_path)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model
    model = BrainTumorSegmentation(
        model_params={
            'in_channels': 3,
            'num_classes': 4,
            'init_features': 16
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
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


if __name__ == "__main__":
    # Training configuration
    DATA_PATH = "path/to/processed_data"
    OUTPUT_PATH = "path/to/output"
    
    model, trainer = train_model(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        batch_size=1,  # Start with 1 for 3D volumes
        num_workers=4,
        max_epochs=100,
        learning_rate=1e-4
    )