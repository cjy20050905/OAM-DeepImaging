"""
OAM Radar Imaging - Training Script

This script trains the end-to-end reconstruction network for OAM radar imaging.

Usage:
    python train.py --config configs/default.yaml

Author: [Your Name]
Date: 2026-03-06
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import json
import time

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent))
from physics import OAMRadarGeometry
from dataset import SparseTargetDataset
from model import EndToEndReconstructionNet, CombinedLoss


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted image
        target: Ground truth image

    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(torch.tensor(max_pixel) / torch.sqrt(mse))
    return psnr.item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0

    for y, x_gt in tqdm(dataloader, desc="Training", leave=False):
        y = y.to(device)
        x_gt = x_gt.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            x_pred = model(y)
            loss = criterion(x_pred, x_gt)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate the model.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        tuple: (average_loss, average_psnr)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for y, x_gt in tqdm(dataloader, desc="Validation", leave=False):
            y = y.to(device)
            x_gt = x_gt.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                x_pred = model(y)
                loss = criterion(x_pred, x_gt)

            total_loss += loss.item()

            # Calculate PSNR for each sample
            for i in range(x_pred.shape[0]):
                total_psnr += calculate_psnr(x_pred[i], x_gt[i])

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader.dataset)

    return avg_loss, avg_psnr


def main():
    """Main training function."""

    # ========== Configuration ==========
    IMAGE_SIZE = 24
    OAM_MODES = (-3, 3)
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 5e-4
    NUM_TRAIN_SAMPLES = 30000
    NUM_VAL_SAMPLES = 3000
    HIDDEN_DIM = 768
    MAX_GRAD_NORM = 1.0
    WARMUP_EPOCHS = 10

    # Output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ========== Generate Observation Matrix ==========
    print("\n[1/5] Generating observation matrix...")
    radar = OAMRadarGeometry(image_size=IMAGE_SIZE, oam_modes=OAM_MODES)
    print(radar)

    Phi = radar.generate_observation_matrix()
    measurement_dim = Phi.shape[0]

    # ========== Create Datasets ==========
    print("\n[2/5] Creating datasets...")
    train_dataset = SparseTargetDataset(
        Phi=Phi,
        image_size=IMAGE_SIZE,
        num_samples=NUM_TRAIN_SAMPLES,
        noise_std=0.001,
        augment=True
    )
    val_dataset = SparseTargetDataset(
        Phi=Phi,
        image_size=IMAGE_SIZE,
        num_samples=NUM_VAL_SAMPLES,
        noise_std=0.001,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ========== Create Model ==========
    print("\n[3/5] Building model...")
    model = EndToEndReconstructionNet(
        measurement_dim=measurement_dim,
        image_size=IMAGE_SIZE,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    print(model)

    # ========== Training Configuration ==========
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ========== Training Loop ==========
    print("\n[4/5] Training...")
    best_psnr = 0.0
    train_losses = []
    val_losses = []
    val_psnrs = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, MAX_GRAD_NORM
        )

        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # Print results
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr,
                'config': {
                    'image_size': IMAGE_SIZE,
                    'measurement_dim': measurement_dim,
                    'hidden_dim': HIDDEN_DIM
                }
            }, output_dir / 'best_model.pth')
            print(f"✓ Best model saved (PSNR: {val_psnr:.2f} dB)")

    # ========== Training Complete ==========
    elapsed_time = time.time() - start_time
    print(f"\n[5/5] Training completed!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total training time: {elapsed_time/3600:.2f} hours")
    print(f"Results saved to: {output_dir}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'best_psnr': best_psnr,
        'training_time': elapsed_time
    }
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
