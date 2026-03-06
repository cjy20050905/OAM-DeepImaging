"""
OAM Radar Imaging - Visualization Script

This script generates comprehensive visualizations for paper figures:
1. Reconstruction results comparison
2. Training curves
3. Error heatmaps
4. Ablation study results

Usage:
    python visualize.py --model_path outputs/best_model.pth --output_dir figures/

Author: Dryoung
Date: 2026-03-06
License: MIT
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path
import json
import argparse

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from physics import OAMRadarGeometry
from dataset import SparseTargetDataset
from model import EndToEndReconstructionNet


def calculate_psnr(pred, target):
    """Calculate PSNR."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)).item()


def plot_reconstruction_results(model, val_loader, device, output_dir, num_samples=8):
    """
    Plot reconstruction results for paper Figure 1.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        y, x_gt = next(iter(val_loader))
        y = y[:num_samples].to(device)
        x_gt = x_gt[:num_samples].to(device)

        x_pred = model(y)

        # Create figure
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))

        for i in range(num_samples):
            # Ground truth
            axes[0, i].imshow(x_gt[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Ground Truth', fontsize=12, rotation=90, labelpad=10)

            # Reconstruction
            axes[1, i].imshow(x_pred[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstruction', fontsize=12, rotation=90, labelpad=10)

            # Error map
            error = torch.abs(x_pred[i] - x_gt[i]).cpu().numpy()
            im = axes[2, i].imshow(error, cmap='hot', vmin=0, vmax=0.5)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Error Map', fontsize=12, rotation=90, labelpad=10)

            # Add PSNR value
            psnr = calculate_psnr(x_pred[i], x_gt[i])
            axes[1, i].set_title(f'PSNR: {psnr:.1f} dB', fontsize=10)

        # Add colorbar for error map
        fig.colorbar(im, ax=axes[2, :], orientation='horizontal', pad=0.05, fraction=0.046)

        plt.tight_layout()
        plt.savefig(output_dir / 'reconstruction_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'reconstruction_results.pdf', bbox_inches='tight')
        plt.close()

    print(f"✓ Saved reconstruction results to {output_dir}")


def plot_training_curves(history_path, output_dir):
    """
    Plot training curves for paper Figure 2.

    Args:
        history_path: Path to training history JSON
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history['val_losses']
    val_psnrs = history['val_psnrs']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # PSNR curve
    ax2.plot(epochs, val_psnrs, label='Validation PSNR', color='green', linewidth=2)
    ax2.axhline(y=history['best_psnr'], color='r', linestyle='--',
                label=f'Best PSNR: {history["best_psnr"]:.2f} dB', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Validation PSNR', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved training curves to {output_dir}")


def plot_psnr_distribution(model, val_loader, device, output_dir):
    """
    Plot PSNR distribution histogram.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device
        output_dir: Output directory
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psnr_list = []

    with torch.no_grad():
        for y, x_gt in val_loader:
            y = y.to(device)
            x_gt = x_gt.to(device)

            x_pred = model(y)

            for i in range(x_pred.shape[0]):
                psnr = calculate_psnr(x_pred[i], x_gt[i])
                psnr_list.append(psnr)

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(psnr_list, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(psnr_list), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(psnr_list):.2f} dB')
    ax.axvline(np.median(psnr_list), color='g', linestyle='--', linewidth=2,
               label=f'Median: {np.median(psnr_list):.2f} dB')

    ax.set_xlabel('PSNR (dB)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('PSNR Distribution on Validation Set', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'psnr_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'psnr_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PSNR distribution to {output_dir}")
    print(f"  Mean PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"  Std PSNR: {np.std(psnr_list):.2f} dB")
    print(f"  Min PSNR: {np.min(psnr_list):.2f} dB")
    print(f"  Max PSNR: {np.max(psnr_list):.2f} dB")


def main():
    """Main visualization function."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize OAM radar imaging results')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to visualize')
    args = parser.parse_args()

    # Configuration
    IMAGE_SIZE = 24
    OAM_MODES = (-3, 3)
    BATCH_SIZE = 64
    MODEL_PATH = args.checkpoint_path
    HISTORY_PATH = str(Path(args.checkpoint_path).parent / 'training_history.json')
    OUTPUT_DIR = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate observation matrix
    print("\n[1/4] Generating observation matrix...")
    radar = OAMRadarGeometry(image_size=IMAGE_SIZE, oam_modes=OAM_MODES)
    Phi = radar.generate_observation_matrix()

    # Create validation dataset
    print("\n[2/4] Creating validation dataset...")
    val_dataset = SparseTargetDataset(
        Phi=Phi,
        image_size=IMAGE_SIZE,
        num_samples=1000,
        noise_std=0.001,
        augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    print("\n[3/4] Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = EndToEndReconstructionNet(
        measurement_dim=Phi.shape[0],
        image_size=IMAGE_SIZE,
        hidden_dim=checkpoint['config']['hidden_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with PSNR: {checkpoint['psnr']:.2f} dB")

    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    plot_reconstruction_results(model, val_loader, device, OUTPUT_DIR, num_samples=8)
    plot_training_curves(HISTORY_PATH, OUTPUT_DIR)
    plot_psnr_distribution(model, val_loader, device, OUTPUT_DIR)

    print(f"\n✓ All visualizations saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - reconstruction_results.png/pdf")
    print("  - training_curves.png/pdf")
    print("  - psnr_distribution.png/pdf")


if __name__ == "__main__":
    main()
