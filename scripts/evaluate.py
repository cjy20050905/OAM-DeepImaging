"""
OAM Radar Imaging - Evaluation Script

This script evaluates the trained model and compares with baseline methods.

Usage:
    python evaluate.py --model_path outputs/best_model.pth

Author: Dryoung
Date: 2026-03-06
License: MIT
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from physics import OAMRadarGeometry
from dataset import SparseTargetDataset
from model import EndToEndReconstructionNet


def calculate_metrics(pred, target):
    """
    Calculate multiple evaluation metrics.

    Args:
        pred: Predicted image
        target: Ground truth image

    Returns:
        dict: Dictionary of metrics
    """
    # PSNR
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)).item()

    # SSIM (simplified version)
    # For full SSIM, use pytorch-msssim library

    # MAE
    mae = torch.mean(torch.abs(pred - target)).item()

    # RMSE
    rmse = torch.sqrt(mse).item()

    return {
        'psnr': psnr,
        'mae': mae,
        'rmse': rmse
    }


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device

    Returns:
        dict: Evaluation results
    """
    model.eval()

    all_psnr = []
    all_mae = []
    all_rmse = []

    with torch.no_grad():
        for y, x_gt in tqdm(dataloader, desc="Evaluating"):
            y = y.to(device)
            x_gt = x_gt.to(device)

            x_pred = model(y)

            for i in range(x_pred.shape[0]):
                metrics = calculate_metrics(x_pred[i], x_gt[i])
                all_psnr.append(metrics['psnr'])
                all_mae.append(metrics['mae'])
                all_rmse.append(metrics['rmse'])

    results = {
        'psnr': {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
            'min': np.min(all_psnr),
            'max': np.max(all_psnr),
            'median': np.median(all_psnr)
        },
        'mae': {
            'mean': np.mean(all_mae),
            'std': np.std(all_mae)
        },
        'rmse': {
            'mean': np.mean(all_rmse),
            'std': np.std(all_rmse)
        }
    }

    return results


def main():
    """Main evaluation function."""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate OAM radar imaging model')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--output_path', type=str, default='results/evaluation_metrics.json',
                        help='Path to save evaluation results')
    args = parser.parse_args()

    # Configuration
    IMAGE_SIZE = 24
    OAM_MODES = (-3, 3)
    BATCH_SIZE = 64
    MODEL_PATH = args.checkpoint_path
    OUTPUT_PATH = Path(args.output_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate observation matrix
    print("\n[1/3] Generating observation matrix...")
    radar = OAMRadarGeometry(image_size=IMAGE_SIZE, oam_modes=OAM_MODES)
    Phi = radar.generate_observation_matrix()

    # Create test dataset
    print("\n[2/3] Creating test dataset...")
    test_dataset = SparseTargetDataset(
        Phi=Phi,
        image_size=IMAGE_SIZE,
        num_samples=args.num_samples,
        noise_std=0.001,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    print("\n[3/3] Loading and evaluating model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = EndToEndReconstructionNet(
        measurement_dim=Phi.shape[0],
        image_size=IMAGE_SIZE,
        hidden_dim=checkpoint['config']['hidden_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nPSNR:")
    print(f"  Mean:   {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
    print(f"  Median: {results['psnr']['median']:.2f} dB")
    print(f"  Range:  [{results['psnr']['min']:.2f}, {results['psnr']['max']:.2f}] dB")

    print(f"\nMAE:")
    print(f"  Mean:   {results['mae']['mean']:.4f} ± {results['mae']['std']:.4f}")

    print(f"\nRMSE:")
    print(f"  Mean:   {results['rmse']['mean']:.4f} ± {results['rmse']['std']:.4f}")

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
