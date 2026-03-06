"""
OAM Radar Imaging - Neural Network Model

This module implements the end-to-end deep learning model for OAM radar imaging.
The model directly learns the mapping from measurement vectors to target images.

Author: Dryoung
Date: 2026-03-06
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional


class EndToEndReconstructionNet(nn.Module):
    """
    End-to-End Reconstruction Network for OAM Radar Imaging.

    This network directly learns the mapping from complex measurement vectors
    to real-valued target images without relying on physical model iterations.

    Architecture:
        1. Complex-to-Real Feature Extraction (Fully Connected)
        2. Feature Transformation (Fully Connected with Dropout)
        3. Image Decoder (Transposed Convolutions + Refinement)

    Attributes:
        measurement_dim (int): Dimension of input measurement vector
        image_size (int): Size of output image (M×M)
        hidden_dim (int): Dimension of hidden feature space
    """

    def __init__(
        self,
        measurement_dim: int = 224,
        image_size: int = 24,
        hidden_dim: int = 768
    ):
        """
        Initialize the end-to-end reconstruction network.

        Args:
            measurement_dim: Dimension of measurement vector (default: 224)
            image_size: Size of output image (default: 24)
            hidden_dim: Dimension of hidden features (default: 768)
        """
        super().__init__()

        self.measurement_dim = measurement_dim
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        # Complex-to-Real Feature Extraction
        # Input: [batch, measurement_dim * 2] (real and imaginary parts)
        # Output: [batch, hidden_dim]
        self.feature_extractor = nn.Sequential(
            nn.Linear(measurement_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Feature to Spatial Transformation
        # Output: [batch, 64, 6, 6]
        self.fc_to_spatial = nn.Linear(hidden_dim, 6 * 6 * 64)

        # Image Decoder (Upsampling Network)
        # 6×6 → 12×12 → 24×24
        self.decoder = nn.Sequential(
            # Upsample: 6×6 → 12×12
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Upsample: 12×12 → 24×24
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Refinement layers
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Output layer
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the reconstruction network.

        Args:
            y: Complex measurement vector of shape [batch, measurement_dim]

        Returns:
            torch.Tensor: Reconstructed image of shape [batch, image_size, image_size]
        """
        batch_size = y.shape[0]

        # Split complex measurements into real and imaginary parts
        # [batch, measurement_dim] → [batch, measurement_dim * 2]
        y_real = torch.cat([y.real, y.imag], dim=1)

        # Extract features
        # [batch, measurement_dim * 2] → [batch, hidden_dim]
        features = self.feature_extractor(y_real)

        # Transform to spatial features
        # [batch, hidden_dim] → [batch, 6*6*64]
        spatial_features = self.fc_to_spatial(features)

        # Reshape to 4D tensor
        # [batch, 6*6*64] → [batch, 64, 6, 6]
        spatial_features = spatial_features.view(batch_size, 64, 6, 6)

        # Decode to image
        # [batch, 64, 6, 6] → [batch, 1, 24, 24]
        output = self.decoder(spatial_features)

        # Remove channel dimension
        # [batch, 1, 24, 24] → [batch, 24, 24]
        output = output.squeeze(1)

        return output

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        num_params = self.count_parameters()
        return (
            f"EndToEndReconstructionNet(\n"
            f"  measurement_dim={self.measurement_dim},\n"
            f"  image_size={self.image_size}×{self.image_size},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  trainable_parameters={num_params:,}\n"
            f")"
        )


class CombinedLoss(nn.Module):
    """
    Combined Loss Function: L1 + L2.

    This loss function combines L1 loss (promoting sparsity) and L2 loss
    (promoting smoothness) for better reconstruction quality.

    Attributes:
        alpha (float): Weight for L1 loss (1-alpha for L2 loss)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize the combined loss function.

        Args:
            alpha: Weight for L1 loss, range [0, 1] (default: 0.5)
        """
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            pred: Predicted image of shape [batch, H, W]
            target: Ground truth image of shape [batch, H, W]

        Returns:
            torch.Tensor: Scalar loss value
        """
        return self.alpha * self.l1_loss(pred, target) + (1 - self.alpha) * self.l2_loss(pred, target)


if __name__ == "__main__":
    # Example usage
    model = EndToEndReconstructionNet(
        measurement_dim=224,
        image_size=24,
        hidden_dim=768
    )
    print(model)

    # Test forward pass
    batch_size = 4
    y = torch.randn(batch_size, 224, dtype=torch.complex64)
    x_pred = model(y)
    print(f"\nInput shape: {y.shape}")
    print(f"Output shape: {x_pred.shape}")

    # Test loss
    x_true = torch.rand(batch_size, 24, 24)
    criterion = CombinedLoss(alpha=0.5)
    loss = criterion(x_pred, x_true)
    print(f"Loss: {loss.item():.6f}")
