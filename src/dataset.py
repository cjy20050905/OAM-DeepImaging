"""
OAM Radar Imaging - Dataset Module

This module implements the dataset generation for OAM radar sparse target imaging.
It generates synthetic sparse targets and simulates the forward measurement process.

Author: Dryoung
Date: 2026-03-06
License: MIT
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional


class SparseTargetDataset(Dataset):
    """
    Sparse Target Dataset for OAM Radar Imaging.

    This dataset generates synthetic sparse targets consisting of simple geometric shapes
    (circles and rectangles) and simulates the forward measurement process:
        y = Φx + noise

    Attributes:
        Phi (torch.Tensor): Observation matrix of shape [measurement_dim, image_dim]
        image_size (int): Size of the imaging grid (M×M)
        num_samples (int): Number of samples in the dataset
        noise_std (float): Standard deviation of additive Gaussian noise
        num_targets (Tuple[int, int]): Range of number of targets per image
        augment (bool): Whether to apply data augmentation
    """

    def __init__(
        self,
        Phi: torch.Tensor,
        image_size: int = 24,
        num_samples: int = 30000,
        noise_std: float = 0.001,
        num_targets: Tuple[int, int] = (1, 3),
        augment: bool = True
    ):
        """
        Initialize the sparse target dataset.

        Args:
            Phi: Observation matrix of shape [measurement_dim, image_dim]
            image_size: Size of the imaging grid (default: 24)
            num_samples: Number of samples in the dataset (default: 30000)
            noise_std: Standard deviation of additive noise (default: 0.001)
            num_targets: Range of number of targets per image (default: (1, 3))
            augment: Whether to apply data augmentation (default: True)
        """
        self.Phi = Phi
        self.M = image_size
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.num_targets = num_targets
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single sample.

        Args:
            idx: Sample index (not used, samples are generated randomly)

        Returns:
            Tuple containing:
                - y: Measurement vector of shape [measurement_dim], dtype complex64
                - x: Ground truth image of shape [M, M], dtype float32
        """
        # Generate sparse target image
        x = self._generate_sparse_target()

        # Apply data augmentation if enabled
        if self.augment:
            x = self._apply_augmentation(x)

        # Forward measurement process: y = Φx + noise
        x_vec = x.flatten().to(torch.complex64)
        y = torch.matmul(self.Phi, x_vec)

        # Add complex Gaussian noise
        noise_real = torch.randn_like(y.real) * self.noise_std
        noise_imag = torch.randn_like(y.imag) * self.noise_std
        noise = torch.complex(noise_real, noise_imag)
        y = y + noise

        return y, x

    def _generate_sparse_target(self) -> torch.Tensor:
        """
        Generate a sparse target image with random geometric shapes.

        Returns:
            torch.Tensor: Image of shape [M, M] with values in [0, 1]
        """
        # Initialize black background
        image = torch.zeros(self.M, self.M, dtype=torch.float32)

        # Randomly determine number of targets
        num_objs = np.random.randint(self.num_targets[0], self.num_targets[1] + 1)

        for _ in range(num_objs):
            shape_type = np.random.choice(['circle', 'rectangle'])

            if shape_type == 'circle':
                # Generate circular target
                cx = np.random.randint(6, self.M - 6)
                cy = np.random.randint(6, self.M - 6)
                radius = np.random.randint(3, 7)

                for i in range(self.M):
                    for j in range(self.M):
                        if (i - cy)**2 + (j - cx)**2 <= radius**2:
                            image[i, j] = 1.0

            elif shape_type == 'rectangle':
                # Generate rectangular target
                x1 = np.random.randint(0, self.M - 12)
                y1 = np.random.randint(0, self.M - 12)
                w = np.random.randint(6, 12)
                h = np.random.randint(6, 12)
                image[y1:y1+h, x1:x1+w] = 1.0

        return image

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation (rotation and flipping).

        Args:
            x: Input image of shape [M, M]

        Returns:
            torch.Tensor: Augmented image of shape [M, M]
        """
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        x = torch.rot90(x, k=k, dims=[0, 1])

        # Random horizontal flip
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[1])

        # Random vertical flip
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[0])

        return x


if __name__ == "__main__":
    # Example usage
    from physics import OAMRadarGeometry

    # Generate observation matrix
    radar = OAMRadarGeometry(image_size=24)
    Phi = radar.generate_observation_matrix()

    # Create dataset
    dataset = SparseTargetDataset(
        Phi=Phi,
        image_size=24,
        num_samples=100,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    y, x = dataset[0]
    print(f"Measurement shape: {y.shape}, dtype: {y.dtype}")
    print(f"Image shape: {x.shape}, dtype: {x.dtype}")
    print(f"Number of non-zero pixels: {torch.count_nonzero(x).item()}")
