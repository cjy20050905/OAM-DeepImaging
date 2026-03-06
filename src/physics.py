"""
OAM Radar Imaging - Physical Model Module

This module implements the physical model for Orbital Angular Momentum (OAM) radar imaging.
It generates the observation matrix based on electromagnetic wave propagation theory.

Author: Dryoung
Date: 2026-03-06
License: MIT
"""

import numpy as np
import torch
from scipy.special import jv
from typing import Tuple, Optional


class OAMRadarGeometry:
    """
    OAM Radar Geometry Configuration and Observation Matrix Generation.

    This class implements the physical model for OAM radar imaging, including:
    - Bessel function computation for OAM modes
    - Phase encoding for vortex beams
    - Observation matrix generation with normalization

    Attributes:
        image_size (int): Size of the imaging grid (M×M)
        oam_modes (Tuple[int, int]): Range of OAM mode indices (l_min, l_max)
        array_radius (float): Radius of the receiver array in meters
        wavelength (float): Electromagnetic wavelength in meters
        target_distance (float): Distance to target plane in meters
        num_receivers (int): Number of receiver antennas
    """

    def __init__(
        self,
        image_size: int = 24,
        oam_modes: Tuple[int, int] = (-3, 3),
        array_radius: float = 0.5,
        wavelength: float = 0.03,
        target_distance: float = 10.0,
        num_receivers: int = 32
    ):
        """
        Initialize OAM radar geometry configuration.

        Args:
            image_size: Size of the imaging grid (default: 24)
            oam_modes: Range of OAM mode indices (default: (-3, 3))
            array_radius: Radius of receiver array in meters (default: 0.5)
            wavelength: Electromagnetic wavelength in meters (default: 0.03)
            target_distance: Distance to target plane in meters (default: 10.0)
            num_receivers: Number of receiver antennas (default: 32)
        """
        self.M = image_size
        self.l_min, self.l_max = oam_modes
        self.a = array_radius
        self.lam = wavelength
        self.R = target_distance
        self.N = num_receivers

        # Derived parameters
        self.k = 2 * np.pi / self.lam  # Wave number
        self.oam_list = list(range(self.l_min, self.l_max + 1))
        self.L = len(self.oam_list)

    def generate_observation_matrix(self) -> torch.Tensor:
        """
        Generate the observation matrix Φ ∈ C^{(L×N) × (M×M)}.

        The observation matrix relates the target image x to the received signal y:
            y = Φx + noise

        The matrix elements are computed based on:
            Φ_{(l,n), j} = J_l(ka·sinθ_j) · exp(i·l·(φ_j - φ_n))

        where:
            - J_l: Bessel function of the first kind
            - k: Wave number
            - a: Array radius
            - θ_j: Elevation angle to pixel j
            - φ_j: Azimuth angle to pixel j
            - φ_n: Azimuth angle of receiver n

        Returns:
            torch.Tensor: Complex observation matrix of shape [L*N, M*M]
        """
        # Generate target grid coordinates (Cartesian)
        x_coords = np.linspace(-1, 1, self.M)
        y_coords = np.linspace(-1, 1, self.M)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Convert to polar coordinates
        rho = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)  # Azimuth angle [-π, π]

        # Compute elevation angle
        theta = np.arctan(rho / self.R)

        # Receiver array angular positions
        receiver_angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)

        # Initialize observation matrix
        Phi = np.zeros((self.L * self.N, self.M * self.M), dtype=np.complex64)

        # Compute matrix elements for each OAM mode and receiver
        for l_idx, l in enumerate(self.oam_list):
            for n_idx, phi_n in enumerate(receiver_angles):
                row_idx = l_idx * self.N + n_idx

                # Bessel function J_l(ka·sinθ)
                bessel_arg = self.k * self.a * np.sin(theta)
                J_l = jv(l, bessel_arg)

                # Phase term exp(i·l·(φ - φ_n))
                phase = np.exp(1j * l * (phi - phi_n))

                # Combine terms
                Phi[row_idx, :] = (J_l * phase).flatten()

        # Normalize each row to unit norm (critical for numerical stability)
        for i in range(Phi.shape[0]):
            row_norm = np.linalg.norm(Phi[i, :])
            if row_norm > 1e-10:
                Phi[i, :] = Phi[i, :] / row_norm

        # Convert to PyTorch tensor
        Phi_torch = torch.from_numpy(Phi)

        return Phi_torch

    def get_measurement_dimension(self) -> int:
        """Get the dimension of measurement vector (L×N)."""
        return self.L * self.N

    def get_image_dimension(self) -> int:
        """Get the dimension of image vector (M×M)."""
        return self.M * self.M

    def get_compression_ratio(self) -> float:
        """Get the compression ratio (measurement_dim / image_dim)."""
        return self.get_measurement_dimension() / self.get_image_dimension()

    def __repr__(self) -> str:
        """String representation of the radar geometry."""
        return (
            f"OAMRadarGeometry(\n"
            f"  image_size={self.M}×{self.M},\n"
            f"  oam_modes={self.l_min} to {self.l_max} ({self.L} modes),\n"
            f"  num_receivers={self.N},\n"
            f"  measurement_dim={self.get_measurement_dimension()},\n"
            f"  compression_ratio={self.get_compression_ratio()*100:.1f}%\n"
            f")"
        )


if __name__ == "__main__":
    # Example usage
    radar = OAMRadarGeometry(image_size=24, oam_modes=(-3, 3))
    print(radar)

    Phi = radar.generate_observation_matrix()
    print(f"\nObservation matrix shape: {Phi.shape}")
    print(f"Matrix dtype: {Phi.dtype}")
    print(f"Matrix norm: {torch.norm(Phi).item():.4f}")
