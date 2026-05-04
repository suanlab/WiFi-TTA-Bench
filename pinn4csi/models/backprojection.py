"""Classical backprojection baseline for WiFi imaging.

Implements a simple, non-learnable backprojection method that reconstructs
field values at query coordinates from CSI measurements at TX/RX pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ClassicalBackprojection(nn.Module):
    """Classical backprojection baseline for WiFi field reconstruction.

    This is a non-learnable baseline that reconstructs field values at query
    coordinates by weighted superposition of CSI measurements from TX/RX pairs.
    The weight is inversely proportional to distance (1/r weighting).

    No learnable parameters are used; the method is purely geometric.
    """

    def __init__(self) -> None:
        """Initialize classical backprojection baseline.

        This baseline has no learnable parameters.
        """
        super().__init__()

    def forward(
        self,
        csi_features: Tensor,
        tx_rx_positions: Tensor,
        query_coordinates: Tensor,
    ) -> dict[str, Tensor]:
        """Reconstruct field and permittivity at query coordinates.

        Args:
            csi_features: CSI feature vector per sample.
                Shape: (batch, num_pairs * pair_feature_dim).
                For imaging, typically flattened amplitude features.
            tx_rx_positions: TX/RX antenna positions.
                Shape: (batch, num_pairs, 2, coordinate_dim).
                Encodes transmitter and receiver coordinates for each pair.
            query_coordinates: Spatial query points.
                Shape: (batch, num_points, coordinate_dim).

        Returns:
            Dictionary with keys:
            - `field`: Reconstructed field values. Shape: (batch, num_points).
            - `permittivity`: Constant permittivity estimate (1.0 baseline).
                Shape: (batch, num_points).

        Raises:
            ValueError: If input shapes are incompatible.
        """
        self._validate_inputs(
            csi_features=csi_features,
            tx_rx_positions=tx_rx_positions,
            query_coordinates=query_coordinates,
        )

        batch_size, num_pairs, _, coordinate_dim = tx_rx_positions.shape
        _, num_points, _ = query_coordinates.shape

        # Reshape CSI features: (batch, num_pairs * pair_feature_dim)
        # -> (batch, num_pairs, pair_feature_dim)
        pair_feature_dim = csi_features.shape[1] // num_pairs
        csi_reshaped = csi_features.reshape(batch_size, num_pairs, pair_feature_dim)

        # Use first feature (amplitude) as the measurement magnitude
        csi_magnitude = csi_reshaped[..., 0]  # Shape: (batch, num_pairs)

        # Compute midpoint of each TX/RX pair
        tx_positions = tx_rx_positions[:, :, 0, :]  # (batch, num_pairs, coord_dim)
        rx_positions = tx_rx_positions[:, :, 1, :]  # (batch, num_pairs, coord_dim)
        pair_midpoints = (tx_positions + rx_positions) / 2.0  # (batch, num_pairs, d)

        # Compute distances from each query point to each pair midpoint
        # query_coordinates: (batch, num_points, coordinate_dim)
        # pair_midpoints: (batch, num_pairs, coordinate_dim)
        # distances: (batch, num_points, num_pairs)
        query_expanded = query_coordinates.unsqueeze(2)  # (batch, num_points, 1, d)
        midpoints_expanded = pair_midpoints.unsqueeze(1)  # (batch, 1, num_pairs, d)
        delta = query_expanded - midpoints_expanded  # (batch, num_points, pairs, d)
        distances = torch.norm(delta, dim=-1)  # (batch, num_points, num_pairs)

        # Avoid division by zero: add small epsilon
        distances = torch.clamp(distances, min=1e-6)

        # Compute inverse-distance weights: w_ij = 1 / d_ij
        weights = 1.0 / distances  # Shape: (batch, num_points, num_pairs)

        # Normalize weights per query point
        weight_sum = weights.sum(dim=-1, keepdim=True)  # (batch, num_points, 1)
        normalized_weights = weights / (weight_sum + 1e-8)  # (batch, num_points, pairs)

        # Backproject: field(q) = sum_i w_i * csi_magnitude_i
        csi_magnitude_expanded = csi_magnitude.unsqueeze(1)  # (batch, 1, pairs)
        field = (normalized_weights * csi_magnitude_expanded).sum(dim=-1)  # (batch, q)

        # Permittivity: constant baseline (no estimation from CSI)
        permittivity = torch.ones_like(field)  # (batch, num_points)

        return {
            "field": field,
            "permittivity": permittivity,
        }

    def _validate_inputs(
        self,
        csi_features: Tensor,
        tx_rx_positions: Tensor,
        query_coordinates: Tensor,
    ) -> None:
        """Validate input tensor shapes and compatibility.

        Args:
            csi_features: Shape (batch, num_pairs * pair_feature_dim).
            tx_rx_positions: Shape (batch, num_pairs, 2, coordinate_dim).
            query_coordinates: Shape (batch, num_points, coordinate_dim).

        Raises:
            ValueError: If shapes are incompatible.
        """
        if csi_features.dim() != 2:
            raise ValueError(
                "csi_features must have shape (batch, num_pairs * pair_feature_dim), "
                f"got {tuple(csi_features.shape)}"
            )

        if tx_rx_positions.dim() != 4:
            raise ValueError(
                "tx_rx_positions must have shape "
                "(batch, num_pairs, 2, coordinate_dim), "
                f"got {tuple(tx_rx_positions.shape)}"
            )

        if query_coordinates.dim() != 3:
            raise ValueError(
                "query_coordinates must have shape "
                "(batch, num_points, coordinate_dim), "
                f"got {tuple(query_coordinates.shape)}"
            )

        batch_size_csi = csi_features.shape[0]
        batch_size_tx_rx = tx_rx_positions.shape[0]
        batch_size_query = query_coordinates.shape[0]

        if not (batch_size_csi == batch_size_tx_rx == batch_size_query):
            raise ValueError(
                "csi_features, tx_rx_positions, and query_coordinates "
                "must share batch size. "
                f"Got {batch_size_csi}, {batch_size_tx_rx}, {batch_size_query}"
            )

        num_pairs_tx_rx = tx_rx_positions.shape[1]
        pair_feature_dim = csi_features.shape[1] // num_pairs_tx_rx

        if csi_features.shape[1] != num_pairs_tx_rx * pair_feature_dim:
            raise ValueError(
                "csi_features size must be divisible by num_pairs. "
                f"Got csi_features.shape[1]={csi_features.shape[1]}, "
                f"num_pairs={num_pairs_tx_rx}"
            )

        if tx_rx_positions.shape[2] != 2:
            raise ValueError(
                "tx_rx_positions must encode transmitter and receiver (dim 2 = 2), "
                f"got {tx_rx_positions.shape[2]}"
            )

        coordinate_dim_tx_rx = tx_rx_positions.shape[3]
        coordinate_dim_query = query_coordinates.shape[2]

        if coordinate_dim_tx_rx != coordinate_dim_query:
            raise ValueError(
                "tx_rx_positions and query_coordinates must share "
                "coordinate dimension. "
                f"Got {coordinate_dim_tx_rx} and {coordinate_dim_query}"
            )
