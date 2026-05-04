"""Unit tests for classical backprojection baseline."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from pinn4csi.models.backprojection import ClassicalBackprojection


class TestClassicalBackprojection:
    """Test suite for ClassicalBackprojection baseline."""

    @pytest.fixture
    def model(self) -> ClassicalBackprojection:
        """Create a backprojection model instance."""
        return ClassicalBackprojection()

    @pytest.fixture
    def sample_inputs(self) -> dict[str, Tensor]:
        """Create sample inputs for testing.

        Returns:
            Dictionary with:
            - csi_features: (batch=2, num_pairs * pair_feature_dim=12)
            - tx_rx_positions: (batch=2, num_pairs=3, 2, coordinate_dim=2)
            - query_coordinates: (batch=2, num_points=5, coordinate_dim=2)
        """
        batch_size = 2
        num_pairs = 3
        pair_feature_dim = 4
        num_points = 5
        coordinate_dim = 2

        csi_features = torch.randn(batch_size, num_pairs * pair_feature_dim)
        tx_rx_positions = torch.randn(batch_size, num_pairs, 2, coordinate_dim)
        query_coordinates = torch.randn(batch_size, num_points, coordinate_dim)

        return {
            "csi_features": csi_features,
            "tx_rx_positions": tx_rx_positions,
            "query_coordinates": query_coordinates,
        }

    def test_forward_output_shapes(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that forward pass produces correct output shapes."""
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        batch_size = sample_inputs["csi_features"].shape[0]
        num_points = sample_inputs["query_coordinates"].shape[1]

        assert "field" in outputs
        assert "permittivity" in outputs
        assert outputs["field"].shape == (batch_size, num_points)
        assert outputs["permittivity"].shape == (batch_size, num_points)

    def test_forward_output_dtypes(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that forward pass produces correct output dtypes."""
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        assert outputs["field"].dtype == torch.float32
        assert outputs["permittivity"].dtype == torch.float32

    def test_permittivity_is_ones(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that permittivity output is constant 1.0 (baseline)."""
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        assert torch.allclose(
            outputs["permittivity"],
            torch.ones_like(outputs["permittivity"]),
        )

    def test_field_is_finite(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that field output contains finite values."""
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        assert torch.isfinite(outputs["field"]).all()

    def test_field_is_normalized(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that field values are in reasonable range.

        Weighted average of CSI features should be roughly in [-1, 1].
        """
        # CSI features are standard normal, so weighted average should be ~[-1, 1]
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        # Field should be bounded by the range of CSI magnitudes
        field_max = outputs["field"].abs().max().item()
        assert field_max < 10.0  # Sanity check: not exploding

    def test_batch_independence(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that batch samples are processed independently."""
        outputs = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        # Process each batch sample separately
        batch_size = sample_inputs["csi_features"].shape[0]
        for i in range(batch_size):
            single_output = model(
                csi_features=sample_inputs["csi_features"][i : i + 1],
                tx_rx_positions=sample_inputs["tx_rx_positions"][i : i + 1],
                query_coordinates=sample_inputs["query_coordinates"][i : i + 1],
            )
            assert torch.allclose(
                outputs["field"][i : i + 1],
                single_output["field"],
                atol=1e-5,
            )

    def test_invalid_csi_features_shape(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that invalid CSI features shape raises ValueError."""
        invalid_csi = torch.randn(2, 3, 4)  # Wrong: 3D instead of 2D
        with pytest.raises(ValueError, match="csi_features must have shape"):
            model(
                csi_features=invalid_csi,
                tx_rx_positions=sample_inputs["tx_rx_positions"],
                query_coordinates=sample_inputs["query_coordinates"],
            )

    def test_invalid_tx_rx_positions_shape(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that invalid TX/RX positions shape raises ValueError."""
        invalid_tx_rx = torch.randn(2, 3, 2)  # Wrong: 3D instead of 4D
        with pytest.raises(ValueError, match="tx_rx_positions must have shape"):
            model(
                csi_features=sample_inputs["csi_features"],
                tx_rx_positions=invalid_tx_rx,
                query_coordinates=sample_inputs["query_coordinates"],
            )

    def test_invalid_query_coordinates_shape(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that invalid query coordinates shape raises ValueError."""
        invalid_query = torch.randn(2, 5)  # Wrong: 2D instead of 3D
        with pytest.raises(ValueError, match="query_coordinates must have shape"):
            model(
                csi_features=sample_inputs["csi_features"],
                tx_rx_positions=sample_inputs["tx_rx_positions"],
                query_coordinates=invalid_query,
            )

    def test_batch_size_mismatch(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that batch size mismatch raises ValueError."""
        mismatched_csi = torch.randn(3, 12)  # Different batch size
        with pytest.raises(ValueError, match="must share batch size"):
            model(
                csi_features=mismatched_csi,
                tx_rx_positions=sample_inputs["tx_rx_positions"],
                query_coordinates=sample_inputs["query_coordinates"],
            )

    def test_coordinate_dim_mismatch(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that coordinate dimension mismatch raises ValueError."""
        mismatched_query = torch.randn(2, 5, 3)  # Different coordinate dim
        with pytest.raises(ValueError, match="must share coordinate dimension"):
            model(
                csi_features=sample_inputs["csi_features"],
                tx_rx_positions=sample_inputs["tx_rx_positions"],
                query_coordinates=mismatched_query,
            )

    def test_no_learnable_parameters(
        self,
        model: ClassicalBackprojection,
    ) -> None:
        """Test that the model has no learnable parameters."""
        params = list(model.parameters())
        assert len(params) == 0, (
            "ClassicalBackprojection should have no learnable parameters"
        )

    def test_deterministic_output(
        self,
        model: ClassicalBackprojection,
        sample_inputs: dict[str, Tensor],
    ) -> None:
        """Test that the same inputs produce the same outputs."""
        output1 = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )
        output2 = model(
            csi_features=sample_inputs["csi_features"],
            tx_rx_positions=sample_inputs["tx_rx_positions"],
            query_coordinates=sample_inputs["query_coordinates"],
        )

        assert torch.allclose(output1["field"], output2["field"])
        assert torch.allclose(output1["permittivity"], output2["permittivity"])

    def test_single_pair(
        self,
        model: ClassicalBackprojection,
    ) -> None:
        """Test with a single TX/RX pair."""
        batch_size = 1
        num_pairs = 1
        pair_feature_dim = 4
        num_points = 3
        coordinate_dim = 2

        csi_features = torch.randn(batch_size, num_pairs * pair_feature_dim)
        tx_rx_positions = torch.randn(batch_size, num_pairs, 2, coordinate_dim)
        query_coordinates = torch.randn(batch_size, num_points, coordinate_dim)

        outputs = model(
            csi_features=csi_features,
            tx_rx_positions=tx_rx_positions,
            query_coordinates=query_coordinates,
        )

        assert outputs["field"].shape == (batch_size, num_points)
        assert outputs["permittivity"].shape == (batch_size, num_points)

    def test_3d_coordinates(
        self,
        model: ClassicalBackprojection,
    ) -> None:
        """Test with 3D spatial coordinates."""
        batch_size = 2
        num_pairs = 3
        pair_feature_dim = 4
        num_points = 5
        coordinate_dim = 3

        csi_features = torch.randn(batch_size, num_pairs * pair_feature_dim)
        tx_rx_positions = torch.randn(batch_size, num_pairs, 2, coordinate_dim)
        query_coordinates = torch.randn(batch_size, num_points, coordinate_dim)

        outputs = model(
            csi_features=csi_features,
            tx_rx_positions=tx_rx_positions,
            query_coordinates=query_coordinates,
        )

        assert outputs["field"].shape == (batch_size, num_points)
        assert outputs["permittivity"].shape == (batch_size, num_points)

    def test_zero_csi_features(
        self,
        model: ClassicalBackprojection,
    ) -> None:
        """Test with zero CSI features (should produce zero field)."""
        batch_size = 1
        num_pairs = 2
        pair_feature_dim = 4
        num_points = 3
        coordinate_dim = 2

        csi_features = torch.zeros(batch_size, num_pairs * pair_feature_dim)
        tx_rx_positions = torch.randn(batch_size, num_pairs, 2, coordinate_dim)
        query_coordinates = torch.randn(batch_size, num_points, coordinate_dim)

        outputs = model(
            csi_features=csi_features,
            tx_rx_positions=tx_rx_positions,
            query_coordinates=query_coordinates,
        )

        assert torch.allclose(outputs["field"], torch.zeros_like(outputs["field"]))
