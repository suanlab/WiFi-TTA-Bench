# pyright: basic, reportUnknownVariableType=false
"""Data loading and preprocessing."""

from pinn4csi.data.csi_dataset import (
    CSIDataset,
    cross_environment_split,
    train_val_test_split,
)
from pinn4csi.data.esp32_loader import (
    ESP32CaptureMetadata,
    load_esp32_prepared_dataset,
)
from pinn4csi.data.paper1 import (
    PAPER1_DATASET_REGISTRY,
    Paper1DatasetConfig,
    Paper1Splits,
    PreparedCSIBundle,
    PreparedCSIDataset,
    create_mock_paper1_prepared_data,
    create_paper1_splits,
    get_paper1_dataset_config,
    list_paper1_datasets,
    load_array_file,
    load_prepared_paper1_dataset,
    save_prepared_paper1_dataset,
)
from pinn4csi.data.prepared_collection import (
    CollectionMetadataBase,
    CollectionRepresentation,
    PreparedCollectionBundle,
)
from pinn4csi.data.transforms import (
    amplitude_phase_split,
    has_meaningful_phase,
    normalize_amplitude,
    select_subcarriers,
)
from pinn4csi.data.wifi6_loader import (
    WiFi6CaptureMetadata,
    load_wifi6_prepared_dataset,
)
from pinn4csi.data.wifi_imaging_dataset import (
    CoordinateNormalizationStats,
    FieldNormalizationStats,
    WiFiImagingDataset,
    WiFiImagingMetadata,
    WiFiImagingPreparedBundle,
    WiFiImagingRepresentation,
    WiFiImagingSplits,
    create_wifi_imaging_splits,
    load_wifi_imaging_prepared_dataset,
)

__all__ = [
    "CSIDataset",
    "train_val_test_split",
    "cross_environment_split",
    "CollectionRepresentation",
    "CollectionMetadataBase",
    "PreparedCollectionBundle",
    "ESP32CaptureMetadata",
    "WiFi6CaptureMetadata",
    "load_esp32_prepared_dataset",
    "load_wifi6_prepared_dataset",
    "Paper1DatasetConfig",
    "PreparedCSIBundle",
    "PreparedCSIDataset",
    "Paper1Splits",
    "PAPER1_DATASET_REGISTRY",
    "get_paper1_dataset_config",
    "load_array_file",
    "list_paper1_datasets",
    "load_prepared_paper1_dataset",
    "create_paper1_splits",
    "create_mock_paper1_prepared_data",
    "save_prepared_paper1_dataset",
    "amplitude_phase_split",
    "has_meaningful_phase",
    "normalize_amplitude",
    "select_subcarriers",
    "WiFiImagingRepresentation",
    "CoordinateNormalizationStats",
    "FieldNormalizationStats",
    "WiFiImagingMetadata",
    "WiFiImagingPreparedBundle",
    "WiFiImagingDataset",
    "WiFiImagingSplits",
    "load_wifi_imaging_prepared_dataset",
    "create_wifi_imaging_splits",
]
