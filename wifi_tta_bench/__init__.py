"""WiFi-TTA-Bench: a harm-aware benchmark for test-time adaptation on WiFi CSI.

Quick start:
    from wifi_tta_bench import load_dataset, list_methods, evaluate

    dataset = load_dataset("widar_bvp")
    result = evaluate("tent", dataset, seeds=5)
    print(result.mean_gain, result.ci_lower, result.ci_upper,
          result.negative_adaptation_rate)
"""

from wifi_tta_bench.api import evaluate, list_datasets, list_methods, load_dataset

__all__ = ["load_dataset", "list_datasets", "list_methods", "evaluate"]
__version__ = "2.0.0"
