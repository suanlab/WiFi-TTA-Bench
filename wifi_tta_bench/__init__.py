"""WiFi-TTA-Bench: A Benchmark for TTA Under Physics-Structured Shift.

Quick start:
    from wifi_tta_bench import load_dataset, list_methods, evaluate

    dataset = load_dataset("widar_bvp")
    results = evaluate("tent", dataset, seeds=5)
    print(results["mean_gain"], results["ci"])
"""

from wifi_tta_bench.api import evaluate, list_datasets, list_methods, load_dataset

__all__ = ["load_dataset", "list_datasets", "list_methods", "evaluate"]
__version__ = "0.1.0"
