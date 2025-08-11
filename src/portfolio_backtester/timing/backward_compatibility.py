"""
Backward compatibility for timing configurations.
"""

from typing import Dict, Any


def ensure_backward_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures backward compatibility for timing configurations.
    """
    if "rebalance_frequency" in config and "timing_config" not in config:
        config["timing_config"] = {
            "mode": "time_based",
            "rebalance_frequency": config.pop("rebalance_frequency"),
        }
    return config
