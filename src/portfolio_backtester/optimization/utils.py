"""Utility helpers for optimization layer."""
from __future__ import annotations

from typing import Dict, Any, Optional


def discrete_space_size(parameter_space: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Return the number of unique combinations if the space is fully discrete.

    Parameters comprised only of categorical choices or integer ranges with
    explicit *step* are considered discrete. Any float or integer parameter
    without a *step* (i.e. continuous sampling) renders the space effectively
    infinite and the function returns *None*.
    """
    size = 1
    for conf in parameter_space.values():
        ptype = conf.get("type", "float")
        if ptype == "categorical":
            size *= len(conf["choices"])
        elif ptype == "int" and "step" in conf:
            low = conf["low"]
            high = conf["high"]
            step = conf.get("step", 1)
            size *= (high - low) // step + 1
        else:
            # float parameter or int without step ⇒ continuous
            return None
    return size